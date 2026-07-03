//! Table policy: act from the trained blueprint, with an optional
//! time-budgeted online subgame resolve for postflop decisions
//! (`play --search`).
//!
//! The resolve samples hidden cards from Bayes-tracked ranges when a
//! RangeTracker is provided (uniformly otherwise). Flop subgames are
//! depth-limited to the end of the street with biased-continuation leaf
//! values; river spots with two live players use the exact range-vs-range
//! CFR+ solver in `river.rs`; everything else solves to the end of the hand
//! with MCCFR.

use crate::abstraction::{AbsAction, Abstraction};
use crate::cfr::{sample_index, Blueprint, LeafCfg, TrainConfig, Trainer};
use crate::engine::{Hand, Street};
use crate::search::RangeTracker;
use crate::valuenet::ValueNet;
use rand::rngs::SmallRng;
use std::sync::Arc;

/// Bet sizes used by the value-net flop solver (33%, 75%, 150% + all-in):
/// richer than the turn solver's data-generation menu, still small enough
/// to keep leaf-refresh network queries affordable.
const FLOP_MENU: [u8; 3] = [1, 3, 5];
/// Bet-size menu for exact turn resolves at the table (75% pot + all-in,
/// matching the value net's training menu): full menus put the turn+river
/// vector tree at tens of GB.
const TURN_MENU: [u8; 1] = [3];
/// Turn cards sampled per leaf refresh (of 49) — bounds net queries per
/// refresh at real-time budgets.
const FLOP_QUERY_TURNS: usize = 16;

pub struct Policy {
    pub blueprint: Arc<Blueprint>,
    pub abs: Arc<Abstraction>,
    /// Belief-state value net: enables ReBeL-style flop solving.
    pub value_net: Option<Arc<ValueNet>>,
}

#[derive(Debug, Clone, Copy)]
pub struct SearchParams {
    pub time_ms: u64,
    pub max_iters: u64,
    /// Model opponents as lambda-rational (logit QRE) instead of fully
    /// rational. None = solve toward equilibrium (default).
    pub qre_lambda: Option<f64>,
    /// Safe (gadget) river resolving: the opponent may take a blueprint
    /// safety value instead of entering the subgame, bounding how much
    /// resolving with wrong beliefs can be exploited.
    pub safe_resolve: bool,
    /// Metareasoning: probe with 1/8 of the budget and stop early when the
    /// hero's decision is already near-pure — trivial spots take
    /// milliseconds, the saved time is available to hard ones.
    pub adaptive: bool,
}

impl Default for SearchParams {
    fn default() -> Self {
        SearchParams {
            time_ms: 2_000,
            max_iters: 2_000_000,
            qre_lambda: None,
            safe_resolve: false,
            adaptive: false,
        }
    }
}

/// Early-exit purity threshold for adaptive search.
const ADAPTIVE_PURITY: f64 = 0.97;

/// Per-hand search state for continual resolving: the most recent exact
/// turn resolve's carry (opponent CFVs at river entries) plus where in the
/// action history that resolve was rooted. Reset every hand.
#[derive(Default)]
pub struct SearchSession {
    pub carry: Option<crate::turn::TurnCarry>,
    pub root_hist_len: usize,
}

impl SearchSession {
    pub fn new() -> SearchSession {
        SearchSession::default()
    }

    /// Carried gadget alternatives for a river resolve at `h` given the
    /// full table history: valid when the line since the turn resolve is on
    /// the solve's tree and the bot opens the river betting (one street
    /// separator, at the end — after an opponent river bet the entry CFVs
    /// no longer describe the resolve root).
    fn river_alt(&self, h: &Hand, hist: &[u8]) -> Option<Vec<f64>> {
        let carry = self.carry.as_ref()?;
        let suffix = hist.get(self.root_hist_len..)?;
        let (&last, line) = suffix.split_last()?;
        if last != crate::abstraction::TOKEN_STREET_SEP
            || line.contains(&crate::abstraction::TOKEN_STREET_SEP)
        {
            return None;
        }
        carry.alt_for(line, *h.board().get(4)?)
    }
}

impl Policy {
    pub fn new(blueprint: Blueprint, abs: Arc<Abstraction>) -> Self {
        Policy {
            blueprint: Arc::new(blueprint),
            abs,
            value_net: None,
        }
    }

    pub fn with_value_net(mut self, net: Option<Arc<ValueNet>>) -> Self {
        self.value_net = net;
        self
    }

    /// Cheap clone (shared Arcs) with the value net removed — for paired
    /// with-net vs without-net comparisons.
    pub fn clone_without_net(&self) -> Policy {
        Policy {
            blueprint: self.blueprint.clone(),
            abs: self.abs.clone(),
            value_net: None,
        }
    }

    /// The blueprint's full action distribution at the current infoset —
    /// exactly what `act_blueprint` samples from, including its check/call
    /// fallback on unseen or mismatched infosets.
    pub fn blueprint_dist(&self, h: &Hand, hist: &[u8], rng: &mut SmallRng) -> (Vec<AbsAction>, Vec<f64>) {
        let p = h.to_act();
        let acts = self.abs.abstract_actions(h);
        let bucket = self.abs.bucket(h.hole(p), h.board(), rng);
        if let Some(s) = self.blueprint.get(bucket, hist) {
            if s.len() == acts.len() {
                let probs: Vec<f64> = s.iter().map(|&x| x as f64).collect();
                let total: f64 = probs.iter().sum();
                if total > 0.0 {
                    let norm: Vec<f64> = probs.iter().map(|x| x / total).collect();
                    return (acts, norm);
                }
            }
        }
        let mut probs = vec![0.0; acts.len()];
        let call = acts
            .iter()
            .position(|&a| a == AbsAction::CheckCall)
            .expect("check/call is always legal");
        probs[call] = 1.0;
        (acts, probs)
    }

    /// Pick an abstract action for the player to act on `h` from the
    /// blueprint. `hist` is the abstract action history (tokens).
    /// Falls back to check/call on infosets the blueprint never visited or
    /// whose action menu no longer matches.
    pub fn act_blueprint(&self, h: &Hand, hist: &[u8], rng: &mut SmallRng) -> AbsAction {
        let (acts, probs) = self.blueprint_dist(h, hist, rng);
        acts[sample_index(&probs, rng)]
    }

    /// Resolve the current subgame with MCCFR for a time/iteration budget and
    /// act from the resolved strategy; falls back to the blueprint.
    /// Preflop always plays the blueprint (as Pluribus did). With a
    /// RangeTracker, opponents' hidden cards are sampled from their tracked
    /// ranges; flop subgames are depth-limited to the end of the street with
    /// biased-continuation leaf values.
    ///
    /// Subgames are rooted at `real` — the true table state — so an
    /// opponent's off-tree bet is re-solved at its actual size (nested
    /// re-solving) instead of being priced at the nearest abstract size.
    /// `shadow` (the on-tree mirror) is used only for blueprint infoset
    /// lookups; pass the same hand twice when they cannot diverge.
    pub fn act_with_search(
        &self,
        real: &Hand,
        shadow: &Hand,
        hist: &[u8],
        params: SearchParams,
        train_cfg: &TrainConfig,
        tracker: Option<&RangeTracker>,
        session: Option<&mut SearchSession>,
        rng: &mut SmallRng,
    ) -> AbsAction {
        let (acts, probs) =
            self.search_dist(real, shadow, hist, params, train_cfg, tracker, session, rng);
        acts[sample_index(&probs, rng)]
    }

    /// The full action distribution the search would act from — the routing
    /// behind `act_with_search`, exposed for distillation (the resolved
    /// distribution is the teacher signal, not just a sampled action).
    /// Falls back to the blueprint distribution when no solver claims the
    /// spot.
    pub fn search_dist(
        &self,
        real: &Hand,
        shadow: &Hand,
        hist: &[u8],
        params: SearchParams,
        train_cfg: &TrainConfig,
        tracker: Option<&RangeTracker>,
        mut session: Option<&mut SearchSession>,
        rng: &mut SmallRng,
    ) -> (Vec<AbsAction>, Vec<f64>) {
        if real.street() == Street::Preflop {
            return self.blueprint_dist(shadow, hist, rng);
        }
        // River spots with two live players: exact range-vs-range CFR+
        // (river.rs) instead of sampled MCCFR.
        if real.street() == Street::River && real.live_count() == 2 {
            if let Some(tr) = tracker {
                let sess = session.as_deref();
                if let Some(d) = self.river_dist(real, hist, tr, params, sess, rng) {
                    return d;
                }
            }
        }
        // Turn spots with two live players: exact turn+river vector solve
        // (slim bet menu), optionally inside the safety gadget. The solve's
        // river-entry CFVs are carried in the session for the river
        // resolve's gadget (continual resolving).
        if real.street() == Street::Turn && real.live_count() == 2 {
            if let Some(tr) = tracker {
                let sess = session.as_deref_mut();
                if let Some(d) = self.turn_dist(real, hist, tr, params, sess, rng) {
                    return d;
                }
            }
        }
        // Flop spots with two live players and a value net: ReBeL-style
        // depth-limited vector solving with learned leaf values.
        if real.street() == Street::Flop && real.live_count() == 2 {
            if let (Some(tr), Some(net)) = (tracker, &self.value_net) {
                if let Some(d) = self.flop_net_dist(real, tr, net, params, rng) {
                    return d;
                }
            }
        }
        let leaf = (real.street() == Street::Flop).then(|| LeafCfg {
            blueprint: self.blueprint.clone(),
            limit: Street::Flop,
        });
        let solver = resolve_subgame(self.abs.clone(), train_cfg, real, hist, params, tracker, leaf);
        let p = real.to_act();
        let bucket = solver.abs.bucket(real.hole(p), real.board(), rng);
        if let Some(s) = solver.avg_strategy(bucket, hist) {
            let acts = solver.abs.abstract_actions(real);
            if s.len() == acts.len() {
                return (acts, s);
            }
        }
        self.blueprint_dist(shadow, hist, rng)
    }

    /// Depth-limited flop resolve over both players' tracked ranges with
    /// value-net leaves. None when the spot doesn't qualify or the hero's
    /// combo got no strategy weight (caller falls back to MCCFR search).
    fn flop_net_dist(
        &self,
        h: &Hand,
        tracker: &RangeTracker,
        net: &Arc<ValueNet>,
        params: SearchParams,
        _rng: &mut SmallRng,
    ) -> Option<(Vec<AbsAction>, Vec<f64>)> {
        let hero = h.to_act();
        let villain = (0..h.num_players()).find(|&p| p != hero && !h.folded(p))?;
        let net_ref: &ValueNet = net;
        let mut solver = crate::flop::FlopSolver::build_sampled(
            h,
            &self.abs,
            [tracker.seat_weights(hero), tracker.seat_weights(villain)],
            &FLOP_MENU,
            net_ref,
            25,
            FLOP_QUERY_TURNS,
        )?;
        if params.adaptive {
            solver.solve_adaptive(10_000, params.time_ms, h.hole(hero), ADAPTIVE_PURITY);
        } else {
            solver.solve(10_000, params.time_ms);
        }
        Some(solver.root_strategy(h.hole(hero))?)
    }

    /// Exact turn+river resolve over both players' tracked ranges (vector
    /// CFR+ with the slim `TURN_MENU`, so the tree stays table-budget
    /// sized). With `safe_resolve`, runs inside the Burch gadget using
    /// rollout-estimated safety values. None when the spot doesn't qualify
    /// (caller falls back to sampled MCCFR).
    fn turn_dist(
        &self,
        h: &Hand,
        hist: &[u8],
        tracker: &RangeTracker,
        params: SearchParams,
        session: Option<&mut SearchSession>,
        rng: &mut SmallRng,
    ) -> Option<(Vec<AbsAction>, Vec<f64>)> {
        let hero = h.to_act();
        let villain = (0..h.num_players()).find(|&p| p != hero && !h.folded(p))?;
        let mut solver = crate::turn::TurnSolver::build(
            h,
            &self.abs,
            [tracker.seat_weights(hero), tracker.seat_weights(villain)],
            &TURN_MENU,
        )?;
        if params.safe_resolve {
            let alt = self.estimate_alt(h, hist, tracker, villain, rng);
            solver = solver.with_gadget(alt);
        }
        solver.solve(200, params.time_ms);
        let dist = solver.root_strategy(h.hole(hero))?;
        if let Some(sess) = session {
            sess.carry = Some(solver.extract_carry());
            sess.root_hist_len = hist.len();
        }
        Some(dist)
    }

    /// Exact river resolve over both players' tracked ranges. None when the
    /// spot doesn't qualify or the hero's combo got no strategy weight, in
    /// which case the caller falls back to sampled MCCFR. With
    /// `safe_resolve`, the solve runs the Burch resolving gadget using
    /// rollout-estimated blueprint safety values for the opponent.
    fn river_dist(
        &self,
        h: &Hand,
        hist: &[u8],
        tracker: &RangeTracker,
        params: SearchParams,
        session: Option<&SearchSession>,
        rng: &mut SmallRng,
    ) -> Option<(Vec<AbsAction>, Vec<f64>)> {
        let hero = h.to_act();
        let villain = (0..h.num_players()).find(|&p| p != hero && !h.folded(p))?;
        let mut solver = crate::river::RiverSolver::build(
            h,
            &self.abs,
            [tracker.seat_weights(hero), tracker.seat_weights(villain)],
        )?;
        if params.safe_resolve {
            // Continual resolving: prefer the turn resolve's carried CFVs
            // as the gadget alternatives; fall back to rollout estimates
            // when the line left the carry's tree.
            let alt = session
                .and_then(|s| s.river_alt(h, hist))
                .unwrap_or_else(|| self.estimate_alt(h, hist, tracker, villain, rng));
            solver = solver.with_gadget(alt);
        }
        if params.adaptive {
            solver.solve_adaptive(
                10_000,
                params.time_ms,
                params.qre_lambda,
                h.hole(hero),
                ADAPTIVE_PURITY,
            );
        } else {
            solver.solve(10_000, params.time_ms, params.qre_lambda);
        }
        Some(solver.root_strategy(h.hole(hero))?)
    }

    /// Rollout-estimated safety values for the opponent: for each of its
    /// combos, the mean utility of blueprint-vs-blueprint playouts from this
    /// state with the opponent pinned to that combo and everyone else drawn
    /// from the tracked ranges — an estimate of what the combo was worth
    /// before we replaced the blueprint with a resolved strategy.
    /// Street-agnostic: rollouts run from `h` to the end of the hand.
    fn estimate_alt(
        &self,
        h: &Hand,
        hist: &[u8],
        tracker: &RangeTracker,
        villain: usize,
        rng: &mut SmallRng,
    ) -> Vec<f64> {
        use rayon::prelude::*;
        const ROLLOUTS: usize = 6;
        let combos = crate::search::all_combos();
        let base_seed: u64 = rand::Rng::random(rng);
        let board = h.board().to_vec();

        (0..crate::search::NUM_COMBOS)
            .into_par_iter()
            .map(|ci| {
                let c = combos[ci];
                if board.contains(&c[0]) || board.contains(&c[1]) {
                    return 0.0;
                }
                let mut rng = <SmallRng as rand::SeedableRng>::seed_from_u64(
                    base_seed ^ (ci as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                );
                let mut sum = 0.0;
                for _ in 0..ROLLOUTS {
                    let mut sim = h.clone();
                    // Sample everyone from the tracker, then pin the villain
                    // (rejecting samples that collide with the pinned combo).
                    let mut want = tracker.sample_holes(&sim, &mut rng);
                    for _ in 0..10 {
                        let clash = want
                            .iter()
                            .enumerate()
                            .any(|(p, w)| {
                                p != villain
                                    && w.is_some_and(|w| {
                                        w.contains(&c[0]) || w.contains(&c[1])
                                    })
                            });
                        if !clash {
                            break;
                        }
                        want = tracker.sample_holes(&sim, &mut rng);
                    }
                    want[villain] = Some(c);
                    let clash = want.iter().enumerate().any(|(p, w)| {
                        p != villain
                            && w.is_some_and(|w| w.contains(&c[0]) || w.contains(&c[1]))
                    });
                    if clash {
                        for (p, w) in want.iter_mut().enumerate() {
                            if p != villain {
                                *w = None; // uniform fallback
                            }
                        }
                    }
                    sim.resample_hidden_with(&want, &mut rng);

                    let mut hist2 = hist.to_vec();
                    let mut guard = 0;
                    while !sim.is_terminal() {
                        guard += 1;
                        debug_assert!(guard < 100, "alt rollout did not terminate");
                        if guard >= 100 {
                            break;
                        }
                        let a = self.act_blueprint(&sim, &hist2, &mut rng);
                        let street_before = sim.street();
                        sim.apply(self.abs.concrete(&sim, a));
                        hist2.push(a.token());
                        if !sim.is_terminal() && sim.street() != street_before {
                            hist2.push(crate::abstraction::TOKEN_STREET_SEP);
                        }
                    }
                    if sim.is_terminal() {
                        sum += sim.utilities()[villain] as f64;
                    }
                }
                sum / ROLLOUTS as f64
            })
            .collect()
    }
}

/// Build a fresh subgame solver rooted at `root` (hidden cards get resampled
/// every traversal — from tracked ranges when a tracker is given) and train
/// it for the given budget. The abstraction is shared with the caller so
/// equity/bucket caches persist across decisions.
pub fn resolve_subgame(
    abs: Arc<Abstraction>,
    train_cfg: &TrainConfig,
    root: &Hand,
    root_hist: &[u8],
    params: SearchParams,
    tracker: Option<&RangeTracker>,
    leaf: Option<LeafCfg>,
) -> Trainer {
    let t = Trainer::new(abs, train_cfg.clone())
        .with_leaf(leaf)
        .with_plus(true)
        .with_qre(params.qre_lambda);
    match tracker {
        Some(tr) => {
            let sampler = move |h: &mut Hand, rng: &mut SmallRng| {
                let holes = tr.sample_holes(h, rng);
                h.resample_hidden_with(&holes, rng);
            };
            t.run_subgame(root, root_hist, params.time_ms, params.max_iters, Some(&sampler));
        }
        None => t.run_subgame(root, root_hist, params.time_ms, params.max_iters, None),
    }
    t
}

// ---------------------------------------------------------------------------
// Tests (written first, TDD)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::AbsConfig;
    use crate::cards::{fresh_deck, parse_cards};
    use crate::cfr::make_key;
    use crate::engine::{HandConfig, PlayerAction};
    use rand::SeedableRng;
    use std::collections::HashMap;

    fn abs_small() -> Abstraction {
        Abstraction::new(AbsConfig {
            postflop_buckets: 6,
            equity_rollouts: 50,
            cache_cap: 100_000,
            ..AbsConfig::default()
        })
    }

    #[test]
    fn policy_follows_blueprint_when_available() {
        let abs = abs_small();
        let h = Hand::new(&HandConfig::default(), 0, fresh_deck());
        let acts = abs.abstract_actions(&h);
        let bucket = crate::abstraction::preflop_bucket(h.hole(3));

        // Blueprint that always folds at this exact infoset.
        let mut strategies = HashMap::new();
        let mut probs = vec![0.0f32; acts.len()];
        probs[0] = 1.0; // Fold is always first when facing a bet
        strategies.insert(make_key(bucket, &[]).to_vec(), probs);
        let bp = Blueprint {
            strategies,
            iterations: 1,
            num_players: 6,
            abs_cfg: AbsConfig::default(),
            centroids: None,
        };

        let policy = Policy::new(bp, Arc::new(abs));
        let mut rng = SmallRng::seed_from_u64(1);
        for _ in 0..20 {
            assert_eq!(policy.act_blueprint(&h, &[], &mut rng), AbsAction::Fold);
        }
    }

    #[test]
    fn policy_falls_back_to_checkcall_when_unseen() {
        let abs = abs_small();
        let h = Hand::new(&HandConfig::default(), 0, fresh_deck());
        let bp = Blueprint {
            strategies: HashMap::new(),
            iterations: 0,
            num_players: 6,
            abs_cfg: AbsConfig::default(),
            centroids: None,
        };
        let policy = Policy::new(bp, Arc::new(abs));
        let mut rng = SmallRng::seed_from_u64(1);
        assert_eq!(
            policy.act_blueprint(&h, &[], &mut rng),
            AbsAction::CheckCall
        );
    }

    /// Subgame resolving on a rigged river: hero holds the nuts facing an
    /// all-in. The resolved strategy must call nearly always.
    #[test]
    fn search_calls_the_nuts_on_the_river() {
        // Heads-up. p0 (button/SB) has a royal flush by the river.
        let front = parse_cards("As Ks 2c 7d Qs Js Ts 3h 4d").unwrap();
        let mut deck = fresh_deck();
        let mut used = [false; 52];
        for (i, &c) in front.iter().enumerate() {
            deck[i] = c;
            used[c as usize] = true;
        }
        let mut idx = front.len();
        for c in 0..52u8 {
            if !used[c as usize] {
                deck[idx] = c;
                idx += 1;
            }
        }

        let hand_cfg = HandConfig {
            num_players: 2,
            stack: 2_000,
            sb: 50,
            bb: 100,
        };
        let mut h = Hand::new(&hand_cfg, 0, deck);
        let abs = abs_small();
        let mut hist: Vec<u8> = Vec::new();

        // p0 calls, p1 checks -> flop; checks to the river; p1 shoves.
        let do_act = |h: &mut Hand, hist: &mut Vec<u8>, a: AbsAction, abs: &Abstraction| {
            let street_before = h.street();
            h.apply(abs.concrete(h, a));
            hist.push(a.token());
            if !h.is_terminal() && h.street() != street_before {
                hist.push(crate::abstraction::TOKEN_STREET_SEP);
            }
        };
        do_act(&mut h, &mut hist, AbsAction::CheckCall, &abs); // p0 limps
        do_act(&mut h, &mut hist, AbsAction::CheckCall, &abs); // p1 checks
        for _ in 0..2 {
            do_act(&mut h, &mut hist, AbsAction::CheckCall, &abs); // p1 checks
            do_act(&mut h, &mut hist, AbsAction::CheckCall, &abs); // p0 checks
        }
        assert_eq!(h.street(), crate::engine::Street::River);
        assert_eq!(h.to_act(), 1);
        do_act(&mut h, &mut hist, AbsAction::AllIn, &abs); // p1 shoves
        assert_eq!(h.to_act(), 0);
        assert!(!h.is_terminal());

        let train_cfg = TrainConfig {
            hand: hand_cfg,
            prune_after: u64::MAX,
            ..TrainConfig::default()
        };
        let solver = resolve_subgame(
            Arc::new(Abstraction::new(AbsConfig {
                postflop_buckets: 6,
                equity_rollouts: 50,
                cache_cap: 100_000,
                ..AbsConfig::default()
            })),
            &train_cfg,
            &h,
            &hist,
            SearchParams {
                time_ms: 60_000, // generous: survives CPU contention in CI
                max_iters: 20_000,
                qre_lambda: None,
                safe_resolve: false,
                adaptive: false,
            },
            None,
            None,
        );

        let mut rng = SmallRng::seed_from_u64(2);
        let bucket = solver.abs.bucket(h.hole(0), h.board(), &mut rng);
        assert_eq!(bucket, 5, "royal flush must be in the top bucket");
        let strat = solver
            .avg_strategy(bucket, &hist)
            .expect("root infoset must be trained");
        // Actions facing a shove: [Fold, CheckCall].
        assert!(
            strat[1] > 0.9,
            "must call all-in with the nuts, got {:?}",
            strat
        );
        // Sanity: engine agrees calling ends the hand with hero winning.
        let mut done = h.clone();
        done.apply(PlayerAction::CheckCall);
        assert!(done.is_terminal());
        assert!(done.utilities()[0] > 0);
    }

    /// Safe (gadget) river resolving end to end through act_with_search:
    /// rollout-estimated safety values + gadget solve must still call the
    /// shove with the nuts.
    #[test]
    fn safe_resolve_calls_the_nuts_on_the_river() {
        use crate::search::RangeTracker;
        let front = parse_cards("As Ks 2c 7d Qs Js Ts 3h 4d").unwrap();
        let mut deck = fresh_deck();
        let mut used = [false; 52];
        for (i, &c) in front.iter().enumerate() {
            deck[i] = c;
            used[c as usize] = true;
        }
        let mut idx = front.len();
        for c in 0..52u8 {
            if !used[c as usize] {
                deck[idx] = c;
                idx += 1;
            }
        }
        let hand_cfg = HandConfig {
            num_players: 2,
            stack: 2_000,
            sb: 50,
            bb: 100,
        };
        let mut h = Hand::new(&hand_cfg, 0, deck);
        let abs = Arc::new(abs_small());
        let policy = Policy::new(
            Blueprint {
                strategies: HashMap::new(),
                iterations: 0,
                num_players: 2,
                abs_cfg: AbsConfig::default(),
                centroids: None,
            },
            abs.clone(),
        );
        let mut hist: Vec<u8> = Vec::new();
        let mut tracker = RangeTracker::new(2);
        let mut act = |h: &mut Hand, hist: &mut Vec<u8>, a: AbsAction| {
            let street_before = h.street();
            h.apply(abs.concrete(h, a));
            hist.push(a.token());
            if !h.is_terminal() && h.street() != street_before {
                hist.push(crate::abstraction::TOKEN_STREET_SEP);
            }
        };
        for _ in 0..3 {
            act(&mut h, &mut hist, AbsAction::CheckCall);
            act(&mut h, &mut hist, AbsAction::CheckCall);
        }
        assert_eq!(h.street(), Street::River);
        act(&mut h, &mut hist, AbsAction::AllIn); // p1 shoves
        assert_eq!(h.to_act(), 0);
        tracker.exclude(h.board());

        let train_cfg = TrainConfig {
            hand: hand_cfg,
            prune_after: u64::MAX,
            ..TrainConfig::default()
        };
        let mut rng = SmallRng::seed_from_u64(5);
        let a = policy.act_with_search(
            &h,
            &h,
            &hist,
            SearchParams {
                time_ms: 5_000,
                max_iters: 300,
                qre_lambda: None,
                safe_resolve: true,
                adaptive: false,
            },
            &train_cfg,
            Some(&tracker),
            None,
            &mut rng,
        );
        assert_eq!(a, AbsAction::CheckCall, "royal must call even with the gadget");
    }

    /// Turn decisions with two live players route to the exact turn+river
    /// vector solver: a made royal facing a turn shove must call, in both
    /// plain and gadget (safe) resolving modes.
    #[test]
    fn exact_turn_resolve_calls_the_nuts_facing_a_shove() {
        use crate::search::RangeTracker;
        use crate::table::Table;
        let front = parse_cards("As Ks 2c 7d Qs Js Ts 3h 4d").unwrap();
        let mut deck = fresh_deck();
        let mut used = [false; 52];
        for (i, &c) in front.iter().enumerate() {
            deck[i] = c;
            used[c as usize] = true;
        }
        let mut idx = front.len();
        for c in 0..52u8 {
            if !used[c as usize] {
                deck[idx] = c;
                idx += 1;
            }
        }
        let hand_cfg = HandConfig {
            num_players: 2,
            stack: 2_000,
            sb: 50,
            bb: 100,
        };
        let abs = Arc::new(abs_small());
        let policy = Policy::new(
            Blueprint {
                strategies: HashMap::new(),
                iterations: 0,
                num_players: 2,
                abs_cfg: AbsConfig::default(),
                centroids: None,
            },
            abs.clone(),
        );
        let train_cfg = TrainConfig {
            hand: hand_cfg.clone(),
            prune_after: u64::MAX,
            ..TrainConfig::default()
        };
        for safe in [false, true] {
            let mut table = Table::new(&hand_cfg, 0, deck);
            for _ in 0..4 {
                table.apply_abs(AbsAction::CheckCall, &abs);
            }
            assert_eq!(table.real.street(), Street::Turn);
            table.apply_abs(AbsAction::AllIn, &abs); // p1 shoves the turn
            assert_eq!(table.real.to_act(), 0);
            let mut tracker = RangeTracker::new(2);
            tracker.exclude(table.real.board());

            let mut rng = SmallRng::seed_from_u64(11);
            let a = policy.act_with_search(
                &table.real,
                &table.shadow,
                &table.hist,
                SearchParams {
                    time_ms: 30_000,
                    max_iters: 20_000,
                    qre_lambda: None,
                    safe_resolve: safe,
                    adaptive: false,
                },
                &train_cfg,
                Some(&tracker),
                None,
                &mut rng,
            );
            assert_eq!(
                a,
                AbsAction::CheckCall,
                "made royal must call the turn shove (safe_resolve={safe})"
            );
        }
    }

    /// Continual resolving end to end: the bot's exact turn resolve must
    /// populate the session carry, and its river decision must be able to
    /// consume the carried CFVs as gadget alternatives (the line/river
    /// lookup succeeds on the play path actually taken).
    #[test]
    fn turn_resolve_carries_cfvs_into_the_river_gadget() {
        use crate::search::RangeTracker;
        use crate::table::Table;
        // Bot is p1 (first to act postflop) with a medium hand; we are p0.
        let front = parse_cards("2c 7d Ah Kd Qs Js Ts 3h 4d").unwrap();
        let mut deck = fresh_deck();
        let mut used = [false; 52];
        for (i, &c) in front.iter().enumerate() {
            deck[i] = c;
            used[c as usize] = true;
        }
        let mut idx = front.len();
        for c in 0..52u8 {
            if !used[c as usize] {
                deck[idx] = c;
                idx += 1;
            }
        }
        let hand_cfg = HandConfig {
            num_players: 2,
            stack: 2_000,
            sb: 50,
            bb: 100,
        };
        let abs = Arc::new(abs_small());
        let policy = Policy::new(
            Blueprint {
                strategies: HashMap::new(),
                iterations: 0,
                num_players: 2,
                abs_cfg: AbsConfig::default(),
                centroids: None,
            },
            abs.clone(),
        );
        let train_cfg = TrainConfig {
            hand: hand_cfg.clone(),
            prune_after: u64::MAX,
            ..TrainConfig::default()
        };
        let params = SearchParams {
            time_ms: 30_000,
            max_iters: 20_000,
            qre_lambda: None,
            safe_resolve: true,
            adaptive: false,
        };
        let mut table = Table::new(&hand_cfg, 0, deck);
        for _ in 0..4 {
            table.apply_abs(AbsAction::CheckCall, &abs);
        }
        assert_eq!(table.real.street(), Street::Turn);
        assert_eq!(table.real.to_act(), 1, "bot (p1) opens the turn");

        let mut tracker = RangeTracker::new(2);
        tracker.exclude(table.real.board());
        let mut session = SearchSession::new();
        let mut rng = SmallRng::seed_from_u64(3);

        // Bot's turn decision: must populate the carry.
        let a1 = policy.act_with_search(
            &table.real,
            &table.shadow,
            &table.hist,
            params,
            &train_cfg,
            Some(&tracker),
            Some(&mut session),
            &mut rng,
        );
        assert!(session.carry.is_some(), "turn resolve must store the carry");
        let root_len = session.root_hist_len;
        assert_eq!(root_len, table.hist.len());
        table.apply_abs(a1, &abs);
        if table.real.street() == Street::Turn && table.real.to_act() == 0 {
            // We call/check behind to close the turn.
            table.apply_abs(AbsAction::CheckCall, &abs);
        }
        if table.real.is_terminal() || table.real.street() != Street::River {
            panic!("rig must reach the river, bot chose {a1:?}");
        }
        assert_eq!(table.real.to_act(), 1, "bot opens the river");

        // The exact lookup the river resolve performs must succeed on this
        // line: the carried alternatives exist for the realized river card.
        let alt = session
            .river_alt(&table.real, &table.hist)
            .expect("carried CFVs must cover the line actually played");
        assert_eq!(alt.len(), crate::search::NUM_COMBOS);
        assert!(alt.iter().all(|v| v.is_finite()));

        // And the river decision itself completes through the gadget path.
        let a2 = policy.act_with_search(
            &table.real,
            &table.shadow,
            &table.hist,
            params,
            &train_cfg,
            Some(&tracker),
            Some(&mut session),
            &mut rng,
        );
        let menu = abs.abstract_actions(&table.real);
        assert!(menu.contains(&a2), "river action must be legal, got {a2:?}");
    }

    /// Nested re-solving of off-tree bets: the bot must price an off-tree
    /// bet at its REAL amount, not the nearest abstract size. Villain bets
    /// a real 800 into a 200 pot, which log-maps to the 400 abstract bet.
    /// Hero's bluff-catcher has 42% equity against the tracked range: a
    /// call at the mapped price (needs 40%) but a fold at the real price
    /// (needs 44.4%). Solving from the real state must fold; the control
    /// (solving from the shadow, the old behavior) must call — which also
    /// validates that the pricing window is set up correctly.
    #[test]
    fn search_prices_off_tree_bets_from_the_real_state() {
        use crate::search::RangeTracker;
        use crate::table::Table;
        let front = parse_cards("Ah 3c Kd 9c Qs Js Ts 3h 4d").unwrap();
        let mut deck = fresh_deck();
        let mut used = [false; 52];
        for (i, &c) in front.iter().enumerate() {
            deck[i] = c;
            used[c as usize] = true;
        }
        let mut idx = front.len();
        for c in 0..52u8 {
            if !used[c as usize] {
                deck[idx] = c;
                idx += 1;
            }
        }
        let hand_cfg = HandConfig {
            num_players: 2,
            stack: 2_000,
            sb: 50,
            bb: 100,
        };
        let abs = Arc::new(abs_small());
        let policy = Policy::new(
            Blueprint {
                strategies: HashMap::new(),
                iterations: 0,
                num_players: 2,
                abs_cfg: AbsConfig::default(),
                centroids: None,
            },
            abs.clone(),
        );
        let mut table = Table::new(&hand_cfg, 0, deck);
        // Check down to the river.
        for _ in 0..6 {
            table.apply_abs(AbsAction::CheckCall, &abs);
        }
        assert_eq!(table.real.street(), Street::River);
        assert_eq!(table.real.to_act(), 1);
        // Villain bets an OFF-TREE 800 into the 200 pot.
        table.apply_concrete(PlayerAction::RaiseTo(800), &abs);
        assert_eq!(table.real.to_act(), 0);
        assert_eq!(table.real.to_call(), 800, "real price is the off-tree bet");
        assert_eq!(
            table.shadow.to_call(),
            400,
            "shadow must have mapped the bet to the 200%-pot abstract size"
        );

        // Villain's tracked range: 42% a bluff hero beats (pair of twos),
        // 58% the nuts (royal). Hero (Ah 3c, pair of threes) has 42% equity.
        let mut tracker = RangeTracker::new(2);
        tracker.set_all(1, 0.0);
        let bluff = parse_cards("2c 2h").unwrap();
        let nuts = parse_cards("As Ks").unwrap();
        tracker.set_weight(1, [bluff[0], bluff[1]], 0.42);
        tracker.set_weight(1, [nuts[0], nuts[1]], 0.58);

        let train_cfg = TrainConfig {
            hand: hand_cfg,
            prune_after: u64::MAX,
            ..TrainConfig::default()
        };
        let params = SearchParams {
            time_ms: 10_000,
            max_iters: 20_000,
            qre_lambda: None,
            safe_resolve: false,
            adaptive: false,
        };
        // EV of calling 800 real: .42(+900) - .58(900) = -144 < fold -100.
        let mut rng = SmallRng::seed_from_u64(9);
        let a = policy.act_with_search(
            &table.real,
            &table.shadow,
            &table.hist,
            params,
            &train_cfg,
            Some(&tracker),
            None,
            &mut rng,
        );
        assert_eq!(a, AbsAction::Fold, "must fold at the real 800 price");

        // Control (old behavior): priced at the mapped 400, calling is
        // correct: .42(+500) - .58(500) = -80 > fold -100.
        let mut rng = SmallRng::seed_from_u64(9);
        let a = policy.act_with_search(
            &table.shadow,
            &table.shadow,
            &table.hist,
            params,
            &train_cfg,
            Some(&tracker),
            None,
            &mut rng,
        );
        assert_eq!(
            a,
            AbsAction::CheckCall,
            "at the mapped price the same spot is a call — pricing window sanity"
        );
    }

    /// Depth-limited, range-tracked flop resolve: must produce a trained
    /// root strategy (leaf rollouts + range sampling wired end to end).
    #[test]
    fn range_tracked_depth_limited_flop_search_smoke() {
        use crate::search::RangeTracker;
        let abs = Arc::new(abs_small());
        let hand_cfg = HandConfig {
            num_players: 2,
            ..HandConfig::default()
        };
        let train_cfg = TrainConfig {
            hand: hand_cfg.clone(),
            prune_after: u64::MAX,
            ..TrainConfig::default()
        };

        // Tiny blueprint so leaf rollouts and range updates have something
        // to look up (missing infosets fall back to uniform anyway).
        let trainer = Trainer::new(abs.clone(), train_cfg.clone());
        trainer.run(2_000, &|_| {});
        let bp = Arc::new(trainer.to_blueprint());

        // Play to the flop: button calls, BB checks.
        let mut rng = SmallRng::seed_from_u64(77);
        let mut h = Hand::new(&hand_cfg, 0, fresh_deck());
        let mut hist: Vec<u8> = Vec::new();
        let mut tracker = RangeTracker::new(2);
        for a in [AbsAction::CheckCall, AbsAction::CheckCall] {
            tracker.observe(h.to_act(), a, &h, &hist, &bp, &abs);
            let street_before = h.street();
            h.apply(abs.concrete(&h, a));
            hist.push(a.token());
            if h.street() != street_before {
                hist.push(crate::abstraction::TOKEN_STREET_SEP);
                tracker.exclude(h.board());
            }
        }
        assert_eq!(h.street(), Street::Flop);

        let solver = resolve_subgame(
            abs.clone(),
            &train_cfg,
            &h,
            &hist,
            SearchParams {
                time_ms: 5_000,
                max_iters: 3_000,
                qre_lambda: None,
                safe_resolve: false,
                adaptive: false,
            },
            Some(&tracker),
            Some(LeafCfg {
                blueprint: bp.clone(),
                limit: Street::Flop,
            }),
        );
        assert!(solver.node_count() > 0, "search must create infosets");
        let p = h.to_act();
        let bucket = solver.abs.bucket(h.hole(p), h.board(), &mut rng);
        let strat = solver
            .avg_strategy(bucket, &hist)
            .expect("root infoset must be trained");
        let acts = abs.abstract_actions(&h);
        assert_eq!(strat.len(), acts.len());
        assert!((strat.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    }
}
