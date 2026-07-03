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
    pub fn act_with_search(
        &self,
        h: &Hand,
        hist: &[u8],
        params: SearchParams,
        train_cfg: &TrainConfig,
        tracker: Option<&RangeTracker>,
        rng: &mut SmallRng,
    ) -> AbsAction {
        if h.street() == Street::Preflop {
            return self.act_blueprint(h, hist, rng);
        }
        // River spots with two live players: exact range-vs-range CFR+
        // (river.rs) instead of sampled MCCFR.
        if h.street() == Street::River && h.live_count() == 2 {
            if let Some(tr) = tracker {
                if let Some(a) = self.act_river_exact(h, hist, tr, params, rng) {
                    return a;
                }
            }
        }
        // Flop spots with two live players and a value net: ReBeL-style
        // depth-limited vector solving with learned leaf values.
        if h.street() == Street::Flop && h.live_count() == 2 {
            if let (Some(tr), Some(net)) = (tracker, &self.value_net) {
                if let Some(a) = self.act_flop_net(h, tr, net, params, rng) {
                    return a;
                }
            }
        }
        let leaf = (h.street() == Street::Flop).then(|| LeafCfg {
            blueprint: self.blueprint.clone(),
            limit: Street::Flop,
        });
        let solver = resolve_subgame(self.abs.clone(), train_cfg, h, hist, params, tracker, leaf);
        let p = h.to_act();
        let bucket = solver.abs.bucket(h.hole(p), h.board(), rng);
        if let Some(s) = solver.avg_strategy(bucket, hist) {
            let acts = solver.abs.abstract_actions(h);
            if s.len() == acts.len() {
                return acts[sample_index(&s, rng)];
            }
        }
        self.act_blueprint(h, hist, rng)
    }

    /// Depth-limited flop resolve over both players' tracked ranges with
    /// value-net leaves. None when the spot doesn't qualify or the hero's
    /// combo got no strategy weight (caller falls back to MCCFR search).
    fn act_flop_net(
        &self,
        h: &Hand,
        tracker: &RangeTracker,
        net: &Arc<ValueNet>,
        params: SearchParams,
        rng: &mut SmallRng,
    ) -> Option<AbsAction> {
        let hero = h.to_act();
        let villain = (0..h.num_players()).find(|&p| p != hero && !h.folded(p))?;
        let net_ref: &ValueNet = net;
        let mut solver = crate::flop::FlopSolver::build(
            h,
            &self.abs,
            [tracker.seat_weights(hero), tracker.seat_weights(villain)],
            &FLOP_MENU,
            net_ref,
            25,
        )?;
        if params.adaptive {
            solver.solve_adaptive(10_000, params.time_ms, h.hole(hero), ADAPTIVE_PURITY);
        } else {
            solver.solve(10_000, params.time_ms);
        }
        let (acts, s) = solver.root_strategy(h.hole(hero))?;
        Some(acts[sample_index(&s, rng)])
    }

    /// Exact river resolve over both players' tracked ranges. None when the
    /// spot doesn't qualify or the hero's combo got no strategy weight, in
    /// which case the caller falls back to sampled MCCFR. With
    /// `safe_resolve`, the solve runs the Burch resolving gadget using
    /// rollout-estimated blueprint safety values for the opponent.
    fn act_river_exact(
        &self,
        h: &Hand,
        hist: &[u8],
        tracker: &RangeTracker,
        params: SearchParams,
        rng: &mut SmallRng,
    ) -> Option<AbsAction> {
        let hero = h.to_act();
        let villain = (0..h.num_players()).find(|&p| p != hero && !h.folded(p))?;
        let mut solver = crate::river::RiverSolver::build(
            h,
            &self.abs,
            [tracker.seat_weights(hero), tracker.seat_weights(villain)],
        )?;
        if params.safe_resolve {
            let alt = self.estimate_river_alt(h, hist, tracker, villain, rng);
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
        let (acts, s) = solver.root_strategy(h.hole(hero))?;
        Some(acts[sample_index(&s, rng)])
    }

    /// Rollout-estimated safety values for the opponent: for each of its
    /// combos, the mean utility of blueprint-vs-blueprint playouts from this
    /// state with the opponent pinned to that combo and everyone else drawn
    /// from the tracked ranges — an estimate of what the combo was worth
    /// before we replaced the blueprint with a resolved strategy.
    fn estimate_river_alt(
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
            &mut rng,
        );
        assert_eq!(a, AbsAction::CheckCall, "royal must call even with the gadget");
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
