//! Local Best Response (LBR) — a lower bound on a policy's exploitability
//! (Lisý & Bowling 2017, "Eqilibrium Approximation Quality of Current
//! No-Limit Poker Bots").
//!
//! The LBR agent knows the bot's exact policy. It tracks the bot's range with
//! Bayes updates using the bot's true per-combo action probabilities (no
//! floor — the policy is known, not estimated), and at each of its own
//! decisions greedily maximizes expected value under the assumption that both
//! players check/call to showdown afterwards: calls are valued by showdown
//! equity against the tracked range, bets by fold equity against the range's
//! fold probability plus showdown equity against the continuing (non-folding)
//! part of the range. Its average winnings lower-bound what a true best
//! response would win — i.e. how exploitable the bot really is.
//!
//! Harness: heads-up blind-vs-blind inside the blueprint's native game (any
//! other seats fold on their first action), LBR alternating between the two
//! blind seats. Reported in mbb/hand: 0 = unexploitable by this probe.

use crate::abstraction::AbsAction;
use crate::bot::Policy;
use crate::cards::{fresh_deck, Card};
use crate::engine::{Hand, HandConfig, PlayerAction};
use crate::eval::eval_hole_board;
use crate::search::{all_combos, NUM_COMBOS};
use crate::table::Table;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// The LBR agent's belief over the bot's hole cards.
pub struct BotRange {
    pub weights: Vec<f64>,
}

impl Default for BotRange {
    fn default() -> Self {
        Self::new()
    }
}

impl BotRange {
    pub fn new() -> BotRange {
        BotRange {
            weights: vec![1.0; NUM_COMBOS],
        }
    }

    /// Zero out combos containing any of `cards` (board, or LBR's own holes).
    pub fn exclude(&mut self, combos: &[[Card; 2]], cards: &[Card]) {
        for (i, combo) in combos.iter().enumerate() {
            if combo.iter().any(|c| cards.contains(c)) {
                self.weights[i] = 0.0;
            }
        }
    }
}

pub struct Lbr<'a> {
    policy: &'a Policy,
    combos: Vec<[Card; 2]>,
    /// Board completions sampled per equity estimate (exact on the river).
    runouts: u32,
}

impl Lbr<'_> {
    pub fn new(policy: &Policy, runouts: u32) -> Lbr<'_> {
        Lbr {
            policy,
            combos: all_combos(),
            runouts,
        }
    }

    /// Probability the bot plays `acts[idx]` at this infoset holding `combo`
    /// — mirrors `Policy::act_blueprint` exactly, including its check/call
    /// fallback on unseen or mismatched infosets.
    fn action_prob(&self, ci: usize, board: &[Card], hist: &[u8], acts: &[AbsAction], idx: usize) -> f64 {
        let mut rng = SmallRng::seed_from_u64(0x1B12_5EED ^ ci as u64);
        let bucket = self.policy.abs.bucket(self.combos[ci], board, &mut rng);
        if let Some(s) = self.policy.blueprint.get(bucket, hist) {
            if s.len() == acts.len() {
                let total: f64 = s.iter().map(|&x| x as f64).sum();
                if total > 0.0 {
                    return s[idx] as f64 / total;
                }
            }
        }
        if acts[idx] == AbsAction::CheckCall {
            1.0
        } else {
            0.0
        }
    }

    /// Bayes-update the bot's range after watching it take `acts[taken_idx]`
    /// at the (pre-action) state `h` with history `hist`.
    pub fn observe(&self, range: &mut BotRange, h: &Hand, hist: &[u8], acts: &[AbsAction], taken_idx: usize) {
        let board = h.board().to_vec();
        range.weights = range
            .weights
            .par_iter()
            .enumerate()
            .map(|(ci, &w)| {
                if w <= 0.0 {
                    return 0.0;
                }
                w * self.action_prob(ci, &board, hist, acts, taken_idx)
            })
            .collect();
    }

    /// Hero's probability of winning at showdown (ties count half) against a
    /// weighted range if both check down: remaining board cards are sampled
    /// (`runouts` completions), exact when the board is already complete.
    pub fn range_equity(&self, hole: [Card; 2], board: &[Card], weights: &[f64], rng: &mut SmallRng) -> f64 {
        let mut used = [false; 52];
        used[hole[0] as usize] = true;
        used[hole[1] as usize] = true;
        for &c in board {
            used[c as usize] = true;
        }
        let mut stock: Vec<Card> = (0..52).filter(|&c| !used[c as usize]).collect();
        let need = 5 - board.len();
        let mut full = [0u8; 5];
        full[..board.len()].copy_from_slice(board);
        let iters = if need == 0 { 1 } else { self.runouts.max(1) };

        let mut num = 0.0;
        let mut den = 0.0;
        for _ in 0..iters {
            let mut on_runout = [false; 52];
            for k in 0..need {
                let j = rng.random_range(k..stock.len());
                stock.swap(k, j);
                full[board.len() + k] = stock[k];
                on_runout[stock[k] as usize] = true;
            }
            let hero = eval_hole_board(&hole, &full);
            for (ci, combo) in self.combos.iter().enumerate() {
                let w = weights[ci];
                if w <= 0.0
                    || used[combo[0] as usize]
                    || used[combo[1] as usize]
                    || on_runout[combo[0] as usize]
                    || on_runout[combo[1] as usize]
                {
                    continue;
                }
                let v = eval_hole_board(combo, &full);
                num += w * if hero > v {
                    1.0
                } else if hero == v {
                    0.5
                } else {
                    0.0
                };
                den += w;
            }
        }
        if den > 0.0 {
            num / den
        } else {
            0.5
        }
    }

    /// Greedy LBR action at the table's current decision (LBR to act).
    ///
    /// All EVs are deltas against folding (= 0): chips already in the pot are
    /// sunk, so winning them back counts as gain.
    pub fn action(&self, table: &Table, range: &BotRange, bot_seat: usize, rng: &mut SmallRng) -> AbsAction {
        let h = &table.real;
        let me = h.to_act();
        let hole = h.hole(me);
        let board = h.board();
        let wp = self.range_equity(hole, board, &range.weights, rng);
        let pot0 = h.pot() as f64;
        let asked = h.to_call().min(h.stack(me)) as f64;

        let mut best = AbsAction::CheckCall;
        let mut best_ev = wp * pot0 - (1.0 - wp) * asked;
        if h.to_call() > 0 && best_ev < 0.0 {
            best = AbsAction::Fold;
            best_ev = 0.0;
        }

        if h.all_in(bot_seat) {
            return best; // no live opponent decision: betting adds nothing
        }
        for a in self.policy.abs.abstract_actions(&table.shadow) {
            if !matches!(a, AbsAction::Bet(_) | AbsAction::AllIn) {
                continue;
            }
            let act = self.policy.abs.concrete(&table.shadow, a);
            let PlayerAction::RaiseTo(x) = act else {
                continue;
            };
            let mut sim = table.shadow.clone();
            sim.apply(act);
            if sim.is_terminal() || sim.to_act() != bot_seat {
                continue;
            }
            let resp_acts = self.policy.abs.abstract_actions(&sim);
            let Some(fold_idx) = resp_acts.iter().position(|&r| r == AbsAction::Fold) else {
                continue;
            };
            let mut hist2 = table.hist.clone();
            hist2.push(a.token());

            let board_v = board.to_vec();
            let res: Vec<(f64, f64)> = range
                .weights
                .par_iter()
                .enumerate()
                .map(|(ci, &w)| {
                    if w <= 0.0 {
                        return (0.0, 0.0);
                    }
                    let pf = self.action_prob(ci, &board_v, &hist2, &resp_acts, fold_idx);
                    (w * pf, w * (1.0 - pf))
                })
                .collect();
            let folded: f64 = res.iter().map(|r| r.0).sum();
            let cont: Vec<f64> = res.iter().map(|r| r.1).collect();
            let total = folded + cont.iter().sum::<f64>();
            if total <= 0.0 {
                continue;
            }
            let fp = folded / total;

            // Effective raise: capped by what the opponent can match.
            let opp_commit = h.street_commit(bot_seat);
            let x_eff = x.min(opp_commit + h.stack(bot_seat));
            let risk = x_eff.saturating_sub(h.street_commit(me)) as f64;
            let gain = x_eff.saturating_sub(opp_commit) as f64;
            let wp2 = if fp < 1.0 {
                self.range_equity(hole, board, &cont, rng)
            } else {
                0.5
            };
            let ev = fp * pot0 + (1.0 - fp) * (wp2 * (pot0 + gain) - (1.0 - wp2) * risk);
            if ev > best_ev {
                best_ev = ev;
                best = a;
            }
        }
        best
    }
}

#[derive(Debug)]
pub struct LbrResult {
    pub hands: u64,
    /// LBR's mean winnings in milli-big-blinds per hand: a lower bound on
    /// the policy's exploitability.
    pub mbb_per_hand: f64,
    pub ci95: f64,
}

/// Run the LBR probe: LBR and the bot in the two blind seats (alternating by
/// hand), every other seat folds. Stacks reset per hand, button random.
pub fn run_lbr(policy: &Policy, cfg: &HandConfig, hands: u64, runouts: u32, seed: u64) -> LbrResult {
    let n = cfg.num_players;
    let bb = cfg.bb as f64;
    let lbr = Lbr::new(policy, runouts);
    let results: Vec<f64> = (0..hands)
        .into_par_iter()
        .map(|i| {
            let mut rng =
                SmallRng::seed_from_u64(seed ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1B12);
            let button = rng.random_range(0..n);
            let (sb, bbs) = if n == 2 {
                (button, (button + 1) % n)
            } else {
                ((button + 1) % n, (button + 2) % n)
            };
            let (lbr_seat, bot_seat) = if i % 2 == 0 { (sb, bbs) } else { (bbs, sb) };
            let mut deck = fresh_deck();
            deck.shuffle(&mut rng);
            let mut table = Table::new(cfg, button, deck);
            let mut range = BotRange::new();
            range.exclude(&lbr.combos, &table.real.hole(lbr_seat));

            let mut guard = 0;
            while !table.real.is_terminal() {
                guard += 1;
                assert!(guard < 200, "LBR hand did not terminate");
                let p = table.real.to_act();
                let street_before = table.real.street();
                if p == bot_seat {
                    let acts = policy.abs.abstract_actions(&table.shadow);
                    let a = policy.act_blueprint(&table.shadow, &table.hist, &mut rng);
                    if let Some(idx) = acts.iter().position(|&x| x == a) {
                        lbr.observe(&mut range, &table.shadow, &table.hist, &acts, idx);
                    }
                    table.apply_abs(a, &policy.abs);
                } else if p == lbr_seat {
                    let a = lbr.action(&table, &range, bot_seat, &mut rng);
                    table.apply_abs(a, &policy.abs);
                } else {
                    table.apply_abs(AbsAction::Fold, &policy.abs);
                }
                if table.real.street() != street_before {
                    range.exclude(&lbr.combos, table.real.board());
                }
            }
            table.real.utilities()[lbr_seat] as f64 / bb * 1000.0
        })
        .collect();

    let mean = results.iter().sum::<f64>() / results.len() as f64;
    let var = results.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>()
        / (results.len().saturating_sub(1)) as f64;
    LbrResult {
        hands,
        mbb_per_hand: mean,
        ci95: 1.96 * (var / results.len() as f64).sqrt(),
    }
}

// ---------------------------------------------------------------------------
// Tests (written first, TDD)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::{preflop_bucket, AbsConfig, Abstraction};
    use crate::cards::parse_cards;
    use crate::cfr::{make_key, Blueprint};
    use std::collections::HashMap;
    use std::sync::Arc;

    fn abs_small() -> Abstraction {
        Abstraction::new(AbsConfig {
            postflop_buckets: 6,
            equity_rollouts: 40,
            dist_runouts: 8,
            runout_rollouts: 20,
            cache_cap: 500_000,
        })
    }

    fn empty_policy() -> Policy {
        Policy::new(
            Blueprint {
                strategies: HashMap::new(),
                iterations: 0,
                num_players: 2,
                abs_cfg: AbsConfig::default(),
                centroids: None,
            },
            Arc::new(abs_small()),
        )
    }

    /// Rig a deck: `front` cards first, the rest in order.
    fn rigged_deck(front: &str) -> [Card; 52] {
        let front = parse_cards(front).unwrap();
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
        deck
    }

    /// HU table walked to the river with p1 shoving: p0 (LBR) to act.
    /// Deck layout: p0 = first two cards, p1 = next two, board = next five.
    fn river_facing_shove(front: &str, policy: &Policy) -> Table {
        let cfg = HandConfig {
            num_players: 2,
            stack: 2_000,
            sb: 50,
            bb: 100,
        };
        let mut t = Table::new(&cfg, 0, rigged_deck(front));
        // p0 limps, p1 checks -> flop; p1/p0 check down to the river.
        for _ in 0..6 {
            t.apply_abs(AbsAction::CheckCall, &policy.abs);
        }
        assert_eq!(t.real.street(), crate::engine::Street::River);
        assert_eq!(t.real.to_act(), 1);
        t.apply_abs(AbsAction::AllIn, &policy.abs);
        assert_eq!(t.real.to_act(), 0);
        assert!(!t.real.is_terminal());
        t
    }

    #[test]
    fn range_equity_is_exact_on_the_river() {
        let policy = empty_policy();
        let lbr = Lbr::new(&policy, 10);
        let mut rng = SmallRng::seed_from_u64(7);
        let board = parse_cards("Qs Js Ts 3h 4d").unwrap();
        let uniform = vec![1.0; NUM_COMBOS];

        // Royal flush: beats every combo, no ties possible.
        let nuts = parse_cards("As Ks").unwrap();
        let wp = lbr.range_equity([nuts[0], nuts[1]], &board, &uniform, &mut rng);
        assert!((wp - 1.0).abs() < 1e-12, "royal flush wp must be 1, got {wp}");

        // Six-high air: loses to almost everything.
        let air = parse_cards("6c 2h").unwrap();
        let wp = lbr.range_equity([air[0], air[1]], &board, &uniform, &mut rng);
        assert!(wp < 0.1, "six-high wp must be tiny, got {wp}");
    }

    /// After watching a bot that only raises AA take the raise action, the
    /// tracked range must put zero weight on junk and keep AA.
    #[test]
    fn observe_concentrates_range_exactly() {
        let abs = abs_small();
        let h = Hand::new(
            &HandConfig::default(),
            0,
            rigged_deck("2c 7d"), // cards irrelevant: we observe seat 3
        );
        let acts = abs.abstract_actions(&h);
        let aa = preflop_bucket([51, 50]);
        let raise_idx = acts
            .iter()
            .position(|a| matches!(a, AbsAction::Bet(_)))
            .unwrap();
        let mut strategies = HashMap::new();
        for bucket in 0..169u16 {
            let mut s = vec![0.0f32; acts.len()];
            if bucket == aa {
                s[raise_idx] = 1.0;
            } else {
                s[0] = 1.0;
            }
            strategies.insert(make_key(bucket, &[]).to_vec(), s);
        }
        let policy = Policy::new(
            Blueprint {
                strategies,
                iterations: 1,
                num_players: 6,
                abs_cfg: AbsConfig::default(),
                centroids: None,
            },
            Arc::new(abs),
        );
        let lbr = Lbr::new(&policy, 10);
        let mut range = BotRange::new();
        lbr.observe(&mut range, &h, &[], &acts, raise_idx);

        let aa_combo = parse_cards("As Ah").unwrap();
        let junk = parse_cards("7c 2d").unwrap();
        assert!(range.weights[crate::search::combo_index(aa_combo[0], aa_combo[1])] > 0.99);
        assert_eq!(
            range.weights[crate::search::combo_index(junk[0], junk[1])],
            0.0,
            "exact Bayes update must zero out non-raising hands"
        );
    }

    #[test]
    fn lbr_calls_shove_with_nuts_and_folds_air() {
        let policy = empty_policy();
        let lbr = Lbr::new(&policy, 20);
        let mut rng = SmallRng::seed_from_u64(3);

        // Hero holds the royal flush: must call.
        let t = river_facing_shove("As Ks 2c 7d Qs Js Ts 3h 4d", &policy);
        let mut range = BotRange::new();
        range.exclude(&lbr.combos, &t.real.hole(0));
        range.exclude(&lbr.combos, t.real.board());
        assert_eq!(lbr.action(&t, &range, 1, &mut rng), AbsAction::CheckCall);

        // Hero holds six-high air: must fold.
        let t = river_facing_shove("6c 2h Ah Kd Qs Js Ts 3h 4d", &policy);
        let mut range = BotRange::new();
        range.exclude(&lbr.combos, &t.real.hole(0));
        range.exclude(&lbr.combos, t.real.board());
        assert_eq!(lbr.action(&t, &range, 1, &mut rng), AbsAction::Fold);
    }

    /// A calling station (empty blueprint -> check/call fallback) must be
    /// heavily exploited by LBR: pure value betting with exact equities.
    #[test]
    fn lbr_crushes_a_calling_station() {
        let policy = empty_policy();
        let cfg = HandConfig {
            num_players: 2,
            ..HandConfig::default()
        };
        let r = run_lbr(&policy, &cfg, 600, 16, 99);
        assert_eq!(r.hands, 600);
        assert!(
            r.mbb_per_hand - r.ci95 > 1_000.0,
            "LBR must crush a calling station, got {:+.0} +/- {:.0} mbb/hand",
            r.mbb_per_hand,
            r.ci95
        );
    }
}
