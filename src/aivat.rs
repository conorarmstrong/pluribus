//! AIVAT — an unbiased, low-variance winrate estimator (Burch, Schmid,
//! Moravčík, Morrill & Bowling, AAAI 2018).
//!
//! The estimate for one hand is `u + Σ corrections`, with one correction per
//! "known-distribution" event along the realized trajectory:
//!
//! - the hero's hole-card deal (uniform given the other players' cards),
//! - every board reveal (uniform over cards unseen by the omniscient
//!   evaluator),
//! - every decision whose mixed strategy we know exactly: the hero's
//!   (blueprint distribution) and Random baselines' (uniform); Caller is
//!   deterministic so its corrections vanish.
//!
//! Each correction is `E_event[v(s·e)] − v(s·e_actual)` for a value function
//! `v` — zero-mean conditional on the trajectory prefix for ANY `v`, so the
//! estimator stays unbiased no matter how crude `v` is; `v`'s quality only
//! affects how much variance is removed. We use an omniscient heuristic:
//! hero's showdown equity against the opponents' ACTUAL hands (board rolled
//! out at random) times the pot, minus the hero's commitment. On the river
//! this is exact, which cancels all showdown luck.
//!
//! Chance expectations are themselves Monte-Carlo sampled (`alt_samples`
//! alternative deals); independent sampling noise keeps the estimator
//! unbiased and only slightly widens the CI.

use crate::bot::Policy;
use crate::cards::{fresh_deck, Card};
use crate::engine::{Hand, HandConfig, PlayerAction};
use crate::eval::eval_hole_board;
use crate::table::{baseline_action, Baseline, EvalResult, Table};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// Rebuild the hand with some deck positions holding different cards
/// (swapping each desired card into place), replaying `actions` — the
/// deterministic public-API way to realize an alternative chance outcome.
fn alt_hand(
    cfg: &HandConfig,
    button: usize,
    deck: [Card; 52],
    put: &[(usize, Card)],
    actions: &[PlayerAction],
) -> Hand {
    let mut d = deck;
    for &(pos, card) in put {
        let cur = d.iter().position(|&c| c == card).unwrap();
        d.swap(pos, cur);
    }
    let mut h = Hand::new(cfg, button, d);
    for &a in actions {
        h.apply(a);
    }
    h
}

/// Hero's expected pot share against the opponents' actual hands, with any
/// missing board cards sampled uniformly from cards unseen by the evaluator.
/// Exact on the river.
fn equity_vs_actual(h: &Hand, hero: usize, runouts: u32, rng: &mut SmallRng) -> f64 {
    let live: Vec<usize> = (0..h.num_players())
        .filter(|&p| p != hero && !h.folded(p))
        .collect();
    if live.is_empty() {
        return 1.0;
    }
    let board = h.board();
    let mut used = [false; 52];
    for p in 0..h.num_players() {
        for c in h.hole(p) {
            used[c as usize] = true;
        }
    }
    for &c in board {
        used[c as usize] = true;
    }
    let mut stock: Vec<Card> = (0..52).filter(|&c| !used[c as usize]).collect();
    let need = 5 - board.len();
    let mut full = [0u8; 5];
    full[..board.len()].copy_from_slice(board);
    let iters = if need == 0 { 1 } else { runouts.max(1) };

    let mut share = 0.0;
    for _ in 0..iters {
        for k in 0..need {
            let j = rng.random_range(k..stock.len());
            stock.swap(k, j);
            full[board.len() + k] = stock[k];
        }
        let mine = eval_hole_board(&h.hole(hero), &full);
        let mut better = 0;
        let mut tied = 1u32;
        for &p in &live {
            let v = eval_hole_board(&h.hole(p), &full);
            if v > mine {
                better += 1;
            } else if v == mine {
                tied += 1;
            }
        }
        if better == 0 {
            share += 1.0 / tied as f64;
        }
    }
    share / iters as f64
}

/// Heuristic state value for the hero: exact utility when terminal, sunk
/// commitment when folded, otherwise equity-vs-actual-hands × pot − commit.
fn v_est(h: &Hand, hero: usize, runouts: u32, rng: &mut SmallRng) -> f64 {
    if h.is_terminal() {
        return h.utilities()[hero] as f64;
    }
    if h.folded(hero) {
        return -(h.hand_commit(hero) as f64);
    }
    let wp = equity_vs_actual(h, hero, runouts, rng);
    wp * h.pot() as f64 - h.hand_commit(hero) as f64
}

/// Cards not in any player's hole nor on `hand`'s board.
fn unseen(h: &Hand) -> Vec<Card> {
    let mut used = [false; 52];
    for p in 0..h.num_players() {
        for c in h.hole(p) {
            used[c as usize] = true;
        }
    }
    for &c in h.board() {
        used[c as usize] = true;
    }
    (0..52).filter(|&c| !used[c as usize]).collect()
}

struct AivatCfg {
    alt_samples: u32,
    runouts: u32,
}

#[allow(clippy::too_many_arguments)]
fn play_one_aivat(
    policy: &Policy,
    cfg: &HandConfig,
    baseline: Baseline,
    deck: [Card; 52],
    button: usize,
    hero: usize,
    ac: &AivatCfg,
    rng: &mut SmallRng,
) -> f64 {
    let n = cfg.num_players;
    let mut table = Table::new(cfg, button, deck);
    let mut actions: Vec<PlayerAction> = Vec::with_capacity(24);
    let mut corr = 0.0;

    // --- Hero hole-card correction: alternatives drawn given the other
    // players' actual holes (board not yet conditioned on — it is corrected
    // at its own reveals).
    {
        let actual_v = v_est(&table.real, hero, ac.runouts, rng);
        let mut allowed = [true; 52];
        for p in 0..n {
            if p != hero {
                for c in table.real.hole(p) {
                    allowed[c as usize] = false;
                }
            }
        }
        let pool: Vec<Card> = (0..52).filter(|&c| allowed[c as usize]).collect();
        let mut sum = 0.0;
        for _ in 0..ac.alt_samples {
            let a = pool[rng.random_range(0..pool.len())];
            let mut b = a;
            while b == a {
                b = pool[rng.random_range(0..pool.len())];
            }
            let alt = alt_hand(
                cfg,
                button,
                deck,
                &[(2 * hero, a), (2 * hero + 1, b)],
                &[],
            );
            sum += v_est(&alt, hero, ac.runouts, rng);
        }
        corr += sum / ac.alt_samples as f64 - actual_v;
    }

    let mut guard = 0;
    while !table.real.is_terminal() {
        guard += 1;
        assert!(guard < 500, "aivat hand did not terminate");
        let p = table.real.to_act();
        let board_before = table.real.board().len();

        // Known-distribution decision correction (hero / Random baseline).
        let (a, decision_corr) = if p == hero {
            let (acts, probs) = policy.blueprint_dist(&table.shadow, &table.hist, rng);
            let idx = crate::cfr::sample_index(&probs, rng);
            (acts[idx], Some((acts, probs, idx)))
        } else {
            let a = baseline_action(baseline, &table.shadow, &policy.abs, rng);
            match baseline {
                Baseline::Random => {
                    let acts = policy.abs.abstract_actions(&table.shadow);
                    let idx = acts.iter().position(|&x| x == a).unwrap();
                    let probs = vec![1.0 / acts.len() as f64; acts.len()];
                    (a, Some((acts, probs, idx)))
                }
                Baseline::Caller => (a, None), // deterministic: correction = 0
            }
        };
        if let Some((acts, probs, idx)) = decision_corr {
            if !table.real.folded(hero) && probs.iter().filter(|&&q| q > 0.0).count() > 1 {
                let mut expect = 0.0;
                let mut taken_v = 0.0;
                for (j, &cand) in acts.iter().enumerate() {
                    if probs[j] <= 0.0 && j != idx {
                        continue;
                    }
                    let mut sim = table.real.clone();
                    sim.apply(policy.abs.concrete(&table.real, cand));
                    let v = v_est(&sim, hero, ac.runouts, rng);
                    expect += probs[j] * v;
                    if j == idx {
                        taken_v = v;
                    }
                }
                corr += expect - taken_v;
            }
        }

        actions.push(table.apply_abs(a, &policy.abs));

        // Board-reveal correction (one event per batch of new cards; an
        // all-in fast-forward may reveal several streets at once).
        let board_after = table.real.board().len();
        if board_after > board_before && !table.real.folded(hero) {
            let actual_v = v_est(&table.real, hero, ac.runouts, rng);
            // Unseen from the evaluator's view, minus the newly revealed
            // cards themselves (they are the event being corrected).
            let pool = unseen(&table.real);
            let count = board_after - board_before;
            let mut pool = pool;
            let mut sum = 0.0;
            for _ in 0..ac.alt_samples {
                let mut put = Vec::with_capacity(count);
                for k in 0..count {
                    let j = rng.random_range(k..pool.len());
                    pool.swap(k, j);
                    put.push((2 * n + board_before + k, pool[k]));
                }
                let alt = alt_hand(cfg, button, deck, &put, &actions);
                sum += v_est(&alt, hero, ac.runouts, rng);
            }
            corr += sum / ac.alt_samples as f64 - actual_v;
        }
    }

    let u = table.real.utilities()[hero] as f64;
    (u + corr) / cfg.bb as f64 * 1000.0
}

/// AIVAT evaluation: identical matchup and estimand to `run_eval`, but each
/// hand's score is the AIVAT-corrected value.
pub fn run_eval_aivat(
    policy: &Policy,
    cfg: &HandConfig,
    baseline: Baseline,
    hands: u64,
    seed: u64,
) -> EvalResult {
    let n = cfg.num_players;
    let ac = AivatCfg {
        alt_samples: 24,
        runouts: 128,
    };
    let results: Vec<f64> = (0..hands)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(seed ^ i.wrapping_mul(0x2545_F491_4F6C_DD1D));
            let hero = (i % n as u64) as usize;
            let button = rng.random_range(0..n);
            let mut deck = fresh_deck();
            deck.shuffle(&mut rng);
            play_one_aivat(policy, cfg, baseline, deck, button, hero, &ac, &mut rng)
        })
        .collect();

    let mean = results.iter().sum::<f64>() / results.len() as f64;
    let var = results.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>()
        / (results.len().saturating_sub(1)) as f64;
    EvalResult {
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
    use crate::abstraction::{AbsConfig, Abstraction};
    use crate::cards::parse_cards;
    use crate::cfr::Blueprint;
    use crate::table::run_eval;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn empty_policy() -> Policy {
        Policy::new(
            Blueprint {
                strategies: HashMap::new(),
                iterations: 0,
                num_players: 6,
                abs_cfg: AbsConfig::default(),
                centroids: None,
            },
            Arc::new(Abstraction::new(AbsConfig {
                postflop_buckets: 6,
                equity_rollouts: 40,
                dist_runouts: 8,
                runout_rollouts: 20,
                cache_cap: 500_000,
            })),
        )
    }

    /// v_est must be exact at showdown-ready states: on the river with the
    /// board complete, a hero holding the stone-cold nuts values the state at
    /// pot − commit.
    #[test]
    fn v_est_is_exact_on_the_river() {
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
        let cfg = HandConfig {
            num_players: 2,
            stack: 2_000,
            sb: 50,
            bb: 100,
        };
        let mut h = Hand::new(&cfg, 0, deck);
        for _ in 0..6 {
            h.apply(PlayerAction::CheckCall); // check down to the river
        }
        assert_eq!(h.street(), crate::engine::Street::River);
        let mut rng = SmallRng::seed_from_u64(1);
        let v = v_est(&h, 0, 8, &mut rng);
        // Royal flush: wins the whole 200 pot, committed 100.
        assert!((v - 100.0).abs() < 1e-9, "nuts on river: v must be exactly pot·1 − commit, got {v}");
        let v1 = v_est(&h, 1, 8, &mut rng);
        assert!((v1 + 100.0).abs() < 1e-9, "dead hand: v must be −commit, got {v1}");
    }

    /// alt_hand must place the requested cards and keep the deck a valid
    /// permutation (replays legally).
    #[test]
    fn alt_hand_swaps_cards_consistently() {
        let cfg = HandConfig::default();
        let deck = fresh_deck();
        let want = parse_cards("Ah Kd").unwrap();
        let h = alt_hand(&cfg, 0, deck, &[(4, want[0]), (5, want[1])], &[]);
        assert_eq!(h.hole(2), [want[0], want[1]]);
        // Full deck still a permutation: every card seen exactly once across
        // holes of all players (12 cards) — spot-check no duplicates.
        let mut seen = [false; 52];
        for p in 0..6 {
            for c in h.hole(p) {
                assert!(!seen[c as usize], "duplicate card after swap");
                seen[c as usize] = true;
            }
        }
    }

    /// AIVAT must agree with the plain estimator in expectation and shrink
    /// the CI dramatically. Caller-vs-caller: true value 0; plain CI is
    /// hundreds of mbb, AIVAT should cut it by well over half.
    #[test]
    fn aivat_is_unbiased_and_low_variance() {
        let policy = empty_policy();
        let cfg = HandConfig::default();
        let hands = 1_500;
        let plain = run_eval(&policy, &cfg, Baseline::Caller, hands, 5);
        let aivat = run_eval_aivat(&policy, &cfg, Baseline::Caller, hands, 5);
        assert!(
            aivat.mbb_per_hand.abs() < 3.0 * aivat.ci95.max(10.0),
            "AIVAT estimate must stay near the true value 0, got {} ± {}",
            aivat.mbb_per_hand,
            aivat.ci95
        );
        assert!(
            aivat.ci95 < 0.5 * plain.ci95,
            "AIVAT must shrink the CI by >2x: plain ±{:.1}, aivat ±{:.1}",
            plain.ci95,
            aivat.ci95
        );
    }
}
