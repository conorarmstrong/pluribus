//! Belief-state value network for turn states (ReBeL-lite).
//!
//! Training data: heads-up turn spots reached by blueprint self-play with
//! Bayes-tracked ranges (perturbed toward uniform for coverage), solved
//! EXACTLY by the turn solver (`turn.rs`). Each sample stores the public
//! state, both range vectors and both players' per-combo root values.
//!
//! The network maps (board, pot, stacks, both ranges) → per-combo *forward*
//! values for both players — chips won from this point on, i.e. the solver's
//! net values plus the player's current commitment, which makes the target a
//! function of the public state alone (sunk chips shift all terminal
//! utilities by a constant and cancel out of the strategy). Targets are
//! scaled by `pot + min(stacks)` (the maximum forward swing) into [-1, 1].
//!
//! This net is the leaf evaluator for depth-limited flop solving: an
//! end-of-flop leaf is exactly a turn street-start public state.

use crate::bot::Policy;
use crate::cards::{fresh_deck, Card};
use crate::engine::{HandConfig, Street};
use crate::net::{Adam, Example, Mlp};
use crate::search::{RangeTracker, NUM_COMBOS};
use crate::table::Table;
use crate::turn::TurnSolver;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub const INPUT_DIM: usize = 52 + 3 + 2 * NUM_COMBOS;
pub const OUTPUT_DIM: usize = 2 * NUM_COMBOS;

/// Bet-size menu (indices into BET_SIZES) used for exact turn solves —
/// 75% pot plus all-in keeps the turn+river tree ~100MB.
pub const SOLVE_MENU: [u8; 1] = [3];

#[derive(Serialize, Deserialize)]
pub struct TurnSample {
    pub board: [Card; 4],
    pub pot: u32,
    /// [first-to-act, other] at the turn street start.
    pub stack: [u32; 2],
    pub commit: [u32; 2],
    /// Normalized range per player (sums to 1 over valid combos).
    pub range: [Vec<f32>; 2],
    /// Forward values in chips per combo (net value + own commitment).
    pub values: [Vec<f32>; 2],
}

/// Target scale: the maximum chips a player can win from a street start.
pub fn value_scale(pot: u32, stack: [u32; 2]) -> f32 {
    (pot + stack[0].min(stack[1])) as f32
}

/// Feature vector: board multi-hot, pot/stack scalars, both ranges.
pub fn encode(board: &[Card], pot: u32, stack: [u32; 2], range: [&[f32]; 2]) -> Vec<f32> {
    let mut x = vec![0.0f32; INPUT_DIM];
    for &c in board {
        x[c as usize] = 1.0;
    }
    x[52] = pot as f32 / 10_000.0;
    x[53] = stack[0] as f32 / 10_000.0;
    x[54] = stack[1] as f32 / 10_000.0;
    for (p, r) in range.iter().enumerate() {
        let total: f32 = r.iter().sum();
        let norm = if total > 0.0 { 1.0 / total } else { 0.0 };
        let base = 55 + p * NUM_COMBOS;
        for (ci, &w) in r.iter().enumerate() {
            x[base + ci] = w * norm;
        }
    }
    x
}

/// Training example: inputs as above; targets = forward values / scale;
/// loss weights = the owning player's range mass (half per player).
pub fn to_example(s: &TurnSample) -> Example {
    let x = encode(
        &s.board,
        s.pot,
        s.stack,
        [&s.range[0], &s.range[1]],
    );
    let scale = value_scale(s.pot, s.stack);
    let mut t = vec![0.0f32; OUTPUT_DIM];
    let mut wt = vec![0.0f32; OUTPUT_DIM];
    for p in 0..2 {
        let mass: f32 = s.range[p].iter().sum();
        for ci in 0..NUM_COMBOS {
            t[p * NUM_COMBOS + ci] = s.values[p][ci] / scale;
            if mass > 0.0 {
                wt[p * NUM_COMBOS + ci] = 0.5 * s.range[p][ci] / mass;
            }
        }
    }
    (x, t, wt)
}

/// Play one blueprint self-play hand; if it reaches the turn with exactly
/// two live players, solve the spot exactly and return the sample.
fn sample_one(
    policy: &Policy,
    cfg: &HandConfig,
    solve_iters: u64,
    solve_ms: u64,
    seed: u64,
) -> Option<TurnSample> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let n = cfg.num_players;
    let button = rng.random_range(0..n);
    let mut deck = fresh_deck();
    deck.shuffle(&mut rng);
    let mut table = Table::new(cfg, button, deck);
    let mut tracker = RangeTracker::new(n);

    let mut guard = 0;
    while table.real.street() != Street::Turn {
        if table.real.is_terminal() {
            return None;
        }
        guard += 1;
        assert!(guard < 200, "sample hand did not terminate");
        let p = table.real.to_act();
        let a = policy.act_blueprint(&table.shadow, &table.hist, &mut rng);
        tracker.observe(p, a, &table.shadow, &table.hist, &policy.blueprint, &policy.abs);
        let street_before = table.real.street();
        table.apply_abs(a, &policy.abs);
        if table.real.street() != street_before {
            tracker.exclude(table.real.board());
        }
    }
    let h = &table.real;
    if h.is_terminal() || h.live_count() != 2 {
        return None;
    }
    let hero = h.to_act();
    let villain = (0..n).find(|&p| p != hero && !h.folded(p))?;

    // Perturb tracked ranges toward uniform for input-space coverage
    // (mostly small alpha, occasionally large).
    let alpha = rng.random::<f64>().powi(2);
    let mut on_board = [false; 52];
    for &c in h.board() {
        on_board[c as usize] = true;
    }
    let combos = crate::search::all_combos();
    let valid: Vec<bool> = combos
        .iter()
        .map(|c| !on_board[c[0] as usize] && !on_board[c[1] as usize])
        .collect();
    let n_valid = valid.iter().filter(|&&v| v).count() as f64;
    let mix = |w: &[f64]| -> Vec<f64> {
        let total: f64 = w
            .iter()
            .zip(&valid)
            .filter(|(_, &v)| v)
            .map(|(&x, _)| x)
            .sum();
        (0..NUM_COMBOS)
            .map(|ci| {
                if !valid[ci] {
                    return 0.0;
                }
                let base = if total > 0.0 { w[ci] / total } else { 1.0 / n_valid };
                (1.0 - alpha) * base + alpha / n_valid
            })
            .collect()
    };
    let r0 = mix(tracker.seat_weights(hero));
    let r1 = mix(tracker.seat_weights(villain));

    let mut solver = TurnSolver::build(h, &policy.abs, [&r0, &r1], &SOLVE_MENU)?;
    solver.solve(solve_iters, solve_ms);

    let commit = [h.hand_commit(hero), h.hand_commit(villain)];
    let values: [Vec<f32>; 2] = [0, 1].map(|p| {
        let v = solver.root_values(p);
        v.iter()
            .enumerate()
            .map(|(ci, &x)| {
                if r_of(p, &r0, &r1)[ci] > 0.0 {
                    (x + commit[p] as f64) as f32
                } else {
                    0.0
                }
            })
            .collect()
    });
    Some(TurnSample {
        board: h.board().try_into().ok()?,
        pot: h.pot(),
        stack: [h.stack(hero), h.stack(villain)],
        commit,
        range: [
            r0.iter().map(|&x| x as f32).collect(),
            r1.iter().map(|&x| x as f32).collect(),
        ],
        values,
    })
}

fn r_of<'a>(p: usize, r0: &'a [f64], r1: &'a [f64]) -> &'a [f64] {
    if p == 0 {
        r0
    } else {
        r1
    }
}

/// Generate `n` exactly-solved turn samples in parallel.
pub fn generate(
    policy: &Policy,
    solve_iters: u64,
    solve_ms: u64,
    n: usize,
    seed: u64,
    progress: &(dyn Fn(usize) + Sync),
) -> Vec<TurnSample> {
    let cfg = HandConfig {
        num_players: policy.blueprint.num_players,
        ..HandConfig::default()
    };
    let mut out: Vec<TurnSample> = Vec::with_capacity(n);
    let mut next_seed = seed;
    while out.len() < n {
        let want = n - out.len();
        let attempts = (want * 4).max(64);
        let batch: Vec<TurnSample> = (0..attempts)
            .into_par_iter()
            .filter_map(|i| {
                sample_one(
                    policy,
                    &cfg,
                    solve_iters,
                    solve_ms,
                    next_seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                )
            })
            .collect();
        next_seed = next_seed.wrapping_add(0xABCD_EF01_2345_6789);
        for s in batch {
            if out.len() < n {
                out.push(s);
            }
        }
        progress(out.len());
    }
    out
}

pub fn save_samples(path: &str, samples: &[TurnSample]) -> std::io::Result<()> {
    let bytes = bincode::serialize(samples).map_err(std::io::Error::other)?;
    std::fs::write(path, bytes)
}

pub fn load_samples(path: &str) -> std::io::Result<Vec<TurnSample>> {
    let bytes = std::fs::read(path)?;
    bincode::deserialize(&bytes).map_err(std::io::Error::other)
}

/// Train the value net; returns (net, final validation loss).
#[allow(clippy::too_many_arguments)]
pub fn train(
    samples: &[TurnSample],
    hidden: &[usize],
    epochs: usize,
    lr: f32,
    batch: usize,
    seed: u64,
    progress: &mut dyn FnMut(usize, f32, f32),
) -> (Mlp, f32) {
    let mut examples: Vec<Example> = samples.par_iter().map(to_example).collect();
    let mut rng = SmallRng::seed_from_u64(seed);
    examples.shuffle(&mut rng);
    let n_val = (examples.len() / 10).max(1).min(examples.len() - 1);
    let (val, tr) = examples.split_at(n_val);

    let mut sizes = vec![INPUT_DIM];
    sizes.extend_from_slice(hidden);
    sizes.push(OUTPUT_DIM);
    let mut net = Mlp::new(&sizes, seed ^ 0xA11CE);
    let mut adam = Adam::new(&net, lr);
    let mut val_loss = f32::MAX;
    for e in 0..epochs {
        let train_loss = crate::net::train_epoch(&mut net, &mut adam, tr, batch, &mut rng);
        val_loss = eval_loss(&net, val);
        progress(e, train_loss, val_loss);
    }
    (net, val_loss)
}

/// Mean weighted-MSE loss on a dataset (forward only).
pub fn eval_loss(net: &Mlp, data: &[Example]) -> f32 {
    let total: f64 = data
        .par_iter()
        .map(|(x, t, wt)| {
            let y = net.forward(x);
            y.iter()
                .zip(t)
                .zip(wt)
                .map(|((&y, &t), &w)| (w * (y - t) * (y - t)) as f64)
                .sum::<f64>()
        })
        .sum();
    (total / data.len() as f64) as f32
}

/// Query wrapper used by the flop solver: per-combo forward values (chips)
/// for both players at a turn street-start public state.
pub struct ValueNet {
    pub net: Mlp,
}

impl ValueNet {
    pub fn load(path: &str) -> std::io::Result<ValueNet> {
        Ok(ValueNet {
            net: Mlp::load(path)?,
        })
    }

    pub fn values(
        &self,
        board: &[Card],
        pot: u32,
        stack: [u32; 2],
        range: [&[f64]; 2],
    ) -> [Vec<f64>; 2] {
        let r0: Vec<f32> = range[0].iter().map(|&x| x as f32).collect();
        let r1: Vec<f32> = range[1].iter().map(|&x| x as f32).collect();
        let x = encode(board, pot, stack, [&r0, &r1]);
        let y = self.net.forward(&x);
        let scale = value_scale(pot, stack) as f64;
        [0, 1].map(|p| {
            y[p * NUM_COMBOS..(p + 1) * NUM_COMBOS]
                .iter()
                .map(|&v| v as f64 * scale)
                .collect()
        })
    }
}

// ---------------------------------------------------------------------------
// Tests (written first, TDD)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::{AbsConfig, Abstraction};
    use crate::cfr::{TrainConfig, Trainer};
    use std::sync::Arc;

    /// Heads-up test blueprint: guarantees turn spots have exactly 2 live
    /// players (a lightly-trained 6-max blueprint plays too loose to ever
    /// fold 4 seats; the real blueprint has no such trouble).
    fn small_policy() -> Policy {
        let abs = Arc::new(Abstraction::new(AbsConfig {
            postflop_buckets: 6,
            equity_rollouts: 40,
            dist_runouts: 8,
            runout_rollouts: 20,
            cache_cap: 500_000,
        }));
        let cfg = TrainConfig {
            hand: hu_cfg(),
            prune_after: u64::MAX,
            ..TrainConfig::default()
        };
        let trainer = Trainer::new(abs.clone(), cfg);
        trainer.run(3_000, &|_| {});
        Policy::new(trainer.to_blueprint(), abs)
    }

    fn hu_cfg() -> HandConfig {
        HandConfig {
            num_players: 2,
            ..HandConfig::default()
        }
    }

    #[test]
    fn encoding_has_correct_shape_and_normalization() {
        let board = crate::cards::parse_cards("Qs Js Ts 3h").unwrap();
        let r0 = vec![2.0f32; NUM_COMBOS];
        let r1 = vec![1.0f32; NUM_COMBOS];
        let x = encode(&board, 600, [9_700, 9_700], [&r0, &r1]);
        assert_eq!(x.len(), INPUT_DIM);
        assert_eq!(x[..52].iter().sum::<f32>(), 4.0);
        let s0: f32 = x[55..55 + NUM_COMBOS].iter().sum();
        let s1: f32 = x[55 + NUM_COMBOS..].iter().sum();
        assert!((s0 - 1.0).abs() < 1e-4 && (s1 - 1.0).abs() < 1e-4);
    }

    /// End-to-end: blueprint self-play must reach solvable turn spots and
    /// produce internally consistent samples.
    #[test]
    fn generates_valid_samples() {
        let policy = small_policy();
        let cfg = hu_cfg();
        let mut got = None;
        for seed in 0..400 {
            if let Some(s) = sample_one(&policy, &cfg, 5, 60_000, seed) {
                got = Some(s);
                break;
            }
        }
        let s = got.expect("no turn spot found in 400 attempts");
        // Board: 4 distinct cards.
        let mut b = s.board.to_vec();
        b.sort_unstable();
        b.dedup();
        assert_eq!(b.len(), 4);
        // Ranges normalized-ish, no weight on board-conflicting combos.
        for p in 0..2 {
            let total: f32 = s.range[p].iter().sum();
            assert!((total - 1.0).abs() < 1e-3, "range {p} sums to {total}");
        }
        // Values bounded by the maximum forward swing.
        let scale = value_scale(s.pot, s.stack);
        for p in 0..2 {
            for ci in 0..NUM_COMBOS {
                if s.range[p][ci] > 0.0 {
                    let fwd = s.values[p][ci];
                    assert!(
                        fwd.abs() <= 1.05 * scale,
                        "player {p} combo {ci}: forward value {fwd} exceeds scale {scale}"
                    );
                }
            }
        }
        // Sample round-trips through the dataset file format.
        let path = std::env::temp_dir().join("pluribus_turnsample_test.bin");
        let path = path.to_str().unwrap();
        save_samples(path, std::slice::from_ref(&s)).unwrap();
        let back = load_samples(path).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].board, s.board);
        assert_eq!(back[0].values[0], s.values[0]);
        let _ = std::fs::remove_file(path);
    }

    /// The net must learn *something* real from a handful of samples:
    /// training loss decreases and beats predicting zero.
    #[test]
    fn training_reduces_loss() {
        let policy = small_policy();
        let cfg = hu_cfg();
        let mut samples = Vec::new();
        for seed in 1000..1400 {
            if let Some(s) = sample_one(&policy, &cfg, 5, 60_000, seed) {
                samples.push(s);
                if samples.len() >= 12 {
                    break;
                }
            }
        }
        assert!(samples.len() >= 8, "only {} samples", samples.len());
        let examples: Vec<Example> = samples.iter().map(to_example).collect();
        let zero_loss: f32 = {
            let z = Mlp::new(&[INPUT_DIM, 4, OUTPUT_DIM], 1); // near-zero output
            eval_loss(&z, &examples)
        };
        let (net, _) = train(&samples, &[64], 60, 2e-3, 4, 9, &mut |_, _, _| {});
        let trained_loss = eval_loss(&net, &examples);
        assert!(
            trained_loss < 0.5 * zero_loss,
            "training must at least halve the trivial loss: {trained_loss} vs {zero_loss}"
        );
    }
}
