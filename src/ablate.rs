//! Safety ablation: unsafe vs gadget (safe) river resolving under corrupted
//! range beliefs.
//!
//! Deployment model: the blueprint achieves per-combo values `alt` for the
//! opponent (proxied here by an equilibrium solve under the opponent's TRUE
//! range). The resolver, however, only sees a *belief* — the true range
//! mixed with noise at level ε (tracker error). Unsafe resolving trusts the
//! belief outright; safe resolving plays the Burch-style gadget game with
//! `alt` as the opponent's alternative payoffs.
//!
//! Metric: per-combo best-response margin `(BR value − alt)⁺` — how many
//! chips a combo extracts from the resolved strategy beyond its safety
//! value. The gadget's guarantee is that this is ≤ the CFR convergence
//! error for every combo; unsafe resolving has no such bound, and combos
//! the belief underweights blow straight through it.

use crate::abstraction::{AbsConfig, Abstraction};
use crate::cards::fresh_deck;
use crate::engine::{Hand, HandConfig, PlayerAction};
use crate::river::RiverSolver;
use crate::search::NUM_COMBOS;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

#[derive(Debug, Clone, Copy)]
pub struct AblateRow {
    pub epsilon: f64,
    /// True-range-weighted mean margin, in chips.
    pub unsafe_mean: f64,
    pub unsafe_max: f64,
    pub safe_mean: f64,
    pub safe_max: f64,
    /// Mean pot size across spots (for scale).
    pub pot: f64,
}

/// Random heads-up river spot: random board, random preflop raise size
/// (pot 400..4000), checked to p0 on the river.
fn random_river_spot(rng: &mut SmallRng) -> Hand {
    let cfg = HandConfig {
        num_players: 2,
        stack: 10_000,
        sb: 50,
        bb: 100,
    };
    let mut deck = fresh_deck();
    deck.shuffle(rng);
    let mut h = Hand::new(&cfg, 0, deck);
    let raise_to = 200 * rng.random_range(1..=10);
    h.apply(PlayerAction::RaiseTo(raise_to)); // p0 opens
    h.apply(PlayerAction::CheckCall); // p1 calls
    for _ in 0..2 {
        h.apply(PlayerAction::CheckCall); // p1 checks flop/turn
        h.apply(PlayerAction::CheckCall); // p0 checks behind
    }
    h.apply(PlayerAction::CheckCall); // p1 checks the river
    debug_assert_eq!(h.street(), crate::engine::Street::River);
    debug_assert_eq!(h.to_act(), 0);
    h
}

/// Log-normal-ish random range weights.
fn random_range(rng: &mut SmallRng) -> Vec<f64> {
    (0..NUM_COMBOS)
        .map(|_| {
            let u: f64 = rng.random_range(-2.0..2.0);
            u.exp()
        })
        .collect()
}

/// True-range-weighted mean and max of per-combo margins `(br − alt)⁺`.
fn margins(solver: &RiverSolver, alt: &[f64], truth: &[f64]) -> (f64, f64) {
    let br = solver.best_response_values(1);
    let compat = solver.compat_mass(1);
    let mut num = 0.0;
    let mut den = 0.0;
    let mut max = 0.0f64;
    for ci in 0..NUM_COMBOS {
        if truth[ci] <= 0.0 || compat[ci] <= 0.0 {
            continue;
        }
        let m = (br[ci] / compat[ci] - alt[ci]).max(0.0);
        num += truth[ci] * m;
        den += truth[ci];
        max = max.max(m);
    }
    (if den > 0.0 { num / den } else { 0.0 }, max)
}

pub fn run(spots: usize, iters: u64, seed: u64) -> Vec<AblateRow> {
    let abs = Abstraction::new(AbsConfig {
        postflop_buckets: 6,
        equity_rollouts: 40,
        cache_cap: 100_000,
        ..AbsConfig::default()
    });
    let epsilons = [0.25, 0.5, 1.0];

    // Per spot × epsilon: (unsafe_mean, unsafe_max, safe_mean, safe_max, pot)
    let results: Vec<Vec<(f64, f64, f64, f64, f64)>> = (0..spots)
        .into_par_iter()
        .map(|i| {
            let mut rng =
                SmallRng::seed_from_u64(seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let h = random_river_spot(&mut rng);
            let truth = random_range(&mut rng);
            let hero = random_range(&mut rng);

            // Baseline: equilibrium under the true range -> safety values.
            let mut base = RiverSolver::build(&h, &abs, [&hero, &truth]).unwrap();
            base.solve(iters, 120_000, None);
            let alt = base.root_values(1);

            epsilons
                .iter()
                .map(|&eps| {
                    let noise = random_range(&mut rng);
                    let t_sum: f64 = truth.iter().sum();
                    let n_sum: f64 = noise.iter().sum();
                    let belief: Vec<f64> = truth
                        .iter()
                        .zip(&noise)
                        .map(|(&t, &n)| (1.0 - eps) * t / t_sum + eps * n / n_sum)
                        .collect();

                    let mut uns = RiverSolver::build(&h, &abs, [&hero, &belief]).unwrap();
                    uns.solve(iters, 120_000, None);
                    let (um, ux) = margins(&uns, &alt, &truth);

                    let mut safe = RiverSolver::build(&h, &abs, [&hero, &belief])
                        .unwrap()
                        .with_gadget(alt.clone());
                    safe.solve(iters, 120_000, None);
                    let (sm, sx) = margins(&safe, &alt, &truth);

                    (um, ux, sm, sx, h.pot() as f64)
                })
                .collect()
        })
        .collect();

    epsilons
        .iter()
        .enumerate()
        .map(|(ei, &epsilon)| {
            let n = results.len() as f64;
            let mut row = AblateRow {
                epsilon,
                unsafe_mean: 0.0,
                unsafe_max: 0.0,
                safe_mean: 0.0,
                safe_max: 0.0,
                pot: 0.0,
            };
            for spot in &results {
                let (um, ux, sm, sx, pot) = spot[ei];
                row.unsafe_mean += um / n;
                row.unsafe_max = row.unsafe_max.max(ux);
                row.safe_mean += sm / n;
                row.safe_max = row.safe_max.max(sx);
                row.pot += pot / n;
            }
            row
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests (written first, TDD)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// The core safety claim, as a unit test: with a badly wrong belief
    /// (ε = 1: pure noise), unsafe resolving gives the opponent's true range
    /// significantly more than its safety values; gadget resolving does not.
    #[test]
    fn gadget_bounds_margin_under_wrong_beliefs() {
        let rows = run(6, 250, 42);
        let worst = rows.last().unwrap(); // epsilon = 1.0
        assert!(
            worst.unsafe_mean > 2.0 * worst.safe_mean.max(1.0),
            "safe resolving must at least halve the mean margin: unsafe {:.1} vs safe {:.1} (pot {:.0})",
            worst.unsafe_mean,
            worst.safe_mean,
            worst.pot
        );
        assert!(
            worst.safe_mean < 0.05 * worst.pot,
            "safe mean margin should be small relative to the pot: {:.1} vs pot {:.0}",
            worst.safe_mean,
            worst.pot
        );
    }
}
