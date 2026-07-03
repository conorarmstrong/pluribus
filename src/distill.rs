//! Expert-iteration distillation (the flywheel): self-play with online
//! search, recording the RESOLVED action distribution at every searched
//! postflop decision, then blending those distributions back into the
//! blueprint. The search is measurably stronger than the blueprint it
//! reads from (see BASELINES.md), so its outputs are a teacher signal; a
//! distilled blueprint both plays better raw and gives the next round of
//! search better range tracking (beliefs are Bayes updates through
//! blueprint action probabilities).
//!
//! Theory note: resolves respond to tracked ranges, so distillation has no
//! equilibrium-convergence guarantee — it is an empirical bet, gated by
//! the BR probe, cross-play vs the parent, and baseline winrates.

use crate::bot::{Policy, SearchParams, SearchSession};
use crate::cards::fresh_deck;
use crate::cfr::{make_key, Blueprint};
use crate::engine::{HandConfig, Street};
use crate::search::RangeTracker;
use crate::table::Table;
use dashmap::DashMap;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;

pub struct DistillCfg {
    pub hands: u64,
    pub params: SearchParams,
    /// Blend weight toward the search distribution at recorded infosets.
    pub alpha: f64,
    pub seed: u64,
}

/// Accumulated teacher signal: infoset key → (summed distribution, count).
type Records = DashMap<Vec<u8>, (Vec<f64>, f64)>;

/// Self-play `hands` hands with every seat searching, recording resolved
/// distributions at searched postflop decisions. Returns the accumulated
/// records and the number of recorded decisions.
pub fn collect(policy: &Policy, cfg: &HandConfig, dcfg: &DistillCfg) -> (Records, u64) {
    let n = cfg.num_players;
    let train_cfg = crate::cfr::TrainConfig {
        hand: cfg.clone(),
        prune_after: u64::MAX,
        ..crate::cfr::TrainConfig::default()
    };
    let records: Records = DashMap::new();
    let samples: u64 = (0..dcfg.hands)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(
                dcfg.seed ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD157,
            );
            let button = rng.random_range(0..n);
            let mut deck = fresh_deck();
            deck.shuffle(&mut rng);
            let mut table = Table::new(cfg, button, deck);
            let mut tracker = RangeTracker::new(n);
            let mut sessions: Vec<SearchSession> =
                (0..n).map(|_| SearchSession::new()).collect();
            let mut count = 0u64;
            let mut guard = 0;
            while !table.real.is_terminal() {
                guard += 1;
                assert!(guard < 500, "distill hand did not terminate");
                let p = table.real.to_act();
                let (acts, probs) = policy.search_dist(
                    &table.real,
                    &table.shadow,
                    &table.hist,
                    dcfg.params,
                    &train_cfg,
                    Some(&tracker),
                    Some(&mut sessions[p]),
                    &mut rng,
                );
                if table.real.street() != Street::Preflop {
                    // Preflop is pure blueprint (no teacher there); record
                    // every searched postflop distribution.
                    let mut krng = SmallRng::seed_from_u64(0xD157_5EED);
                    let bucket = policy.abs.bucket(
                        table.real.hole(p),
                        table.real.board(),
                        &mut krng,
                    );
                    let key = make_key(bucket, &table.hist).to_vec();
                    let mut e = records.entry(key).or_insert((vec![0.0; probs.len()], 0.0));
                    if e.0.len() == probs.len() {
                        for (a, &x) in e.0.iter_mut().zip(&probs) {
                            *a += x;
                        }
                        e.1 += 1.0;
                        count += 1;
                    }
                }
                let a = acts[crate::cfr::sample_index(&probs, &mut rng)];
                tracker.observe(p, a, &table.shadow, &table.hist, &policy.blueprint, &policy.abs);
                let street_before = table.real.street();
                table.apply_abs(a, &policy.abs);
                if table.real.street() != street_before {
                    tracker.exclude(table.real.board());
                }
            }
            count
        })
        .sum();
    (records, samples)
}

/// Blend the recorded search distributions into a copy of the blueprint:
/// at each recorded key, `new = (1-alpha)·old + alpha·mean(search)` (both
/// normalized first). Keys the blueprint never visited are inserted
/// outright; length-mismatched entries (menu drift) are overwritten with
/// the search distribution. Returns the new blueprint and how many keys
/// changed.
pub fn merge(old: &Blueprint, records: &Records, alpha: f64) -> (Blueprint, u64) {
    let mut strategies: HashMap<Vec<u8>, Vec<f32>> = old.strategies.clone();
    let mut updated = 0u64;
    for e in records.iter() {
        let (sum, count) = e.value();
        if *count <= 0.0 {
            continue;
        }
        let mean: Vec<f64> = sum.iter().map(|x| x / count).collect();
        let blended: Vec<f32> = match strategies.get(e.key()) {
            Some(oldv) if oldv.len() == mean.len() => {
                let total: f64 = oldv.iter().map(|&x| x as f64).sum();
                if total > 0.0 {
                    oldv.iter()
                        .zip(&mean)
                        .map(|(&o, &m)| ((1.0 - alpha) * (o as f64 / total) + alpha * m) as f32)
                        .collect()
                } else {
                    mean.iter().map(|&m| m as f32).collect()
                }
            }
            _ => mean.iter().map(|&m| m as f32).collect(),
        };
        strategies.insert(e.key().clone(), blended);
        updated += 1;
    }
    (
        Blueprint {
            strategies,
            iterations: old.iterations,
            num_players: old.num_players,
            abs_cfg: old.abs_cfg.clone(),
            centroids: old.centroids.clone(),
        },
        updated,
    )
}

// ---------------------------------------------------------------------------
// Tests (written first, TDD)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::{AbsConfig, Abstraction, Centroids};
    use crate::cfr::{TrainConfig, Trainer};
    use std::sync::Arc;

    #[test]
    fn merge_blends_inserts_and_overwrites() {
        let mut strategies = HashMap::new();
        // Existing key, matching length, unnormalized on purpose.
        strategies.insert(vec![1u8], vec![3.0f32, 1.0]); // normalizes to .75/.25
        // Existing key with a stale menu length.
        strategies.insert(vec![2u8], vec![1.0f32, 1.0, 1.0]);
        let old = Blueprint {
            strategies,
            iterations: 7,
            num_players: 2,
            abs_cfg: AbsConfig::default(),
            centroids: None,
        };
        let records: Records = DashMap::new();
        // Two samples at key 1 averaging to (0.25, 0.75).
        records.insert(vec![1u8], (vec![0.5, 1.5], 2.0));
        // One sample at the stale key (new menu has 2 actions).
        records.insert(vec![2u8], (vec![0.9, 0.1], 1.0));
        // A brand-new key.
        records.insert(vec![3u8], (vec![0.6, 0.4], 1.0));

        let (bp, updated) = merge(&old, &records, 0.5);
        assert_eq!(updated, 3);
        let s1 = &bp.strategies[&vec![1u8]];
        assert!((s1[0] - 0.5).abs() < 1e-6, "0.5·0.75 + 0.5·0.25 = 0.5, got {}", s1[0]);
        assert!((s1[1] - 0.5).abs() < 1e-6);
        let s2 = &bp.strategies[&vec![2u8]];
        assert_eq!(s2.len(), 2, "stale-length entries are overwritten");
        assert!((s2[0] - 0.9).abs() < 1e-6);
        let s3 = &bp.strategies[&vec![3u8]];
        assert!((s3[0] - 0.6).abs() < 1e-6);
        // Metadata preserved.
        assert_eq!(bp.iterations, 7);
    }

    /// End to end on a tiny blueprint: collection records searched postflop
    /// decisions, the merged blueprint stays a valid strategy store, and
    /// the round trip is loadable.
    #[test]
    fn distill_smoke_produces_a_valid_blueprint() {
        let abs_cfg = AbsConfig {
            postflop_buckets: 6,
            equity_rollouts: 40,
            dist_runouts: 8,
            runout_rollouts: 20,
            cache_cap: 500_000,
        };
        let cents = Centroids::train(&abs_cfg, 300, 5);
        let abs = Arc::new(Abstraction::with_centroids(abs_cfg, Some(cents)));
        let hand = HandConfig {
            num_players: 2,
            stack: 1_000,
            sb: 50,
            bb: 100,
        };
        let tcfg = TrainConfig {
            hand: hand.clone(),
            prune_after: u64::MAX,
            ..TrainConfig::default()
        };
        let trainer = Trainer::new(abs.clone(), tcfg);
        trainer.run(30_000, &|_| {});
        let bp = trainer.to_blueprint();
        let policy = Policy::new(bp, abs);

        let dcfg = DistillCfg {
            hands: 40,
            params: SearchParams {
                time_ms: 300,
                max_iters: 500,
                qre_lambda: None,
                safe_resolve: false,
                adaptive: false,
            },
            alpha: 0.5,
            seed: 42,
        };
        let (records, samples) = collect(&policy, &hand, &dcfg);
        assert!(samples > 0, "self-play must record searched decisions");
        let (bp2, updated) = merge(&policy.blueprint, &records, dcfg.alpha);
        assert!(updated > 0);
        assert!(bp2.strategies.len() >= policy.blueprint.strategies.len());
        // Every recorded key's entry is a valid distribution.
        for e in records.iter() {
            let s = &bp2.strategies[e.key()];
            let total: f32 = s.iter().sum();
            assert!((total - 1.0).abs() < 1e-3, "must normalize, got {total}");
            assert!(s.iter().all(|&x| (0.0..=1.0).contains(&x)));
        }
    }
}
