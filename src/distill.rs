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

use crate::abstraction::AbsAction;
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

pub struct DistillCfg {
    pub hands: u64,
    pub params: SearchParams,
    /// Blend weight toward the search distribution at recorded infosets.
    pub alpha: f64,
    pub seed: u64,
}

/// Accumulated teacher signal: infoset key → (summed distribution, count).
type Records = DashMap<Vec<u8>, (Vec<f64>, f64)>;

/// Re-express a solver distribution over `slim` actions on the `full`
/// blueprint menu, matching actions by identity (mass on actions the full
/// menu lacks is dropped and the rest renormalized). None when no mass
/// survives. The slim-menu solvers (turn, flop-net) return distributions
/// shorter than the blueprint's stored entries; recording them unaligned
/// would corrupt the blueprint (a length-mismatched entry falls back to
/// pure check/call at play time).
pub fn expand_dist(slim: &[AbsAction], dist: &[f64], full: &[AbsAction]) -> Option<Vec<f64>> {
    debug_assert_eq!(slim.len(), dist.len());
    let mut out = vec![0.0; full.len()];
    let mut kept = 0.0;
    for (a, &p) in slim.iter().zip(dist) {
        if let Some(i) = full.iter().position(|f| f == a) {
            out[i] += p;
            kept += p;
        }
    }
    if kept <= 0.0 {
        return None;
    }
    for o in &mut out {
        *o /= kept;
    }
    Some(out)
}

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
                    Some(&table.round),
                    &mut rng,
                );
                if table.real.street() != Street::Preflop {
                    // Preflop is pure blueprint (no teacher there); record
                    // every searched postflop distribution, re-expressed on
                    // the full blueprint menu (slim-menu solvers return
                    // shorter vectors).
                    let full = policy.abs.abstract_actions(&table.real);
                    if let Some(expanded) = expand_dist(&acts, &probs, &full) {
                        let mut krng = SmallRng::seed_from_u64(0xD157_5EED);
                        let bucket = policy.abs.bucket(
                            table.real.hole(p),
                            table.real.board(),
                            &mut krng,
                        );
                        let key = make_key(bucket, &table.hist).to_vec();
                        let mut e =
                            records.entry(key).or_insert((vec![0.0; expanded.len()], 0.0));
                        if e.0.len() == expanded.len() {
                            for (a, &x) in e.0.iter_mut().zip(&expanded) {
                                *a += x;
                            }
                            e.1 += 1.0;
                            count += 1;
                        }
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

/// Blend the recorded search distributions into the blueprint IN PLACE
/// (taken by value: a multi-GB blueprint must not be cloned): at each
/// recorded key, `new = (1-alpha)·old + alpha·mean(search)` (both
/// normalized first). Keys the blueprint never visited are inserted
/// outright. Length-mismatched entries are SKIPPED, never overwritten —
/// a shorter entry falls back to pure check/call at play time, which is
/// exactly the corruption that sank the first flywheel generation.
/// Returns the blueprint, keys changed, and keys skipped on mismatch.
pub fn merge(mut bp: Blueprint, records: &Records, alpha: f64) -> (Blueprint, u64, u64) {
    let mut updated = 0u64;
    let mut skipped = 0u64;
    for e in records.iter() {
        let (sum, count) = e.value();
        if *count <= 0.0 {
            continue;
        }
        let mean: Vec<f64> = sum.iter().map(|x| x / count).collect();
        let blended: Vec<f32> = match bp.strategies.get(e.key()) {
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
            Some(_) => {
                skipped += 1;
                continue;
            }
            None => mean.iter().map(|&m| m as f32).collect(),
        };
        bp.strategies.insert(e.key().clone(), blended);
        updated += 1;
    }
    (bp, updated, skipped)
}

// ---------------------------------------------------------------------------
// Tests (written first, TDD)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::{AbsConfig, Abstraction, Centroids};
    use crate::cfr::{TrainConfig, Trainer};
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn merge_blends_inserts_and_skips_mismatches() {
        let mut strategies = crate::cfr::StrategyMap::default();
        // Existing key, matching length, unnormalized on purpose.
        strategies.insert(vec![1u8], vec![3.0f32, 1.0]); // normalizes to .75/.25
        // Existing key whose stored menu length differs from the record.
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
        // One sample at the mismatched key (record has 2 actions).
        records.insert(vec![2u8], (vec![0.9, 0.1], 1.0));
        // A brand-new key.
        records.insert(vec![3u8], (vec![0.6, 0.4], 1.0));

        let (bp, updated, skipped) = merge(old, &records, 0.5);
        assert_eq!(updated, 2);
        assert_eq!(skipped, 1, "length mismatches are skipped, never overwritten");
        let s1 = &bp.strategies[&vec![1u8]];
        assert!((s1[0] - 0.5).abs() < 1e-6, "0.5·0.75 + 0.5·0.25 = 0.5, got {}", s1[0]);
        assert!((s1[1] - 0.5).abs() < 1e-6);
        let s2 = &bp.strategies[&vec![2u8]];
        assert_eq!(s2.len(), 3, "mismatched entry must be left intact");
        assert_eq!(s2[0], 1.0f32);
        let s3 = &bp.strategies[&vec![3u8]];
        assert!((s3[0] - 0.6).abs() < 1e-6);
        // Metadata preserved.
        assert_eq!(bp.iterations, 7);
    }

    /// Slim-menu solver distributions must re-express exactly onto the
    /// full blueprint menu by action identity.
    #[test]
    fn expand_dist_aligns_by_action_identity() {
        use crate::abstraction::AbsAction::*;
        let full = vec![Fold, CheckCall, Bet(2), Bet(3), Bet(4), AllIn];
        let slim = vec![Fold, CheckCall, Bet(3), AllIn];
        let d = expand_dist(&slim, &[0.1, 0.4, 0.3, 0.2], &full).unwrap();
        assert_eq!(d, vec![0.1, 0.4, 0.0, 0.3, 0.0, 0.2]);

        // Mass on actions the full menu lacks is dropped and renormalized.
        let slim2 = vec![CheckCall, Bet(7)];
        let d2 = expand_dist(&slim2, &[0.5, 0.5], &full).unwrap();
        assert!((d2[1] - 1.0).abs() < 1e-12);
        // No surviving mass → None.
        assert!(expand_dist(&[Bet(7)], &[1.0], &full).is_none());
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
            menu: crate::abstraction::MenuShape::Wide,
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
                round_root: true,
            },
            alpha: 0.5,
            seed: 42,
        };
        let (records, samples) = collect(&policy, &hand, &dcfg);
        assert!(samples > 0, "self-play must record searched decisions");
        let n_before = policy.blueprint.strategies.len();
        let owned = (*policy.blueprint).clone();
        let (bp2, updated, skipped) = merge(owned, &records, dcfg.alpha);
        assert!(updated > 0);
        assert_eq!(
            skipped, 0,
            "collect-time expansion must leave nothing length-mismatched"
        );
        assert!(bp2.strategies.len() >= n_before);
        // Every recorded key's entry is a valid distribution of the same
        // length as the record (nothing corrupted by menu mismatch).
        for e in records.iter() {
            let s = &bp2.strategies[e.key()];
            assert_eq!(s.len(), e.value().0.len());
            let total: f32 = s.iter().sum();
            assert!((total - 1.0).abs() < 1e-3, "must normalize, got {total}");
            assert!(s.iter().all(|&x| (0.0..=1.0).contains(&x)));
        }
    }
}
