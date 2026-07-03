//! Strategy portfolio with an online bandit: hold several blueprints (e.g.
//! an equilibrium and restricted-Nash-response exploiters at increasing p),
//! and select which to play per hand with UCB1 as evidence about the
//! opposition accumulates. Exploitation slides in only when it demonstrably
//! pays; against opponents that punish the exploiters, the bandit retreats
//! to the equilibrium arm — bounded-downside exploitation.

use crate::bot::Policy;
use crate::engine::HandConfig;
use crate::table::{baseline_action, Baseline, EvalResult, Table};
use crate::cards::fresh_deck;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

pub struct Portfolio {
    pub arms: Vec<Policy>,
    sums: Vec<f64>,
    counts: Vec<u64>,
    total: u64,
    /// Reward scale for UCB exploration (≈ per-hand outcome std, in mbb).
    scale: f64,
}

impl Portfolio {
    pub fn new(arms: Vec<Policy>, scale: f64) -> Portfolio {
        let n = arms.len();
        Portfolio {
            arms,
            sums: vec![0.0; n],
            counts: vec![0; n],
            total: 0,
            scale,
        }
    }

    /// UCB1 arm selection: each arm once, then mean + scale·sqrt(2 ln T / n).
    pub fn choose(&self) -> usize {
        if let Some(unplayed) = self.counts.iter().position(|&c| c == 0) {
            return unplayed;
        }
        let t = (self.total as f64).ln();
        (0..self.arms.len())
            .max_by(|&a, &b| {
                let ucb = |i: usize| {
                    self.sums[i] / self.counts[i] as f64
                        + self.scale * (2.0 * t / self.counts[i] as f64).sqrt()
                };
                ucb(a).total_cmp(&ucb(b))
            })
            .unwrap()
    }

    pub fn update(&mut self, arm: usize, mbb: f64) {
        self.sums[arm] += mbb;
        self.counts[arm] += 1;
        self.total += 1;
    }

    pub fn counts(&self) -> &[u64] {
        &self.counts
    }
}

/// Sequential bandit evaluation: the hero picks a portfolio arm per hand
/// against `baseline` opponents. Returns the overall result and per-arm
/// play counts.
pub fn run_portfolio_eval(
    portfolio: &mut Portfolio,
    cfg: &HandConfig,
    baseline: Baseline,
    hands: u64,
    seed: u64,
) -> (EvalResult, Vec<u64>) {
    let n = cfg.num_players;
    let bb = cfg.bb as f64;
    let mut results = Vec::with_capacity(hands as usize);
    for i in 0..hands {
        let mut rng = SmallRng::seed_from_u64(seed ^ i.wrapping_mul(0x2545_F491_4F6C_DD1D));
        let hero = (i % n as u64) as usize;
        let button = rng.random_range(0..n);
        let mut deck = fresh_deck();
        deck.shuffle(&mut rng);
        let arm = portfolio.choose();
        let policy = &portfolio.arms[arm];
        let mut table = Table::new(cfg, button, deck);
        let mut guard = 0;
        while !table.real.is_terminal() {
            guard += 1;
            assert!(guard < 500, "portfolio hand did not terminate");
            let p = table.real.to_act();
            let a = if p == hero {
                policy.act_blueprint(&table.shadow, &table.hist, &mut rng)
            } else {
                baseline_action(baseline, &table.shadow, &policy.abs, &mut rng)
            };
            table.apply_abs(a, &policy.abs);
        }
        let mbb = table.real.utilities()[hero] as f64 / bb * 1000.0;
        portfolio.update(arm, mbb);
        results.push(mbb);
    }
    let mean = results.iter().sum::<f64>() / results.len() as f64;
    let var = results.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>()
        / (results.len().saturating_sub(1)) as f64;
    (
        EvalResult {
            hands,
            mbb_per_hand: mean,
            ci95: 1.96 * (var / results.len() as f64).sqrt(),
        },
        portfolio.counts().to_vec(),
    )
}

// ---------------------------------------------------------------------------
// Tests (written first, TDD)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::{AbsConfig, Abstraction, Centroids};
    use crate::cfr::{RnrCfg, TrainConfig, Trainer};
    use std::sync::Arc;

    fn train_arm(rnr: Option<RnrCfg>) -> Policy {
        let abs_cfg = AbsConfig {
            postflop_buckets: 6,
            equity_rollouts: 50,
            dist_runouts: 12,
            runout_rollouts: 25,
            cache_cap: 1_000_000,
        };
        let cents = Centroids::train(&abs_cfg, 400, 99);
        let abs = Arc::new(Abstraction::with_centroids(abs_cfg, Some(cents)));
        let cfg = TrainConfig {
            hand: HandConfig {
                num_players: 2,
                stack: 1_000,
                sb: 50,
                bb: 100,
            },
            prune_after: u64::MAX,
            ..TrainConfig::default()
        };
        let t = Trainer::new(abs.clone(), cfg).with_rnr(rnr);
        t.run(120_000, &|_| {});
        let bp = t.to_blueprint();
        Policy::new(bp, abs)
    }

    /// Against a calling station, the bandit must discover and mostly play
    /// the caller-exploiting RNR arm, and end up winning more than the
    /// equilibrium arm would alone.
    #[test]
    fn bandit_converges_to_the_exploitative_arm() {
        let cfg = HandConfig {
            num_players: 2,
            stack: 1_000,
            sb: 50,
            bb: 100,
        };
        let nash = train_arm(None);
        let rnr = train_arm(Some(RnrCfg {
            model: Baseline::Caller,
            p: 0.9,
        }));
        let w_nash = crate::table::run_eval(&nash, &cfg, Baseline::Caller, 30_000, 5);

        let mut portfolio = Portfolio::new(vec![nash, rnr], 4_000.0);
        let (total, counts) =
            run_portfolio_eval(&mut portfolio, &cfg, Baseline::Caller, 30_000, 7);
        assert!(
            counts[1] > counts[0] * 2,
            "bandit must favor the RNR arm vs a caller: counts {counts:?}"
        );
        assert!(
            total.mbb_per_hand > w_nash.mbb_per_hand,
            "portfolio must beat equilibrium-alone vs a caller: \
             portfolio {:+.0}±{:.0} vs nash {:+.0}±{:.0}",
            total.mbb_per_hand,
            total.ci95,
            w_nash.mbb_per_hand,
            w_nash.ci95
        );
    }
}
