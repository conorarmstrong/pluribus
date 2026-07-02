//! Table wrapper (real hand + abstract shadow hand + history tokens) and the
//! evaluation harness that measures a policy's winrate against baselines.
//!
//! The shadow hand mirrors the real hand through *abstract* actions, so that
//! infoset histories stay on-tree even when a human makes an off-tree bet.
//! If the two ever diverge structurally (rare all-in edge cases), the shadow
//! resyncs to the real hand and the bot falls back gracefully on lookups.

use crate::abstraction::{AbsAction, Abstraction, TOKEN_STREET_SEP};
use crate::bot::Policy;
use crate::cards::fresh_deck;
use crate::engine::{Hand, HandConfig, PlayerAction};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

pub struct Table {
    pub real: Hand,
    pub shadow: Hand,
    pub hist: Vec<u8>,
}

impl Table {
    pub fn new(cfg: &HandConfig, button: usize, deck: [u8; 52]) -> Table {
        let real = Hand::new(cfg, button, deck);
        Table {
            shadow: real.clone(),
            real,
            hist: Vec::with_capacity(32),
        }
    }

    /// Apply a bot-chosen abstract action to both hands.
    pub fn apply_abs(&mut self, a: AbsAction, abs: &Abstraction) {
        let street_before = self.shadow.street();
        let shadow_act = abs.concrete(&self.shadow, a);
        self.shadow.apply(shadow_act);
        // Concrete amount computed from the real pot, so real bets stay
        // sensible even if the shadow has drifted.
        let real_act = abs.concrete(&self.real, a);
        self.real.apply(real_act);
        self.push_token(a, self.shadow.street() != street_before);
        self.resync_if_diverged();
    }

    /// The abstract action the shadow hand will follow for a concrete
    /// (possibly off-tree) action — exposed so callers can Bayes-update
    /// tracked ranges before applying.
    pub fn map_concrete(&self, a: PlayerAction, abs: &Abstraction) -> AbsAction {
        match a {
            PlayerAction::Fold => {
                if self.real.can_check() {
                    AbsAction::CheckCall // engine treats fold-when-free as check
                } else {
                    AbsAction::Fold
                }
            }
            PlayerAction::CheckCall => AbsAction::CheckCall,
            PlayerAction::RaiseTo(x) => {
                if self.shadow.raise_bounds().is_some() {
                    abs.map_raise(&self.shadow, x)
                } else {
                    AbsAction::CheckCall
                }
            }
        }
    }

    /// Apply a human's concrete action; the shadow follows the nearest
    /// abstract action.
    pub fn apply_concrete(&mut self, a: PlayerAction, abs: &Abstraction) {
        let street_before = self.shadow.street();
        let abs_a = self.map_concrete(a, abs);
        self.shadow.apply(abs.concrete(&self.shadow, abs_a));
        self.real.apply(a);
        self.push_token(abs_a, self.shadow.street() != street_before);
        self.resync_if_diverged();
    }

    fn push_token(&mut self, a: AbsAction, street_changed: bool) {
        self.hist.push(a.token());
        if street_changed && !self.shadow.is_terminal() {
            self.hist.push(TOKEN_STREET_SEP);
        }
    }

    fn resync_if_diverged(&mut self) {
        if self.shadow.to_act() != self.real.to_act()
            || self.shadow.street() != self.real.street()
            || self.shadow.is_terminal() != self.real.is_terminal()
        {
            self.shadow = self.real.clone();
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Baseline {
    /// Uniform random over the abstract action menu.
    Random,
    /// Always check/call.
    Caller,
}

impl std::str::FromStr for Baseline {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "random" => Ok(Baseline::Random),
            "caller" => Ok(Baseline::Caller),
            other => Err(format!("unknown baseline '{other}' (random|caller)")),
        }
    }
}

pub fn baseline_action(b: Baseline, h: &Hand, abs: &Abstraction, rng: &mut SmallRng) -> AbsAction {
    match b {
        Baseline::Caller => AbsAction::CheckCall,
        Baseline::Random => {
            let acts = abs.abstract_actions(h);
            acts[rng.random_range(0..acts.len())]
        }
    }
}

#[derive(Debug)]
pub struct EvalResult {
    pub hands: u64,
    /// Mean winnings in milli-big-blinds per hand.
    pub mbb_per_hand: f64,
    /// Half-width of the 95% confidence interval, in mbb/hand.
    pub ci95: f64,
}

/// Play `hands` hands of `policy` (one seat, rotating) against baselines in
/// every other seat. Stacks reset each hand (as in the Pluribus experiment);
/// button rotates. Returns winrate in mbb/hand with a 95% CI.
pub fn run_eval(
    policy: &Policy,
    cfg: &HandConfig,
    baseline: Baseline,
    hands: u64,
    seed: u64,
) -> EvalResult {
    let n = cfg.num_players;
    let bb = cfg.bb as f64;
    let results: Vec<f64> = (0..hands)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(seed ^ i.wrapping_mul(0x2545_F491_4F6C_DD1D));
            let hero = (i % n as u64) as usize;
            let button = rng.random_range(0..n);
            let mut deck = fresh_deck();
            deck.shuffle(&mut rng);
            let mut table = Table::new(cfg, button, deck);
            let mut guard = 0;
            while !table.real.is_terminal() {
                guard += 1;
                assert!(guard < 500, "eval hand did not terminate");
                let p = table.real.to_act();
                let a = if p == hero {
                    policy.act_blueprint(&table.shadow, &table.hist, &mut rng)
                } else {
                    baseline_action(baseline, &table.shadow, &policy.abs, &mut rng)
                };
                table.apply_abs(a, &policy.abs);
            }
            table.real.utilities()[hero] as f64 / bb * 1000.0
        })
        .collect();

    let mean = results.iter().sum::<f64>() / results.len() as f64;
    let var = results.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>()
        / (results.len().saturating_sub(1)) as f64;
    let ci95 = 1.96 * (var / results.len() as f64).sqrt();
    EvalResult {
        hands,
        mbb_per_hand: mean,
        ci95,
    }
}

// ---------------------------------------------------------------------------
// Tests (written first, TDD)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::AbsConfig;
    use crate::cfr::Blueprint;
    use std::collections::HashMap;

    fn abs_small() -> Abstraction {
        Abstraction::new(AbsConfig {
            postflop_buckets: 6,
            equity_rollouts: 40,
            cache_cap: 100_000,
            ..AbsConfig::default()
        })
    }

    fn empty_policy() -> Policy {
        Policy::new(
            Blueprint {
                strategies: HashMap::new(),
                iterations: 0,
                num_players: 6,
                abs_cfg: AbsConfig::default(),
                centroids: None,
            },
            std::sync::Arc::new(abs_small()),
        )
    }

    #[test]
    fn shadow_stays_synced_under_abstract_play() {
        use rand::seq::IndexedRandom;
        let abs = abs_small();
        let mut rng = SmallRng::seed_from_u64(9);
        for _ in 0..300 {
            let mut deck = fresh_deck();
            deck.shuffle(&mut rng);
            let mut t = Table::new(&HandConfig::default(), rng.random_range(0..6), deck);
            while !t.real.is_terminal() {
                let acts = abs.abstract_actions(&t.shadow);
                let &a = acts.choose(&mut rng).unwrap();
                t.apply_abs(a, &abs);
                assert_eq!(t.real.to_act(), t.shadow.to_act());
                assert_eq!(t.real.street(), t.shadow.street());
                assert_eq!(t.real.pot(), t.shadow.pot());
                assert_eq!(t.real.is_terminal(), t.shadow.is_terminal());
            }
        }
    }

    #[test]
    fn concrete_offtree_bets_are_tracked() {
        let abs = abs_small();
        let mut deck = fresh_deck();
        let mut rng = SmallRng::seed_from_u64(4);
        deck.shuffle(&mut rng);
        let mut t = Table::new(&HandConfig::default(), 0, deck);
        // Human UTG makes an odd raise; shadow should map it to a raise token
        // and stay structurally in sync.
        t.apply_concrete(PlayerAction::RaiseTo(317), &abs);
        assert_eq!(t.hist.len(), 1);
        assert!(t.hist[0] >= 2, "expected a raise-family token");
        assert_eq!(t.real.to_act(), t.shadow.to_act());
        assert_eq!(t.real.street(), t.shadow.street());
        // Real pot reflects the real bet.
        assert_eq!(t.real.pot(), 150 + 317);
    }

    #[test]
    fn folding_when_free_records_check() {
        let abs = abs_small();
        let mut t = Table::new(&HandConfig::default(), 0, fresh_deck());
        for _ in 0..5 {
            t.apply_concrete(PlayerAction::CheckCall, &abs);
        }
        assert!(t.real.can_check()); // BB option
        t.apply_concrete(PlayerAction::Fold, &abs);
        // Engine treats fold-when-free as a check; history must agree.
        assert_eq!(*t.hist.last().unwrap(), TOKEN_STREET_SEP);
        assert!(!t.real.folded(2));
        assert_eq!(t.real.street(), crate::engine::Street::Flop);
    }

    /// A calling-station policy against calling-station baselines is a
    /// symmetric game: winrate must be statistically near zero, and the
    /// harness must complete quickly.
    #[test]
    fn eval_symmetric_matchup_is_near_zero() {
        let policy = empty_policy();
        let r = run_eval(
            &policy,
            &HandConfig::default(),
            Baseline::Caller,
            4_000,
            123,
        );
        assert_eq!(r.hands, 4_000);
        assert!(
            r.mbb_per_hand.abs() < 4.0 * r.ci95.max(1.0),
            "symmetric matchup should be ~0, got {} +/- {}",
            r.mbb_per_hand,
            r.ci95
        );
    }
}
