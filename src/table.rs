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

    /// Apply a bot-chosen abstract action to both hands. Returns the concrete
    /// action applied to the real hand (used by AIVAT replay).
    pub fn apply_abs(&mut self, a: AbsAction, abs: &Abstraction) -> PlayerAction {
        let street_before = self.shadow.street();
        let shadow_act = abs.concrete(&self.shadow, a);
        self.shadow.apply(shadow_act);
        // Concrete amount computed from the real pot, so real bets stay
        // sensible even if the shadow has drifted.
        let real_act = abs.concrete(&self.real, a);
        self.real.apply(real_act);
        self.push_token(a, self.shadow.street() != street_before);
        self.resync_if_diverged();
        real_act
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

/// Play one hand: `policy` in seat `hero`, `baseline` everywhere else.
/// Returns the hero's net chips in mbb.
fn play_one(
    policy: &Policy,
    cfg: &HandConfig,
    baseline: Baseline,
    deck: [u8; 52],
    button: usize,
    hero: usize,
    rng: &mut SmallRng,
) -> f64 {
    let mut table = Table::new(cfg, button, deck);
    let mut guard = 0;
    while !table.real.is_terminal() {
        guard += 1;
        assert!(guard < 500, "eval hand did not terminate");
        let p = table.real.to_act();
        let a = if p == hero {
            policy.act_blueprint(&table.shadow, &table.hist, rng)
        } else {
            baseline_action(baseline, &table.shadow, &policy.abs, rng)
        };
        table.apply_abs(a, &policy.abs);
    }
    table.real.utilities()[hero] as f64 / cfg.bb as f64 * 1000.0
}

fn mean_ci(results: &[f64]) -> (f64, f64) {
    let mean = results.iter().sum::<f64>() / results.len() as f64;
    let var = results.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>()
        / (results.len().saturating_sub(1)) as f64;
    (mean, 1.96 * (var / results.len() as f64).sqrt())
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
    let results: Vec<f64> = (0..hands)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(seed ^ i.wrapping_mul(0x2545_F491_4F6C_DD1D));
            let hero = (i % n as u64) as usize;
            let button = rng.random_range(0..n);
            let mut deck = fresh_deck();
            deck.shuffle(&mut rng);
            play_one(policy, cfg, baseline, deck, button, hero, &mut rng)
        })
        .collect();

    let (mean, ci95) = mean_ci(&results);
    EvalResult {
        hands,
        mbb_per_hand: mean,
        ci95,
    }
}

/// Cross-play: one "focal" policy in a rotating hero seat against a table
/// of a DIFFERENT policy, duplicate-style deals. Measures how well one
/// self-play equilibrium fares inside another's population — the
/// multiplayer equilibrium-selection question: if the two runs converged
/// to interchangeable equilibria, cross-play is ~0 by symmetry.
pub fn run_crossplay(
    focal: &Policy,
    field: &Policy,
    cfg: &HandConfig,
    hands: u64,
    seed: u64,
) -> EvalResult {
    let n = cfg.num_players;
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
                assert!(guard < 500, "crossplay hand did not terminate");
                let p = table.real.to_act();
                let pol = if p == hero { focal } else { field };
                let a = pol.act_blueprint(&table.shadow, &table.hist, &mut rng);
                table.apply_abs(a, &pol.abs);
            }
            table.real.utilities()[hero] as f64 / cfg.bb as f64 * 1000.0
        })
        .collect();

    let (mean, ci95) = mean_ci(&results);
    EvalResult {
        hands,
        mbb_per_hand: mean,
        ci95,
    }
}

/// Search-mode evaluation: the hero plays with full range-tracked online
/// resolving (`act_with_search`) instead of raw blueprint lookups — the
/// only way to measure what search adds. Much slower than `run_eval`;
/// use a small per-decision budget.
pub fn run_eval_search(
    policy: &Policy,
    cfg: &HandConfig,
    baseline: Baseline,
    hands: u64,
    params: crate::bot::SearchParams,
    seed: u64,
) -> EvalResult {
    use crate::search::RangeTracker;
    let n = cfg.num_players;
    let train_cfg = crate::cfr::TrainConfig {
        hand: cfg.clone(),
        prune_after: u64::MAX,
        ..crate::cfr::TrainConfig::default()
    };
    let results: Vec<f64> = (0..hands)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(seed ^ i.wrapping_mul(0x2545_F491_4F6C_DD1D));
            let hero = (i % n as u64) as usize;
            let button = rng.random_range(0..n);
            let mut deck = fresh_deck();
            deck.shuffle(&mut rng);
            let mut table = Table::new(cfg, button, deck);
            let mut tracker = RangeTracker::new(n);
            let mut guard = 0;
            while !table.real.is_terminal() {
                guard += 1;
                assert!(guard < 500, "search eval hand did not terminate");
                let p = table.real.to_act();
                let a = if p == hero {
                    policy.act_with_search(
                        &table.shadow,
                        &table.hist,
                        params,
                        &train_cfg,
                        Some(&tracker),
                        &mut rng,
                    )
                } else {
                    baseline_action(baseline, &table.shadow, &policy.abs, &mut rng)
                };
                tracker.observe(p, a, &table.shadow, &table.hist, &policy.blueprint, &policy.abs);
                let street_before = table.real.street();
                table.apply_abs(a, &policy.abs);
                if table.real.street() != street_before {
                    tracker.exclude(table.real.board());
                }
            }
            table.real.utilities()[hero] as f64 / cfg.bb as f64 * 1000.0
        })
        .collect();

    let (mean, ci95) = mean_ci(&results);
    EvalResult {
        hands,
        mbb_per_hand: mean,
        ci95,
    }
}

/// Paired search-gain evaluation: each deal is played twice — once with the
/// hero using range-tracked search, once with the hero on the raw blueprint
/// — with every seat drawing from its own per-deal RNG stream so the two
/// playouts stay aligned until the hero actually deviates. The reported
/// value is the mean per-deal difference (search minus blueprint): deals
/// where search never changes an action contribute exactly zero variance,
/// which makes small strategy gains measurable at modest hand counts.
pub fn run_eval_paired(
    policy: &Policy,
    cfg: &HandConfig,
    hands: u64,
    params: crate::bot::SearchParams,
    seed: u64,
) -> EvalResult {
    run_eval_paired_policies(policy, Some(params), policy, None, cfg, hands, seed)
}

/// Paired A-vs-B evaluation: each deal is played once with the hero as
/// (policy_a, search params_a) and once as (policy_b, params_b), everyone
/// else on policy_a's blueprint, with per-seat RNG streams keeping the two
/// playouts aligned until the hero deviates. `None` params = raw blueprint.
/// Reports the mean per-deal A−B difference.
pub fn run_eval_paired_policies(
    policy_a: &Policy,
    params_a: Option<crate::bot::SearchParams>,
    policy_b: &Policy,
    params_b: Option<crate::bot::SearchParams>,
    cfg: &HandConfig,
    hands: u64,
    seed: u64,
) -> EvalResult {
    use crate::search::RangeTracker;
    let n = cfg.num_players;
    let train_cfg = crate::cfr::TrainConfig {
        hand: cfg.clone(),
        prune_after: u64::MAX,
        ..crate::cfr::TrainConfig::default()
    };

    let play_variant = |deck: [u8; 52],
                        button: usize,
                        hero: usize,
                        i: u64,
                        policy: &Policy,
                        search: Option<crate::bot::SearchParams>| {
        let mut seat_rngs: Vec<SmallRng> = (0..n)
            .map(|p| {
                SmallRng::seed_from_u64(
                    seed ^ i.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ ((p as u64) << 48),
                )
            })
            .collect();
        let mut table = Table::new(cfg, button, deck);
        let mut tracker = RangeTracker::new(n);
        let mut guard = 0;
        while !table.real.is_terminal() {
            guard += 1;
            assert!(guard < 500, "paired eval hand did not terminate");
            let p = table.real.to_act();
            let a = match (p == hero, search) {
                (true, Some(params)) => policy.act_with_search(
                    &table.shadow,
                    &table.hist,
                    params,
                    &train_cfg,
                    Some(&tracker),
                    &mut seat_rngs[p],
                ),
                _ => policy.act_blueprint(&table.shadow, &table.hist, &mut seat_rngs[p]),
            };
            if search.is_some() {
                tracker.observe(p, a, &table.shadow, &table.hist, &policy.blueprint, &policy.abs);
            }
            let street_before = table.real.street();
            table.apply_abs(a, &policy.abs);
            if search.is_some() && table.real.street() != street_before {
                tracker.exclude(table.real.board());
            }
        }
        table.real.utilities()[hero] as f64 / cfg.bb as f64 * 1000.0
    };

    let results: Vec<f64> = (0..hands)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(seed ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let hero = (i % n as u64) as usize;
            let button = rng.random_range(0..n);
            let mut deck = fresh_deck();
            deck.shuffle(&mut rng);
            play_variant(deck, button, hero, i, policy_a, params_a)
                - play_variant(deck, button, hero, i, policy_b, params_b)
        })
        .collect();

    let (mean, ci95) = mean_ci(&results);
    EvalResult {
        hands,
        mbb_per_hand: mean,
        ci95,
    }
}

/// Duplicate evaluation (ACPC-style variance reduction): each sampled deal
/// (deck + button) is played `num_players` times with the hero rotated
/// through every seat, and the deal's score is the mean over rotations —
/// card luck largely cancels within a deal. The CI is computed over deals,
/// which are independent, so it remains unbiased.
pub fn run_eval_duplicate(
    policy: &Policy,
    cfg: &HandConfig,
    baseline: Baseline,
    deals: u64,
    seed: u64,
) -> EvalResult {
    let n = cfg.num_players;
    let results: Vec<f64> = (0..deals)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(seed ^ i.wrapping_mul(0x2545_F491_4F6C_DD1D));
            let button = rng.random_range(0..n);
            let mut deck = fresh_deck();
            deck.shuffle(&mut rng);
            let mut sum = 0.0;
            for hero in 0..n {
                let mut hrng = SmallRng::seed_from_u64(
                    seed ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (hero as u64) << 56,
                );
                sum += play_one(policy, cfg, baseline, deck, button, hero, &mut hrng);
            }
            sum / n as f64
        })
        .collect();

    let (mean, ci95) = mean_ci(&results);
    EvalResult {
        hands: deals * n as u64,
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

    /// Duplicate evaluation kills card-luck variance. In the deterministic
    /// caller-vs-caller matchup, every deal's rotation-average is exactly the
    /// zero-sum total / n = 0, so the estimate and CI must both be ~0 — while
    /// plain evaluation of the same matchup has a CI of hundreds of mbb.
    #[test]
    fn duplicate_eval_cancels_card_luck() {
        let policy = empty_policy();
        let cfg = HandConfig::default();
        let dup = run_eval_duplicate(&policy, &cfg, Baseline::Caller, 300, 7);
        assert_eq!(dup.hands, 1_800);
        assert!(
            dup.mbb_per_hand.abs() < 1e-9 && dup.ci95 < 1e-9,
            "duplicate caller-vs-caller must be exactly 0, got {} ± {}",
            dup.mbb_per_hand,
            dup.ci95
        );
        let plain = run_eval(&policy, &cfg, Baseline::Caller, 1_800, 7);
        assert!(
            plain.ci95 > 50.0,
            "plain eval should be noisy here, got ±{}",
            plain.ci95
        );
    }

    /// Search-mode eval smoke: hero resolves with a range tracker on every
    /// decision; the harness must terminate and produce a finite estimate.
    #[test]
    fn search_eval_smoke() {
        let policy = empty_policy();
        let cfg = HandConfig {
            num_players: 2,
            ..HandConfig::default()
        };
        let params = crate::bot::SearchParams {
            time_ms: 20,
            max_iters: 200,
            ..crate::bot::SearchParams::default()
        };
        let r = run_eval_search(&policy, &cfg, Baseline::Caller, 24, params, 3);
        assert_eq!(r.hands, 24);
        assert!(r.mbb_per_hand.is_finite() && r.ci95.is_finite());
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
