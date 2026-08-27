//! External-sampling Monte Carlo CFR with linear weighting (Linear CFR)
//! and Pluribus-style negative-regret pruning.
//!
//! Training always uses button = 0; play mode rotates seats so the button is
//! seat 0 before building infoset keys, which keeps the infoset space aligned.
//!
//! Infoset key layout: [bucket_lo, bucket_hi, history tokens...].
//! History tokens come from AbsAction::token(), with TOKEN_STREET_SEP between
//! streets. Node action order == Abstraction::abstract_actions() order.

use crate::abstraction::{AbsAction, AbsConfig, Abstraction, Centroids, TOKEN_LEAF, TOKEN_STREET_SEP};
use crate::cards::fresh_deck;
use crate::engine::{Hand, HandConfig, PlayerAction, Street, MAX_PLAYERS};
use dashmap::DashMap;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub type InfoKey = Box<[u8]>;

#[derive(Clone, Serialize, Deserialize)]
pub struct Node {
    pub regret: Vec<f64>,
    pub strat: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub hand: HandConfig,
    /// Regret below which an action may be skipped (chips, linearly weighted).
    pub prune_threshold: f64,
    /// Probability a prunable action is actually skipped.
    pub prune_prob: f64,
    /// Iterations before pruning activates.
    pub prune_after: u64,
    /// Traversal RNG seed — distinct seeds give independent self-play runs
    /// (equilibrium-selection studies).
    pub seed: u64,
    /// Pluribus's pruning rules (Science 2019 supplement, Algorithm 1):
    /// whether to prune is decided once per traversal (95% of traversals
    /// prune, 5% explore everything) rather than per action, and pruning
    /// never applies on the final betting round or to actions that lead
    /// directly to a terminal node — the river gains nothing from the
    /// abstraction-refining side effect of pruning, and terminal payoffs
    /// are cheap to examine. Default on: measured to halve the BR
    /// exploitability bound at equal iterations (BASELINES.md, Aug 2026).
    /// `false` restores per-action pruning everywhere.
    pub pluribus_prune: bool,
    /// Pluribus's postflop blueprint: no running average strategy after
    /// the first betting round. Instead `snapshot_strategy` periodically
    /// adds the current regret-matching strategy of every postflop infoset
    /// into `strat`, and the blueprint is the mean of those snapshots. The
    /// average has no convergence guarantee in 6-player poker anyway, and
    /// the current strategy has already zeroed actions the average still
    /// carries residual mass on.
    pub snapshot_avg: bool,
    /// Multiway-focused sampling: at a non-traverser's preflop node the
    /// action is drawn from (1-f)*sigma + f*uniform-over-non-fold and the
    /// traversal's weight is multiplied by sigma/q, so multiway lines are
    /// reached more often while every update stays unbiased (the
    /// fixed point is unchanged). 0 = plain external sampling.
    pub multiway_focus: f64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            hand: HandConfig::default(),
            prune_threshold: -3.0e8,
            prune_prob: 0.95,
            prune_after: 200_000,
            seed: 0,
            pluribus_prune: true,
            snapshot_avg: false,
            multiway_focus: 0.0,
        }
    }
}

/// Depth-limited search: streets beyond `limit` are not solved; instead each
/// player picks one of four continuation strategies (blueprint as-is, or
/// fold-/call-/raise-biased) and the leaf is valued by a blueprint rollout.
/// This is Pluribus's guard against the blueprint's leaf values being
/// exploitable by a single fixed continuation.
pub struct LeafCfg {
    pub blueprint: Arc<Blueprint>,
    /// The blueprint's own abstraction (rollouts look infosets up in it;
    /// the solver's abstraction may be finer).
    pub abs: Arc<Abstraction>,
    pub limit: Street,
}

const LEAF_BIASES: usize = 4; // 0 = blueprint, 1 = fold-, 2 = call-, 3 = raise-biased
const BIAS_MULT: f64 = 5.0;

/// Re-randomizes hidden cards at a subgame root before each traversal.
pub type RootSampler<'a> = dyn Fn(&mut Hand, &mut SmallRng) + Sync + 'a;

/// Round-start rooting for online search (Pluribus): the subgame is rooted
/// at the start of the current betting round and the actions actually
/// taken since form the "spine". The hero's spine actions are held fixed
/// (already taken); every other player's strategy in the round is
/// re-solved, so their ranges at the hero's current decision are the
/// resolved reach-weighted ones rather than blueprint Bayes. Along the
/// spine the tree uses the real concrete actions (an off-tree bet is
/// solved at its actual size, replacing the nearest abstract size at
/// that node).
pub struct Spine {
    pub hero: usize,
    /// `hist` length at the round's start.
    pub base_len: usize,
    /// (seat, real action) for each step taken in the round so far.
    pub steps: Vec<(usize, PlayerAction)>,
    /// The abstract token recorded for each step (shadow mapping).
    pub tokens: Vec<u8>,
}

/// Mixture weight for spine-targeted sampling: at a non-traverser spine
/// node the actor's action is drawn from eps*spine + (1-eps)*sigma, with
/// the sample importance-weighted by sigma/q so regrets stay unbiased,
/// concentrating traversals on the line actually being played.
const SPINE_EPS: f64 = 0.5;

impl Spine {
    /// If the node (`hist` after `h`'s history, menu `acts`) lies on the
    /// spine: (index of the spine action in `acts`, its real concrete
    /// action, whether the actor is the hero).
    fn step_at(&self, hist: &[u8], h: &Hand, acts: &[AbsAction]) -> Option<(usize, PlayerAction, bool)> {
        let k = hist.len().checked_sub(self.base_len)?;
        if k >= self.steps.len() || hist[self.base_len..] != self.tokens[..k] {
            return None;
        }
        let (seat, act) = self.steps[k];
        if h.to_act() != seat {
            return None;
        }
        let si = acts.iter().position(|a| a.token() == self.tokens[k])?;
        Some((si, act, seat == self.hero))
    }
}

/// Restricted Nash response training (Johanson et al. 2008): with
/// probability `p`, decided once per traversal, every opponent plays the
/// fixed `model` for the whole hand (and contributes nothing to the average
/// strategy); otherwise the traversal is ordinary self-play. The learned
/// strategy maximally exploits the model subject to staying an equilibrium
/// against rational play with weight (1 − p) — an exploitation dial with a
/// bounded-exploitability knob.
#[derive(Debug, Clone, Copy)]
pub struct RnrCfg {
    pub model: crate::table::Baseline,
    pub p: f64,
}

pub struct Trainer {
    pub abs: Arc<Abstraction>,
    pub cfg: TrainConfig,
    nodes: DashMap<InfoKey, Node, ahash::RandomState>,
    iters_done: AtomicU64,
    leaf: Option<LeafCfg>,
    /// CFR+ mode (regrets floored at 0). Used for online subgame solves;
    /// blueprint training keeps Pluribus's negative-regret scheme, which the
    /// pruning depends on.
    plus: bool,
    /// Model opponents as lambda-rational (logit QRE) instead of
    /// regret-matching at their nodes. Subgame-only exploitation knob.
    qre_lambda: Option<f64>,
    /// Restricted Nash response mixture (blueprint training only).
    rnr: Option<RnrCfg>,
    /// Round-start rooting for subgame solves.
    spine: Option<Spine>,
    /// RNR opponent model as a fixed blueprint (e.g. a cloned static
    /// opponent). When set, model-opponent traversals sample from this
    /// strategy instead of `rnr.model`'s baseline. Must share this trainer's
    /// abstraction so its (bucket, history) keys line up.
    rnr_opp: Option<Arc<Blueprint>>,
    /// VR-MCCFR (Schmid et al. 2019): control-variate baselines at sampled
    /// opponent nodes. The sampled child's value is replaced by
    /// sum_a sigma(a) b(a) + (v_sampled - b(a_sampled)); unbiased for any
    /// b, and the variance of the traverser's regret updates shrinks as b
    /// learns the expected value. Baselines are keyed by the opponent's
    /// infoset plus the traverser's seat and card bucket (the traverser's
    /// hand drives most of the value's variance), learned as an
    /// exponential average with alpha = 0.5. Not checkpointed: a resumed
    /// run relearns them from zero, which costs variance, not bias.
    vr: bool,
    baselines: DashMap<InfoKey, Vec<f32>, ahash::RandomState>,
}

const BASELINE_ALPHA: f32 = 0.5;

/// Baseline key: opponent infoset key, then the traverser's seat and bucket.
fn baseline_key(key: &[u8], traverser: usize, bucket: u16) -> InfoKey {
    let mut k = Vec::with_capacity(key.len() + 3);
    k.extend_from_slice(key);
    k.push(traverser as u8);
    k.extend_from_slice(&bucket.to_le_bytes());
    k.into_boxed_slice()
}

/// Approximate logit quantal-response distribution over a node's actions:
/// softmax of cumulative regrets normalized to [-1, 1]. lambda = 0 is
/// uniform random; lambda -> infinity approaches the argmax (fully
/// rational). Used to model boundedly rational opponents in search.
pub(crate) fn qre_distribution(regrets: &[f64], lambda: f64, out: &mut Vec<f64>) {
    out.clear();
    let scale = regrets.iter().fold(0.0f64, |m, &r| m.max(r.abs()));
    if scale <= 0.0 || lambda <= 0.0 {
        out.extend(std::iter::repeat_n(
            1.0 / regrets.len() as f64,
            regrets.len(),
        ));
        return;
    }
    let logits: Vec<f64> = regrets.iter().map(|&r| lambda * r / scale).collect();
    let mx = logits.iter().cloned().fold(f64::MIN, f64::max);
    out.extend(logits.iter().map(|&l| (l - mx).exp()));
    let total: f64 = out.iter().sum();
    out.iter_mut().for_each(|p| *p /= total);
}

/// sigma = positive-regret matching; uniform when no positive regret.
pub fn regret_matching(regrets: &[f64], out: &mut Vec<f64>) {
    out.clear();
    let total: f64 = regrets.iter().map(|&r| r.max(0.0)).sum();
    if total > 0.0 {
        out.extend(regrets.iter().map(|&r| r.max(0.0) / total));
    } else {
        out.extend(std::iter::repeat_n(1.0 / regrets.len() as f64, regrets.len()));
    }
}

pub fn make_key(bucket: u16, hist: &[u8]) -> InfoKey {
    let mut k = Vec::with_capacity(2 + hist.len());
    k.push(bucket as u8);
    k.push((bucket >> 8) as u8);
    k.extend_from_slice(hist);
    k.into_boxed_slice()
}

impl Trainer {
    pub fn new(abs: Arc<Abstraction>, cfg: TrainConfig) -> Self {
        Trainer {
            abs,
            cfg,
            nodes: DashMap::with_hasher(ahash::RandomState::new()),
            iters_done: AtomicU64::new(0),
            leaf: None,
            plus: false,
            qre_lambda: None,
            rnr: None,
            spine: None,
            rnr_opp: None,
            vr: false,
            baselines: DashMap::with_hasher(ahash::RandomState::new()),
        }
    }

    pub fn with_qre(mut self, lambda: Option<f64>) -> Self {
        self.qre_lambda = lambda;
        self
    }

    pub fn with_rnr(mut self, rnr: Option<RnrCfg>) -> Self {
        self.rnr = rnr;
        self
    }

    /// Exploit a fixed opponent given as a blueprint (e.g. a cloned static
    /// bot). The opponent must have been built with this trainer's
    /// abstraction so bucket/history keys align.
    pub fn with_vr(mut self, vr: bool) -> Self {
        self.vr = vr;
        self
    }

    pub fn with_rnr_opponent(mut self, opp: Option<Arc<Blueprint>>) -> Self {
        self.rnr_opp = opp;
        self
    }

    pub fn with_leaf(mut self, leaf: Option<LeafCfg>) -> Self {
        self.leaf = leaf;
        self
    }

    pub fn with_spine(mut self, spine: Option<Spine>) -> Self {
        self.spine = spine;
        self
    }

    /// Concrete action for child `i` of `h`: the real action on the spine
    /// step, the abstract mapping elsewhere.
    fn child_action(&self, h: &Hand, a: AbsAction, i: usize, step: Option<(usize, PlayerAction, bool)>) -> PlayerAction {
        match step {
            Some((si, real, _)) if si == i => real,
            _ => self.abs.concrete(h, a),
        }
    }

    pub fn with_plus(mut self, plus: bool) -> Self {
        self.plus = plus;
        self
    }

    fn regret_floor(&self) -> f64 {
        if self.plus {
            0.0
        } else {
            2.0 * self.cfg.prune_threshold
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn iterations(&self) -> u64 {
        self.iters_done.load(Ordering::Relaxed)
    }

    /// Run `iterations` external-sampling traversals in parallel.
    /// `progress` is called periodically with total completed iterations.
    pub fn run(&self, iterations: u64, progress: &(dyn Fn(u64) + Sync)) {
        let start = self.iters_done.load(Ordering::Relaxed);
        let n = self.cfg.hand.num_players;
        // Traversals are batched per rayon task: hundreds of millions of
        // one-traversal tasks pay measurable scheduler overhead. Each
        // traversal keeps its own t-derived rng, so results are identical
        // to unbatched scheduling.
        const BATCH: u64 = 256;
        let chunks = iterations.div_ceil(BATCH);
        (0..chunks).into_par_iter().for_each(|c| {
            let lo = c * BATCH;
            let hi = ((c + 1) * BATCH).min(iterations);
            for i in lo..hi {
                let t = start + i + 1;
                let mut rng =
                    SmallRng::seed_from_u64(t.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ self.cfg.seed);
                let traverser = (t % n as u64) as usize;
                let mut deck = fresh_deck();
                deck.shuffle(&mut rng);
                let hand = Hand::new(&self.cfg.hand, 0, deck);
                if hand.is_terminal() {
                    continue; // degenerate deal (e.g. blinds all-in, tiny stacks)
                }
                let weight = t as f64;
                // Pluribus mode decides pruning once per traversal; the
                // per-action mode keeps prune_ok on and rolls per action.
                let prune_ok = t > self.cfg.prune_after
                    && (!self.cfg.pluribus_prune || rng.random::<f64>() < self.cfg.prune_prob);
                let model_opp = self
                    .rnr
                    .as_ref()
                    .is_some_and(|r| rng.random::<f64>() < r.p);
                let mut hist = Vec::with_capacity(32);
                self.traverse(&hand, &mut hist, traverser, weight, prune_ok, model_opp, &mut rng);
                let done = self.iters_done.fetch_add(1, Ordering::Relaxed) + 1;
                if done.is_multiple_of(4096) {
                    progress(done - start);
                }
            }
        });
        // Make the final count exact even for the skipped terminal deals.
        self.iters_done.store(start + iterations, Ordering::Relaxed);
        progress(iterations);
    }

    /// Train a subgame rooted at `root` for a time/iteration budget.
    /// Hidden cards are resampled every traversal — uniformly by default, or
    /// by `sampler` (range-weighted sampling from a RangeTracker) — so the
    /// solver learns strategies for every bucket; callers query the bucket
    /// they actually hold afterwards.
    pub fn run_subgame(
        &self,
        root: &Hand,
        root_hist: &[u8],
        time_ms: u64,
        max_iters: u64,
        sampler: Option<&RootSampler<'_>>,
    ) {
        let start = std::time::Instant::now();
        let n = root.num_players();
        while self.iterations() < max_iters
            && (start.elapsed().as_millis() as u64) < time_ms
        {
            let t0 = self.iters_done.load(Ordering::Relaxed);
            let batch = 1024.min(max_iters - t0);
            (0..batch).into_par_iter().for_each(|j| {
                let t = t0 + j + 1;
                let mut rng =
                    SmallRng::seed_from_u64(t.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xACE);
                let traverser = (t % n as u64) as usize;
                if root.folded(traverser) {
                    return;
                }
                let mut h = root.clone();
                match sampler {
                    Some(f) => f(&mut h, &mut rng),
                    None => h.resample_hidden(None, &mut rng),
                }
                let mut hist = root_hist.to_vec();
                self.traverse(&h, &mut hist, traverser, t as f64, false, false, &mut rng);
            });
            self.iters_done.fetch_add(batch, Ordering::Relaxed);
        }
    }

    /// One external-sampling traversal. Returns utility (chips) for `traverser`.
    /// `model_opp`: this traversal's opponents play the fixed RNR model.
    #[allow(clippy::too_many_arguments)]
    fn traverse(
        &self,
        h: &Hand,
        hist: &mut Vec<u8>,
        traverser: usize,
        weight: f64,
        prune_ok: bool,
        model_opp: bool,
        rng: &mut SmallRng,
    ) -> f64 {
        if h.is_terminal() {
            return h.utilities()[traverser] as f64;
        }
        if let Some(lc) = &self.leaf {
            if h.street() > lc.limit {
                return self.leaf_value(h, hist, traverser, weight, lc, rng);
            }
        }
        let p = h.to_act();
        let acts = self.abs.abstract_actions(h);
        let bucket = self.abs.bucket(h.hole(p), h.board(), rng);
        let key = make_key(bucket, hist);

        // Snapshot sigma without holding the shard lock during recursion.
        let (sigma, regrets) = {
            let node = self.nodes.entry(key.clone()).or_insert_with(|| Node {
                regret: vec![0.0; acts.len()],
                strat: vec![0.0; acts.len()],
            });
            let mut s = Vec::with_capacity(acts.len());
            regret_matching(&node.regret, &mut s);
            (s, node.regret.clone())
        };

        let step = self.spine.as_ref().and_then(|sp| sp.step_at(hist, h, &acts));
        if let Some((si, real, true)) = step {
            // The hero's already-taken action this round: fixed, no
            // learning at this node.
            let mut child = h.clone();
            child.apply(real);
            let depth = hist.len();
            hist.push(acts[si].token());
            if !child.is_terminal() && child.street() != h.street() {
                hist.push(TOKEN_STREET_SEP);
            }
            let v = self.traverse(&child, hist, traverser, weight, prune_ok, model_opp, rng);
            hist.truncate(depth);
            return v;
        }

        if p == traverser {
            // Full-width over own actions, with negative-regret pruning.
            let pluribus = self.cfg.pluribus_prune;
            let mut explore: Vec<bool> = if prune_ok && !(pluribus && h.street() == Street::River) {
                regrets
                    .iter()
                    .map(|&r| {
                        r >= self.cfg.prune_threshold
                            || (!pluribus && rng.random::<f64>() >= self.cfg.prune_prob)
                    })
                    .collect()
            } else {
                vec![true; acts.len()]
            };
            let mut children: Vec<Hand> = Vec::with_capacity(acts.len());
            for (i, &a) in acts.iter().enumerate() {
                let mut child = h.clone();
                child.apply(self.child_action(h, a, i, step));
                if pluribus && child.is_terminal() {
                    explore[i] = true; // terminal payoffs are never pruned
                }
                children.push(child);
            }
            if !explore.iter().any(|&e| e) {
                explore.iter_mut().for_each(|e| *e = true);
            }

            let mut utils = vec![0.0f64; acts.len()];
            let mut node_util = 0.0;
            for (i, &a) in acts.iter().enumerate() {
                if !explore[i] {
                    continue;
                }
                let child = &children[i];
                let depth = hist.len();
                hist.push(a.token());
                if !child.is_terminal() && child.street() != h.street() {
                    hist.push(TOKEN_STREET_SEP);
                }
                utils[i] =
                    self.traverse(child, hist, traverser, weight, prune_ok, model_opp, rng);
                hist.truncate(depth);
                node_util += sigma[i] * utils[i];
            }

            if let Some(mut node) = self.nodes.get_mut(&key) {
                let floor = self.regret_floor();
                for i in 0..acts.len() {
                    if explore[i] {
                        node.regret[i] =
                            (node.regret[i] + weight * (utils[i] - node_util)).max(floor);
                    }
                }
            }
            node_util
        } else if model_opp {
            // RNR model traversal: the opponent plays the fixed model and
            // contributes nothing to the learned average strategy. A cloned
            // blueprint opponent (rnr_opp) takes precedence over the baseline
            // model; both are fixed, not learned.
            let a = if let Some(opp) = &self.rnr_opp {
                let p = h.to_act();
                let bucket = self.abs.bucket(h.hole(p), h.board(), rng);
                match opp.get(bucket, hist) {
                    Some(s) if s.len() == acts.len() && s.iter().sum::<f32>() > 0.0 => {
                        let probs: Vec<f64> = s.iter().map(|&x| x as f64).collect();
                        acts[sample_index(&probs, rng)]
                    }
                    // Unseen/mismatched: the clone's fallback is check/call.
                    _ => AbsAction::CheckCall,
                }
            } else {
                let model = self.rnr.as_ref().expect("model_opp without rnr").model;
                let a = crate::table::baseline_action(model, h, &self.abs, rng);
                if acts.contains(&a) {
                    a
                } else {
                    acts[0]
                }
            };
            let mut child = h.clone();
            child.apply(self.abs.concrete(h, a));
            hist.push(a.token());
            if !child.is_terminal() && child.street() != h.street() {
                hist.push(TOKEN_STREET_SEP);
            }
            self.traverse(&child, hist, traverser, weight, prune_ok, model_opp, rng)
        } else {
            // Sample one opponent action from their modeled strategy
            // (regret matching, or logit QRE when exploiting) and
            // accumulate it into their average strategy.
            let dist = match self.qre_lambda {
                Some(l) => {
                    let mut q = Vec::with_capacity(regrets.len());
                    qre_distribution(&regrets, l, &mut q);
                    q
                }
                None => sigma,
            };
            if !(self.cfg.snapshot_avg && h.street() != Street::Preflop) {
                if let Some(mut node) = self.nodes.get_mut(&key) {
                    for (st, &d) in node.strat.iter_mut().zip(&dist) {
                        *st += weight * d;
                    }
                }
            }
            // Spine-targeted sampling at an opponent's spine node.
            // Off-policy sampling (spine targeting, multiway focus) draws
            // the action from q instead of sigma. Two corrections keep
            // every update unbiased: descendants' updates carry the reach
            // ratio sigma/q in `weight`, and the value returned to the
            // ancestors is scaled by the same ratio (E_q[ratio * v] =
            // E_sigma[v]).
            let (idx, ratio) = match step {
                Some((si, _, false)) => {
                    let idx = if rng.random::<f64>() < SPINE_EPS {
                        si
                    } else {
                        sample_index(&dist, rng)
                    };
                    let q = (1.0 - SPINE_EPS) * dist[idx] + if idx == si { SPINE_EPS } else { 0.0 };
                    (idx, dist[idx] / q)
                }
                _ if self.cfg.multiway_focus > 0.0 && h.street() == Street::Preflop => {
                    // Multiway focus: mix toward non-fold actions,
                    // importance-weighted (see TrainConfig::multiway_focus).
                    let f = self.cfg.multiway_focus;
                    let live: Vec<usize> = (0..acts.len())
                        .filter(|&i| acts[i] != AbsAction::Fold)
                        .collect();
                    let u = 1.0 / live.len() as f64;
                    let idx = if rng.random::<f64>() < f {
                        live[rng.random_range(0..live.len())]
                    } else {
                        sample_index(&dist, rng)
                    };
                    let q = (1.0 - f) * dist[idx]
                        + if acts[idx] != AbsAction::Fold { f * u } else { 0.0 };
                    (idx, dist[idx] / q)
                }
                _ => (sample_index(&dist, rng), 1.0),
            };
            let weight = weight * ratio;
            let a = acts[idx];
            let mut child = h.clone();
            child.apply(self.child_action(h, a, idx, step));
            hist.push(a.token());
            if !child.is_terminal() && child.street() != h.street() {
                hist.push(TOKEN_STREET_SEP);
            }
            // No truncation needed: the nearest traverser ancestor restores hist.
            let v = ratio * self.traverse(&child, hist, traverser, weight, prune_ok, model_opp, rng);
            if !self.vr {
                return v;
            }
            let tb = self.abs.bucket(h.hole(traverser), h.board(), rng);
            let bkey = baseline_key(&key, traverser, tb);
            let mut b = self
                .baselines
                .entry(bkey)
                .or_insert_with(|| vec![0.0f32; acts.len()]);
            let expected: f64 = b.iter().zip(&dist).map(|(&bi, &d)| bi as f64 * d).sum();
            let corrected = expected + (v - b[idx] as f64);
            b[idx] += BASELINE_ALPHA * (v as f32 - b[idx]);
            corrected
        }
    }

    /// Value of a depth-limit leaf: every live player simultaneously picks
    /// one of four continuation strategies (an extra 4-action infoset keyed
    /// by bucket + history + TOKEN_LEAF); the traverser explores all four
    /// with regret updates, opponents sample theirs; the outcome is a
    /// blueprint rollout to the end of the hand under the chosen biases.
    fn leaf_value(
        &self,
        h: &Hand,
        hist: &[u8],
        traverser: usize,
        weight: f64,
        lc: &LeafCfg,
        rng: &mut SmallRng,
    ) -> f64 {
        let mut biases = [0usize; MAX_PLAYERS];
        for (p, bias) in biases.iter_mut().enumerate().take(h.num_players()) {
            if p == traverser || h.folded(p) || h.all_in(p) {
                continue;
            }
            let bucket = self.abs.bucket(h.hole(p), h.board(), rng);
            let key = leaf_key(bucket, hist);
            let sigma = {
                let node = self.nodes.entry(key).or_insert_with(|| Node {
                    regret: vec![0.0; LEAF_BIASES],
                    strat: vec![0.0; LEAF_BIASES],
                });
                let mut s = Vec::with_capacity(LEAF_BIASES);
                regret_matching(&node.regret, &mut s);
                s
            };
            *bias = sample_index(&sigma, rng);
        }

        if h.folded(traverser) || h.all_in(traverser) {
            return self.leaf_rollout(h, hist, &biases, traverser, lc, rng);
        }

        let bucket = self.abs.bucket(h.hole(traverser), h.board(), rng);
        let key = leaf_key(bucket, hist);
        let sigma = {
            let node = self.nodes.entry(key.clone()).or_insert_with(|| Node {
                regret: vec![0.0; LEAF_BIASES],
                strat: vec![0.0; LEAF_BIASES],
            });
            let mut s = Vec::with_capacity(LEAF_BIASES);
            regret_matching(&node.regret, &mut s);
            s
        };
        let mut utils = [0.0f64; LEAF_BIASES];
        let mut node_util = 0.0;
        for (b, u) in utils.iter_mut().enumerate() {
            let mut bs = biases;
            bs[traverser] = b;
            *u = self.leaf_rollout(h, hist, &bs, traverser, lc, rng);
            node_util += sigma[b] * *u;
        }
        if let Some(mut node) = self.nodes.get_mut(&key) {
            let floor = self.regret_floor();
            for (r, &u) in node.regret.iter_mut().zip(&utils) {
                *r = (*r + weight * (u - node_util)).max(floor);
            }
        }
        node_util
    }

    /// Play the hand out from `h` with every player following the blueprint
    /// under their continuation bias; returns the traverser's chip utility.
    fn leaf_rollout(
        &self,
        h: &Hand,
        hist: &[u8],
        biases: &[usize; MAX_PLAYERS],
        traverser: usize,
        lc: &LeafCfg,
        rng: &mut SmallRng,
    ) -> f64 {
        let mut h = h.clone();
        let mut hist = hist.to_vec();
        while !h.is_terminal() {
            let p = h.to_act();
            let acts = lc.abs.abstract_actions(&h);
            let bucket = lc.abs.bucket(h.hole(p), h.board(), rng);
            let mut probs: Vec<f64> = match lc.blueprint.get(bucket, &hist) {
                Some(s) if s.len() == acts.len() => s.iter().map(|&x| x as f64).collect(),
                _ => vec![1.0 / acts.len() as f64; acts.len()],
            };
            apply_bias(&mut probs, &acts, biases[p]);
            let a = acts[sample_index(&probs, rng)];
            let street_before = h.street();
            h.apply(lc.abs.concrete(&h, a));
            hist.push(a.token());
            if !h.is_terminal() && h.street() != street_before {
                hist.push(TOKEN_STREET_SEP);
            }
        }
        h.utilities()[traverser] as f64
    }

    /// Snapshot mode: add every postflop infoset's current regret-matching
    /// strategy into its `strat` accumulator. Called periodically after a
    /// warm-up; the exported blueprint is the mean of the snapshots.
    /// Returns the number of infosets snapshotted.
    pub fn snapshot_strategy(&self) -> usize {
        let mut n = 0;
        let mut s = Vec::new();
        for mut e in self.nodes.iter_mut() {
            let postflop = e.key()[2..].contains(&TOKEN_STREET_SEP);
            if !postflop {
                continue;
            }
            regret_matching(&e.regret, &mut s);
            for (st, &p) in e.strat.iter_mut().zip(&s) {
                *st += p;
            }
            n += 1;
        }
        n
    }

    /// Normalized average strategy for an infoset, if visited.
    pub fn avg_strategy(&self, bucket: u16, hist: &[u8]) -> Option<Vec<f64>> {
        let key = make_key(bucket, hist);
        let node = self.nodes.get(&key)?;
        normalize(&node.strat)
    }

    /// Export the (much smaller) normalized average strategy for play.
    pub fn to_blueprint(&self) -> Blueprint {
        let mut strategies =
            StrategyMap::with_capacity_and_hasher(self.nodes.len(), Default::default());
        for e in self.nodes.iter() {
            if let Some(s) = normalize(&e.value().strat) {
                strategies.insert(e.key().to_vec(), s.iter().map(|&x| x as f32).collect());
            }
        }
        Blueprint {
            strategies,
            iterations: self.iterations(),
            num_players: self.cfg.hand.num_players,
            abs_cfg: self.abs.cfg.clone(),
            centroids: self.abs.centroids.clone(),
        }
    }

    /// Full checkpoint (regrets + strategy sums) for resuming training.
    pub fn save_checkpoint(&self, path: &str) -> std::io::Result<()> {
        let snapshot: Vec<(Vec<u8>, Node)> = self
            .nodes
            .iter()
            .map(|e| (e.key().to_vec(), e.value().clone()))
            .collect();
        let ckpt = Checkpoint {
            iterations: self.iterations(),
            abs_cfg: self.abs.cfg.clone(),
            centroids: self.abs.centroids.clone(),
            nodes: snapshot,
        };
        let f = std::io::BufWriter::new(std::fs::File::create(path)?);
        bincode::serialize_into(f, &ckpt).map_err(std::io::Error::other)
    }

    /// Resume from a checkpoint. The card abstraction (config + k-means
    /// centroids) is restored from the file so bucketing matches exactly.
    pub fn load_checkpoint(path: &str, cfg: TrainConfig) -> std::io::Result<Trainer> {
        let f = std::io::BufReader::new(std::fs::File::open(path)?);
        let ckpt: Checkpoint = bincode::deserialize_from(f).map_err(std::io::Error::other)?;
        let abs = Abstraction::with_centroids(ckpt.abs_cfg, ckpt.centroids);
        let t = Trainer::new(Arc::new(abs), cfg);
        for (k, n) in ckpt.nodes {
            t.nodes.insert(k.into_boxed_slice(), n);
        }
        t.iters_done.store(ckpt.iterations, Ordering::Relaxed);
        Ok(t)
    }
}

/// On-disk layout of blueprints written before `AbsConfig::menu`.
#[derive(Deserialize)]
struct LegacyAbsConfig {
    postflop_buckets: u16,
    equity_rollouts: u32,
    dist_runouts: u32,
    runout_rollouts: u32,
    cache_cap: usize,
}

#[derive(Deserialize)]
struct LegacyBlueprint {
    strategies: StrategyMap,
    iterations: u64,
    num_players: usize,
    abs_cfg: LegacyAbsConfig,
    centroids: Option<Centroids>,
}

#[derive(Serialize, Deserialize)]
struct Checkpoint {
    iterations: u64,
    abs_cfg: AbsConfig,
    centroids: Option<Centroids>,
    nodes: Vec<(Vec<u8>, Node)>,
}

fn leaf_key(bucket: u16, hist: &[u8]) -> InfoKey {
    let mut k = Vec::with_capacity(3 + hist.len());
    k.push(bucket as u8);
    k.push((bucket >> 8) as u8);
    k.extend_from_slice(hist);
    k.push(TOKEN_LEAF);
    k.into_boxed_slice()
}

/// Reweight a strategy toward folding / calling / raising (bias 1/2/3) by
/// multiplying the matching actions' probabilities by BIAS_MULT, then
/// renormalizing. Bias 0 leaves the blueprint strategy as-is.
pub(crate) fn apply_bias(probs: &mut [f64], acts: &[AbsAction], bias: usize) {
    for (p, a) in probs.iter_mut().zip(acts) {
        let boosted = match bias {
            1 => matches!(a, AbsAction::Fold),
            2 => matches!(a, AbsAction::CheckCall),
            3 => matches!(a, AbsAction::Bet(_) | AbsAction::AllIn),
            _ => false,
        };
        if boosted {
            *p *= BIAS_MULT;
        }
    }
    let total: f64 = probs.iter().sum();
    if total > 0.0 {
        for p in probs.iter_mut() {
            *p /= total;
        }
    } else {
        let u = 1.0 / probs.len() as f64;
        probs.iter_mut().for_each(|p| *p = u);
    }
}

fn normalize(v: &[f64]) -> Option<Vec<f64>> {
    let total: f64 = v.iter().sum();
    if total <= 0.0 {
        return None;
    }
    Some(v.iter().map(|&x| x / total).collect())
}

pub fn sample_index(probs: &[f64], rng: &mut SmallRng) -> usize {
    let mut r: f64 = rng.random();
    for (i, &p) in probs.iter().enumerate() {
        r -= p;
        if r <= 0.0 {
            return i;
        }
    }
    probs.len() - 1
}

/// Strategy store keyed by (bucket, history) bytes. ahash: `get` runs on
/// every decision of every hand everywhere, and the wire format (len +
/// entries) is hasher-independent, so existing .bin files stay compatible.
pub type StrategyMap = HashMap<Vec<u8>, Vec<f32>, ahash::RandomState>;

/// The trained average strategy used at the table, together with the card
/// abstraction it was trained under (config + k-means centroids), so play
/// and benchmarking bucket cards exactly as training did.
#[derive(Clone, Serialize, Deserialize)]
pub struct Blueprint {
    pub strategies: StrategyMap,
    pub iterations: u64,
    pub num_players: usize,
    pub abs_cfg: AbsConfig,
    pub centroids: Option<Centroids>,
}

impl Blueprint {
    pub fn get(&self, bucket: u16, hist: &[u8]) -> Option<&Vec<f32>> {
        let key = make_key(bucket, hist);
        self.strategies.get(key.as_ref() as &[u8])
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let f = std::io::BufWriter::new(std::fs::File::create(path)?);
        bincode::serialize_into(f, self).map_err(std::io::Error::other)
    }

    pub fn load(path: &str) -> std::io::Result<Blueprint> {
        let f = std::io::BufReader::new(std::fs::File::open(path)?);
        match bincode::deserialize_from(f) {
            Ok(bp) => Ok(bp),
            Err(e) => {
                // Blueprints written before AbsConfig::menu existed (all
                // 2026-07/08 artifacts) were trained with the Wide menu.
                let f = std::io::BufReader::new(std::fs::File::open(path)?);
                let legacy: LegacyBlueprint =
                    bincode::deserialize_from(f).map_err(|_| std::io::Error::other(e))?;
                Ok(Blueprint {
                    strategies: legacy.strategies,
                    iterations: legacy.iterations,
                    num_players: legacy.num_players,
                    abs_cfg: AbsConfig {
                        postflop_buckets: legacy.abs_cfg.postflop_buckets,
                        menu: crate::abstraction::MenuShape::Wide,
                        equity_rollouts: legacy.abs_cfg.equity_rollouts,
                        dist_runouts: legacy.abs_cfg.dist_runouts,
                        runout_rollouts: legacy.abs_cfg.runout_rollouts,
                        cache_cap: legacy.abs_cfg.cache_cap,
                    },
                    centroids: legacy.centroids,
                })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests (written first, TDD)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::{preflop_bucket, AbsAction, AbsConfig};
    use crate::cards::make_card;

    fn push_fold_trainer() -> Trainer {
        // 10bb heads-up: near push/fold; converges fast and has known properties.
        let abs_cfg = AbsConfig {
            postflop_buckets: 6,
            equity_rollouts: 50,
            dist_runouts: 12,
            runout_rollouts: 25,
            cache_cap: 1_000_000,
            menu: crate::abstraction::MenuShape::Wide,
        };
        let cents = Centroids::train(&abs_cfg, 400, 99);
        let abs = Abstraction::with_centroids(abs_cfg, Some(cents));
        let cfg = TrainConfig {
            hand: HandConfig {
                num_players: 2,
                stack: 1_000,
                sb: 50,
                bb: 100,
            },
            prune_after: u64::MAX, // no pruning in this small test
            ..TrainConfig::default()
        };
        Trainer::new(Arc::new(abs), cfg)
    }

    #[test]
    fn regret_matching_math() {
        let mut out = Vec::new();
        regret_matching(&[300.0, 100.0, -50.0], &mut out);
        assert_eq!(out, vec![0.75, 0.25, 0.0]);
        regret_matching(&[-5.0, -10.0], &mut out);
        assert_eq!(out, vec![0.5, 0.5]);
    }

    #[test]
    fn qre_distribution_interpolates_rationality() {
        let regrets = [900.0, 300.0, -600.0];
        let mut q = Vec::new();

        qre_distribution(&regrets, 0.0, &mut q);
        for &p in &q {
            assert!((p - 1.0 / 3.0).abs() < 1e-12, "lambda=0 must be uniform");
        }

        qre_distribution(&regrets, 2.0, &mut q);
        assert!(q[0] > q[1] && q[1] > q[2], "must order by regret: {:?}", q);
        assert!((q.iter().sum::<f64>() - 1.0).abs() < 1e-12);

        qre_distribution(&regrets, 50.0, &mut q);
        assert!(q[0] > 0.99, "large lambda approaches argmax: {:?}", q);

        // Degenerate all-zero regrets: uniform.
        qre_distribution(&[0.0, 0.0], 3.0, &mut q);
        assert_eq!(q, vec![0.5, 0.5]);
    }

    #[test]
    fn apply_bias_shifts_mass() {
        let acts = [
            AbsAction::Fold,
            AbsAction::CheckCall,
            AbsAction::Bet(2),
            AbsAction::AllIn,
        ];
        let base = [0.25f64, 0.25, 0.25, 0.25];

        let mut p = base;
        apply_bias(&mut p, &acts, 0);
        assert_eq!(p, base, "bias 0 is the blueprint as-is");

        let mut p = base;
        apply_bias(&mut p, &acts, 1);
        assert!(p[0] > 0.6, "fold bias must boost fold: {:?}", p);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);

        let mut p = base;
        apply_bias(&mut p, &acts, 3);
        assert!(p[2] > 0.3 && p[3] > 0.3, "raise bias boosts bets+allin: {:?}", p);
        assert!(p[0] < 0.1);
    }

    /// RNR against an always-caller model: with no fold equity to exploit,
    /// the near-pure best response must beat a calling station by clearly
    /// more than the equilibrium strategy does at the same training budget.
    #[test]
    fn rnr_exploits_the_modeled_opponent() {
        use crate::bot::Policy;
        use crate::table::{run_eval, Baseline};

        let train = |rnr: Option<RnrCfg>| {
            let t = push_fold_trainer().with_rnr(rnr);
            t.run(120_000, &|_| {});
            let bp = t.to_blueprint();
            let abs = Abstraction::with_centroids(bp.abs_cfg.clone(), bp.centroids.clone());
            Policy::new(bp, Arc::new(abs))
        };
        let cfg = HandConfig {
            num_players: 2,
            stack: 1_000,
            sb: 50,
            bb: 100,
        };
        let nash = train(None);
        let rnr = train(Some(RnrCfg {
            model: Baseline::Caller,
            p: 0.9,
        }));
        let w_nash = run_eval(&nash, &cfg, Baseline::Caller, 40_000, 5);
        let w_rnr = run_eval(&rnr, &cfg, Baseline::Caller, 40_000, 5);
        assert!(
            w_rnr.mbb_per_hand > w_nash.mbb_per_hand + 50.0,
            "RNR(caller, 0.9) must exploit a caller more than equilibrium: \
             rnr {:+.0}±{:.0} vs nash {:+.0}±{:.0}",
            w_rnr.mbb_per_hand,
            w_rnr.ci95,
            w_nash.mbb_per_hand,
            w_nash.ci95
        );
    }

    /// RNR against a blueprint opponent: an empty-strategy blueprint plays
    /// pure check/call by fallback (a calling station), so exploiting it via
    /// the blueprint-opponent path must beat equilibrium by clearly more —
    /// the same result as exploiting Baseline::Caller, validating that the
    /// clone-opponent traversal drives the exploit.
    #[test]
    fn rnr_exploits_a_blueprint_opponent() {
        use crate::bot::Policy;
        use crate::table::{run_eval, Baseline};

        let policy_from = |t: Trainer| {
            t.run(120_000, &|_| {});
            let bp = t.to_blueprint();
            let abs = Abstraction::with_centroids(bp.abs_cfg.clone(), bp.centroids.clone());
            Policy::new(bp, Arc::new(abs))
        };
        let empty_opp = Arc::new(Blueprint {
            strategies: Default::default(),
            iterations: 0,
            num_players: 2,
            abs_cfg: AbsConfig::default(),
            centroids: None,
        });
        let nash = policy_from(push_fold_trainer());
        let rnr = policy_from(
            push_fold_trainer()
                .with_rnr(Some(RnrCfg {
                    model: Baseline::Caller, // ignored when rnr_opp is set
                    p: 0.9,
                }))
                .with_rnr_opponent(Some(empty_opp)),
        );
        let cfg = HandConfig {
            num_players: 2,
            stack: 1_000,
            sb: 50,
            bb: 100,
        };
        let w_nash = run_eval(&nash, &cfg, Baseline::Caller, 40_000, 5);
        let w_rnr = run_eval(&rnr, &cfg, Baseline::Caller, 40_000, 5);
        assert!(
            w_rnr.mbb_per_hand > w_nash.mbb_per_hand + 50.0,
            "RNR vs a check/call blueprint must exploit more than equilibrium: \
             rnr {:+.0}±{:.0} vs nash {:+.0}±{:.0}",
            w_rnr.mbb_per_hand,
            w_rnr.ci95,
            w_nash.mbb_per_hand,
            w_nash.ci95
        );
    }

    #[test]
    fn key_roundtrip_bucket_encoding() {
        let k = make_key(168, &[2, 15, 1]);
        assert_eq!(&*k, &[168, 0, 2, 15, 1]);
        let k = make_key(300, &[]);
        assert_eq!(&*k, &[300u16 as u8, 1]);
    }

    /// After training 10bb heads-up:
    /// - BB with AA facing an all-in must mostly call.
    /// - BB with 32o facing an all-in must mostly fold.
    /// - The button must not be folding AA.
    #[test]
    fn push_fold_convergence() {
        let t = push_fold_trainer();
        t.run(80_000, &|_| {});
        assert!(t.node_count() > 100);

        let aa = preflop_bucket([make_card(12, 0), make_card(12, 1)]);
        let junk = preflop_bucket([make_card(1, 0), make_card(0, 1)]); // 32o

        // Facing an all-in: history = [AllIn token]. Actions: [Fold, CheckCall].
        let shove_hist = [AbsAction::AllIn.token()];
        let aa_call = t.avg_strategy(aa, &shove_hist).expect("AA-vs-shove visited");
        assert_eq!(aa_call.len(), 2);
        assert!(
            aa_call[1] > 0.8,
            "AA should call a 10bb shove, got call prob {:.3}",
            aa_call[1]
        );
        let junk_call = t.avg_strategy(junk, &shove_hist).expect("32o-vs-shove visited");
        assert!(
            junk_call[0] > 0.6,
            "32o should fold to a 10bb shove, got fold prob {:.3}",
            junk_call[0]
        );

        // Button root with AA: actions [Fold, CheckCall, Bet, AllIn] (or without Bet
        // if deduped). Fold must be near zero.
        let btn_aa = t.avg_strategy(aa, &[]).expect("root AA visited");
        assert!(
            btn_aa[0] < 0.1,
            "button must not fold AA, got fold prob {:.3}",
            btn_aa[0]
        );
    }

    /// VR-MCCFR baselines are a control variate: they must be populated and
    /// finite after training, and must not move the fixed point. The
    /// push/fold checks that are stable across seeds at this budget (AA
    /// calls a shove, the button never folds AA) must still hold.
    #[test]
    fn vr_baselines_are_learned_and_unbiased() {
        let t = push_fold_trainer().with_vr(true);
        t.run(80_000, &|_| {});
        assert!(!t.baselines.is_empty(), "baselines never populated");
        assert!(t
            .baselines
            .iter()
            .all(|e| e.value().iter().all(|v| v.is_finite())));
        // Some baseline must have learned a non-zero value: the sampled
        // child values feeding the EMA are chip utilities.
        assert!(t.baselines.iter().any(|e| e.value().iter().any(|&v| v != 0.0)));

        let aa = preflop_bucket([make_card(12, 0), make_card(12, 1)]);
        let shove_hist = [AbsAction::AllIn.token()];
        let aa_call = t.avg_strategy(aa, &shove_hist).expect("AA-vs-shove visited");
        assert!(aa_call[1] > 0.8, "AA should call a shove, got {:.3}", aa_call[1]);
        let btn_aa = t.avg_strategy(aa, &[]).expect("root AA visited");
        assert!(btn_aa[0] < 0.1, "button must not fold AA, got {:.3}", btn_aa[0]);
    }

    /// Spine bookkeeping: a node is on the spine only while the history
    /// since the round start matches the recorded tokens and the recorded
    /// seat is the one to act; the hero flag follows the seat.
    #[test]
    fn spine_step_matches_history_and_seat() {
        let abs = Abstraction::new(AbsConfig::default());
        let cfg = HandConfig {
            num_players: 3,
            ..HandConfig::default()
        };
        let h0 = Hand::new(&cfg, 0, fresh_deck()); // p0 to act preflop
        let acts0 = abs.abstract_actions(&h0);
        let mut h1 = h0.clone();
        h1.apply(PlayerAction::CheckCall); // p1 to act
        let acts1 = abs.abstract_actions(&h1);
        let base = vec![7u8, 7, TOKEN_STREET_SEP];
        let bet = *acts1.iter().find(|a| matches!(a, AbsAction::Bet(_))).unwrap();
        let spine = Spine {
            hero: 1,
            base_len: base.len(),
            steps: vec![(0, PlayerAction::CheckCall), (1, PlayerAction::RaiseTo(333))],
            tokens: vec![AbsAction::CheckCall.token(), bet.token()],
        };
        let mut hist = base.clone();
        let (si, act, hero) = spine.step_at(&hist, &h0, &acts0).expect("root is on the spine");
        assert_eq!(acts0[si], AbsAction::CheckCall);
        assert_eq!(act, PlayerAction::CheckCall);
        assert!(!hero);
        hist.push(AbsAction::CheckCall.token());
        let (si, act, hero) = spine.step_at(&hist, &h1, &acts1).expect("second step");
        assert_eq!(acts1[si], bet);
        assert_eq!(act, PlayerAction::RaiseTo(333));
        assert!(hero, "seat 1 is the hero");
        // Wrong seat to act for the recorded step: off the spine.
        assert!(spine.step_at(&hist, &h0, &acts0).is_none());
        // Deviated history: off the spine.
        let mut dev = base.clone();
        dev.push(AbsAction::Fold.token());
        assert!(spine.step_at(&dev, &h1, &acts1).is_none());
        // Past the last step: off the spine (the current decision).
        hist.push(bet.token());
        let mut h2 = h1.clone();
        h2.apply(PlayerAction::RaiseTo(333));
        assert!(spine.step_at(&hist, &h2, &abs.abstract_actions(&h2)).is_none());
        // Shorter than the base: off the spine.
        assert!(spine.step_at(&base[..1], &h0, &acts0).is_none());
    }

    /// Pluribus pruning rules (per-traversal decision, river and terminal
    /// exemptions) must keep the push/fold fixed point under aggressive
    /// pruning.
    #[test]
    fn pluribus_prune_keeps_the_fixed_point() {
        let mut t = push_fold_trainer();
        t.cfg.prune_after = 1_000; // prune almost from the start
        t.cfg.prune_threshold = -1.0; // ...anything with negative regret
        t.cfg.pluribus_prune = true;
        t.run(60_000, &|_| {});
        let aa = preflop_bucket([make_card(12, 0), make_card(12, 1)]);
        let shove_hist = [AbsAction::AllIn.token()];
        let aa_call = t.avg_strategy(aa, &shove_hist).expect("AA-vs-shove visited");
        assert!(aa_call[1] > 0.8, "AA should call a shove, got {:.3}", aa_call[1]);
        let btn_aa = t.avg_strategy(aa, &[]).expect("root AA visited");
        assert!(btn_aa[0] < 0.1, "button must not fold AA, got {:.3}", btn_aa[0]);
    }

    /// Multiway-focused sampling is importance-weighted, so it must keep
    /// the push/fold fixed point (AA calls a shove, the button never
    /// folds AA) even at a strong focus weight.
    #[test]
    fn multiway_focus_keeps_the_fixed_point() {
        let mut t = push_fold_trainer();
        t.cfg.multiway_focus = 0.6;
        t.run(120_000, &|_| {}); // importance weights add variance: extra margin
        let aa = preflop_bucket([make_card(12, 0), make_card(12, 1)]);
        let shove_hist = [AbsAction::AllIn.token()];
        let aa_call = t.avg_strategy(aa, &shove_hist).expect("AA-vs-shove visited");
        assert!(aa_call[1] > 0.8, "AA should call a shove, got {:.3}", aa_call[1]);
        let btn_aa = t.avg_strategy(aa, &[]).expect("root AA visited");
        assert!(btn_aa[0] < 0.1, "button must not fold AA, got {:.3}", btn_aa[0]);
        // Junk must still fold to a shove: the focus changes which lines
        // are sampled, not what is learned on them.
        let junk = preflop_bucket([make_card(0, 0), make_card(5, 1)]);
        let junk_call = t.avg_strategy(junk, &shove_hist).expect("junk-vs-shove visited");
        assert!(junk_call[1] < 0.2, "72o must fold to a shove, got {:.3}", junk_call[1]);
    }

    /// Snapshot averaging: postflop strategies come only from snapshots
    /// (zero before the first one), preflop keeps the running average.
    #[test]
    fn snapshot_avg_fills_postflop_only_from_snapshots() {
        let mut t = push_fold_trainer();
        t.cfg.snapshot_avg = true;
        t.run(20_000, &|_| {});
        let postflop_before = t
            .nodes
            .iter()
            .filter(|e| e.key()[2..].contains(&TOKEN_STREET_SEP))
            .filter(|e| e.strat.iter().any(|&x| x != 0.0))
            .count();
        assert_eq!(postflop_before, 0, "postflop strat accumulated without a snapshot");
        let aa = preflop_bucket([make_card(12, 0), make_card(12, 1)]);
        assert!(t.avg_strategy(aa, &[]).is_some(), "preflop running average missing");
        let n = t.snapshot_strategy();
        assert!(n > 0);
        let postflop_after = t
            .nodes
            .iter()
            .filter(|e| e.key()[2..].contains(&TOKEN_STREET_SEP))
            .filter(|e| normalize(&e.strat).is_some())
            .count();
        assert_eq!(postflop_after, n);
    }

    /// Subgame solving uses CFR+ (regret matching+): cumulative regrets are
    /// floored at zero, which converges faster on small trees.
    #[test]
    fn subgame_solver_keeps_regrets_nonnegative() {
        let t = push_fold_trainer().with_plus(true);
        let h = Hand::new(&t.cfg.hand, 0, fresh_deck());
        t.run_subgame(&h, &[], 5_000, 2_000, None);
        assert!(t.node_count() > 0);
        for e in t.nodes.iter() {
            assert!(
                e.value().regret.iter().all(|&r| r >= 0.0),
                "CFR+ regrets must be nonnegative, got {:?}",
                e.value().regret
            );
        }
    }

    #[test]
    fn blueprint_and_checkpoint_roundtrip() {
        let t = push_fold_trainer();
        t.run(2_000, &|_| {});
        let n_before = t.node_count();
        let iters_before = t.iterations();
        assert!(n_before > 0);

        let dir = std::env::temp_dir();
        let ckpt = dir.join("pluribus_test_ckpt.bin");
        let bp_path = dir.join("pluribus_test_bp.bin");
        let ckpt_s = ckpt.to_str().unwrap();
        let bp_s = bp_path.to_str().unwrap();

        t.save_checkpoint(ckpt_s).unwrap();
        let t2 = Trainer::load_checkpoint(ckpt_s, t.cfg.clone()).unwrap();
        assert_eq!(t2.node_count(), n_before);
        assert_eq!(t2.iterations(), iters_before);
        // Abstraction restored from the checkpoint, not from caller flags.
        assert_eq!(t2.abs.cfg.postflop_buckets, 6);
        assert_eq!(t2.abs.centroids, t.abs.centroids);
        // Resumed trainer can keep training.
        t2.run(500, &|_| {});
        assert_eq!(t2.iterations(), iters_before + 500);

        let bp = t.to_blueprint();
        bp.save(bp_s).unwrap();
        let bp2 = Blueprint::load(bp_s).unwrap();
        assert_eq!(bp.strategies.len(), bp2.strategies.len());
        assert_eq!(bp2.num_players, 2);
        assert_eq!(bp2.abs_cfg.postflop_buckets, 6);
        assert_eq!(bp2.centroids, t.abs.centroids);
        // Strategies are normalized distributions.
        for (_, s) in bp2.strategies.iter().take(50) {
            let sum: f32 = s.iter().sum();
            assert!((sum - 1.0).abs() < 1e-3, "unnormalized strategy: {}", sum);
            assert!(s.iter().all(|&p| (0.0..=1.0).contains(&p)));
        }

        let _ = std::fs::remove_file(ckpt);
        let _ = std::fs::remove_file(bp_path);
    }

    #[test]
    fn parallel_training_smoke() {
        let t = push_fold_trainer();
        // Runs across rayon's default thread pool without panicking or deadlocking.
        t.run(10_000, &|_| {});
        assert_eq!(t.iterations(), 10_000);
        assert!(t.node_count() > 50);
    }
}
