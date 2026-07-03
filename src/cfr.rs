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
use crate::engine::{Hand, HandConfig, Street, MAX_PLAYERS};
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
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            hand: HandConfig::default(),
            prune_threshold: -3.0e8,
            prune_prob: 0.95,
            prune_after: 200_000,
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
    pub limit: Street,
}

const LEAF_BIASES: usize = 4; // 0 = blueprint, 1 = fold-, 2 = call-, 3 = raise-biased
const BIAS_MULT: f64 = 5.0;

/// Re-randomizes hidden cards at a subgame root before each traversal.
pub type RootSampler<'a> = dyn Fn(&mut Hand, &mut SmallRng) + Sync + 'a;

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

    pub fn with_leaf(mut self, leaf: Option<LeafCfg>) -> Self {
        self.leaf = leaf;
        self
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
        (0..iterations).into_par_iter().for_each(|i| {
            let t = start + i + 1;
            let mut rng = SmallRng::seed_from_u64(t.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let traverser = (t % n as u64) as usize;
            let mut deck = fresh_deck();
            deck.shuffle(&mut rng);
            let hand = Hand::new(&self.cfg.hand, 0, deck);
            if hand.is_terminal() {
                return; // degenerate deal (e.g. blinds all-in with tiny stacks)
            }
            let weight = t as f64;
            let prune_ok = t > self.cfg.prune_after;
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

        if p == traverser {
            // Full-width over own actions, with negative-regret pruning.
            let mut explore: Vec<bool> = if prune_ok {
                regrets
                    .iter()
                    .map(|&r| {
                        r >= self.cfg.prune_threshold
                            || rng.random::<f64>() >= self.cfg.prune_prob
                    })
                    .collect()
            } else {
                vec![true; acts.len()]
            };
            if !explore.iter().any(|&e| e) {
                explore.iter_mut().for_each(|e| *e = true);
            }

            let mut utils = vec![0.0f64; acts.len()];
            let mut node_util = 0.0;
            for (i, &a) in acts.iter().enumerate() {
                if !explore[i] {
                    continue;
                }
                let mut child = h.clone();
                child.apply(self.abs.concrete(h, a));
                let depth = hist.len();
                hist.push(a.token());
                if !child.is_terminal() && child.street() != h.street() {
                    hist.push(TOKEN_STREET_SEP);
                }
                utils[i] =
                    self.traverse(&child, hist, traverser, weight, prune_ok, model_opp, rng);
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
            // contributes nothing to the learned average strategy.
            let model = self.rnr.as_ref().expect("model_opp without rnr").model;
            let a = crate::table::baseline_action(model, h, &self.abs, rng);
            let a = if acts.contains(&a) { a } else { acts[0] };
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
            if let Some(mut node) = self.nodes.get_mut(&key) {
                for (st, &d) in node.strat.iter_mut().zip(&dist) {
                    *st += weight * d;
                }
            }
            let idx = sample_index(&dist, rng);
            let a = acts[idx];
            let mut child = h.clone();
            child.apply(self.abs.concrete(h, a));
            hist.push(a.token());
            if !child.is_terminal() && child.street() != h.street() {
                hist.push(TOKEN_STREET_SEP);
            }
            // No truncation needed: the nearest traverser ancestor restores hist.
            self.traverse(&child, hist, traverser, weight, prune_ok, model_opp, rng)
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
            let acts = self.abs.abstract_actions(&h);
            let bucket = self.abs.bucket(h.hole(p), h.board(), rng);
            let mut probs: Vec<f64> = match lc.blueprint.get(bucket, &hist) {
                Some(s) if s.len() == acts.len() => s.iter().map(|&x| x as f64).collect(),
                _ => vec![1.0 / acts.len() as f64; acts.len()],
            };
            apply_bias(&mut probs, &acts, biases[p]);
            let a = acts[sample_index(&probs, rng)];
            let street_before = h.street();
            h.apply(self.abs.concrete(&h, a));
            hist.push(a.token());
            if !h.is_terminal() && h.street() != street_before {
                hist.push(TOKEN_STREET_SEP);
            }
        }
        h.utilities()[traverser] as f64
    }

    /// Normalized average strategy for an infoset, if visited.
    pub fn avg_strategy(&self, bucket: u16, hist: &[u8]) -> Option<Vec<f64>> {
        let key = make_key(bucket, hist);
        let node = self.nodes.get(&key)?;
        normalize(&node.strat)
    }

    /// Export the (much smaller) normalized average strategy for play.
    pub fn to_blueprint(&self) -> Blueprint {
        let mut strategies = HashMap::with_capacity(self.nodes.len());
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

/// The trained average strategy used at the table, together with the card
/// abstraction it was trained under (config + k-means centroids), so play
/// and benchmarking bucket cards exactly as training did.
#[derive(Serialize, Deserialize)]
pub struct Blueprint {
    pub strategies: HashMap<Vec<u8>, Vec<f32>>,
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
        bincode::deserialize_from(f).map_err(std::io::Error::other)
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
