//! Action and card abstraction.
//!
//! Action abstraction: fold / check-call / a small per-street menu of
//! pot-fraction bets / all-in. Card abstraction: 169 canonical preflop hands;
//! postflop Monte Carlo equity vs one random hand, quantized into buckets,
//! memoized in a lock-free cache keyed exactly (packed cards, no hash collisions).

use crate::cards::{fresh_deck, make_card, rank, Card};
use crate::engine::{Hand, PlayerAction, Street};
use dashmap::DashMap;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Pot-fraction bet sizes, in percent of (pot + amount to call).
pub const BET_SIZES: [u16; 8] = [25, 33, 50, 75, 100, 150, 200, 300];

/// History tokens are u8: Fold=0, CheckCall=1, AllIn=2, Bet(i)=3+i, street separator=15.
pub const TOKEN_STREET_SEP: u8 = 15;
/// Key suffix marking a depth-limit leaf's continuation-strategy infoset.
pub const TOKEN_LEAF: u8 = 14;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbsAction {
    Fold,
    CheckCall,
    /// Index into BET_SIZES.
    Bet(u8),
    AllIn,
}

impl AbsAction {
    pub fn token(self) -> u8 {
        match self {
            AbsAction::Fold => 0,
            AbsAction::CheckCall => 1,
            AbsAction::AllIn => 2,
            AbsAction::Bet(i) => 3 + i,
        }
    }
}

/// Number of quantiles representing an equity distribution (flop/turn).
pub const QUANTILES: usize = 8;

/// Opponent hand-strength clusters for OCHS features (preflop strength tiers).
pub const OCHS_CLUSTERS: usize = 8;

/// Dimension of the combined potential-aware feature: equity-distribution
/// quantiles (shape) followed by opponent-cluster hand-strength (how the
/// hand fares against different parts of the opponent range). Reserved so
/// the dimension-based feature-family dispatch stays unambiguous: it must
/// differ from QUANTILES and from any strategic-fingerprint length.
pub const COMBINED_DIM: usize = QUANTILES + OCHS_CLUSTERS;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbsConfig {
    /// Number of postflop card buckets per street (k-means clusters on
    /// flop/turn, equity quantiles on the river).
    pub postflop_buckets: u16,
    /// Bet-size menu shape. Stored with the blueprint, so play and probes
    /// always use the menu the blueprint was trained with.
    pub menu: MenuShape,
    /// Monte Carlo rollouts per river equity estimate.
    pub equity_rollouts: u32,
    /// Sampled future-board runouts per flop/turn distribution estimate.
    pub dist_runouts: u32,
    /// Equity rollouts per sampled runout.
    pub runout_rollouts: u32,
    /// Stop inserting into the equity cache beyond this many entries.
    pub cache_cap: usize,
}

impl Default for AbsConfig {
    fn default() -> Self {
        AbsConfig {
            postflop_buckets: 12,
            menu: MenuShape::Wide,
            equity_rollouts: 200,
            dist_runouts: 24,
            runout_rollouts: 50,
            cache_cap: 30_000_000,
        }
    }
}

/// Bet-size menu shapes (indices into BET_SIZES = [25,33,50,75,100,150,200,300]).
///
/// - `Wide`: the 2026-07 menu — 4 preflop opens, 6-7 postflop first-in
///   sizes from 25% through 200% overbets, 4-size raises, 2-size re-raises.
///   Measured null vs the earlier narrow menu at 200M (BASELINES.md).
/// - `Pluribus`: Pluribus's postflop shape (Science 2019 supplement): at
///   most 50% / 100% / all-in for the first bet in a round, 100% / all-in
///   for a raise, call / fold / all-in beyond that. Preflop as `Wide`.
///   Search, not the blueprint, is meant to supply sizing at the table.
/// - `PluribusFinePre`: `Pluribus` postflop plus a fine preflop menu
///   (7 opens, 5 three-bet sizes, 3 four-bet sizes), because Pluribus
///   never searches preflop and so made that round's abstraction fine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
pub enum MenuShape {
    Wide,
    Pluribus,
    PluribusFinePre,
}

/// K-medians cluster centers, one set per street. Two feature families
/// share this container, self-described by vector dimension:
/// - equity-distribution quantiles (dimension == QUANTILES), or
/// - strategic fingerprints from a previous blueprint (any other dimension:
///   prior policy's action distribution + rollout value statistics).
/// Trained once before blueprint training and persisted with the blueprint
/// so play-time bucketing matches training exactly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Centroids {
    pub flop: Vec<Vec<f32>>,
    pub turn: Vec<Vec<f32>>,
}

impl Centroids {
    /// Sample random (hole, board) situations per street and cluster their
    /// equity-distribution quantile vectors with EMD k-medians. Deterministic
    /// for a given seed.
    pub fn train(cfg: &AbsConfig, samples: usize, seed: u64) -> Centroids {
        Centroids {
            flop: train_street_centroids(cfg, 3, samples, seed),
            turn: train_street_centroids(cfg, 4, samples, seed ^ 0x5EED_5EED),
        }
    }

    /// Strategy-aware abstraction co-training: cluster hands by how the
    /// PREVIOUS blueprint plays and realizes them (strategic fingerprints)
    /// instead of by their equity distributions. `cfg` is the NEW
    /// abstraction's configuration (bucket count).
    pub fn train_strategic(cfg: &AbsConfig, samples: usize, seed: u64, sc: &StratCtx) -> Centroids {
        Centroids {
            flop: train_street_strategic(cfg, 3, samples, seed, sc),
            turn: train_street_strategic(cfg, 4, samples, seed ^ 0x5EED_5EED, sc),
        }
    }

    /// Sample random situations per street and cluster their combined
    /// potential-aware feature (quantiles + OCHS) with L1 k-medians.
    pub fn train_combined(cfg: &AbsConfig, samples: usize, seed: u64) -> Centroids {
        Centroids {
            flop: train_street_combined(cfg, 3, samples, seed),
            turn: train_street_combined(cfg, 4, samples, seed ^ 0x5EED_5EED),
        }
    }

    /// Feature family, recognized by centroid dimension. Equity quantiles and
    /// the combined OCHS feature are self-contained; strategic fingerprints
    /// need the previous blueprint at bucket time.
    pub fn is_strategic(&self) -> bool {
        !matches!(
            self.flop.first().map(|v| v.len()),
            Some(QUANTILES) | Some(COMBINED_DIM)
        )
    }
}

/// Context for strategic fingerprints: the previous co-training round's
/// blueprint and its abstraction.
pub struct StratCtx {
    pub bp: Arc<crate::cfr::Blueprint>,
    pub abs: Arc<Abstraction>,
    /// Self-play rollouts per fingerprint.
    pub rollouts: u32,
}

/// A hand's strategic fingerprint at a standardized heads-up limped-pot
/// street start: the previous policy's action distribution with this hand,
/// plus mean / std / win-fraction of its blueprint-vs-blueprint rollout
/// values — "what the trained policy does with the hand and what it earns",
/// rather than raw equity. Suit-isomorphism invariant (the previous
/// abstraction's buckets are, and rollouts are distributionally symmetric).
pub fn strategic_fingerprint(
    sc: &StratCtx,
    hole: [Card; 2],
    board: &[Card],
    rng: &mut SmallRng,
) -> Vec<f32> {
    let (h0, hist) = standard_state(hole, board);
    let acts = sc.abs.abstract_actions(&h0);
    let bucket = sc.abs.bucket(hole, board, rng);
    let mut feat: Vec<f32> = match sc.bp.get(bucket, &hist) {
        Some(s) if s.len() == acts.len() && s.iter().sum::<f32>() > 0.0 => {
            let total: f32 = s.iter().sum();
            s.iter().map(|&x| x / total).collect()
        }
        _ => vec![1.0 / acts.len() as f32; acts.len()],
    };

    let hero = h0.to_act();
    let mut vals = Vec::with_capacity(sc.rollouts as usize);
    let mut wins = 0u32;
    for _ in 0..sc.rollouts {
        let mut sim = h0.clone();
        let mut want: [Option<[Card; 2]>; crate::engine::MAX_PLAYERS] =
            [None; crate::engine::MAX_PLAYERS];
        want[hero] = Some(hole);
        sim.resample_hidden_with(&want, rng);
        let mut hist2 = hist.clone();
        let mut guard = 0;
        while !sim.is_terminal() && guard < 60 {
            guard += 1;
            let p_acts = sc.abs.abstract_actions(&sim);
            let b = sc.abs.bucket(sim.hole(sim.to_act()), sim.board(), rng);
            let a = match sc.bp.get(b, &hist2) {
                Some(s) if s.len() == p_acts.len() && s.iter().sum::<f32>() > 0.0 => {
                    let probs: Vec<f64> = s.iter().map(|&x| x as f64).collect();
                    let total: f64 = probs.iter().sum();
                    let norm: Vec<f64> = probs.iter().map(|x| x / total).collect();
                    p_acts[crate::cfr::sample_index(&norm, rng)]
                }
                _ => AbsAction::CheckCall,
            };
            let street_before = sim.street();
            sim.apply(sc.abs.concrete(&sim, a));
            hist2.push(a.token());
            if !sim.is_terminal() && sim.street() != street_before {
                hist2.push(TOKEN_STREET_SEP);
            }
        }
        if sim.is_terminal() {
            let u = sim.utilities()[hero] as f64;
            if u > 0.0 {
                wins += 1;
            }
            vals.push(u);
        }
    }
    let n = vals.len().max(1) as f64;
    let mean = vals.iter().sum::<f64>() / n;
    let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    feat.push(((mean / 2_000.0) as f32).clamp(-2.0, 2.0));
    feat.push(((var.sqrt() / 2_000.0) as f32).clamp(0.0, 2.0));
    feat.push(wins as f32 / sc.rollouts.max(1) as f32);
    feat
}

/// Standardized heads-up limped-pot state at this street start (hero =
/// first to act) with the given hole/board, plus its history tokens.
fn standard_state(hole: [Card; 2], board: &[Card]) -> (crate::engine::Hand, Vec<u8>) {
    debug_assert!(board.len() == 3 || board.len() == 4);
    let cfg = crate::engine::HandConfig {
        num_players: 2,
        stack: 10_000,
        sb: 50,
        bb: 100,
    };
    let mut used = [false; 52];
    used[hole[0] as usize] = true;
    used[hole[1] as usize] = true;
    for &c in board {
        used[c as usize] = true;
    }
    let mut free = (0..52u8).filter(|&c| !used[c as usize]);
    let mut deck = [0u8; 52];
    // p0 = dummy villain (resampled per rollout), p1 = hero (first to act
    // postflop heads-up), board at deck[4..9].
    deck[0] = free.next().unwrap();
    deck[1] = free.next().unwrap();
    deck[2] = hole[0];
    deck[3] = hole[1];
    for (i, &c) in board.iter().enumerate() {
        deck[4 + i] = c;
    }
    let mut next = 4 + board.len();
    for c in free {
        deck[next] = c;
        next += 1;
    }
    let mut h = crate::engine::Hand::new(&cfg, 0, deck);
    let mut hist = Vec::with_capacity(8);
    let mut limp_check = |h: &mut crate::engine::Hand, hist: &mut Vec<u8>| {
        for _ in 0..2 {
            h.apply(crate::engine::PlayerAction::CheckCall);
            hist.push(AbsAction::CheckCall.token());
        }
        hist.push(TOKEN_STREET_SEP);
    };
    limp_check(&mut h, &mut hist); // limp + check -> flop
    if board.len() == 4 {
        limp_check(&mut h, &mut hist); // check-check -> turn
    }
    debug_assert_eq!(h.board().len(), board.len());
    (h, hist)
}

fn train_street_strategic(
    cfg: &AbsConfig,
    board_len: usize,
    samples: usize,
    seed: u64,
    sc: &StratCtx,
) -> Vec<Vec<f32>> {
    let points: Vec<Vec<f32>> = (0..samples)
        .into_par_iter()
        .map(|i| {
            let mut rng =
                SmallRng::seed_from_u64(seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut deck = fresh_deck();
            deck.shuffle(&mut rng);
            let hole = [deck[0], deck[1]];
            let board = deck[2..2 + board_len].to_vec();
            strategic_fingerprint(sc, hole, &board, &mut rng)
        })
        .collect();
    kmedians(&points, cfg.postflop_buckets as usize, 25, seed)
}

fn train_street_centroids(
    cfg: &AbsConfig,
    board_len: usize,
    samples: usize,
    seed: u64,
) -> Vec<Vec<f32>> {
    let points: Vec<Vec<f32>> = (0..samples)
        .into_par_iter()
        .map(|i| {
            let mut rng =
                SmallRng::seed_from_u64(seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut deck = fresh_deck();
            deck.shuffle(&mut rng);
            let hole = [deck[0], deck[1]];
            let board = deck[2..2 + board_len].to_vec();
            equity_quantiles(hole, &board, cfg.dist_runouts, cfg.runout_rollouts, &mut rng)
                .to_vec()
        })
        .collect();
    kmedians(&points, cfg.postflop_buckets as usize, 25, seed)
}

pub struct Abstraction {
    pub cfg: AbsConfig,
    pub centroids: Option<Centroids>,
    /// Exact-key memo (hot path: one packed-u64 lookup per visit).
    cache: DashMap<u64, u16, ahash::RandomState>,
    /// Suit-canonical memo, consulted only on exact-cache misses so the
    /// expensive Monte Carlo work runs once per isomorphism class (up to
    /// 24 suit relabelings share one entry).
    canon: DashMap<u64, u16, ahash::RandomState>,
    /// Exact river buckets, one table per suit-canonical BOARD covering
    /// every hole combo at once (O(N) sweep, ~1326 evals per board). The
    /// canonical river-board space is ~134k entries, so this converges to
    /// pure lookups; per-pair Monte Carlo river equity cost 200 rollouts
    /// per miss and dominated blueprint training (83% of samples).
    river_boards: DashMap<u64, std::sync::Arc<Vec<u16>>, ahash::RandomState>,
    /// Previous-round context, required when `centroids.is_strategic()`.
    strat: Option<StratCtx>,
}

impl Abstraction {
    /// Raw-equity bucketing on every street (no k-means centroids).
    #[allow(dead_code)]
    pub fn new(cfg: AbsConfig) -> Self {
        Self::with_centroids(cfg, None)
    }

    pub fn with_centroids(cfg: AbsConfig, centroids: Option<Centroids>) -> Self {
        Abstraction {
            cfg,
            centroids,
            cache: DashMap::with_hasher(ahash::RandomState::new()),
            canon: DashMap::with_hasher(ahash::RandomState::new()),
            river_boards: DashMap::with_hasher(ahash::RandomState::new()),
            strat: None,
        }
    }

    /// Attach the previous-round blueprint context needed to evaluate
    /// strategic fingerprints at cache misses.
    pub fn with_strat(mut self, sc: StratCtx) -> Self {
        self.strat = Some(sc);
        self
    }

    /// Bet-size menu (indices into BET_SIZES = [25,33,50,75,100,150,200,300])
    /// for a street and raise count. Wider than the Pluribus baseline: first-in
    /// menus span a small (25-33%) bet through overbets so the blueprint can
    /// represent modern sizing, with progressively pruned raise/re-raise menus
    /// to bound the tree.
    pub fn bet_menu(&self, street: Street, n_raises: u8) -> &'static [u8] {
        match (self.cfg.menu, street, n_raises) {
            // Preflop: Wide and Pluribus share the 2026-07 preflop menu.
            (MenuShape::Wide | MenuShape::Pluribus, Street::Preflop, 0) => &[1, 2, 3, 4],
            (MenuShape::Wide | MenuShape::Pluribus, Street::Preflop, 1) => &[2, 4, 6],
            (MenuShape::Wide | MenuShape::Pluribus, Street::Preflop, 2) => &[4, 6],
            (MenuShape::Wide | MenuShape::Pluribus, Street::Preflop, _) => &[],
            // Fine preflop: 7 opens (1.6x..6x), 5 three-bets, 3 four-bets.
            (MenuShape::PluribusFinePre, Street::Preflop, 0) => &[0, 1, 2, 3, 4, 5, 6],
            (MenuShape::PluribusFinePre, Street::Preflop, 1) => &[1, 2, 3, 4, 6],
            (MenuShape::PluribusFinePre, Street::Preflop, 2) => &[2, 4, 6],
            (MenuShape::PluribusFinePre, Street::Preflop, _) => &[4],
            // Wide postflop: 25..200% first-in, 4-size raises, 2-size re-raises.
            (MenuShape::Wide, Street::Flop, 0) => &[0, 1, 2, 3, 4, 5, 6],
            (MenuShape::Wide, _, 0) => &[1, 2, 3, 4, 5, 6],
            (MenuShape::Wide, _, 1) => &[2, 3, 4, 6],
            (MenuShape::Wide, _, 2) => &[4, 6],
            (MenuShape::Wide, _, _) => &[],
            // Pluribus postflop: 50%/100% (+ all-in) first-in, pot (+ all-in)
            // raise, call/fold/all-in beyond.
            (_, _, 0) => &[2, 4],
            (_, _, 1) => &[4],
            (_, _, _) => &[],
        }
    }

    /// Raise-to amount for a pot-fraction bet: current bet plus a fraction of
    /// the pot as it would be after calling.
    fn bet_target(h: &Hand, size_idx: u8) -> u32 {
        let pct = BET_SIZES[size_idx as usize] as u64;
        let pot_after_call = (h.pot() + h.to_call()) as u64;
        h.current_bet() + (pot_after_call * pct / 100) as u32
    }

    /// Legal abstract actions for the player to act. Never empty on a
    /// non-terminal hand; concrete amounts are deduplicated.
    pub fn abstract_actions(&self, h: &Hand) -> Vec<AbsAction> {
        let mut acts = Vec::with_capacity(6);
        if !h.can_check() {
            acts.push(AbsAction::Fold);
        }
        acts.push(AbsAction::CheckCall);

        if let Some((lo, hi)) = h.raise_bounds() {
            let mut seen: Vec<u32> = Vec::with_capacity(4);
            for &i in self.bet_menu(h.street(), h.n_raises()) {
                let t = Self::bet_target(h, i).clamp(lo, hi);
                if t < hi && !seen.contains(&t) {
                    seen.push(t);
                    acts.push(AbsAction::Bet(i));
                }
            }
            acts.push(AbsAction::AllIn);
        }
        acts
    }

    /// Translate an abstract action into a concrete engine action.
    pub fn concrete(&self, h: &Hand, a: AbsAction) -> PlayerAction {
        match a {
            AbsAction::Fold => PlayerAction::Fold,
            AbsAction::CheckCall => PlayerAction::CheckCall,
            AbsAction::Bet(i) => {
                let (lo, hi) = h.raise_bounds().expect("bet with no legal raise");
                PlayerAction::RaiseTo(Self::bet_target(h, i).clamp(lo, hi))
            }
            AbsAction::AllIn => {
                let (_, hi) = h.raise_bounds().expect("all-in with no legal raise");
                PlayerAction::RaiseTo(hi)
            }
        }
    }

    /// Map an arbitrary raise-to amount onto the nearest abstract raise
    /// (log-space distance), for tracking off-tree opponents in play mode.
    pub fn map_raise(&self, h: &Hand, raise_to: u32) -> AbsAction {
        let mut best = AbsAction::AllIn;
        let mut best_d = f64::MAX;
        for a in self.abstract_actions(&h.clone()) {
            if let PlayerAction::RaiseTo(t) = self.concrete(h, a) {
                let d = ((raise_to.max(1) as f64).ln() - (t.max(1) as f64).ln()).abs();
                if d < best_d {
                    best_d = d;
                    best = a;
                }
            }
        }
        best
    }

    /// Card bucket for a hole/board pair. Preflop: canonical 169.
    /// Flop/turn: nearest EMD k-medians centroid of the equity distribution
    /// over future runouts (falls back to quantized raw equity without
    /// centroids). River: quantized MC equity vs one uniform random hand.
    pub fn bucket(&self, hole: [Card; 2], board: &[Card], rng: &mut SmallRng) -> u16 {
        if board.is_empty() {
            return preflop_bucket(hole);
        }
        if board.len() == 5 && self.strat.is_none() {
            return self.river_bucket(hole, board);
        }
        let key = pack_cards_key(hole, board);
        if let Some(b) = self.cache.get(&key) {
            return *b;
        }
        // Miss: fold suit-isomorphic variants onto one canonical entry.
        let ckey = canonical_cards_key(hole, board);
        if let Some(b) = self.canon.get(&ckey) {
            let b = *b;
            if self.cache.len() < self.cfg.cache_cap {
                self.cache.insert(key, b);
            }
            return b;
        }
        let nb = self.cfg.postflop_buckets;
        let street_cents = match board.len() {
            3 => self.centroids.as_ref().map(|c| &c.flop),
            4 => self.centroids.as_ref().map(|c| &c.turn),
            _ => None,
        };
        // Monte Carlo draws are seeded from the canonical key, not the
        // caller's rng: the same (hole, board) always gets the same
        // estimate (and therefore the same bucket, even after an eviction),
        // and a cache miss never consumes caller draws — bucketing is
        // deterministic regardless of thread timing or cache state.
        let _ = rng;
        let mut krng = SmallRng::seed_from_u64(ckey ^ 0xD1CE_5EED_0BAD_F00D);
        let b = match street_cents {
            Some(cents) if cents.first().map(|v| v.len()) == Some(COMBINED_DIM) => {
                let f = combined_features(
                    hole,
                    board,
                    self.cfg.dist_runouts,
                    self.cfg.runout_rollouts,
                    &mut krng,
                );
                nearest_centroid(&f, cents)
            }
            Some(cents) if cents.first().map(|v| v.len()) != Some(QUANTILES) => {
                let sc = self.strat.as_ref().expect(
                    "strategic centroids need the previous blueprint (--strat-prev)",
                );
                let f = strategic_fingerprint(sc, hole, board, &mut krng);
                nearest_centroid(&f, cents)
            }
            Some(cents) => {
                let q = equity_quantiles(
                    hole,
                    board,
                    self.cfg.dist_runouts,
                    self.cfg.runout_rollouts,
                    &mut krng,
                );
                nearest_centroid(&q, cents)
            }
            None => {
                let eq = equity_vs_random(hole, board, self.cfg.equity_rollouts, &mut krng);
                ((eq * nb as f64) as u16).min(nb - 1)
            }
        };
        if self.cache.len() < self.cfg.cache_cap {
            self.cache.insert(key, b);
        }
        if self.canon.len() < self.cfg.cache_cap {
            self.canon.insert(ckey, b);
        }
        b
    }

    /// Exact river bucket via a per-board table: suit-canonicalize the
    /// board, build (once) the exact-equity bucket of EVERY hole combo on
    /// it with the O(N) showdown sweep, then look up the hole through the
    /// same suit permutation.
    ///
    /// Hot path: a traversal queries the same river board thousands of
    /// times, so a thread-local one-slot memo skips both the 24-permutation
    /// canonicalization and the shared-map lookup on repeats, and the
    /// shared map is read-locked (get) before ever write-locking (entry).
    fn river_bucket(&self, hole: [Card; 2], board: &[Card]) -> u16 {
        debug_assert_eq!(board.len(), 5);
        thread_local! {
            static LAST: std::cell::RefCell<Option<([Card; 5], [u8; 4], std::sync::Arc<Vec<u16>>)>> =
                const { std::cell::RefCell::new(None) };
        }
        let b5 = [board[0], board[1], board[2], board[3], board[4]];
        let memo = LAST.with(|l| {
            l.borrow()
                .as_ref()
                .filter(|(b, _, _)| *b == b5)
                .map(|(_, p, t)| (*p, t.clone()))
        });
        let (perm, table) = match memo {
            Some(hit) => hit,
            None => {
                let (bkey, perm) = canonical_board(board);
                let table = match self.river_boards.get(&bkey) {
                    Some(t) => t.clone(),
                    None => self
                        .river_boards
                        .entry(bkey)
                        .or_insert_with(|| {
                            std::sync::Arc::new(build_river_table(
                                &apply_perm(board, &perm),
                                self.cfg.postflop_buckets,
                            ))
                        })
                        .clone(),
                };
                LAST.with(|l| *l.borrow_mut() = Some((b5, perm, table.clone())));
                (perm, table)
            }
        };
        let map = |c: Card| (c & !3) | perm[(c & 3) as usize];
        table[crate::search::combo_index(map(hole[0]), map(hole[1]))]
    }

    #[allow(dead_code)]
    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }

    #[allow(dead_code)]
    pub fn river_tables_len(&self) -> usize {
        self.river_boards.len()
    }

    #[allow(dead_code)]
    pub fn canon_len(&self) -> usize {
        self.canon.len()
    }
}

/// Canonical preflop hand index in 0..169.
/// Pairs on the diagonal, suited above it, offsuit below.
pub fn preflop_bucket(hole: [Card; 2]) -> u16 {
    let (r1, r2) = (rank(hole[0]) as u16, rank(hole[1]) as u16);
    let hi = r1.max(r2);
    let lo = r1.min(r2);
    let suited = (hole[0] & 3) == (hole[1] & 3);
    if suited && hi != lo {
        hi * 13 + lo
    } else {
        lo * 13 + hi
    }
}

/// All 24 permutations of the four suits.
const SUIT_PERMS: [[u8; 4]; 24] = [
    [0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 1, 3], [0, 2, 3, 1], [0, 3, 1, 2], [0, 3, 2, 1],
    [1, 0, 2, 3], [1, 0, 3, 2], [1, 2, 0, 3], [1, 2, 3, 0], [1, 3, 0, 2], [1, 3, 2, 0],
    [2, 0, 1, 3], [2, 0, 3, 1], [2, 1, 0, 3], [2, 1, 3, 0], [2, 3, 0, 1], [2, 3, 1, 0],
    [3, 0, 1, 2], [3, 0, 2, 1], [3, 1, 0, 2], [3, 1, 2, 0], [3, 2, 0, 1], [3, 2, 1, 0],
];

/// Canonical key for a (hole, board) pair under suit isomorphism: the
/// lexicographically smallest packed key over all 24 suit relabelings.
/// Equity (and its distribution over runouts) is invariant under relabeling,
/// so isomorphic pairs may share one cached bucket. Only computed on cache
/// misses — the Monte Carlo work it saves costs ~5 orders of magnitude more.
pub(crate) fn canonical_cards_key(hole: [Card; 2], board: &[Card]) -> u64 {
    let mut best = u64::MAX;
    for perm in &SUIT_PERMS {
        let map = |c: Card| (c & !3) | perm[(c & 3) as usize];
        let h = [map(hole[0]), map(hole[1])];
        let mut b = [0u8; 5];
        for (dst, &src) in b.iter_mut().zip(board) {
            *dst = map(src);
        }
        let b = &mut b[..board.len()];
        b.sort_unstable();
        let mut key = board.len() as u64;
        key = (key << 6) | h[0].min(h[1]) as u64;
        key = (key << 6) | h[0].max(h[1]) as u64;
        for &c in b.iter() {
            key = (key << 6) | c as u64;
        }
        best = best.min(key);
    }
    best
}

/// Exact cache key: up to 7 cards, sorted, 6 bits each, plus board length.
fn pack_cards_key(hole: [Card; 2], board: &[Card]) -> u64 {
    let mut cards = [0u8; 7];
    cards[0] = hole[0].min(hole[1]);
    cards[1] = hole[0].max(hole[1]);
    let mut b: Vec<Card> = board.to_vec();
    b.sort_unstable();
    for (i, &c) in b.iter().enumerate() {
        cards[2 + i] = c;
    }
    let mut key = board.len() as u64;
    for &c in cards.iter().take(2 + board.len()) {
        key = (key << 6) | c as u64;
    }
    key
}

/// Suit-canonical form of a 5-card board: the permutation of suits (from
/// the 24) minimizing the packed sorted-board key, and that key.
fn canonical_board(board: &[Card]) -> (u64, [u8; 4]) {
    let mut best_key = u64::MAX;
    let mut best_perm = SUIT_PERMS[0];
    for perm in &SUIT_PERMS {
        let mut b = [0u8; 5];
        for (dst, &src) in b.iter_mut().zip(board) {
            *dst = (src & !3) | perm[(src & 3) as usize];
        }
        b.sort_unstable();
        let mut key = 0u64;
        for &c in &b {
            key = (key << 6) | c as u64;
        }
        if key < best_key {
            best_key = key;
            best_perm = *perm;
        }
    }
    (best_key, best_perm)
}

fn apply_perm(board: &[Card], perm: &[u8; 4]) -> Vec<Card> {
    board.iter().map(|&c| (c & !3) | perm[(c & 3) as usize]).collect()
}

/// Exact river-equity bucket for every hole combo on `board` in one O(N)
/// sweep: rank all combos, then equity = win + tie/2 fraction of the
/// card-removal-compatible opponent combos (uniform opponent — the same
/// estimand as `equity_vs_random`, exactly).
fn build_river_table(board: &[Card], nb: u16) -> Vec<u16> {
    use crate::eval::eval_hole_board;
    use crate::search::NUM_COMBOS;
    let combos = crate::search::all_combos();
    let board5 = [board[0], board[1], board[2], board[3], board[4]];
    let mut on_board = [false; 52];
    for &c in board {
        on_board[c as usize] = true;
    }
    let rank: Vec<u32> = combos
        .iter()
        .map(|c| {
            if on_board[c[0] as usize] || on_board[c[1] as usize] {
                0
            } else {
                eval_hole_board(c, &board5)
            }
        })
        .collect();
    let mut sorted: Vec<u32> = (0..NUM_COMBOS as u32)
        .filter(|&ci| rank[ci as usize] > 0)
        .collect();
    sorted.sort_unstable_by_key(|&ci| rank[ci as usize]);

    let mut reach = vec![0.0f64; NUM_COMBOS];
    let mut total = 0.0f64;
    let mut card = [0.0f64; 52];
    for &ci in &sorted {
        let c = combos[ci as usize];
        reach[ci as usize] = 1.0;
        total += 1.0;
        card[c[0] as usize] += 1.0;
        card[c[1] as usize] += 1.0;
    }
    // matched = 1, dead = 0: sweep value = (weaker − stronger) mass, so
    // equity = (weaker + tie/2) / compat = 0.5 + value / (2·compat).
    let v = crate::river::showdown_sweep(&combos, &sorted, &rank, &reach, total, &card, 1.0, 0.0);
    let mut out = vec![0u16; NUM_COMBOS];
    for &ci in &sorted {
        let ci = ci as usize;
        let c = combos[ci];
        let compat = total - card[c[0] as usize] - card[c[1] as usize] + 1.0;
        let eq = 0.5 + v[ci] / (2.0 * compat);
        out[ci] = ((eq * nb as f64) as u16).min(nb - 1);
    }
    out
}

/// Monte Carlo equity of hole+board vs one uniform random opponent hand,
/// with random board runout. Ties count half.
fn equity_vs_random(hole: [Card; 2], board: &[Card], rollouts: u32, rng: &mut SmallRng) -> f64 {
    use crate::eval::eval_hole_board;
    use rand::Rng;

    let mut used = [false; 52];
    used[hole[0] as usize] = true;
    used[hole[1] as usize] = true;
    for &c in board {
        used[c as usize] = true;
    }
    let mut rem = [0u8; 52];
    let mut rem_len = 0;
    for c in 0..52u8 {
        if !used[c as usize] {
            rem[rem_len] = c;
            rem_len += 1;
        }
    }
    let need_board = 5 - board.len();
    let draw = 2 + need_board;

    let mut full_board = [0u8; 5];
    full_board[..board.len()].copy_from_slice(board);

    let mut score = 0.0f64;
    for _ in 0..rollouts {
        // Partial Fisher-Yates: draw `draw` cards from rem.
        for k in 0..draw {
            let j = rng.random_range(k..rem_len);
            rem.swap(k, j);
        }
        let opp = [rem[0], rem[1]];
        for i in 0..need_board {
            full_board[board.len() + i] = rem[2 + i];
        }
        let ours = eval_hole_board(&hole, &full_board);
        let theirs = eval_hole_board(&opp, &full_board);
        if ours > theirs {
            score += 1.0;
        } else if ours == theirs {
            score += 0.5;
        }
    }
    score / rollouts as f64
}

/// Quantile vector of the hand's river-equity distribution: sample `runouts`
/// future boards, estimate equity vs one random hand on each with
/// `rollouts` Monte Carlo rollouts, and take QUANTILES evenly spaced order
/// statistics. Sorted by construction.
pub fn equity_quantiles(
    hole: [Card; 2],
    board: &[Card],
    runouts: u32,
    rollouts: u32,
    rng: &mut SmallRng,
) -> [f32; QUANTILES] {
    use rand::Rng;
    debug_assert!(!board.is_empty() && board.len() < 5);

    let mut used = [false; 52];
    used[hole[0] as usize] = true;
    used[hole[1] as usize] = true;
    for &c in board {
        used[c as usize] = true;
    }
    let mut unseen = [0u8; 52];
    let mut unseen_len = 0;
    for c in 0..52u8 {
        if !used[c as usize] {
            unseen[unseen_len] = c;
            unseen_len += 1;
        }
    }
    let need = 5 - board.len();
    let mut full = [0u8; 5];
    full[..board.len()].copy_from_slice(board);

    let mut eqs: Vec<f32> = Vec::with_capacity(runouts as usize);
    for _ in 0..runouts {
        for k in 0..need {
            let j = rng.random_range(k..unseen_len);
            unseen.swap(k, j);
        }
        for i in 0..need {
            full[board.len() + i] = unseen[i];
        }
        eqs.push(equity_vs_random(hole, &full, rollouts, rng) as f32);
    }
    eqs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let mut q = [0f32; QUANTILES];
    for (i, qi) in q.iter_mut().enumerate() {
        let pos = ((i as f64 + 0.5) / QUANTILES as f64 * eqs.len() as f64) as usize;
        *qi = eqs[pos.min(eqs.len() - 1)];
    }
    q
}

/// Preflop strength tier (0 = weakest .. OCHS_CLUSTERS-1 = strongest) for
/// each of the 169 canonical preflop buckets, split into equal-count tiers
/// by all-in preflop equity vs a random hand. Computed once, deterministic.
fn preflop_tiers() -> &'static [u8; 169] {
    use std::sync::OnceLock;
    static TIERS: OnceLock<[u8; 169]> = OnceLock::new();
    TIERS.get_or_init(|| {
        // Representative hole cards for each preflop bucket, then MC equity.
        let mut eq = [0f32; 169];
        for (b, e) in eq.iter_mut().enumerate() {
            let a = (b / 13) as u8;
            let c = (b % 13) as u8;
            let hole = if a == c {
                [make_card(a, 0), make_card(a, 1)] // pair
            } else if a > c {
                [make_card(a, 0), make_card(c, 0)] // suited hi,lo
            } else {
                [make_card(c, 0), make_card(a, 1)] // offsuit hi=c, lo=a
            };
            let mut rng = SmallRng::seed_from_u64(0x0C45_0000 ^ b as u64);
            *e = equity_vs_random(hole, &[], 800, &mut rng) as f32;
        }
        // Rank buckets by equity, assign to OCHS_CLUSTERS equal-count tiers.
        let mut order: Vec<usize> = (0..169).collect();
        order.sort_by(|&i, &j| eq[i].partial_cmp(&eq[j]).unwrap());
        let mut tiers = [0u8; 169];
        for (rank, &b) in order.iter().enumerate() {
            tiers[b] = (rank * OCHS_CLUSTERS / 169).min(OCHS_CLUSTERS - 1) as u8;
        }
        tiers
    })
}

/// Combined potential-aware feature: the QUANTILES equity-distribution
/// quantiles (already potential-aware over runouts) concatenated with an
/// OCHS_CLUSTERS-vector of the hand's equity against each opponent strength
/// tier. The OCHS part makes the feature opponent-relative: hands that beat
/// weak ranges but lose to strong ones separate from uniformly-mediocre
/// hands, which the equity-vs-uniform quantiles alone cannot express.
pub fn combined_features(
    hole: [Card; 2],
    board: &[Card],
    runouts: u32,
    rollouts: u32,
    rng: &mut SmallRng,
) -> [f32; COMBINED_DIM] {
    use crate::eval::eval_hole_board;
    use rand::Rng;
    debug_assert!(!board.is_empty() && board.len() < 5);

    let q = equity_quantiles(hole, board, runouts, rollouts, rng);

    let mut used = [false; 52];
    used[hole[0] as usize] = true;
    used[hole[1] as usize] = true;
    for &c in board {
        used[c as usize] = true;
    }
    let mut unseen = [0u8; 52];
    let mut unseen_len = 0;
    for c in 0..52u8 {
        if !used[c as usize] {
            unseen[unseen_len] = c;
            unseen_len += 1;
        }
    }
    let need = 5 - board.len();
    let mut full = [0u8; 5];
    full[..board.len()].copy_from_slice(board);
    let tiers = preflop_tiers();

    // One MC pass; each opponent hand is credited to its preflop tier.
    let mut sum = [0f64; OCHS_CLUSTERS];
    let mut cnt = [0u32; OCHS_CLUSTERS];
    let samples = runouts * rollouts;
    for _ in 0..samples {
        // Draw an opponent hand + the remaining board from `unseen`.
        for k in 0..(need + 2) {
            let j = rng.random_range(k..unseen_len);
            unseen.swap(k, j);
        }
        let opp = [unseen[0], unseen[1]];
        for i in 0..need {
            full[board.len() + i] = unseen[2 + i];
        }
        let tier = tiers[preflop_bucket(opp) as usize] as usize;
        let ours = eval_hole_board(&hole, &full);
        let theirs = eval_hole_board(&opp, &full);
        let s = if ours > theirs {
            1.0
        } else if ours == theirs {
            0.5
        } else {
            0.0
        };
        sum[tier] += s;
        cnt[tier] += 1;
    }

    let mut out = [0f32; COMBINED_DIM];
    out[..QUANTILES].copy_from_slice(&q);
    for t in 0..OCHS_CLUSTERS {
        out[QUANTILES + t] = if cnt[t] > 0 {
            (sum[t] / cnt[t] as f64) as f32
        } else {
            0.5
        };
    }
    out
}

fn train_street_combined(
    cfg: &AbsConfig,
    board_len: usize,
    samples: usize,
    seed: u64,
) -> Vec<Vec<f32>> {
    let points: Vec<Vec<f32>> = (0..samples)
        .into_par_iter()
        .map(|i| {
            let mut rng =
                SmallRng::seed_from_u64(seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut deck = fresh_deck();
            deck.shuffle(&mut rng);
            let hole = [deck[0], deck[1]];
            let board = deck[2..2 + board_len].to_vec();
            combined_features(hole, &board, cfg.dist_runouts, cfg.runout_rollouts, &mut rng).to_vec()
        })
        .collect();
    kmedians(&points, cfg.postflop_buckets as usize, 25, seed)
}

/// EMD between two 1-D equity distributions in quantile form = mean L1
/// distance between the quantile vectors.
pub(crate) fn l1(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .sum()
}

fn nearest_centroid(q: &[f32], cents: &[Vec<f32>]) -> u16 {
    let mut best = 0u16;
    let mut best_d = f64::MAX;
    for (i, c) in cents.iter().enumerate() {
        let d = l1(q, c);
        if d < best_d {
            best_d = d;
            best = i as u16;
        }
    }
    best
}

/// K-medians clustering under L1 (= EMD on quantile vectors): k-means++-style
/// seeding, Lloyd iterations with per-coordinate median centroid updates.
/// Returned centroids are sorted by mean so bucket index roughly orders by
/// hand strength. Deterministic for a given seed.
pub(crate) fn kmedians(points: &[Vec<f32>], k: usize, iters: usize, seed: u64) -> Vec<Vec<f32>> {
    use rand::Rng;
    assert!(!points.is_empty() && k >= 1);
    let mut rng = SmallRng::seed_from_u64(seed);
    let dim = points[0].len();

    // Seeding: first centroid random, then proportional to distance.
    let mut cents: Vec<Vec<f32>> = vec![points[rng.random_range(0..points.len())].clone()];
    let mut dists: Vec<f64> = points.iter().map(|p| l1(p, &cents[0])).collect();
    while cents.len() < k.min(points.len()) {
        let total: f64 = dists.iter().sum();
        let next = if total <= 0.0 {
            rng.random_range(0..points.len())
        } else {
            let mut r = rng.random::<f64>() * total;
            let mut pick = points.len() - 1;
            for (i, &d) in dists.iter().enumerate() {
                r -= d;
                if r <= 0.0 {
                    pick = i;
                    break;
                }
            }
            pick
        };
        cents.push(points[next].clone());
        let c = cents.last().unwrap().clone();
        for (d, p) in dists.iter_mut().zip(points) {
            *d = d.min(l1(p, &c));
        }
    }

    let mut assign: Vec<usize> = vec![0; points.len()];
    for _ in 0..iters {
        let new_assign: Vec<usize> = points
            .par_iter()
            .map(|p| {
                let mut best = 0;
                let mut best_d = f64::MAX;
                for (i, c) in cents.iter().enumerate() {
                    let d = l1(p, c);
                    if d < best_d {
                        best_d = d;
                        best = i;
                    }
                }
                best
            })
            .collect();
        let changed = new_assign != assign;
        assign = new_assign;

        for (ci, cent) in cents.iter_mut().enumerate() {
            let members: Vec<&Vec<f32>> = points
                .iter()
                .zip(&assign)
                .filter(|(_, &a)| a == ci)
                .map(|(p, _)| p)
                .collect();
            if members.is_empty() {
                *cent = points[rng.random_range(0..points.len())].clone();
                continue;
            }
            for d in 0..dim {
                let mut vals: Vec<f32> = members.iter().map(|m| m[d]).collect();
                vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                cent[d] = vals[vals.len() / 2];
            }
        }
        if !changed {
            break;
        }
    }

    cents.sort_by(|a, b| {
        a.iter()
            .sum::<f32>()
            .partial_cmp(&b.iter().sum::<f32>())
            .unwrap()
    });
    cents
}

// ---------------------------------------------------------------------------
// Tests (written first, TDD)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::{fresh_deck, make_card, parse_cards};
    use crate::engine::{HandConfig, PlayerAction};
    use rand::SeedableRng;
    use std::collections::HashSet;

    fn abs() -> Abstraction {
        Abstraction::new(AbsConfig::default())
    }

    fn h6() -> Hand {
        Hand::new(&HandConfig::default(), 0, fresh_deck())
    }

    #[test]
    fn preflop_buckets_are_canonical_169() {
        let mut set = HashSet::new();
        for a in 0..52u8 {
            for b in 0..52u8 {
                if a != b {
                    set.insert(preflop_bucket([a, b]));
                }
            }
        }
        assert_eq!(set.len(), 169);
        // AA same bucket regardless of suits
        let aa1 = preflop_bucket([make_card(12, 0), make_card(12, 1)]);
        let aa2 = preflop_bucket([make_card(12, 2), make_card(12, 3)]);
        assert_eq!(aa1, aa2);
        // AKs != AKo, order-independent
        let aks = preflop_bucket([make_card(12, 0), make_card(11, 0)]);
        let ako = preflop_bucket([make_card(12, 0), make_card(11, 1)]);
        let aks_rev = preflop_bucket([make_card(11, 0), make_card(12, 0)]);
        assert_ne!(aks, ako);
        assert_eq!(aks, aks_rev);
    }

    #[test]
    fn utg_actions_have_fold_call_bet_allin_no_dupes() {
        let a = abs();
        let h = h6();
        let acts = a.abstract_actions(&h);
        assert!(acts.contains(&AbsAction::Fold));
        assert!(acts.contains(&AbsAction::CheckCall));
        assert!(acts.contains(&AbsAction::AllIn));
        assert!(acts.iter().any(|x| matches!(x, AbsAction::Bet(_))));
        // Concrete raise amounts must be unique and legal.
        let mut amounts = HashSet::new();
        for &x in &acts {
            if let PlayerAction::RaiseTo(t) = a.concrete(&h, x) {
                assert!(amounts.insert(t), "duplicate concrete raise {}", t);
                let (lo, hi) = h.raise_bounds().unwrap();
                assert!(t >= lo && t <= hi);
            }
        }
    }

    #[test]
    fn no_fold_when_check_is_free() {
        let a = abs();
        let mut h = h6();
        for _ in 0..5 {
            h.apply(PlayerAction::CheckCall); // limps + SB completes
        }
        assert!(h.can_check()); // BB option
        let acts = a.abstract_actions(&h);
        assert!(!acts.contains(&AbsAction::Fold));
        assert!(acts.contains(&AbsAction::CheckCall));
    }

    /// Modern action abstraction: >=4 open sizes preflop, >=7 first-in
    /// sizes postflop spanning a small bet through overbets on flop AND
    /// turn/river.
    #[test]
    fn pluribus_menus_are_coarse_postflop_and_fine_preflop_is_finer() {
        let mk = |menu| {
            Abstraction::new(AbsConfig {
                menu,
                cache_cap: 1000,
                ..AbsConfig::default()
            })
        };
        let wide = mk(MenuShape::Wide);
        let plu = mk(MenuShape::Pluribus);
        let fine = mk(MenuShape::PluribusFinePre);
        let bets = |a: &Abstraction, h: &Hand| {
            a.abstract_actions(h)
                .iter()
                .filter(|x| matches!(x, AbsAction::Bet(_)))
                .count()
        };
        let cfg = HandConfig::default();
        // Flop first-in: UTG open-limps around to a checked flop.
        let mut h = Hand::new(&cfg, 0, crate::cards::fresh_deck());
        for _ in 0..cfg.num_players {
            h.apply(PlayerAction::CheckCall);
        }
        assert_eq!(h.street(), Street::Flop);
        assert_eq!(bets(&plu, &h), 2, "Pluribus flop first-in: 50% and 100%");
        assert_eq!(bets(&fine, &h), 2);
        assert!(bets(&wide, &h) >= 5);
        // Facing a flop bet: one raise size under Pluribus, none after that.
        let mut h2 = h.clone();
        h2.apply(plu.concrete(&h2, AbsAction::Bet(4)));
        assert_eq!(bets(&plu, &h2), 1, "Pluribus raise: pot only");
        let mut h3 = h2.clone();
        h3.apply(plu.concrete(&h3, AbsAction::Bet(4)));
        assert_eq!(bets(&plu, &h3), 0, "Pluribus re-raise: call/fold/all-in");
        // Preflop open: fine menu strictly more sizes than the shared one.
        let root = Hand::new(&cfg, 0, crate::cards::fresh_deck());
        assert!(bets(&fine, &root) > bets(&plu, &root));
        assert_eq!(bets(&plu, &root), bets(&wide, &root));
    }

    #[test]
    fn menus_are_wide() {
        use crate::engine::Street;
        let a = abs();
        let n_bets = |acts: &[AbsAction]| {
            acts.iter()
                .filter(|x| matches!(x, AbsAction::Bet(_)))
                .count()
        };
        // A first-in menu must span a small (<=40% pot) bet and an overbet
        // (> pot), neither being the all-in.
        let spans_small_to_overbet = |h: &Hand, acts: &[AbsAction]| {
            let pot = h.pot() as f64;
            let mut small = false;
            let mut over = false;
            for &x in acts {
                if matches!(x, AbsAction::AllIn) {
                    continue;
                }
                if let PlayerAction::RaiseTo(t) = a.concrete(h, x) {
                    let added = t as f64 - h.current_bet() as f64;
                    if added <= 0.40 * pot {
                        small = true;
                    }
                    if t as f64 > pot {
                        over = true;
                    }
                }
            }
            (small, over)
        };

        let h = h6();
        assert!(
            n_bets(&a.abstract_actions(&h)) >= 4,
            "preflop open needs >=4 sizes"
        );

        let mut h = h6();
        for _ in 0..6 {
            h.apply(PlayerAction::CheckCall);
        }
        assert_eq!(h.street(), Street::Flop);
        let acts = a.abstract_actions(&h);
        assert!(n_bets(&acts) >= 7, "flop first-in needs >=7 sizes");
        let (small, over) = spans_small_to_overbet(&h, &acts);
        assert!(small, "flop menu must include a small (<=40% pot) bet");
        assert!(over, "flop menu must include an overbet");

        for _ in 0..6 {
            h.apply(PlayerAction::CheckCall);
        }
        assert_eq!(h.street(), Street::Turn);
        let acts = a.abstract_actions(&h);
        assert!(n_bets(&acts) >= 6, "turn first-in needs >=6 sizes");
        let (small, over) = spans_small_to_overbet(&h, &acts);
        assert!(small, "turn menu must include a small bet");
        assert!(over, "turn menu must include an overbet");
    }

    #[test]
    fn pot_bet_math() {
        let a = abs();
        let h = h6();
        // UTG: pot 150, to_call 100 -> pot-after-call 250 -> raise to 100+250=350.
        assert_eq!(
            a.concrete(&h, AbsAction::Bet(4)),
            PlayerAction::RaiseTo(350)
        );
        // Half-pot open: 100 + 125 = 225.
        assert_eq!(
            a.concrete(&h, AbsAction::Bet(2)),
            PlayerAction::RaiseTo(225)
        );
        assert_eq!(a.concrete(&h, AbsAction::AllIn), PlayerAction::RaiseTo(10_000));
        assert_eq!(a.concrete(&h, AbsAction::CheckCall), PlayerAction::CheckCall);
    }

    #[test]
    fn map_raise_picks_nearest_in_log_space() {
        let a = abs();
        let h = h6();
        // Abstract raises available: 182, 225, 287, 350 and 10_000 (all-in).
        assert_eq!(a.map_raise(&h, 360), AbsAction::Bet(4));
        // 200 is nearer 182 (33% open) than 225 (50%) in log space.
        assert_eq!(a.map_raise(&h, 200), AbsAction::Bet(1));
        assert_eq!(a.map_raise(&h, 9_000), AbsAction::AllIn);
    }

    #[test]
    fn preflop_tiers_rank_premiums_top_and_trash_bottom() {
        let t = preflop_tiers();
        let aa = preflop_bucket([make_card(12, 0), make_card(12, 1)]); // pocket aces
        let kk = preflop_bucket([make_card(11, 0), make_card(11, 1)]);
        let trash = preflop_bucket([make_card(5, 0), make_card(0, 1)]); // 72o
        assert_eq!(t[aa as usize], (OCHS_CLUSTERS - 1) as u8, "AA top tier");
        assert!(t[kk as usize] >= (OCHS_CLUSTERS - 2) as u8, "KK near top");
        assert_eq!(t[trash as usize], 0, "72o bottom tier");
    }

    #[test]
    fn combined_feature_shape_and_opponent_relativity() {
        let mut rng = SmallRng::seed_from_u64(7);
        // Flop where our hand crushes: top set on a dry board.
        let board = parse_cards("As Kd 7c").unwrap();
        let nuts = parse_cards("Ah Ac").unwrap(); // set of aces
        let air = parse_cards("5h 4d").unwrap(); // no pair, no draw
        let fn_ = combined_features([nuts[0], nuts[1]], &board, 24, 40, &mut rng);
        let fa = combined_features([air[0], air[1]], &board, 24, 40, &mut rng);
        // Quantile block is sorted (a distribution's order statistics).
        for w in fn_[..QUANTILES].windows(2) {
            assert!(w[1] >= w[0] - 1e-6, "quantile block must be sorted");
        }
        // OCHS block: the set beats every opponent tier more often than air.
        for t in 0..OCHS_CLUSTERS {
            assert!(
                fn_[QUANTILES + t] >= fa[QUANTILES + t],
                "set must fare >= air vs tier {t}"
            );
        }
        // Against the strongest tier the set still wins most of the time;
        // air loses most of the time.
        assert!(fn_[QUANTILES + OCHS_CLUSTERS - 1] > 0.6);
        assert!(fa[QUANTILES + OCHS_CLUSTERS - 1] < 0.4);
    }

    #[test]
    fn combined_centroids_are_not_strategic_and_bucket_orders_strength() {
        let cfg = AbsConfig {
            postflop_buckets: 8,
            dist_runouts: 12,
            runout_rollouts: 20,
            ..AbsConfig::default()
        };
        let cents = Centroids::train_combined(&cfg, 400, 0xC0FFEE);
        assert!(!cents.is_strategic(), "combined family is self-contained");
        assert_eq!(cents.flop[0].len(), COMBINED_DIM);
        let a = Abstraction::with_centroids(cfg, Some(cents));
        let mut rng = SmallRng::seed_from_u64(1);
        let board = parse_cards("As Kd 7c").unwrap();
        let strong = parse_cards("Ah Ac").unwrap();
        let weak = parse_cards("5h 4d").unwrap();
        let bs = a.bucket([strong[0], strong[1]], &board, &mut rng);
        let bw = a.bucket([weak[0], weak[1]], &board, &mut rng);
        assert!(bs > bw, "set must bucket above air (got {bs} vs {bw})");
    }

    #[test]
    fn bucket_orders_nuts_above_air() {
        let a = abs();
        let mut rng = SmallRng::seed_from_u64(3);
        let board = parse_cards("As Ks Qs Js 9h").unwrap();
        let nuts = parse_cards("Ts 3s").unwrap(); // royal flush
        let air = parse_cards("3d 2c").unwrap();
        let bn = a.bucket([nuts[0], nuts[1]], &board, &mut rng);
        let ba = a.bucket([air[0], air[1]], &board, &mut rng);
        assert_eq!(bn, a.cfg.postflop_buckets - 1, "nuts must be top bucket");
        assert!(bn > ba);
        // Memoized: the same query (either card order) returns the same
        // bucket. River queries live in the per-board table, not the
        // pair cache.
        let bn2 = a.bucket([nuts[1], nuts[0]], &board, &mut rng);
        assert_eq!(bn, bn2);
        assert!(a.river_tables_len() >= 1);
    }

    #[test]
    fn preflop_bucket_used_when_board_empty() {
        let a = abs();
        let mut rng = SmallRng::seed_from_u64(3);
        let hole = [make_card(12, 0), make_card(12, 1)];
        assert_eq!(a.bucket(hole, &[], &mut rng), preflop_bucket(hole));
    }

    /// Suit-isomorphic (hole, board) pairs must canonicalize to the same key,
    /// so the expensive equity work is done once per equivalence class.
    #[test]
    fn suit_isomorphism_canonical_keys() {
        let key = |hole: &str, board: &str| {
            let h = parse_cards(hole).unwrap();
            let b = parse_cards(board).unwrap();
            canonical_cards_key([h[0], h[1]], &b)
        };
        // Straight suit relabeling (h->s, c->d).
        assert_eq!(key("Kh Qh", "Ah 7h 2c"), key("Ks Qs", "As 7s 2d"));
        // Hole order must not matter.
        assert_eq!(key("Qh Kh", "Ah 7h 2c"), key("Kh Qh", "Ah 7h 2c"));
        // Mirror case that first-appearance canonicalization misses:
        // board is symmetric under h<->s, holes map onto each other.
        assert_eq!(key("8h 3c", "7h 7s 2d"), key("8s 3c", "7h 7s 2d"));
        // Non-isomorphic: different ranks or different suit structure.
        assert_ne!(key("Kh Qh", "Ah 7h 2c"), key("Kh Jh", "Ah 7h 2c"));
        assert_ne!(
            key("Kh Qh", "Ah 7h 2c"), // flush draw
            key("Kh Qs", "Ah 7h 2c"), // no flush draw
            "suitedness pattern must be preserved"
        );
        // Board length differentiates streets even with same cards prefix.
        assert_ne!(key("Kh Qh", "Ah 7h 2c"), key("Kh Qh", "Ah 7h 2c 2h"));
    }

    /// Two isomorphic hands must trigger only one canonical computation.
    #[test]
    fn isomorphic_hands_share_bucket_and_computation() {
        let a = abs();
        let mut rng = SmallRng::seed_from_u64(41);
        let b1 = {
            let h = parse_cards("Kh Qh").unwrap();
            let b = parse_cards("Ah 7h 2c").unwrap();
            a.bucket([h[0], h[1]], &b, &mut rng)
        };
        let b2 = {
            let h = parse_cards("Ks Qs").unwrap();
            let b = parse_cards("As 7s 2d").unwrap();
            a.bucket([h[0], h[1]], &b, &mut rng)
        };
        assert_eq!(b1, b2, "isomorphic hands must share a bucket");
        assert_eq!(a.canon_len(), 1, "one canonical entry for the class");
        assert_eq!(a.cache_len(), 2, "both exact keys memoized");
    }

    #[test]
    fn quantile_vector_sorted_and_bounded() {
        let mut rng = SmallRng::seed_from_u64(21);
        let board = parse_cards("Qs 7s 2d").unwrap();
        let hole = parse_cards("As 9s").unwrap();
        let q = equity_quantiles([hole[0], hole[1]], &board, 32, 50, &mut rng);
        for w in q.windows(2) {
            assert!(w[0] <= w[1], "quantiles must be nondecreasing: {:?}", q);
        }
        assert!(q[0] >= 0.0 && q[QUANTILES - 1] <= 1.0);
        assert!(q[QUANTILES - 1] > q[0], "distribution must have spread");
    }

    /// A flush draw's equity distribution is bimodal (hit or miss) while a
    /// made pair's is tight: the draw must show a much wider quantile spread.
    #[test]
    fn draw_distribution_wider_than_made_hand() {
        let mut rng = SmallRng::seed_from_u64(22);
        let board = parse_cards("Qs 7s 2d").unwrap();
        let draw = parse_cards("As 9s").unwrap(); // nut flush draw
        let pair = parse_cards("Qd 8h").unwrap(); // top pair
        let qd = equity_quantiles([draw[0], draw[1]], &board, 48, 100, &mut rng);
        let qp = equity_quantiles([pair[0], pair[1]], &board, 48, 100, &mut rng);
        let spread = |q: &[f32; QUANTILES]| q[QUANTILES - 1] - q[0];
        assert!(
            spread(&qd) > spread(&qp) + 0.1,
            "draw spread {:.3} must exceed pair spread {:.3}",
            spread(&qd),
            spread(&qp)
        );
    }

    #[test]
    fn kmedians_recovers_separated_clusters() {
        let mut rng = SmallRng::seed_from_u64(23);
        use rand::Rng;
        let centers: [[f32; QUANTILES]; 3] = [[0.1; QUANTILES], [0.5; QUANTILES], [0.9; QUANTILES]];
        let mut points = Vec::new();
        for _ in 0..300 {
            let c = centers[rng.random_range(0..3)];
            points.push(
                c.iter()
                    .map(|&x| x + rng.random_range(-0.03..0.03))
                    .collect::<Vec<f32>>(),
            );
        }
        let cents = kmedians(&points, 3, 30, 7);
        assert_eq!(cents.len(), 3);
        // Every true center must have a learned centroid within L1 0.05*Q.
        for c in &centers {
            let best = cents
                .iter()
                .map(|k| l1(k, c))
                .fold(f64::MAX, f64::min);
            assert!(best < 0.05 * QUANTILES as f64, "centroid too far: {}", best);
        }
        // Centroids come out sorted by mean (bucket index ~ strength order).
        let means: Vec<f32> = cents.iter().map(|c| c.iter().sum::<f32>()).collect();
        assert!(means.windows(2).all(|w| w[0] <= w[1]));
    }

    /// With trained centroids, hands with clearly different distributions land
    /// in different buckets, and bucketing is stable via the cache.
    #[test]
    /// The exact river table must reproduce naive per-hole exact equity
    /// (direct loop over all compatible opponent combos) bucket-for-bucket.
    #[test]
    fn river_table_matches_naive_exact_equity() {
        use crate::eval::eval_hole_board;
        let abs = Abstraction::new(AbsConfig {
            postflop_buckets: 12,
            ..AbsConfig::default()
        });
        let board = parse_cards("Qs 7h 2d 9c Kd").unwrap();
        let board5 = [board[0], board[1], board[2], board[3], board[4]];
        let mut rng = rand::rngs::SmallRng::seed_from_u64(1);
        for hole_s in ["As Ad", "Kh Qc", "7c 7s", "6h 5h", "3c 2c", "Ah 4s"] {
            let h = parse_cards(hole_s).unwrap();
            let hole = [h[0], h[1]];
            let hero = eval_hole_board(&hole, &board5);
            let mut used = [false; 52];
            for &c in board.iter().chain(hole.iter()) {
                used[c as usize] = true;
            }
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for a in 0..52u8 {
                for b in (a + 1)..52 {
                    if used[a as usize] || used[b as usize] {
                        continue;
                    }
                    let v = eval_hole_board(&[a, b], &board5);
                    num += if hero > v {
                        1.0
                    } else if hero == v {
                        0.5
                    } else {
                        0.0
                    };
                    den += 1.0;
                }
            }
            let expect = (((num / den) * 12.0) as u16).min(11);
            let got = abs.bucket(hole, &board, &mut rng);
            assert_eq!(got, expect, "bucket mismatch for {hole_s}");
        }
    }

    /// Suit-isomorphic (hole, board) pairs must land in the same river
    /// bucket through the per-board canonical table.
    #[test]
    fn river_buckets_are_suit_isomorphic() {
        let abs = Abstraction::new(AbsConfig {
            postflop_buckets: 12,
            ..AbsConfig::default()
        });
        let mut rng = rand::rngs::SmallRng::seed_from_u64(2);
        let board = parse_cards("Ah Kh 7h 4d 2c").unwrap();
        let hole = parse_cards("Qh Jh").unwrap();
        let b0 = abs.bucket([hole[0], hole[1]], &board, &mut rng);
        // Swap hearts and spades, and clubs and diamonds, everywhere.
        let swap = |c: Card| {
            let (r, s) = (c & !3, c & 3);
            r | match s {
                0 => 1,
                1 => 0,
                2 => 3,
                _ => 2,
            }
        };
        let board2: Vec<Card> = board.iter().map(|&c| swap(c)).collect();
        let hole2 = [swap(hole[0]), swap(hole[1])];
        let b1 = abs.bucket(hole2, &board2, &mut rng);
        assert_eq!(b0, b1, "suit relabeling must not change the bucket");
        // Sanity: a made flush is a top bucket, stone air is a low one.
        assert_eq!(b0, 11, "nut flush must be in the top bucket");
        let air = parse_cards("9c 3d").unwrap();
        let ba = abs.bucket([air[0], air[1]], &board, &mut rng);
        assert!(ba <= 4, "nine-high air must be in a low bucket, got {ba}");
    }

    #[test]
    fn kmeans_buckets_separate_monster_from_air() {
        let cfg = AbsConfig {
            postflop_buckets: 8,
            dist_runouts: 24,
            runout_rollouts: 50,
            ..AbsConfig::default()
        };
        let cents = Centroids::train(&cfg, 2_000, 42);
        assert_eq!(cents.flop.len(), 8);
        assert_eq!(cents.turn.len(), 8);
        let a = Abstraction::with_centroids(cfg, Some(cents));
        let mut rng = SmallRng::seed_from_u64(24);
        let board = parse_cards("Ah 7h 2c").unwrap();
        let set = parse_cards("As Ad").unwrap(); // top set
        let air = parse_cards("9c 4d").unwrap();
        let bs = a.bucket([set[0], set[1]], &board, &mut rng);
        let ba = a.bucket([air[0], air[1]], &board, &mut rng);
        assert!(bs > ba, "top set bucket {} must beat air bucket {}", bs, ba);
        // Cache stability.
        assert_eq!(bs, a.bucket([set[1], set[0]], &board, &mut rng));
    }

    fn small_strat_ctx() -> StratCtx {
        let abs = Arc::new(Abstraction::new(AbsConfig {
            postflop_buckets: 6,
            equity_rollouts: 40,
            dist_runouts: 8,
            runout_rollouts: 20,
            cache_cap: 500_000,
            menu: crate::abstraction::MenuShape::Wide,
        }));
        let cfg = crate::cfr::TrainConfig {
            hand: HandConfig {
                num_players: 2,
                ..HandConfig::default()
            },
            prune_after: u64::MAX,
            ..crate::cfr::TrainConfig::default()
        };
        let trainer = crate::cfr::Trainer::new(abs.clone(), cfg);
        trainer.run(3_000, &|_| {});
        StratCtx {
            bp: Arc::new(trainer.to_blueprint()),
            abs,
            rollouts: 16,
        }
    }

    /// Strategic fingerprints: fixed dimension per street (never QUANTILES,
    /// so dimension-sniffing works), rollout value features order a monster
    /// above air, and the whole vector is suit-relabeling stable enough to
    /// share cache entries.
    #[test]
    fn strategic_fingerprint_shape_and_separation() {
        let sc = small_strat_ctx();
        let mut rng = SmallRng::seed_from_u64(9);
        let board = parse_cards("Ah 7h 2c").unwrap();
        let set = parse_cards("As Ad").unwrap();
        let air = parse_cards("9c 4d").unwrap();
        let f_set = strategic_fingerprint(&sc, [set[0], set[1]], &board, &mut rng);
        let f_air = strategic_fingerprint(&sc, [air[0], air[1]], &board, &mut rng);
        assert_eq!(f_set.len(), f_air.len());
        assert_ne!(f_set.len(), QUANTILES, "dimension must not collide with equity features");
        // Value features live in the last three slots: [mean, std, winfrac].
        let n = f_set.len();
        assert!(
            f_set[n - 3] > f_air[n - 3] && f_set[n - 1] > f_air[n - 1],
            "top set must out-earn air in blueprint rollouts: set {:?} air {:?}",
            &f_set[n - 3..],
            &f_air[n - 3..]
        );
        // Action-probability prefix is a distribution.
        let p: f32 = f_set[..n - 3].iter().sum();
        assert!((p - 1.0).abs() < 1e-4);
    }

    /// End to end: strategic centroids train, are recognized as strategic,
    /// and an abstraction using them buckets hands without panicking while
    /// separating monsters from air.
    #[test]
    fn strategic_centroids_bucket_hands() {
        let sc = small_strat_ctx();
        let cfg = AbsConfig {
            postflop_buckets: 6,
            cache_cap: 500_000,
            ..AbsConfig::default()
        };
        let cents = Centroids::train_strategic(&cfg, 300, 7, &sc);
        assert!(cents.is_strategic());
        assert_eq!(cents.flop.len(), 6);
        let a = Abstraction::with_centroids(cfg, Some(cents)).with_strat(sc);
        let mut rng = SmallRng::seed_from_u64(31);
        let board = parse_cards("Ah 7h 2c").unwrap();
        let set = parse_cards("As Ad").unwrap();
        let air = parse_cards("9c 4d").unwrap();
        let bs = a.bucket([set[0], set[1]], &board, &mut rng);
        let ba = a.bucket([air[0], air[1]], &board, &mut rng);
        assert!(bs < 6 && ba < 6);
        assert_ne!(bs, ba, "monster and air should land in different strategic buckets");
        // Cache stability across the suit-symmetric lookup.
        assert_eq!(bs, a.bucket([set[1], set[0]], &board, &mut rng));
    }

    /// Every non-terminal state reached by random abstract play must offer at
    /// least one abstract action whose concrete form is legal.
    #[test]
    fn fuzz_abstract_playouts_always_have_actions() {
        use rand::seq::{IndexedRandom, SliceRandom};
        use rand::Rng;
        let a = abs();
        let mut rng = SmallRng::seed_from_u64(11);
        for n in 2..=6usize {
            for _ in 0..500 {
                let mut deck = fresh_deck();
                deck.shuffle(&mut rng);
                let cfg = HandConfig {
                    num_players: n,
                    ..HandConfig::default()
                };
                let mut h = Hand::new(&cfg, rng.random_range(0..n), deck);
                let mut steps = 0;
                while !h.is_terminal() {
                    steps += 1;
                    assert!(steps < 400);
                    let acts = a.abstract_actions(&h);
                    assert!(!acts.is_empty());
                    let &pick = acts.choose(&mut rng).unwrap();
                    h.apply(a.concrete(&h, pick));
                }
                let u = h.utilities();
                assert_eq!(u[..n].iter().sum::<i64>(), 0);
            }
        }
    }
}
