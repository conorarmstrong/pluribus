//! Depth-limited flop resolving with a learned belief-state value function
//! at the leaves — the ReBeL search architecture.
//!
//! Vector-form CFR+ over the flop betting tree for both players' full
//! 1326-combo ranges. When flop betting closes, the subgame is truncated:
//! an end-of-flop leaf is a turn street-start public state, valued by
//! querying a `LeafEval` (the trained value network, or an exact oracle in
//! tests) once per candidate turn card (weight 1/45: every combo pair
//! blocks exactly 4 of the 49 candidates). Leaf queries are refreshed every
//! `refresh` iterations as the leaf ranges evolve, and cached in between.
//!
//! Fold terminals are exact; all-in-on-the-flop showdowns are evaluated on
//! a fixed sample of full runouts with the shared O(N) sweep.

use crate::abstraction::{AbsAction, Abstraction};
use crate::cards::Card;
use crate::engine::{Hand, Street};
use crate::eval::eval_hole_board;
use crate::river::showdown_sweep;
use crate::search::{all_combos, combo_index, NUM_COMBOS};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// Per-combo forward values (chips from this point on) for both players of
/// a turn street-start public state — `[first-to-act, other]` order.
pub trait LeafEval: Sync {
    fn values(
        &self,
        board: &[Card],
        pot: u32,
        stack: [u32; 2],
        range: [&[f64]; 2],
    ) -> [Vec<f64>; 2];
}

impl LeafEval for crate::valuenet::ValueNet {
    fn values(
        &self,
        board: &[Card],
        pot: u32,
        stack: [u32; 2],
        range: [&[f64]; 2],
    ) -> [Vec<f64>; 2] {
        crate::valuenet::ValueNet::values(self, board, pot, stack, range)
    }
}

enum Kid {
    Node(usize),
    End(usize),
    Leaf(usize),
}

struct Node {
    player: usize,
    acts: Vec<AbsAction>,
    kids: Vec<Kid>,
    regret: Vec<f64>,
    /// Allocated only at the root.
    strat: Vec<f64>,
}

enum Terminal {
    Fold {
        util: [f64; 2],
    },
    /// All-in on the flop: showdown over `RUNOUT_SAMPLES` fixed runouts.
    Runout {
        matched: f64,
        dead: f64,
        /// (turn, river, rank, sorted) per sampled runout.
        tables: Vec<RunoutTable>,
    },
}

struct RunoutTable {
    cards: [Card; 2],
    rank: Vec<u32>,
    sorted: Vec<u32>,
}

const RUNOUT_SAMPLES: usize = 24;

/// End-of-flop leaf: a turn street-start public state, minus the turn card
/// (queried once per candidate).
struct Leaf {
    pot: u32,
    /// Stacks in the leaf evaluator's order ([first-to-act on turn, other]).
    stack_net: [u32; 2],
    /// Commitment per SOLVER player (subtracted from forward values).
    commit: [f64; 2],
    /// Solver player 0 is the leaf evaluator's player 1.
    swap: bool,
    /// Cached evaluator output per turn-card index: [solver player][combo],
    /// forward chips. Refreshed every `refresh` iterations.
    cached: Vec<[Vec<f64>; 2]>,
}

pub struct FlopSolver<'a> {
    combos: Vec<[Card; 2]>,
    range: [Vec<f64>; 2],
    /// Candidate turn cards (board conflicts excluded).
    turns: Vec<Card>,
    /// Indices into `turns` actually queried at leaves — either all of them
    /// or a fixed random subsample (`query_turns.len() < turns.len()`) to
    /// keep leaf-refresh network cost within real-time budgets.
    query_turns: Vec<usize>,
    flop: [Card; 3],
    nodes: Vec<Node>,
    terminals: Vec<Terminal>,
    leaves: Vec<Leaf>,
    eval: &'a dyn LeafEval,
    /// Re-query the leaf evaluator every this many iterations.
    refresh: u64,
    root_vals: [Vec<f64>; 2],
    weight_sum: f64,
    iters: u64,
}

impl<'a> FlopSolver<'a> {
    /// Build the depth-limited flop tree from a decision point `h` (flop
    /// street, exactly two live players; `h.to_act()` becomes solver player
    /// 0). `keep` restricts bet sizes as in the turn solver.
    pub fn build(
        h: &Hand,
        abs: &Abstraction,
        ranges: [&[f64]; 2],
        keep: &[u8],
        eval: &'a dyn LeafEval,
        refresh: u64,
    ) -> Option<FlopSolver<'a>> {
        Self::build_sampled(h, abs, ranges, keep, eval, refresh, usize::MAX)
    }

    /// Like `build`, but leaf evaluation queries only `max_query_turns`
    /// randomly chosen (deterministic per spot) turn cards instead of all
    /// 49 — a sampled chance approximation trading a little accuracy for a
    /// large cut in network queries.
    #[allow(clippy::too_many_arguments)]
    pub fn build_sampled(
        h: &Hand,
        abs: &Abstraction,
        ranges: [&[f64]; 2],
        keep: &[u8],
        eval: &'a dyn LeafEval,
        refresh: u64,
        max_query_turns: usize,
    ) -> Option<FlopSolver<'a>> {
        if h.street() != Street::Flop || h.is_terminal() || h.live_count() != 2 {
            return None;
        }
        let hero = h.to_act();
        let villain = (0..h.num_players()).find(|&p| p != hero && !h.folded(p))?;
        let seats = [hero, villain];

        let combos = all_combos();
        let board: Vec<Card> = h.board().to_vec();
        debug_assert_eq!(board.len(), 3);
        let mut on_board = [false; 52];
        for &c in &board {
            on_board[c as usize] = true;
        }
        let valid = |c: &[Card; 2]| !on_board[c[0] as usize] && !on_board[c[1] as usize];

        let mut range = [vec![0.0; NUM_COMBOS], vec![0.0; NUM_COMBOS]];
        for (pi, r) in range.iter_mut().enumerate() {
            let src = ranges[pi];
            let mut total = 0.0;
            for (ci, combo) in combos.iter().enumerate() {
                if valid(combo) && src[ci] > 0.0 {
                    r[ci] = src[ci];
                    total += src[ci];
                }
            }
            if total <= 0.0 {
                for (ci, combo) in combos.iter().enumerate() {
                    r[ci] = if valid(combo) { 1.0 } else { 0.0 };
                }
            }
        }

        let turns: Vec<Card> = (0..52u8).filter(|&c| !on_board[c as usize]).collect();
        let query_turns: Vec<usize> = if max_query_turns >= turns.len() {
            (0..turns.len()).collect()
        } else {
            // Deterministic per spot: seed from the flop cards.
            let seed = board.iter().fold(0xF10Cu64, |a, &c| a * 53 + c as u64);
            let mut rng = SmallRng::seed_from_u64(seed);
            let mut idx: Vec<usize> = (0..turns.len()).collect();
            for k in 0..max_query_turns {
                let j = rng.random_range(k..idx.len());
                idx.swap(k, j);
            }
            idx.truncate(max_query_turns);
            idx
        };
        let mut solver = FlopSolver {
            combos,
            range,
            turns,
            query_turns,
            flop: [board[0], board[1], board[2]],
            nodes: Vec::new(),
            terminals: Vec::new(),
            leaves: Vec::new(),
            eval,
            refresh: refresh.max(1),
            root_vals: [vec![0.0; NUM_COMBOS], vec![0.0; NUM_COMBOS]],
            weight_sum: 0.0,
            iters: 0,
        };
        let root = solver.expand(h.clone(), abs, seats, keep);
        let Kid::Node(root_ix) = root else {
            return None;
        };
        let n = solver.nodes[root_ix].acts.len();
        solver.nodes[root_ix].strat = vec![0.0; NUM_COMBOS * n];
        Some(solver)
    }

    fn expand(&mut self, h: Hand, abs: &Abstraction, seats: [usize; 2], keep: &[u8]) -> Kid {
        if h.is_terminal() {
            if h.live_count() == 1 {
                let u = h.utilities();
                self.terminals.push(Terminal::Fold {
                    util: [u[seats[0]] as f64, u[seats[1]] as f64],
                });
                return Kid::End(self.terminals.len() - 1);
            }
            // All-in on the flop: fixed-sample runout showdown.
            let (c0, c1) = (
                h.hand_commit(seats[0]) as f64,
                h.hand_commit(seats[1]) as f64,
            );
            let matched = c0.min(c1);
            let dead = h.pot() as f64 - c0 - c1;
            let tables = self.runout_tables(self.terminals.len() as u64);
            self.terminals.push(Terminal::Runout {
                matched,
                dead,
                tables,
            });
            return Kid::End(self.terminals.len() - 1);
        }

        if h.street() == Street::Turn {
            // Flop betting closed: depth-limit leaf (turn street start).
            let first = h.to_act();
            let other = (0..h.num_players())
                .find(|&p| p != first && !h.folded(p))
                .expect("two live players");
            let swap = first != seats[0];
            self.leaves.push(Leaf {
                pot: h.pot(),
                stack_net: [h.stack(first), h.stack(other)],
                commit: [
                    h.hand_commit(seats[0]) as f64,
                    h.hand_commit(seats[1]) as f64,
                ],
                swap,
                cached: Vec::new(),
            });
            return Kid::Leaf(self.leaves.len() - 1);
        }

        let seat = h.to_act();
        let player = if seat == seats[0] { 0 } else { 1 };
        let acts: Vec<AbsAction> = abs
            .abstract_actions(&h)
            .into_iter()
            .filter(|a| match a {
                AbsAction::Bet(i) => keep.contains(i),
                _ => true,
            })
            .collect();
        let kids: Vec<Kid> = acts
            .iter()
            .map(|&a| {
                let mut child = h.clone();
                child.apply(abs.concrete(&h, a));
                self.expand(child, abs, seats, keep)
            })
            .collect();
        let n_acts = acts.len();
        self.nodes.push(Node {
            player,
            acts,
            kids,
            regret: vec![0.0; NUM_COMBOS * n_acts],
            strat: Vec::new(),
        });
        Kid::Node(self.nodes.len() - 1)
    }

    /// Deterministic fixed sample of full runouts for an all-in terminal.
    fn runout_tables(&self, salt: u64) -> Vec<RunoutTable> {
        let mut rng = SmallRng::seed_from_u64(0xF10B ^ salt);
        let mut tables = Vec::with_capacity(RUNOUT_SAMPLES);
        for _ in 0..RUNOUT_SAMPLES {
            let a = self.turns[rng.random_range(0..self.turns.len())];
            let mut b = a;
            while b == a {
                b = self.turns[rng.random_range(0..self.turns.len())];
            }
            let board5 = [self.flop[0], self.flop[1], self.flop[2], a, b];
            let rank: Vec<u32> = self
                .combos
                .iter()
                .map(|c| {
                    let blocked = c.contains(&a)
                        || c.contains(&b)
                        || self.flop.contains(&c[0])
                        || self.flop.contains(&c[1]);
                    if blocked {
                        0
                    } else {
                        eval_hole_board(c, &board5)
                    }
                })
                .collect();
            let mut sorted: Vec<u32> = (0..NUM_COMBOS as u32)
                .filter(|&ci| rank[ci as usize] > 0)
                .collect();
            sorted.sort_by_key(|&ci| rank[ci as usize]);
            tables.push(RunoutTable {
                cards: [a, b],
                rank,
                sorted,
            });
        }
        tables
    }

    fn root(&self) -> usize {
        self.nodes.len() - 1
    }

    #[allow(dead_code)] // diagnostics + tests
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    #[allow(dead_code)]
    pub fn leaf_count(&self) -> usize {
        self.leaves.len()
    }

    #[allow(dead_code)]
    pub fn iterations(&self) -> u64 {
        self.iters
    }

    pub fn solve(&mut self, max_iters: u64, time_ms: u64) {
        let start = std::time::Instant::now();
        while self.iters < max_iters && (start.elapsed().as_millis() as u64) < time_ms {
            self.iters += 1;
            let t = self.iters as f64;
            for u in 0..2 {
                let reach_own = self.range[u].clone();
                let reach_opp = self.range[1 - u].clone();
                let v = self.walk(Kid::Node(self.root()), u, &reach_own, &reach_opp, t);
                for (acc, &x) in self.root_vals[u].iter_mut().zip(&v) {
                    *acc += t * x;
                }
            }
            self.weight_sum += t;
        }
    }

    /// Metareasoning staged solve (see `RiverSolver::solve_adaptive`).
    pub fn solve_adaptive(
        &mut self,
        max_iters: u64,
        time_ms: u64,
        hole: [Card; 2],
        threshold: f64,
    ) -> bool {
        const PROBE_FRAC: u64 = 8;
        self.solve((max_iters / PROBE_FRAC).max(1), (time_ms / PROBE_FRAC).max(1));
        if let Some((_, s)) = self.root_strategy(hole) {
            if s.iter().cloned().fold(0.0, f64::max) >= threshold {
                return true;
            }
        }
        self.solve(max_iters, time_ms.saturating_sub(time_ms / PROBE_FRAC));
        false
    }

    pub fn root_strategy(&self, hole: [Card; 2]) -> Option<(Vec<AbsAction>, Vec<f64>)> {
        let node = &self.nodes[self.root()];
        debug_assert_eq!(node.player, 0);
        let n_acts = node.acts.len();
        let ci = combo_index(hole[0], hole[1]);
        let s = &node.strat[ci * n_acts..(ci + 1) * n_acts];
        let total: f64 = s.iter().sum();
        if total <= 0.0 {
            return None;
        }
        Some((node.acts.clone(), s.iter().map(|&x| x / total).collect()))
    }

    /// Per-combo expected chip value at the root, normalized by compatible
    /// opponent mass (as in the turn solver).
    #[allow(dead_code)] // used by tests and value bootstrapping
    pub fn root_values(&self, u: usize) -> Vec<f64> {
        let opp = &self.range[1 - u];
        let mut total = 0.0f64;
        let mut card = [0.0f64; 52];
        for (ci, combo) in self.combos.iter().enumerate() {
            let w = opp[ci];
            if w > 0.0 {
                total += w;
                card[combo[0] as usize] += w;
                card[combo[1] as usize] += w;
            }
        }
        (0..NUM_COMBOS)
            .map(|ci| {
                if self.weight_sum <= 0.0 {
                    return 0.0;
                }
                let c = self.combos[ci];
                let compat = total - card[c[0] as usize] - card[c[1] as usize] + opp[ci];
                if compat <= 0.0 {
                    return 0.0;
                }
                self.root_vals[u][ci] / self.weight_sum / compat
            })
            .collect()
    }

    fn walk(
        &mut self,
        at: Kid,
        u: usize,
        reach_own: &[f64],
        reach_opp: &[f64],
        t: f64,
    ) -> Vec<f64> {
        let ni = match at {
            Kid::End(ti) => return self.terminal_values(ti, u, reach_opp),
            Kid::Leaf(li) => {
                if self.leaves[li].cached.is_empty()
                    || (u == 0 && self.iters % self.refresh == 1)
                {
                    self.refresh_leaf(li, u, reach_own, reach_opp);
                }
                return self.leaf_values(li, u, reach_opp);
            }
            Kid::Node(ni) => ni,
        };
        let n_acts = self.nodes[ni].acts.len();
        let player = self.nodes[ni].player;
        let sigma = self.node_sigma(ni);

        if player == u {
            let mut vals = vec![0.0; NUM_COMBOS];
            let mut act_vals: Vec<Vec<f64>> = Vec::with_capacity(n_acts);
            for a in 0..n_acts {
                let mut own = vec![0.0; NUM_COMBOS];
                for ci in 0..NUM_COMBOS {
                    own[ci] = reach_own[ci] * sigma[ci * n_acts + a];
                }
                let kid = self.nodes[ni].kids[a].shallow();
                let v = self.walk(kid, u, &own, reach_opp, t);
                for ci in 0..NUM_COMBOS {
                    vals[ci] += sigma[ci * n_acts + a] * v[ci];
                }
                act_vals.push(v);
            }
            let node = &mut self.nodes[ni];
            let averaged = !node.strat.is_empty();
            for ci in 0..NUM_COMBOS {
                for (a, av) in act_vals.iter().enumerate() {
                    let idx = ci * n_acts + a;
                    node.regret[idx] = (node.regret[idx] + av[ci] - vals[ci]).max(0.0);
                    if averaged {
                        node.strat[idx] += t * reach_own[ci] * sigma[idx];
                    }
                }
            }
            vals
        } else {
            let mut vals = vec![0.0; NUM_COMBOS];
            for a in 0..n_acts {
                let mut opp = vec![0.0; NUM_COMBOS];
                for ci in 0..NUM_COMBOS {
                    opp[ci] = reach_opp[ci] * sigma[ci * n_acts + a];
                }
                let kid = self.nodes[ni].kids[a].shallow();
                let v = self.walk(kid, u, reach_own, &opp, t);
                for (o, &x) in vals.iter_mut().zip(&v) {
                    *o += x;
                }
            }
            vals
        }
    }

    /// Query the leaf evaluator for every candidate turn card with the
    /// current leaf ranges and cache the per-combo forward values.
    fn refresh_leaf(&mut self, li: usize, u: usize, reach_own: &[f64], reach_opp: &[f64]) {
        // Solver-player order.
        let (r0, r1) = if u == 0 {
            (reach_own, reach_opp)
        } else {
            (reach_opp, reach_own)
        };
        let leaf = &self.leaves[li];
        let (pot, stack_net, swap) = (leaf.pot, leaf.stack_net, leaf.swap);
        let flop = self.flop;
        let combos = &self.combos;
        let eval = self.eval;

        let cached: Vec<[Vec<f64>; 2]> = self
            .query_turns
            .par_iter()
            .map(|&ti| {
                let tc = self.turns[ti];
                let board = [flop[0], flop[1], flop[2], tc];
                // Mask combos containing the turn card out of both ranges.
                let mask = |r: &[f64]| -> Vec<f64> {
                    r.iter()
                        .enumerate()
                        .map(|(ci, &w)| {
                            let c = combos[ci];
                            if c[0] == tc || c[1] == tc {
                                0.0
                            } else {
                                w
                            }
                        })
                        .collect()
                };
                let (m0, m1) = (mask(r0), mask(r1));
                // Evaluator wants [first-to-act, other].
                if swap {
                    let [a, b] = eval.values(&board, pot, stack_net, [&m1, &m0]);
                    [b, a]
                } else {
                    eval.values(&board, pot, stack_net, [&m0, &m1])
                }
            })
            .collect();
        self.leaves[li].cached = cached;
    }

    /// Counterfactual leaf values for player `u`: chance over the queried
    /// turn cards, forward values minus own commitment, times compatible
    /// opponent mass. With the full turn set this is the exact 1/(T−4)
    /// chance weighting; with a subsample the per-combo average over its
    /// valid queried turns is rescaled by (T−2)/(T−4) — identical to the
    /// full formula when nothing is subsampled.
    fn leaf_values(&self, li: usize, u: usize, reach_opp: &[f64]) -> Vec<f64> {
        let leaf = &self.leaves[li];
        let t_all = self.turns.len() as f64;
        let mut vals = vec![0.0; NUM_COMBOS];
        let mut counts = vec![0u32; NUM_COMBOS];
        for (qi, &ti) in self.query_turns.iter().enumerate() {
            let tc = self.turns[ti];
            let vhat = &leaf.cached[qi][u];
            // Aggregates of reach_opp excluding combos blocked by the turn.
            let mut total = 0.0f64;
            let mut card = [0.0f64; 52];
            for (ci, combo) in self.combos.iter().enumerate() {
                let w = reach_opp[ci];
                if w > 0.0 && combo[0] != tc && combo[1] != tc {
                    total += w;
                    card[combo[0] as usize] += w;
                    card[combo[1] as usize] += w;
                }
            }
            for (ci, combo) in self.combos.iter().enumerate() {
                if combo[0] == tc || combo[1] == tc {
                    continue;
                }
                counts[ci] += 1;
                let compat =
                    total - card[combo[0] as usize] - card[combo[1] as usize] + reach_opp[ci];
                if compat > 0.0 {
                    vals[ci] += (vhat[ci] - leaf.commit[u]) * compat;
                }
            }
        }
        let scale = (t_all - 2.0) / (t_all - 4.0);
        for (v, &n) in vals.iter_mut().zip(&counts) {
            if n > 0 {
                *v *= scale / n as f64;
            }
        }
        vals
    }

    fn node_sigma(&self, ni: usize) -> Vec<f64> {
        let node = &self.nodes[ni];
        let n_acts = node.acts.len();
        let mut sigma = vec![0.0; NUM_COMBOS * n_acts];
        for ci in 0..NUM_COMBOS {
            let r = &node.regret[ci * n_acts..(ci + 1) * n_acts];
            let total: f64 = r.iter().sum();
            let s = &mut sigma[ci * n_acts..(ci + 1) * n_acts];
            if total > 0.0 {
                for (o, &x) in s.iter_mut().zip(r) {
                    *o = x / total;
                }
            } else {
                s.fill(1.0 / n_acts as f64);
            }
        }
        sigma
    }

    fn terminal_values(&self, ti: usize, u: usize, reach_opp: &[f64]) -> Vec<f64> {
        match &self.terminals[ti] {
            Terminal::Fold { util } => {
                let mut total = 0.0f64;
                let mut card = [0.0f64; 52];
                for (ci, combo) in self.combos.iter().enumerate() {
                    let w = reach_opp[ci];
                    if w > 0.0 {
                        total += w;
                        card[combo[0] as usize] += w;
                        card[combo[1] as usize] += w;
                    }
                }
                let mut vals = vec![0.0; NUM_COMBOS];
                for (ci, combo) in self.combos.iter().enumerate() {
                    let excl =
                        total - card[combo[0] as usize] - card[combo[1] as usize] + reach_opp[ci];
                    if excl != 0.0 {
                        vals[ci] = util[u] * excl;
                    }
                }
                vals
            }
            Terminal::Runout {
                matched,
                dead,
                tables,
            } => {
                let mut vals = vec![0.0; NUM_COMBOS];
                let mut counts = vec![0u32; NUM_COMBOS];
                for tab in tables {
                    // Mask combos blocked by this runout.
                    let (a, b) = (tab.cards[0], tab.cards[1]);
                    let mut masked = reach_opp.to_vec();
                    let mut total = 0.0f64;
                    let mut card = [0.0f64; 52];
                    for (ci, combo) in self.combos.iter().enumerate() {
                        if combo[0] == a || combo[1] == a || combo[0] == b || combo[1] == b {
                            masked[ci] = 0.0;
                        } else if masked[ci] > 0.0 {
                            total += masked[ci];
                            card[combo[0] as usize] += masked[ci];
                            card[combo[1] as usize] += masked[ci];
                        }
                    }
                    let sv = showdown_sweep(
                        &self.combos,
                        &tab.sorted,
                        &tab.rank,
                        &masked,
                        total,
                        &card,
                        *matched,
                        *dead,
                    );
                    for (ci, combo) in self.combos.iter().enumerate() {
                        if combo[0] != a && combo[1] != a && combo[0] != b && combo[1] != b {
                            vals[ci] += sv[ci];
                            counts[ci] += 1;
                        }
                    }
                }
                // Per-runout sweeps only count opponent combos surviving
                // that runout: a hero-valid runout excludes the opponent's
                // combo with probability 1 − C(T−4,2)/C(T−2,2). Rescale by
                // that exact constant so values are per full compatible
                // opponent mass (mirrors the turn solver's 1/44 chance
                // weighting).
                let t = self.turns.len() as f64;
                let q = ((t - 4.0) * (t - 5.0)) / ((t - 2.0) * (t - 3.0));
                for (v, &n) in vals.iter_mut().zip(&counts) {
                    if n > 0 {
                        *v /= n as f64 * q;
                    }
                }
                vals
            }
        }
    }
}

impl Kid {
    fn shallow(&self) -> Kid {
        match self {
            Kid::Node(x) => Kid::Node(*x),
            Kid::End(x) => Kid::End(*x),
            Kid::Leaf(x) => Kid::Leaf(*x),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests (written first, TDD)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::AbsConfig;
    use crate::cards::{fresh_deck, parse_cards};
    use crate::engine::{HandConfig, PlayerAction};

    fn abs_small() -> Abstraction {
        Abstraction::new(AbsConfig {
            postflop_buckets: 6,
            equity_rollouts: 40,
            cache_cap: 100_000,
            ..AbsConfig::default()
        })
    }

    /// Exact-equity oracle leaf: forward value = equity vs the opponent
    /// range × pot (a "both check down from here" valuation) — crude but
    /// directionally correct, and needs no trained net.
    struct EquityOracle;

    impl LeafEval for EquityOracle {
        fn values(
            &self,
            board: &[Card],
            pot: u32,
            _stack: [u32; 2],
            range: [&[f64]; 2],
        ) -> [Vec<f64>; 2] {
            let combos = all_combos();
            [0, 1].map(|p| {
                let opp = range[1 - p];
                // Equity via sampled river completion for each combo.
                let mut rng = SmallRng::seed_from_u64(7);
                (0..NUM_COMBOS)
                    .map(|ci| {
                        let c = combos[ci];
                        if range[p][ci] <= 0.0 {
                            return 0.0;
                        }
                        let mut used = [false; 52];
                        for &b in board {
                            used[b as usize] = true;
                        }
                        used[c[0] as usize] = true;
                        used[c[1] as usize] = true;
                        let stock: Vec<Card> =
                            (0..52).filter(|&x| !used[x as usize]).collect();
                        let mut num = 0.0;
                        let mut den = 0.0;
                        for _ in 0..3 {
                            let river = stock[rng.random_range(0..stock.len())];
                            let b5 = [board[0], board[1], board[2], board[3], river];
                            let mine = eval_hole_board(&c, &b5);
                            for (oi, o) in combos.iter().enumerate().step_by(4) {
                                let w = opp[oi];
                                if w <= 0.0
                                    || o[0] == c[0]
                                    || o[0] == c[1]
                                    || o[1] == c[0]
                                    || o[1] == c[1]
                                    || o.contains(&river)
                                    || board.contains(&o[0])
                                    || board.contains(&o[1])
                                {
                                    continue;
                                }
                                let theirs = eval_hole_board(o, &b5);
                                num += w * if mine > theirs {
                                    1.0
                                } else if mine == theirs {
                                    0.5
                                } else {
                                    0.0
                                };
                                den += w;
                            }
                        }
                        let eq = if den > 0.0 { num / den } else { 0.5 };
                        eq * pot as f64
                    })
                    .collect()
            })
        }
    }

    /// HU hand played to p1's flop decision on board Qs Js Ts.
    /// p0 = As Ks (flopped royal flush), p1 = 2c 7d.
    fn flop_p1_to_act() -> (Hand, Abstraction) {
        let front = parse_cards("As Ks 2c 7d Qs Js Ts 3h 4d").unwrap();
        let mut deck = fresh_deck();
        let mut used = [false; 52];
        for (i, &c) in front.iter().enumerate() {
            deck[i] = c;
            used[c as usize] = true;
        }
        let mut idx = front.len();
        for c in 0..52u8 {
            if !used[c as usize] {
                deck[idx] = c;
                idx += 1;
            }
        }
        let cfg = HandConfig {
            num_players: 2,
            stack: 2_000,
            sb: 50,
            bb: 100,
        };
        let mut h = Hand::new(&cfg, 0, deck);
        h.apply(PlayerAction::CheckCall); // p0 limps
        h.apply(PlayerAction::CheckCall); // p1 checks -> flop
        assert_eq!(h.street(), Street::Flop);
        assert_eq!(h.to_act(), 1);
        (h, abs_small())
    }

    #[test]
    fn builds_with_leaves_and_runout_terminals() {
        let (h, abs) = flop_p1_to_act();
        let uni = vec![1.0; NUM_COMBOS];
        let oracle = EquityOracle;
        let s = FlopSolver::build(&h, &abs, [&uni, &uni], &[3], &oracle, 10).unwrap();
        assert!(s.node_count() > 3);
        assert!(s.leaf_count() > 0, "must have depth-limit leaves");
        assert_eq!(s.turns.len(), 49);
        // At least one all-in line ends in a runout terminal.
        assert!(s
            .terminals
            .iter()
            .any(|t| matches!(t, Terminal::Runout { .. })));
    }

    #[test]
    fn strategies_normalize_and_solve_completes() {
        let (h, abs) = flop_p1_to_act();
        let uni = vec![1.0; NUM_COMBOS];
        let oracle = EquityOracle;
        let mut s = FlopSolver::build(&h, &abs, [&uni, &uni], &[3], &oracle, 10).unwrap();
        s.solve(25, 120_000);
        assert!(s.iterations() >= 25);
        let mut checked = 0;
        for ci in (0..NUM_COMBOS).step_by(131) {
            if let Some((_, p)) = s.root_strategy(s.combos[ci]) {
                assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9);
                assert!(p.iter().all(|&x| (0.0..=1.0).contains(&x)));
                checked += 1;
            }
        }
        assert!(checked > 5);
    }

    /// Turn-card subsampling: sampled leaf evaluation must still produce
    /// normalized strategies and reduce to fewer cached queries.
    #[test]
    fn sampled_leaf_queries_still_normalize() {
        let (h, abs) = flop_p1_to_act();
        let uni = vec![1.0; NUM_COMBOS];
        let oracle = EquityOracle;
        let mut s =
            FlopSolver::build_sampled(&h, &abs, [&uni, &uni], &[3], &oracle, 10, 12).unwrap();
        assert_eq!(s.query_turns.len(), 12);
        s.solve(15, 120_000);
        let mut checked = 0;
        for ci in (0..NUM_COMBOS).step_by(173) {
            if let Some((_, p)) = s.root_strategy(s.combos[ci]) {
                assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9);
                checked += 1;
            }
        }
        assert!(checked > 3);
    }

    /// A flopped royal flush facing an all-in must call (the runout-terminal
    /// path: it wins every sampled runout).
    #[test]
    fn flopped_royal_calls_a_shove() {
        let (mut h, abs) = flop_p1_to_act();
        let (_, hi) = h.raise_bounds().unwrap();
        h.apply(PlayerAction::RaiseTo(hi)); // p1 open-shoves the flop
        assert_eq!(h.to_act(), 0);
        let uni = vec![1.0; NUM_COMBOS];
        let oracle = EquityOracle;
        let mut s = FlopSolver::build(&h, &abs, [&uni, &uni], &[3], &oracle, 10).unwrap();
        s.solve(120, 120_000);
        let nuts = parse_cards("As Ks").unwrap();
        let (acts, p) = s.root_strategy([nuts[0], nuts[1]]).unwrap();
        assert_eq!(acts[0], AbsAction::Fold);
        assert!(p[1] > 0.9, "flopped royal must call the shove, got {p:?}");
        let v = s.root_values(0);
        let ci = combo_index(nuts[0], nuts[1]);
        assert!(
            (v[ci] - 2000.0).abs() < 80.0,
            "flopped royal facing shove must value ~+2000, got {}",
            v[ci]
        );
    }
}
