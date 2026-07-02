//! Exact turn resolving over range vectors: vector-form CFR+ over the turn
//! betting tree, an explicit chance node over every possible river card, and
//! the full river betting tree beneath each — the ReBeL public-belief-state
//! computation, one street deeper than `river.rs`.
//!
//! Tractability comes from a configurable slim bet menu (default: 75%-pot
//! plus all-in) — with full menus a turn+river tree over 1326-combo vectors
//! needs tens of GB; with a slim menu it is ~100MB and solves in seconds.
//! River play at the table still uses the full-menu exact river solver.
//!
//! Chance is weighted 1/(rivers − 4): every (hero, villain) combo pair
//! blocks exactly 4 of the unseen river cards, so this is the exact
//! conditional river distribution for every pair simultaneously — root
//! counterfactual values divide cleanly by compatible opponent mass into
//! per-combo expected chip values, the training targets for the belief-state
//! value network.

use crate::abstraction::{AbsAction, Abstraction};
use crate::cards::Card;
use crate::engine::{Hand, Street};
use crate::eval::eval_hole_board;
use crate::river::showdown_sweep;
use crate::search::{all_combos, combo_index, NUM_COMBOS};

enum Kid {
    Node(usize),
    End(usize),
    Chance(usize),
}

struct Node {
    /// Solver player (0/1) to act.
    player: usize,
    acts: Vec<AbsAction>,
    kids: Vec<Kid>,
    /// Flattened [combo][action].
    regret: Vec<f64>,
    /// Linear-averaged strategy — allocated only at the root (that is all
    /// we extract; inner nodes save the memory).
    strat: Vec<f64>,
}

enum Terminal {
    /// Betting ended by a fold: per-solver-player net chips.
    Fold { util: [f64; 2] },
    /// Showdown on a known river (`river_ix` indexes the rank tables).
    Showdown {
        river_ix: usize,
        matched: f64,
        dead: f64,
    },
}

/// One child per possible river card, aligned with `TurnSolver::rivers`.
struct ChanceNode {
    kids: Vec<Kid>,
}

pub struct TurnSolver {
    combos: Vec<[Card; 2]>,
    /// Initial (unnormalized) ranges, board conflicts zeroed.
    range: [Vec<f64>; 2],
    /// The unseen candidate river cards (board conflicts excluded).
    rivers: Vec<Card>,
    /// Per river index: combo hand ranks and rank-sorted valid combos.
    rank: Vec<Vec<u32>>,
    sorted: Vec<Vec<u32>>,
    nodes: Vec<Node>,
    terminals: Vec<Terminal>,
    chances: Vec<ChanceNode>,
    /// Linear-weighted root counterfactual value sums per solver player.
    root_vals: [Vec<f64>; 2],
    weight_sum: f64,
    iters: u64,
}

impl TurnSolver {
    /// Build the turn+river tree from a decision point `h` (turn street,
    /// exactly two live players; `h.to_act()` becomes solver player 0).
    /// `ranges` are tracked weights indexed by `combo_index` for
    /// [to_act, other]; `keep` are the allowed `BET_SIZES` indices (fold,
    /// check/call and all-in always stay).
    pub fn build(
        h: &Hand,
        abs: &Abstraction,
        ranges: [&[f64]; 2],
        keep: &[u8],
    ) -> Option<TurnSolver> {
        if h.street() != Street::Turn || h.is_terminal() || h.live_count() != 2 {
            return None;
        }
        let hero = h.to_act();
        let villain = (0..h.num_players()).find(|&p| p != hero && !h.folded(p))?;
        let seats = [hero, villain];

        let combos = all_combos();
        let board: Vec<Card> = h.board().to_vec();
        debug_assert_eq!(board.len(), 4);
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

        let rivers: Vec<Card> = (0..52u8).filter(|&c| !on_board[c as usize]).collect();
        let mut rank: Vec<Vec<u32>> = Vec::with_capacity(rivers.len());
        let mut sorted: Vec<Vec<u32>> = Vec::with_capacity(rivers.len());
        for &r in &rivers {
            let board5 = [board[0], board[1], board[2], board[3], r];
            let rk: Vec<u32> = combos
                .iter()
                .map(|c| {
                    if valid(c) && c[0] != r && c[1] != r {
                        eval_hole_board(c, &board5)
                    } else {
                        0
                    }
                })
                .collect();
            let mut st: Vec<u32> = (0..NUM_COMBOS as u32)
                .filter(|&ci| rk[ci as usize] > 0)
                .collect();
            st.sort_by_key(|&ci| rk[ci as usize]);
            rank.push(rk);
            sorted.push(st);
        }

        let mut solver = TurnSolver {
            combos,
            range,
            rivers,
            rank,
            sorted,
            nodes: Vec::new(),
            terminals: Vec::new(),
            chances: Vec::new(),
            root_vals: [vec![0.0; NUM_COMBOS], vec![0.0; NUM_COMBOS]],
            weight_sum: 0.0,
            iters: 0,
        };
        let root = solver.expand(h.clone(), abs, seats, keep, None);
        let Kid::Node(root_ix) = root else {
            return None; // degenerate root (shouldn't happen on a live turn)
        };
        let n = solver.nodes[root_ix].acts.len();
        solver.nodes[root_ix].strat = vec![0.0; NUM_COMBOS * n];
        Some(solver)
    }

    /// Slim action menu: the abstraction's menu with only `keep` bet sizes.
    fn menu(abs: &Abstraction, h: &Hand, keep: &[u8]) -> Vec<AbsAction> {
        abs.abstract_actions(h)
            .into_iter()
            .filter(|a| match a {
                AbsAction::Bet(i) => keep.contains(i),
                _ => true,
            })
            .collect()
    }

    /// Recursively expand. `river_ix` is Some once below the chance node.
    fn expand(
        &mut self,
        h: Hand,
        abs: &Abstraction,
        seats: [usize; 2],
        keep: &[u8],
        river_ix: Option<usize>,
    ) -> Kid {
        if h.is_terminal() {
            if h.live_count() == 1 {
                let u = h.utilities();
                self.terminals.push(Terminal::Fold {
                    util: [u[seats[0]] as f64, u[seats[1]] as f64],
                });
                return Kid::End(self.terminals.len() - 1);
            }
            let (c0, c1) = (
                h.hand_commit(seats[0]) as f64,
                h.hand_commit(seats[1]) as f64,
            );
            let matched = c0.min(c1);
            let dead = h.pot() as f64 - c0 - c1;
            return match river_ix {
                Some(ri) => {
                    self.terminals.push(Terminal::Showdown {
                        river_ix: ri,
                        matched,
                        dead,
                    });
                    Kid::End(self.terminals.len() - 1)
                }
                None => {
                    // All-in before the river: runout showdown, one terminal
                    // per candidate river card.
                    let kids: Vec<Kid> = (0..self.rivers.len())
                        .map(|ri| {
                            self.terminals.push(Terminal::Showdown {
                                river_ix: ri,
                                matched,
                                dead,
                            });
                            Kid::End(self.terminals.len() - 1)
                        })
                        .collect();
                    self.chances.push(ChanceNode { kids });
                    Kid::Chance(self.chances.len() - 1)
                }
            };
        }

        if river_ix.is_none() && h.street() == Street::River {
            // Turn betting just closed: branch on every candidate river.
            let kids: Vec<Kid> = (0..self.rivers.len())
                .map(|ri| {
                    let mut child = h.clone();
                    child.force_river(self.rivers[ri]);
                    self.expand(child, abs, seats, keep, Some(ri))
                })
                .collect();
            self.chances.push(ChanceNode { kids });
            return Kid::Chance(self.chances.len() - 1);
        }

        let seat = h.to_act();
        let player = if seat == seats[0] { 0 } else { 1 };
        let acts = Self::menu(abs, &h, keep);
        let kids: Vec<Kid> = acts
            .iter()
            .map(|&a| {
                let mut child = h.clone();
                child.apply(abs.concrete(&h, a));
                self.expand(child, abs, seats, keep, river_ix)
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

    fn root(&self) -> usize {
        self.nodes.len() - 1
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn iterations(&self) -> u64 {
        self.iters
    }

    /// Run alternating vector CFR+ for a time/iteration budget, accumulating
    /// linear-weighted root counterfactual values.
    pub fn solve(&mut self, max_iters: u64, time_ms: u64) {
        let start = std::time::Instant::now();
        while self.iters < max_iters && (start.elapsed().as_millis() as u64) < time_ms {
            self.iters += 1;
            let t = self.iters as f64;
            for u in 0..2 {
                let reach_own = self.range[u].clone();
                let reach_opp = self.range[1 - u].clone();
                let v = self.walk(Kid::Node(self.root()), u, &reach_own, &reach_opp, t);
                for ci in 0..NUM_COMBOS {
                    self.root_vals[u][ci] += t * v[ci];
                }
            }
            self.weight_sum += t;
        }
    }

    /// Average strategy at the root for solver player 0's `hole`.
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

    /// Per-combo expected chip value for solver player `u` holding each
    /// combo at the root (counterfactual value normalized by the opponent's
    /// compatible range mass). NaN-free: combos with no weight report 0.
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

    /// One CFR pass updating player `u`; returns u's counterfactual values
    /// per combo (weighted by opponent reach with exact card removal).
    fn walk(&mut self, at: Kid, u: usize, reach_own: &[f64], reach_opp: &[f64], t: f64) -> Vec<f64> {
        let ni = match at {
            Kid::End(ti) => return self.terminal_values(ti, u, reach_opp),
            Kid::Chance(xi) => {
                // Exact conditional river probability: every combo pair
                // blocks exactly 4 of the candidate cards.
                let p = 1.0 / (self.rivers.len() as f64 - 4.0);
                let mut vals = vec![0.0; NUM_COMBOS];
                for ri in 0..self.chances[xi].kids.len() {
                    let r = self.rivers[ri] as usize;
                    let mut own = reach_own.to_vec();
                    let mut opp = reach_opp.to_vec();
                    for (ci, combo) in self.combos.iter().enumerate() {
                        if combo[0] as usize == r || combo[1] as usize == r {
                            own[ci] = 0.0;
                            opp[ci] = 0.0;
                        }
                    }
                    let kid = self.chances[xi].kids[ri].shallow();
                    let v = self.walk(kid, u, &own, &opp, t);
                    for (ci, combo) in self.combos.iter().enumerate() {
                        if combo[0] as usize != r && combo[1] as usize != r {
                            vals[ci] += p * v[ci];
                        }
                    }
                }
                return vals;
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
                for ci in 0..NUM_COMBOS {
                    vals[ci] += v[ci];
                }
            }
            vals
        }
    }

    /// Per-combo regret-matching+ strategy at a node.
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
        match &self.terminals[ti] {
            Terminal::Fold { util } => {
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
            Terminal::Showdown {
                river_ix,
                matched,
                dead,
            } => showdown_sweep(
                &self.combos,
                &self.sorted[*river_ix],
                &self.rank[*river_ix],
                reach_opp,
                total,
                &card,
                *matched,
                *dead,
            ),
        }
    }
}

impl Kid {
    /// Cheap copy of the enum tag+index (children are indices, not owners).
    fn shallow(&self) -> Kid {
        match self {
            Kid::Node(x) => Kid::Node(*x),
            Kid::End(x) => Kid::End(*x),
            Kid::Chance(x) => Kid::Chance(*x),
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

    /// Heads-up hand checked to p1's turn decision on board Qs Js Ts 3h.
    /// Deck: p0 = As Ks (royal already made), p1 = 2c 7d.
    fn turn_p1_to_act() -> (Hand, Abstraction) {
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
        h.apply(PlayerAction::CheckCall); // p1 checks the option
        h.apply(PlayerAction::CheckCall); // p1 checks flop
        h.apply(PlayerAction::CheckCall); // p0 checks behind -> turn
        assert_eq!(h.street(), Street::Turn);
        assert_eq!(h.to_act(), 1);
        (h, abs_small())
    }

    /// p1 shoves the turn; p0 to act facing the all-in.
    fn turn_facing_shove() -> (Hand, Abstraction) {
        let (mut h, abs) = turn_p1_to_act();
        let (_, hi) = h.raise_bounds().unwrap();
        h.apply(PlayerAction::RaiseTo(hi));
        assert_eq!(h.to_act(), 0);
        (h, abs)
    }

    #[test]
    fn builds_a_multi_street_tree() {
        let (h, abs) = turn_p1_to_act();
        let uni = vec![1.0; NUM_COMBOS];
        let s = TurnSolver::build(&h, &abs, [&uni, &uni], &[3]).unwrap();
        // Must contain river subtrees under chance nodes: far more nodes
        // than a single-street tree, and at least one chance node with one
        // branch per unseen river card.
        assert!(s.node_count() > 100, "got {} nodes", s.node_count());
        assert!(!s.chances.is_empty());
        assert_eq!(s.rivers.len(), 48);
        assert!(s.chances.iter().all(|c| c.kids.len() == 48));
    }

    #[test]
    fn root_strategies_are_normalized() {
        let (h, abs) = turn_p1_to_act();
        let uni = vec![1.0; NUM_COMBOS];
        let mut s = TurnSolver::build(&h, &abs, [&uni, &uni], &[3]).unwrap();
        s.solve(30, 60_000);
        assert!(s.iterations() >= 5, "too slow: {} iters", s.iterations());
        let mut checked = 0;
        for ci in (0..NUM_COMBOS).step_by(97) {
            if let Some((_, p)) = s.root_strategy(s.combos[ci]) {
                assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9);
                assert!(p.iter().all(|&x| (0.0..=1.0).contains(&x)));
                checked += 1;
            }
        }
        assert!(checked > 5);
    }

    /// Facing a turn shove holding an already-completed royal flush (wins on
    /// EVERY river): must call, and the normalized root value must equal the
    /// win payoff (matched + dead = whole pot gain) almost exactly — this
    /// exercises chance weighting, blocker masking and the showdown sweep
    /// end to end.
    #[test]
    fn made_royal_calls_shove_and_values_full_pot() {
        let (h, abs) = turn_facing_shove();
        let uni = vec![1.0; NUM_COMBOS];
        let mut s = TurnSolver::build(&h, &abs, [&uni, &uni], &[3]).unwrap();
        s.solve(60, 60_000);

        let nuts = parse_cards("As Ks").unwrap();
        let (acts, p) = s.root_strategy([nuts[0], nuts[1]]).unwrap();
        assert_eq!(acts[0], AbsAction::Fold);
        assert!(p[1] > 0.95, "royal must call the shove, got {p:?}");

        // p1 shoved 2000 total, p0 has 100 in: matched = 2000 after call...
        // matched portion each = min(commit) = 2000 when called; the root
        // value averages fold/call lines under the average strategy, but a
        // pure-call royal's value is the showdown: matched + dead = 2000
        // (villain's matched stack) + 0 dead... expressed as net win at
        // showdown terminals: win_v = matched + dead where matched is the
        // post-call commitment 2000. The royal wins every river against
        // every combo, so normalized value ≈ +2000 − (already-sunk
        // accounting is inside the terminal: win_v = +2000).
        let v = s.root_values(0);
        let ci = combo_index(nuts[0], nuts[1]);
        assert!(
            (v[ci] - 2000.0).abs() < 60.0,
            "made royal must value ~+2000 (wins every river), got {}",
            v[ci]
        );
    }

    /// Root counterfactual values must be (approximately) zero-sum across
    /// the two players when weighted by their ranges.
    #[test]
    fn root_values_are_zero_sum() {
        let (h, abs) = turn_p1_to_act();
        let uni = vec![1.0; NUM_COMBOS];
        let mut s = TurnSolver::build(&h, &abs, [&uni, &uni], &[3]).unwrap();
        s.solve(40, 60_000);
        let (mut e0, mut e1, mut w0, mut w1) = (0.0, 0.0, 0.0, 0.0);
        let (v0, v1) = (s.root_values(0), s.root_values(1));
        for ci in 0..NUM_COMBOS {
            e0 += s.range[0][ci] * v0[ci];
            w0 += s.range[0][ci];
            e1 += s.range[1][ci] * v1[ci];
            w1 += s.range[1][ci];
        }
        let (m0, m1) = (e0 / w0, e1 / w1);
        // Pot is 200: per-player expectations must roughly cancel.
        assert!(
            (m0 + m1).abs() < 20.0,
            "range-weighted values must be ~zero-sum, got {m0:.1} + {m1:.1}"
        );
    }
}
