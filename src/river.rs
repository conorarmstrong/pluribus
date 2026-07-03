//! Exact river resolving over range vectors (ReBeL-style public belief
//! state solving, specialized to the river where no cards remain to come).
//!
//! With two live players, the remaining game is a small public betting tree.
//! Instead of sampling one hole-card combo per traversal (MCCFR), we solve
//! for *all* 1326 combos of both players simultaneously with vector-form
//! CFR+ over their tracked ranges. Showdown terminals are evaluated in O(N)
//! per pass with a sorted sweep plus blocker inclusion-exclusion, verified
//! against a naive O(N^2) evaluator in the tests.
//!
//! With `qre_lambda`, the opponent plays a logit quantal response (softmax
//! over its regrets) instead of regret matching — an exploitation model of
//! bounded rationality; the hero still regret-matches and thus best-responds
//! to it.

use crate::abstraction::{AbsAction, Abstraction};
use crate::cards::Card;
use crate::cfr::qre_distribution;
use crate::engine::Hand;
use crate::eval::eval_hole_board;
use crate::search::{all_combos, combo_index, NUM_COMBOS};

enum Kid {
    Node(usize),
    End(usize),
}

struct Node {
    /// Solver player (0/1) to act.
    player: usize,
    acts: Vec<AbsAction>,
    kids: Vec<Kid>,
    /// Flattened [combo][action].
    regret: Vec<f64>,
    strat: Vec<f64>,
}

enum Terminal {
    /// Betting ended by a fold: per-solver-player net chips (hole-independent).
    Fold { util: [f64; 2] },
    /// Showdown: `matched` = the called portion each live player has in,
    /// `dead` = folded players' money. Net: win -> matched + dead,
    /// lose -> -matched, tie -> dead / 2.
    Showdown { matched: f64, dead: f64 },
}

pub struct RiverSolver {
    combos: Vec<[Card; 2]>,
    /// Initial (unnormalized) range per solver player, board conflicts zeroed.
    range: [Vec<f64>; 2],
    /// Combo indices sorted by river hand rank; invalid combos excluded.
    sorted: Vec<u32>,
    rank: Vec<u32>,
    nodes: Vec<Node>,
    terminals: Vec<Terminal>,
    iters: u64,
}

impl RiverSolver {
    /// Build the river tree from a decision point `h` (river, exactly two
    /// live players, `h.to_act()` becomes solver player 0). `ranges` are the
    /// tracked weights (indexed by `combo_index`) for [to_act, other].
    pub fn build(h: &Hand, abs: &Abstraction, ranges: [&[f64]; 2]) -> Option<RiverSolver> {
        if h.board().len() != 5 || h.is_terminal() || h.live_count() != 2 {
            return None;
        }
        let hero = h.to_act();
        let villain = (0..h.num_players()).find(|&p| p != hero && !h.folded(p))?;
        let seats = [hero, villain];

        let combos = all_combos();
        let board: Vec<Card> = h.board().to_vec();
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
                // Degenerate tracked range: fall back to uniform over valid.
                for (ci, combo) in combos.iter().enumerate() {
                    r[ci] = if valid(combo) { 1.0 } else { 0.0 };
                }
            }
        }

        let board5: [Card; 5] = board.as_slice().try_into().ok()?;
        let rank: Vec<u32> = combos
            .iter()
            .map(|c| {
                if valid(c) {
                    eval_hole_board(c, &board5)
                } else {
                    0
                }
            })
            .collect();
        let mut sorted: Vec<u32> = (0..NUM_COMBOS as u32)
            .filter(|&ci| valid(&combos[ci as usize]))
            .collect();
        sorted.sort_by_key(|&ci| rank[ci as usize]);

        let mut solver = RiverSolver {
            combos,
            range,
            sorted,
            rank,
            nodes: Vec::new(),
            terminals: Vec::new(),
            iters: 0,
        };
        solver.expand(h.clone(), abs, seats);
        Some(solver)
    }

    /// Recursively expand the betting tree. Returns the root ref.
    fn expand(&mut self, h: Hand, abs: &Abstraction, seats: [usize; 2]) -> Kid {
        if h.is_terminal() {
            let t = if h.live_count() == 1 {
                let u = h.utilities();
                Terminal::Fold {
                    util: [u[seats[0]] as f64, u[seats[1]] as f64],
                }
            } else {
                let (c0, c1) = (
                    h.hand_commit(seats[0]) as f64,
                    h.hand_commit(seats[1]) as f64,
                );
                let matched = c0.min(c1);
                Terminal::Showdown {
                    matched,
                    dead: h.pot() as f64 - c0 - c1,
                }
            };
            self.terminals.push(t);
            return Kid::End(self.terminals.len() - 1);
        }
        let seat = h.to_act();
        let player = if seat == seats[0] { 0 } else { 1 };
        debug_assert!(seat == seats[player]);
        let acts = abs.abstract_actions(&h);
        let kids: Vec<Kid> = acts
            .iter()
            .map(|&a| {
                let mut child = h.clone();
                child.apply(abs.concrete(&h, a));
                self.expand(child, abs, seats)
            })
            .collect();
        let n_acts = acts.len();
        self.nodes.push(Node {
            player,
            acts,
            kids,
            regret: vec![0.0; NUM_COMBOS * n_acts],
            strat: vec![0.0; NUM_COMBOS * n_acts],
        });
        Kid::Node(self.nodes.len() - 1)
    }

    fn root(&self) -> usize {
        self.nodes.len() - 1 // root is pushed last by post-order expansion
    }

    /// Run alternating-updates vector CFR+ for a time/iteration budget.
    /// `qre_lambda`: model solver player 1 (the opponent) as lambda-rational.
    pub fn solve(&mut self, max_iters: u64, time_ms: u64, qre_lambda: Option<f64>) {
        let start = std::time::Instant::now();
        while self.iters < max_iters && (start.elapsed().as_millis() as u64) < time_ms {
            self.iters += 1;
            let t = self.iters as f64;
            for u in 0..2 {
                let reach_own = self.range[u].clone();
                let reach_opp = self.range[1 - u].clone();
                self.walk(Kid::Node(self.root()), u, &reach_own, &reach_opp, t, qre_lambda);
            }
        }
    }

    #[allow(dead_code)] // used by tests; useful for diagnostics
    pub fn iterations(&self) -> u64 {
        self.iters
    }

    /// Average strategy at the root for solver player 0's `hole`, aligned
    /// with the root's action list. None if the combo never got weight.
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

    /// One CFR pass updating player `u`. Returns counterfactual values per
    /// `u`-combo (already weighted by opponent reach and card removal).
    fn walk(
        &mut self,
        at: Kid,
        u: usize,
        reach_own: &[f64],
        reach_opp: &[f64],
        t: f64,
        qre_lambda: Option<f64>,
    ) -> Vec<f64> {
        let ni = match at {
            Kid::End(ti) => return self.terminal_values(ti, u, reach_opp),
            Kid::Node(ni) => ni,
        };
        let n_acts = self.nodes[ni].acts.len();
        let player = self.nodes[ni].player;

        // Current strategy per combo: RM+ for the updating player and the
        // hero; logit QRE for the modeled opponent (player 1) if enabled.
        let sigma = self.node_sigma(ni, qre_lambda);

        if player == u {
            let mut vals = vec![0.0; NUM_COMBOS];
            let mut act_vals: Vec<Vec<f64>> = Vec::with_capacity(n_acts);
            for a in 0..n_acts {
                let mut own = vec![0.0; NUM_COMBOS];
                for ci in 0..NUM_COMBOS {
                    own[ci] = reach_own[ci] * sigma[ci * n_acts + a];
                }
                let kid = match &self.nodes[ni].kids[a] {
                    Kid::Node(x) => Kid::Node(*x),
                    Kid::End(x) => Kid::End(*x),
                };
                let v = self.walk(kid, u, &own, reach_opp, t, qre_lambda);
                for ci in 0..NUM_COMBOS {
                    vals[ci] += sigma[ci * n_acts + a] * v[ci];
                }
                act_vals.push(v);
            }
            // RM+ regret update and linear strategy averaging.
            let node = &mut self.nodes[ni];
            for ci in 0..NUM_COMBOS {
                for (a, av) in act_vals.iter().enumerate() {
                    let idx = ci * n_acts + a;
                    node.regret[idx] = (node.regret[idx] + av[ci] - vals[ci]).max(0.0);
                    node.strat[idx] += t * reach_own[ci] * sigma[idx];
                }
            }
            vals
        } else {
            // Opponent node on u's pass: weight opponent reach by their
            // strategy and sum child values.
            let mut vals = vec![0.0; NUM_COMBOS];
            for a in 0..n_acts {
                let mut opp = vec![0.0; NUM_COMBOS];
                for ci in 0..NUM_COMBOS {
                    opp[ci] = reach_opp[ci] * sigma[ci * n_acts + a];
                }
                let kid = match &self.nodes[ni].kids[a] {
                    Kid::Node(x) => Kid::Node(*x),
                    Kid::End(x) => Kid::End(*x),
                };
                let v = self.walk(kid, u, reach_own, &opp, t, qre_lambda);
                for ci in 0..NUM_COMBOS {
                    vals[ci] += v[ci];
                }
            }
            vals
        }
    }

    /// Per-combo strategy at a node: regret matching+ (or logit QRE for the
    /// modeled opponent, solver player 1).
    fn node_sigma(&self, ni: usize, qre_lambda: Option<f64>) -> Vec<f64> {
        let node = &self.nodes[ni];
        let n_acts = node.acts.len();
        let mut sigma = vec![0.0; NUM_COMBOS * n_acts];
        let qre = qre_lambda.filter(|_| node.player == 1);
        let mut buf = Vec::with_capacity(n_acts);
        for ci in 0..NUM_COMBOS {
            let r = &node.regret[ci * n_acts..(ci + 1) * n_acts];
            match qre {
                Some(l) => qre_distribution(r, l, &mut buf),
                None => {
                    buf.clear();
                    let total: f64 = r.iter().sum(); // RM+: regrets already >= 0
                    if total > 0.0 {
                        buf.extend(r.iter().map(|&x| x / total));
                    } else {
                        buf.extend(std::iter::repeat_n(1.0 / n_acts as f64, n_acts));
                    }
                }
            }
            sigma[ci * n_acts..(ci + 1) * n_acts].copy_from_slice(&buf);
        }
        sigma
    }

    /// Counterfactual values per u-combo at a terminal, given the opponent's
    /// reach vector. Card removal is exact via inclusion-exclusion.
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
                for &ci in &self.sorted {
                    let ci = ci as usize;
                    let c = self.combos[ci];
                    let excl = total - card[c[0] as usize] - card[c[1] as usize] + reach_opp[ci];
                    vals[ci] = util[u] * excl;
                }
                vals
            }
            Terminal::Showdown { matched, dead } => {
                self.showdown_values(u, reach_opp, total, &card, *matched, *dead)
            }
        }
    }

    fn showdown_values(
        &self,
        _u: usize,
        reach_opp: &[f64],
        total: f64,
        card: &[f64; 52],
        matched: f64,
        dead: f64,
    ) -> Vec<f64> {
        showdown_sweep(
            &self.combos,
            &self.sorted,
            &self.rank,
            reach_opp,
            total,
            card,
            matched,
            dead,
        )
    }

    #[cfg(test)]
    /// Naive O(N^2) showdown evaluation for differential testing.
    fn showdown_values_naive(
        &self,
        reach_opp: &[f64],
        matched: f64,
        dead: f64,
    ) -> Vec<f64> {
        let mut vals = vec![0.0; NUM_COMBOS];
        for &ci in &self.sorted {
            let ci = ci as usize;
            let c = self.combos[ci];
            let mut v = 0.0;
            for &oi in &self.sorted {
                let oi = oi as usize;
                if oi == ci {
                    continue;
                }
                let o = self.combos[oi];
                if o[0] == c[0] || o[0] == c[1] || o[1] == c[0] || o[1] == c[1] {
                    continue;
                }
                let w = reach_opp[oi];
                let payoff = match self.rank[ci].cmp(&self.rank[oi]) {
                    std::cmp::Ordering::Greater => matched + dead,
                    std::cmp::Ordering::Equal => dead / 2.0,
                    std::cmp::Ordering::Less => -matched,
                };
                v += w * payoff;
            }
            vals[ci] = v;
        }
        vals
    }
}

/// O(N) sorted-sweep showdown evaluation with exact card-removal blocker
/// effects via inclusion-exclusion — shared by the river and turn solvers.
/// `sorted` lists valid combo indices ascending by `rank`; `total`/`card`
/// aggregate `reach_opp` overall and per card.
#[allow(clippy::too_many_arguments)]
pub(crate) fn showdown_sweep(
    combos: &[[Card; 2]],
    sorted: &[u32],
    rank: &[u32],
    reach_opp: &[f64],
    total: f64,
    card: &[f64; 52],
    matched: f64,
    dead: f64,
) -> Vec<f64> {
    let win_v = matched + dead;
    let tie_v = dead / 2.0;
    let lose_v = -matched;

    let mut vals = vec![0.0; NUM_COMBOS];
    let mut below_total = 0.0f64;
    let mut below_card = [0.0f64; 52];

    let mut i = 0;
    while i < sorted.len() {
        // Group of equal rank.
        let r = rank[sorted[i] as usize];
        let mut j = i;
        let mut grp_total = 0.0f64;
        let mut grp_card = [0.0f64; 52];
        while j < sorted.len() && rank[sorted[j] as usize] == r {
            let ci = sorted[j] as usize;
            let w = reach_opp[ci];
            if w > 0.0 {
                grp_total += w;
                let c = combos[ci];
                grp_card[c[0] as usize] += w;
                grp_card[c[1] as usize] += w;
            }
            j += 1;
        }
        for &ck in &sorted[i..j] {
            let ci = ck as usize;
            let c = combos[ci];
            let (c0, c1) = (c[0] as usize, c[1] as usize);
            // Opponent combos not sharing a card with ours:
            let valid_total = total - card[c0] - card[c1] + reach_opp[ci];
            let weaker = below_total - below_card[c0] - below_card[c1];
            let tie = grp_total - grp_card[c0] - grp_card[c1] + reach_opp[ci];
            let stronger = valid_total - weaker - tie;
            vals[ci] = weaker * win_v + tie * tie_v + stronger * lose_v;
        }
        below_total += grp_total;
        for c in 0..52 {
            below_card[c] += grp_card[c];
        }
        i = j;
    }
    vals
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
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    fn abs_small() -> Abstraction {
        Abstraction::new(AbsConfig {
            postflop_buckets: 6,
            equity_rollouts: 40,
            cache_cap: 100_000,
            ..AbsConfig::default()
        })
    }

    /// Heads-up hand (board Qs Js Ts 3h 4d) checked to p1's river decision:
    /// p0 limps, BB checks, flop and turn check through. p1 is to act on the
    /// river (non-button acts first postflop).
    fn river_p1_to_act() -> (Hand, Abstraction) {
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
        for _ in 0..2 {
            h.apply(PlayerAction::CheckCall); // p1 checks flop/turn
            h.apply(PlayerAction::CheckCall); // p0 checks behind
        }
        assert_eq!(h.street(), crate::engine::Street::River);
        assert_eq!(h.to_act(), 1);
        (h, abs_small())
    }

    /// p1 checks the river; p0 to act with the full betting tree ahead.
    fn river_checked_to_p0() -> (Hand, Abstraction) {
        let (mut h, abs) = river_p1_to_act();
        h.apply(PlayerAction::CheckCall);
        assert_eq!(h.to_act(), 0);
        (h, abs)
    }

    /// p1 shoves the river; p0 faces an all-in with only fold/call.
    fn river_facing_shove() -> (Hand, Abstraction) {
        let (mut h, abs) = river_p1_to_act();
        let (_, hi) = h.raise_bounds().unwrap();
        h.apply(PlayerAction::RaiseTo(hi));
        assert_eq!(h.to_act(), 0);
        (h, abs)
    }

    #[test]
    fn showdown_sweep_matches_naive() {
        let (h, abs) = river_facing_shove();
        let uni = vec![1.0; NUM_COMBOS];
        let solver = RiverSolver::build(&h, &abs, [&uni, &uni]).unwrap();

        let mut rng = SmallRng::seed_from_u64(55);
        for _ in 0..3 {
            let reach: Vec<f64> = (0..NUM_COMBOS)
                .map(|_| if rng.random::<f64>() < 0.3 { 0.0 } else { rng.random::<f64>() })
                .collect();
            let (matched, dead) = (1900.0, 0.0);
            let mut total = 0.0;
            let mut card = [0.0f64; 52];
            for (ci, c) in solver.combos.iter().enumerate() {
                let w = if solver.rank[ci] > 0 { reach[ci] } else { 0.0 };
                if w > 0.0 {
                    total += w;
                    card[c[0] as usize] += w;
                    card[c[1] as usize] += w;
                }
            }
            let masked: Vec<f64> = (0..NUM_COMBOS)
                .map(|ci| if solver.rank[ci] > 0 { reach[ci] } else { 0.0 })
                .collect();
            let fast = solver.showdown_values(0, &masked, total, &card, matched, dead);
            let naive = solver.showdown_values_naive(&masked, matched, dead);
            for &ci in &solver.sorted {
                let ci = ci as usize;
                assert!(
                    (fast[ci] - naive[ci]).abs() < 1e-6 * (1.0 + naive[ci].abs()),
                    "combo {ci}: fast {} vs naive {}",
                    fast[ci],
                    naive[ci]
                );
            }
        }
    }

    /// Facing a river shove with the stone-cold nuts, the solved strategy
    /// must call; with the worst hands it must mostly fold.
    #[test]
    fn nuts_call_and_air_folds_facing_shove() {
        let (h, abs) = river_facing_shove();
        let uni = vec![1.0; NUM_COMBOS];
        let mut solver = RiverSolver::build(&h, &abs, [&uni, &uni]).unwrap();
        solver.solve(400, 10_000, None);
        assert!(solver.iterations() >= 100, "must complete iterations");

        // Board Qs Js Ts 3h 4d. Holding Ks blocks the royal, so Ks 9s (a
        // king-high straight flush) is the effective nuts — and we can query
        // ANY combo, that's the point of range-vector solving.
        let nuts = parse_cards("Ks 9s").unwrap();
        let (acts, s) = solver.root_strategy([nuts[0], nuts[1]]).unwrap();
        assert_eq!(acts[0], AbsAction::Fold);
        assert!(
            s[1] > 0.95,
            "nuts must call the shove, got {:?} for {:?}",
            s,
            acts
        );

        let air = parse_cards("2d 6h").unwrap(); // six-high, no pair
        let (_, s_air) = solver.root_strategy([air[0], air[1]]).unwrap();
        assert!(
            s_air[0] > 0.7,
            "worthless hand should usually fold, got {:?}",
            s_air
        );
    }

    /// Strategies are proper distributions for every valid combo at the root.
    #[test]
    fn root_strategies_are_normalized() {
        let (h, abs) = river_facing_shove();
        let uni = vec![1.0; NUM_COMBOS];
        let mut solver = RiverSolver::build(&h, &abs, [&uni, &uni]).unwrap();
        solver.solve(50, 5_000, None);
        let mut checked = 0;
        for &ci in solver.sorted.clone().iter().step_by(37) {
            let c = solver.combos[ci as usize];
            if let Some((_, s)) = solver.root_strategy(c) {
                assert!((s.iter().sum::<f64>() - 1.0).abs() < 1e-9);
                assert!(s.iter().all(|&p| (0.0..=1.0).contains(&p)));
                checked += 1;
            }
        }
        assert!(checked > 10);
    }

    /// lambda = 0 models a uniform-random opponent (folds to bets far too
    /// often, calls with anything). The hero's exploit response must bet the
    /// river dramatically more than the equilibrium strategy does.
    #[test]
    fn qre_lambda_changes_hero_response() {
        let (h, abs) = river_checked_to_p0();
        let uni = vec![1.0; NUM_COMBOS];

        let mut nash = RiverSolver::build(&h, &abs, [&uni, &uni]).unwrap();
        nash.solve(200, 20_000, None);
        let mut exploit = RiverSolver::build(&h, &abs, [&uni, &uni]).unwrap();
        exploit.solve(200, 20_000, Some(0.0));

        // Range-weighted frequency of betting (any non-check action) at root.
        let bet_freq = |s: &RiverSolver| {
            let (mut bet, mut total) = (0.0, 0.0);
            for &ci in &s.sorted {
                let combo = s.combos[ci as usize];
                if let Some((acts, probs)) = s.root_strategy(combo) {
                    for (a, p) in acts.iter().zip(&probs) {
                        if !matches!(a, AbsAction::CheckCall) {
                            bet += *p;
                        }
                    }
                    total += 1.0;
                }
            }
            bet / total
        };
        let (f_nash, f_exp) = (bet_freq(&nash), bet_freq(&exploit));
        assert!(
            f_exp > f_nash + 0.1,
            "vs a uniform-random opponent the hero must bet much more: \
             nash {f_nash:.3} exploit {f_exp:.3}"
        );
    }
}
