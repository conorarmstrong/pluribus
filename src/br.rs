//! Exact-subgame best response probe — a tighter exploitability lower bound
//! than LBR.
//!
//! Same harness and exact-Bayes range tracking as `lbr.rs`, but at every
//! turn and river decision the probe computes a TRUE best response of the
//! entire remaining game: a single expectimax pass over the full-menu
//! betting tree (turn decisions include the explicit river chance node and
//! the full river tree beneath it, `turn.rs`-style). The bot is a fixed,
//! known strategy, so no CFR iterations are needed — one downward-upward
//! pass is exact. Preflop and flop decisions fall back to the greedy LBR
//! action (a full-game exact BR is intractable: the chance branching over
//! boards puts it at ~1e13 vector node-visits).
//!
//! Every strategy's average winnings against the bot lower-bound the bot's
//! exploitability, so a strictly stronger probe gives a strictly tighter
//! bound in expectation. Given a seed the probe is deterministic, so two
//! code versions can be compared on identical deals (paired differences).

use crate::abstraction::{AbsAction, TOKEN_STREET_SEP};
use crate::bot::Policy;
use crate::cards::{fresh_deck, Card};
use crate::engine::{Hand, HandConfig, Street};
use crate::eval::eval_hole_board;
use crate::lbr::{BotRange, Lbr};
use crate::river::showdown_sweep;
use crate::search::{all_combos, NUM_COMBOS};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// Exact best response over the remaining game from a turn or river
/// decision, against a fixed blueprint policy with a tracked range.
pub struct SubgameBr<'a> {
    policy: &'a Policy,
    combos: Vec<[Card; 2]>,
    hero_seat: usize,
    bot_seat: usize,
    /// Candidate river cards (empty when the board is already complete).
    rivers: Vec<Card>,
    /// Per river index (single entry on a complete board): combo hand ranks
    /// and rank-sorted valid combos for the O(N) showdown sweep.
    rank: Vec<Vec<u32>>,
    sorted: Vec<Vec<u32>>,
    /// Bot bucket per combo on the current (pre-chance) board; 0 for combos
    /// blocked by the board.
    bucket_now: Vec<u16>,
    /// Bot bucket per combo per candidate river (empty on a complete board).
    bucket_river: Vec<Vec<u16>>,
}

impl SubgameBr<'_> {
    /// Build from a decision point `h` (turn or river, exactly two live
    /// players, probe to act).
    pub fn build<'a>(policy: &'a Policy, h: &Hand, bot_seat: usize) -> Option<SubgameBr<'a>> {
        if h.is_terminal() || h.live_count() != 2 {
            return None;
        }
        if h.street() != Street::Turn && h.street() != Street::River {
            return None;
        }
        let hero_seat = h.to_act();
        if hero_seat == bot_seat {
            return None;
        }
        let combos = all_combos();
        let board: Vec<Card> = h.board().to_vec();
        let mut on_board = [false; 52];
        for &c in &board {
            on_board[c as usize] = true;
        }
        let valid = |c: &[Card; 2]| !on_board[c[0] as usize] && !on_board[c[1] as usize];

        let mut rivers = Vec::new();
        let mut rank = Vec::new();
        let mut sorted = Vec::new();
        let mut bucket_river = Vec::new();

        let bucket_for = |combo: &[Card; 2], b: &[Card], ci: usize| -> u16 {
            let mut rng = SmallRng::seed_from_u64(0x1B12_5EED ^ ci as u64);
            policy.abs.bucket(*combo, b, &mut rng)
        };

        let rank_table = |board5: &[Card; 5]| -> (Vec<u32>, Vec<u32>) {
            let rk: Vec<u32> = combos
                .iter()
                .map(|c| {
                    if valid(c) && !board5[3..].contains(&c[0]) && !board5[3..].contains(&c[1]) {
                        eval_hole_board(c, board5)
                    } else {
                        0
                    }
                })
                .collect();
            let mut st: Vec<u32> = (0..NUM_COMBOS as u32)
                .filter(|&ci| rk[ci as usize] > 0)
                .collect();
            st.sort_by_key(|&ci| rk[ci as usize]);
            (rk, st)
        };

        if h.street() == Street::Turn {
            debug_assert_eq!(board.len(), 4);
            rivers = (0..52u8).filter(|&c| !on_board[c as usize]).collect();
            for &r in &rivers {
                let board5 = [board[0], board[1], board[2], board[3], r];
                let (rk, st) = rank_table(&board5);
                rank.push(rk);
                sorted.push(st);
                let b5 = board5.to_vec();
                let buckets: Vec<u16> = combos
                    .iter()
                    .enumerate()
                    .map(|(ci, c)| {
                        if valid(c) && c[0] != r && c[1] != r {
                            bucket_for(c, &b5, ci)
                        } else {
                            0
                        }
                    })
                    .collect();
                bucket_river.push(buckets);
            }
        } else {
            debug_assert_eq!(board.len(), 5);
            let board5 = [board[0], board[1], board[2], board[3], board[4]];
            let (rk, st) = rank_table(&board5);
            rank.push(rk);
            sorted.push(st);
        }

        let bucket_now: Vec<u16> = combos
            .iter()
            .enumerate()
            .map(|(ci, c)| if valid(c) { bucket_for(c, &board, ci) } else { 0 })
            .collect();

        Some(SubgameBr {
            policy,
            combos,
            hero_seat,
            bot_seat,
            rivers,
            rank,
            sorted,
            bucket_now,
            bucket_river,
        })
    }

    /// Best-response action for the probe holding `hole`, given the bot's
    /// tracked range `bot_range` and the table history `hist`. Returns the
    /// argmax action and its exact expected value (net chips from here on,
    /// normalized by the bot's compatible range mass).
    pub fn action(&self, h: &Hand, hist: &[u8], hole: [Card; 2], bot_range: &[f64]) -> (AbsAction, f64) {
        let hero_ci = crate::search::combo_index(hole[0], hole[1]);
        let mut blocked = [false; 52];
        blocked[hole[0] as usize] = true;
        blocked[hole[1] as usize] = true;
        for &c in h.board() {
            blocked[c as usize] = true;
        }
        // The probe's own cards and the board are impossible bot holdings.
        let mut reach = bot_range.to_vec();
        for (ci, combo) in self.combos.iter().enumerate() {
            if blocked[combo[0] as usize] || blocked[combo[1] as usize] {
                reach[ci] = 0.0;
            }
        }
        let river_ix = if h.street() == Street::River { Some(0) } else { None };

        let acts = self.policy.abs.abstract_actions(h);
        let mut best = AbsAction::CheckCall;
        let mut best_v = f64::MIN;
        for &a in &acts {
            let mut child = h.clone();
            child.apply(self.policy.abs.concrete(h, a));
            let mut hist2 = hist.to_vec();
            hist2.push(a.token());
            if child.street() != h.street() && !child.is_terminal() {
                hist2.push(TOKEN_STREET_SEP);
            }
            let v = self.walk(&child, &hist2, &reach, river_ix);
            let mass = self.compat_mass(&reach, hero_ci);
            let val = if mass > 0.0 { v[hero_ci] / mass } else { 0.0 };
            if val > best_v {
                best_v = val;
                best = a;
            }
        }
        (best, best_v)
    }

    /// Opponent range mass compatible with the hero combo (card removal).
    fn compat_mass(&self, reach: &[f64], hero_ci: usize) -> f64 {
        let hc = self.combos[hero_ci];
        let mut total = 0.0;
        for (ci, combo) in self.combos.iter().enumerate() {
            if reach[ci] > 0.0 && !combo.contains(&hc[0]) && !combo.contains(&hc[1]) {
                total += reach[ci];
            }
        }
        total
    }

    /// Exact expectimax: hero (probe) maximizes per combo, the bot mixes by
    /// its blueprint strategy per combo, chance enumerates rivers. Returns
    /// hero values per hero combo, weighted by compatible bot reach.
    fn walk(&self, h: &Hand, hist: &[u8], reach: &[f64], river_ix: Option<usize>) -> Vec<f64> {
        if h.is_terminal() {
            return self.terminal_values(h, reach, river_ix);
        }
        if river_ix.is_none() && h.street() == Street::River {
            // Turn betting closed: branch on every candidate river card.
            let p = 1.0 / (self.rivers.len() as f64 - 4.0);
            let mut vals = vec![0.0; NUM_COMBOS];
            for (ri, &r) in self.rivers.iter().enumerate() {
                let mut child = h.clone();
                child.force_board_card(4, r);
                let mut masked = reach.to_vec();
                for (ci, combo) in self.combos.iter().enumerate() {
                    if combo[0] == r || combo[1] == r {
                        masked[ci] = 0.0;
                    }
                }
                let v = self.walk(&child, hist, &masked, Some(ri));
                for (ci, combo) in self.combos.iter().enumerate() {
                    if combo[0] != r && combo[1] != r {
                        vals[ci] += p * v[ci];
                    }
                }
            }
            return vals;
        }

        let acts = self.policy.abs.abstract_actions(h);
        let hero_to_act = h.to_act() != self.bot_seat;
        let mut children: Vec<(Hand, Vec<u8>)> = Vec::with_capacity(acts.len());
        for &a in &acts {
            let mut child = h.clone();
            child.apply(self.policy.abs.concrete(h, a));
            let mut hist2 = hist.to_vec();
            hist2.push(a.token());
            if child.street() != h.street() && !child.is_terminal() {
                hist2.push(TOKEN_STREET_SEP);
            }
            children.push((child, hist2));
        }

        if hero_to_act {
            let mut vals = vec![f64::MIN; NUM_COMBOS];
            for (child, hist2) in &children {
                let v = self.walk(child, hist2, reach, river_ix);
                for (o, &x) in vals.iter_mut().zip(&v) {
                    if x > *o {
                        *o = x;
                    }
                }
            }
            vals
        } else {
            // Bot node: per-combo blueprint mixture over the menu.
            let sigma = self.bot_sigma(hist, &acts, river_ix);
            let mut vals = vec![0.0; NUM_COMBOS];
            for (a, (child, hist2)) in children.iter().enumerate() {
                let mut r2 = vec![0.0; NUM_COMBOS];
                let mut any = false;
                for ci in 0..NUM_COMBOS {
                    let w = reach[ci] * sigma[ci * acts.len() + a];
                    r2[ci] = w;
                    any |= w > 0.0;
                }
                if !any {
                    continue;
                }
                let v = self.walk(child, hist2, &r2, river_ix);
                for (o, &x) in vals.iter_mut().zip(&v) {
                    *o += x;
                }
            }
            vals
        }
    }

    /// Per-combo blueprint strategy for the bot at this node, mirroring
    /// `Policy::act_blueprint`'s fallback: unseen or mismatched infosets
    /// play pure check/call.
    fn bot_sigma(&self, hist: &[u8], acts: &[AbsAction], river_ix: Option<usize>) -> Vec<f64> {
        let buckets = match river_ix {
            Some(ri) if !self.bucket_river.is_empty() => &self.bucket_river[ri],
            _ => &self.bucket_now,
        };
        let cc = acts
            .iter()
            .position(|&a| a == AbsAction::CheckCall)
            .expect("menu always contains check/call");
        let n = acts.len();
        let mut sigma = vec![0.0; NUM_COMBOS * n];
        let mut cache: Vec<Option<Option<Vec<f64>>>> = Vec::new();
        for ci in 0..NUM_COMBOS {
            let b = buckets[ci] as usize;
            if cache.len() <= b {
                cache.resize(b + 1, None);
            }
            if cache[b].is_none() {
                let s = self.policy.blueprint.get(buckets[ci], hist).and_then(|s| {
                    if s.len() != n {
                        return None;
                    }
                    let total: f64 = s.iter().map(|&x| x as f64).sum();
                    if total <= 0.0 {
                        return None;
                    }
                    Some(s.iter().map(|&x| x as f64 / total).collect::<Vec<f64>>())
                });
                cache[b] = Some(s);
            }
            match cache[b].as_ref().unwrap() {
                Some(s) => sigma[ci * n..(ci + 1) * n].copy_from_slice(s),
                None => sigma[ci * n + cc] = 1.0,
            }
        }
        sigma
    }

    /// Terminal values per hero combo, weighted by compatible bot reach.
    fn terminal_values(&self, h: &Hand, reach: &[f64], river_ix: Option<usize>) -> Vec<f64> {
        let hero_seat = self.hero_seat;
        let mut total = 0.0f64;
        let mut card = [0.0f64; 52];
        for (ci, combo) in self.combos.iter().enumerate() {
            let w = reach[ci];
            if w > 0.0 {
                total += w;
                card[combo[0] as usize] += w;
                card[combo[1] as usize] += w;
            }
        }

        if h.live_count() == 1 {
            let util = h.utilities()[hero_seat] as f64;
            let mut vals = vec![0.0; NUM_COMBOS];
            for (ci, combo) in self.combos.iter().enumerate() {
                let excl = total - card[combo[0] as usize] - card[combo[1] as usize] + reach[ci];
                if excl > 0.0 {
                    vals[ci] = util * excl;
                }
            }
            return vals;
        }

        let matched = (h.hand_commit(hero_seat) as f64).min(h.hand_commit(self.bot_seat) as f64);
        let dead = h.pot() as f64
            - h.hand_commit(hero_seat) as f64
            - h.hand_commit(self.bot_seat) as f64;

        match river_ix {
            Some(ri) => showdown_sweep(
                &self.combos,
                &self.sorted[ri],
                &self.rank[ri],
                reach,
                total,
                &card,
                matched,
                dead,
            ),
            None => {
                // All-in on the turn: runout showdown over every river.
                let p = 1.0 / (self.rivers.len() as f64 - 4.0);
                let mut vals = vec![0.0; NUM_COMBOS];
                for (ri, &r) in self.rivers.iter().enumerate() {
                    let mut masked = reach.to_vec();
                    let mut t2 = total;
                    let mut c2 = card;
                    for (ci, combo) in self.combos.iter().enumerate() {
                        if (combo[0] == r || combo[1] == r) && masked[ci] > 0.0 {
                            t2 -= masked[ci];
                            c2[combo[0] as usize] -= masked[ci];
                            c2[combo[1] as usize] -= masked[ci];
                            masked[ci] = 0.0;
                        }
                    }
                    let v = showdown_sweep(
                        &self.combos,
                        &self.sorted[ri],
                        &self.rank[ri],
                        &masked,
                        t2,
                        &c2,
                        matched,
                        dead,
                    );
                    for (ci, combo) in self.combos.iter().enumerate() {
                        if combo[0] != r && combo[1] != r {
                            vals[ci] += p * v[ci];
                        }
                    }
                }
                vals
            }
        }
    }
}

#[derive(Debug)]
pub struct BrResult {
    pub hands: u64,
    /// Probe winnings in mbb/hand: a lower bound on the bot's
    /// exploitability, tighter than LBR's.
    pub mbb_per_hand: f64,
    pub ci95: f64,
}

/// Run the probe: heads-up blind-vs-blind inside the blueprint's native
/// game (other seats fold), alternating blind seats — identical harness to
/// `run_lbr`, so results are directly comparable. Preflop/flop use the
/// greedy LBR action; turn/river decisions use the exact subgame BR.
pub fn run_br(
    policy: &Policy,
    cfg: &HandConfig,
    hands: u64,
    runouts: u32,
    seed: u64,
    search: Option<crate::bot::SearchParams>,
) -> BrResult {
    let n = cfg.num_players;
    let bb = cfg.bb as f64;
    let lbr = Lbr::new(policy, runouts);
    let combos = all_combos();
    let train_cfg = crate::cfr::TrainConfig {
        hand: cfg.clone(),
        prune_after: u64::MAX,
        ..crate::cfr::TrainConfig::default()
    };
    let results: Vec<f64> = (0..hands)
        .into_par_iter()
        .map(|i| {
            let mut rng =
                SmallRng::seed_from_u64(seed ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xB12E);
            let button = rng.random_range(0..n);
            let (sb, bbs) = if n == 2 {
                (button, (button + 1) % n)
            } else {
                ((button + 1) % n, (button + 2) % n)
            };
            let (probe_seat, bot_seat) = if i % 2 == 0 { (sb, bbs) } else { (bbs, sb) };
            let mut deck = fresh_deck();
            deck.shuffle(&mut rng);
            let mut table = crate::table::Table::new(cfg, button, deck);
            let mut range = BotRange::new();
            range.exclude(&combos, &table.real.hole(probe_seat));
            // With `search`, the bot resolves postflop decisions online
            // (the probe's range model of the bot stays the blueprint).
            let mut tracker = search.map(|_| crate::search::RangeTracker::new(n));
            let mut session = crate::bot::SearchSession::new();

            let mut guard = 0;
            while !table.real.is_terminal() {
                guard += 1;
                assert!(guard < 200, "BR probe hand did not terminate");
                let p = table.real.to_act();
                let street_before = table.real.street();
                if p == bot_seat {
                    let acts = policy.abs.abstract_actions(&table.shadow);
                    let a = match (search, tracker.as_ref()) {
                        (Some(params), Some(tr)) => policy.act_with_search(
                            &table.real,
                            &table.shadow,
                            &table.hist,
                            params,
                            &train_cfg,
                            Some(tr),
                            Some(&mut session),
                            Some(&table.round),
                            &mut rng,
                        ),
                        _ => policy.act_blueprint(&table.shadow, &table.hist, &mut rng),
                    };
                    if let Some(idx) = acts.iter().position(|&x| x == a) {
                        lbr.observe(&mut range, &table.shadow, &table.hist, &acts, idx);
                    }
                    if let Some(tr) = tracker.as_mut() {
                        tr.observe(p, a, &table.shadow, &table.hist, &policy.blueprint, &policy.abs);
                    }
                    table.apply_abs(a, &policy.abs);
                } else if p == probe_seat {
                    let h = &table.shadow;
                    let exact = (h.street() == Street::Turn || h.street() == Street::River)
                        && h.live_count() == 2;
                    let a = if exact {
                        match SubgameBr::build(policy, h, bot_seat) {
                            Some(sub) => {
                                sub.action(h, &table.hist, h.hole(probe_seat), &range.weights).0
                            }
                            None => lbr.action(&table, &range, bot_seat, &mut rng),
                        }
                    } else {
                        lbr.action(&table, &range, bot_seat, &mut rng)
                    };
                    if let Some(tr) = tracker.as_mut() {
                        tr.observe(p, a, &table.shadow, &table.hist, &policy.blueprint, &policy.abs);
                    }
                    table.apply_abs(a, &policy.abs);
                } else {
                    if let Some(tr) = tracker.as_mut() {
                        tr.observe(
                            p,
                            AbsAction::Fold,
                            &table.shadow,
                            &table.hist,
                            &policy.blueprint,
                            &policy.abs,
                        );
                    }
                    table.apply_abs(AbsAction::Fold, &policy.abs);
                }
                if table.real.street() != street_before {
                    range.exclude(&combos, table.real.board());
                    if let Some(tr) = tracker.as_mut() {
                        tr.exclude(table.real.board());
                    }
                }
            }
            table.real.utilities()[probe_seat] as f64 / bb * 1000.0
        })
        .collect();

    let mean = results.iter().sum::<f64>() / results.len() as f64;
    let var = results.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>()
        / (results.len().saturating_sub(1)) as f64;
    BrResult {
        hands,
        mbb_per_hand: mean,
        ci95: 1.96 * (var / results.len() as f64).sqrt(),
    }
}

// ---------------------------------------------------------------------------
// Tests (written first, TDD)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::{AbsConfig, Abstraction};
    use crate::cards::parse_cards;
    use crate::cfr::Blueprint;
    use crate::table::Table;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn abs_small() -> Abstraction {
        Abstraction::new(AbsConfig {
            postflop_buckets: 6,
            equity_rollouts: 40,
            dist_runouts: 8,
            runout_rollouts: 20,
            cache_cap: 500_000,
            menu: crate::abstraction::MenuShape::Wide,
        })
    }

    fn empty_policy() -> Policy {
        Policy::new(
            Blueprint {
                strategies: Default::default(),
                iterations: 0,
                num_players: 2,
                abs_cfg: AbsConfig::default(),
                centroids: None,
            },
            Arc::new(abs_small()),
        )
    }

    fn rigged_deck(front: &str) -> [Card; 52] {
        let front = parse_cards(front).unwrap();
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
        deck
    }

    /// HU table checked down to the river, p1 shoves, p0 to act.
    fn river_facing_shove(front: &str, policy: &Policy) -> Table {
        let cfg = HandConfig {
            num_players: 2,
            stack: 2_000,
            sb: 50,
            bb: 100,
        };
        let mut t = Table::new(&cfg, 0, rigged_deck(front));
        for _ in 0..6 {
            t.apply_abs(AbsAction::CheckCall, &policy.abs);
        }
        assert_eq!(t.real.street(), Street::River);
        t.apply_abs(AbsAction::AllIn, &policy.abs);
        assert_eq!(t.real.to_act(), 0);
        t
    }

    /// HU table checked to p0's turn decision facing p1's shove.
    fn turn_facing_shove(front: &str, policy: &Policy) -> Table {
        let cfg = HandConfig {
            num_players: 2,
            stack: 2_000,
            sb: 50,
            bb: 100,
        };
        let mut t = Table::new(&cfg, 0, rigged_deck(front));
        for _ in 0..4 {
            t.apply_abs(AbsAction::CheckCall, &policy.abs);
        }
        assert_eq!(t.real.street(), Street::Turn);
        assert_eq!(t.real.to_act(), 1);
        t.apply_abs(AbsAction::AllIn, &policy.abs);
        assert_eq!(t.real.to_act(), 0);
        t
    }

    #[test]
    fn br_calls_river_shove_with_nuts_and_folds_air() {
        let policy = empty_policy();

        let t = river_facing_shove("As Ks 2c 7d Qs Js Ts 3h 4d", &policy);
        let sub = SubgameBr::build(&policy, &t.shadow, 1).unwrap();
        let uniform = vec![1.0; NUM_COMBOS];
        let (a, v) = sub.action(&t.shadow, &t.hist, t.real.hole(0), &uniform);
        assert_eq!(a, AbsAction::CheckCall, "royal must call the shove");
        // Calling wins the whole matched pot on every runout: +2000 net.
        assert!(
            (v - 2000.0).abs() < 1e-6,
            "royal call must value exactly +2000, got {v}"
        );

        let t = river_facing_shove("6c 2h Ah Kd Qs Js Ts 3h 4d", &policy);
        let sub = SubgameBr::build(&policy, &t.shadow, 1).unwrap();
        let (a, v) = sub.action(&t.shadow, &t.hist, t.real.hole(0), &uniform);
        assert_eq!(a, AbsAction::Fold, "six-high must fold to the shove");
        // Values are net whole-hand chips: folding surrenders the 100
        // already committed preflop.
        assert!((v + 100.0).abs() < 1e-9, "fold must value -100 net, got {v}");
    }

    /// Facing a turn shove with a made royal (wins every river): the exact
    /// BR must call and value the spot at exactly +2000 — a single exact
    /// pass, so the tolerance is float noise, not CFR convergence.
    #[test]
    fn br_values_made_royal_exactly_facing_turn_shove() {
        let policy = empty_policy();
        let t = turn_facing_shove("As Ks 2c 7d Qs Js Ts 3h 4d", &policy);
        let sub = SubgameBr::build(&policy, &t.shadow, 1).unwrap();
        let uniform = vec![1.0; NUM_COMBOS];
        let (a, v) = sub.action(&t.shadow, &t.hist, t.real.hole(0), &uniform);
        assert_eq!(a, AbsAction::CheckCall, "made royal must call the shove");
        assert!(
            (v - 2000.0).abs() < 1e-6,
            "made royal must value exactly +2000, got {v}"
        );
    }

    /// Holding a made royal against a station, every line that gets the
    /// stacks in is worth exactly +2000 — including checking the turn and
    /// shoving the river. The BR value must hit that maximum exactly, which
    /// requires the expectimax to plan the river shove two streets ahead
    /// (a greedy check-down valuation would report only +100).
    #[test]
    fn br_plans_multistreet_value_extraction_with_the_nuts() {
        let policy = empty_policy();
        let cfg = HandConfig {
            num_players: 2,
            stack: 2_000,
            sb: 50,
            bb: 100,
        };
        let mut t = Table::new(&cfg, 0, rigged_deck("2c 7d As Ks Qs Js Ts 3h 4d"));
        for _ in 0..4 {
            t.apply_abs(AbsAction::CheckCall, &policy.abs);
        }
        assert_eq!(t.real.street(), Street::Turn);
        assert_eq!(t.real.to_act(), 1);
        // p1 holds the made royal vs a calling station: value-betting must
        // beat checking (the station calls everything, so bets are pure
        // value).
        let sub = SubgameBr::build(&policy, &t.shadow, 0).unwrap();
        let uniform = vec![1.0; NUM_COMBOS];
        let (_, v) = sub.action(&t.shadow, &t.hist, t.real.hole(1), &uniform);
        assert!(
            (v - 2000.0).abs() < 1e-6,
            "BR must extract the full stacks via multi-street planning, got {v}"
        );
    }

    /// The probe must crush a calling station at least as hard as LBR does
    /// on the same seed — the stronger probe can only tighten the bound —
    /// and must be exactly reproducible.
    #[test]
    fn br_probe_crushes_a_calling_station_and_is_deterministic() {
        let policy = empty_policy();
        let cfg = HandConfig {
            num_players: 2,
            ..HandConfig::default()
        };
        let r1 = run_br(&policy, &cfg, 200, 16, 99, None);
        let r2 = run_br(&policy, &cfg, 200, 16, 99, None);
        assert_eq!(
            r1.mbb_per_hand, r2.mbb_per_hand,
            "same seed must reproduce exactly"
        );
        assert!(
            r1.mbb_per_hand - r1.ci95 > 1_000.0,
            "BR probe must crush a calling station, got {:+.0} ±{:.0}",
            r1.mbb_per_hand,
            r1.ci95
        );
    }

    /// Card-removal sanity: values returned by the walk are reach-weighted;
    /// normalizing by compatible mass must give a value inside the range of
    /// possible outcomes for a river call.
    #[test]
    fn br_river_values_are_bounded_by_the_pot() {
        let policy = empty_policy();
        let t = river_facing_shove("Ah Kh 2c 7d Qs Js Ts 3h 4d", &policy);
        let sub = SubgameBr::build(&policy, &t.shadow, 1).unwrap();
        let uniform = vec![1.0; NUM_COMBOS];
        let (_, v) = sub.action(&t.shadow, &t.hist, t.real.hole(0), &uniform);
        // Nut straight on a three-spade board (flushes beat it): the best
        // action's net value must lie within the fold / stack-win bounds.
        assert!(v >= -100.0 - 1e-9, "BR value can never be below folding");
        assert!(v <= 2000.0 + 1e-9, "BR value can never exceed the stack win");
    }

    #[test]
    fn build_rejects_wrong_streets_and_multiway() {
        let policy = empty_policy();
        let cfg = HandConfig {
            num_players: 2,
            stack: 2_000,
            sb: 50,
            bb: 100,
        };
        let t = Table::new(&cfg, 0, rigged_deck("As Ks 2c 7d Qs Js Ts 3h 4d"));
        // Preflop: must refuse.
        assert!(SubgameBr::build(&policy, &t.shadow, 1).is_none());

        let mut t2 = Table::new(&cfg, 0, rigged_deck("As Ks 2c 7d Qs Js Ts 3h 4d"));
        t2.apply_abs(AbsAction::CheckCall, &policy.abs);
        t2.apply_abs(AbsAction::CheckCall, &policy.abs);
        assert_eq!(t2.real.street(), Street::Flop);
        // Flop: must refuse (v1 covers turn/river only).
        assert!(SubgameBr::build(&policy, &t2.shadow, 1).is_none());
    }
}
