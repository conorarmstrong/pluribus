//! No-limit Texas Hold'em hand state machine for 2-6 players.
//!
//! Correctness-critical invariants (all covered by tests below):
//! - Every hand terminates (the Python predecessor infinite-looped on multi-way all-ins).
//! - Utilities are net chips and zero-sum across seats.
//! - Side pots: a short stack can only win chips it covered.
//! - Min-raise rules, big-blind option, heads-up blind order.

use crate::cards::Card;
use crate::eval::eval_hole_board;
use rand::rngs::SmallRng;
use rand::Rng;

pub const MAX_PLAYERS: usize = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Street {
    Preflop = 0,
    Flop = 1,
    Turn = 2,
    River = 3,
}

impl Street {
    pub fn next(self) -> Street {
        match self {
            Street::Preflop => Street::Flop,
            Street::Flop => Street::Turn,
            Street::Turn => Street::River,
            Street::River => Street::River,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayerAction {
    Fold,
    /// Check when nothing to call, otherwise call (all-in call if short).
    CheckCall,
    /// Raise TO this total street commitment. Clamped into legal bounds.
    RaiseTo(u32),
}

#[derive(Debug, Clone)]
pub struct HandConfig {
    pub num_players: usize,
    pub stack: u32,
    pub sb: u32,
    pub bb: u32,
}

impl Default for HandConfig {
    fn default() -> Self {
        HandConfig {
            num_players: 6,
            stack: 10_000,
            sb: 50,
            bb: 100,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Hand {
    n: usize,
    button: usize,
    bb: u32,
    holes: [[Card; 2]; MAX_PLAYERS],
    board: [Card; 5],
    board_len: usize,
    deck: [Card; 52],
    stacks: [u32; MAX_PLAYERS],
    street_commit: [u32; MAX_PLAYERS],
    hand_commit: [u32; MAX_PLAYERS],
    folded: [bool; MAX_PLAYERS],
    allin: [bool; MAX_PLAYERS],
    acted: [bool; MAX_PLAYERS],
    street: Street,
    to_act: usize,
    current_bet: u32,
    min_raise_inc: u32,
    n_raises: u8,
    terminal: bool,
}

impl Hand {
    /// Deal a new hand. Holes: player p gets deck[2p], deck[2p+1].
    /// Board comes from deck[2n..2n+5] as streets are dealt.
    pub fn new(cfg: &HandConfig, button: usize, deck: [Card; 52]) -> Hand {
        let stacks = vec![cfg.stack; cfg.num_players];
        Hand::new_with_stacks(cfg, button, deck, &stacks)
    }

    pub fn new_with_stacks(
        cfg: &HandConfig,
        button: usize,
        deck: [Card; 52],
        stacks: &[u32],
    ) -> Hand {
        let n = cfg.num_players;
        assert!((2..=MAX_PLAYERS).contains(&n));
        assert_eq!(stacks.len(), n);
        assert!(button < n);

        let mut holes = [[0u8; 2]; MAX_PLAYERS];
        for (p, hole) in holes.iter_mut().enumerate().take(n) {
            hole[0] = deck[2 * p];
            hole[1] = deck[2 * p + 1];
        }

        let mut h = Hand {
            n,
            button,
            bb: cfg.bb,
            holes,
            board: [0; 5],
            board_len: 0,
            deck,
            stacks: [0; MAX_PLAYERS],
            street_commit: [0; MAX_PLAYERS],
            hand_commit: [0; MAX_PLAYERS],
            folded: [false; MAX_PLAYERS],
            allin: [false; MAX_PLAYERS],
            acted: [false; MAX_PLAYERS],
            street: Street::Preflop,
            to_act: 0,
            current_bet: 0,
            min_raise_inc: cfg.bb,
            n_raises: 0,
            terminal: false,
        };
        h.stacks[..n].copy_from_slice(stacks);

        // Blinds. Heads-up: the button posts the small blind.
        let (sb_seat, bb_seat) = if n == 2 {
            (button, (button + 1) % n)
        } else {
            ((button + 1) % n, (button + 2) % n)
        };
        h.commit(sb_seat, cfg.sb.min(h.stacks[sb_seat]));
        h.commit(bb_seat, cfg.bb.min(h.stacks[bb_seat]));
        h.current_bet = h.street_commit[..n].iter().copied().max().unwrap();

        match h.next_actor(bb_seat) {
            Some(p) => h.to_act = p,
            None => h.fast_forward_to_showdown(),
        }
        h
    }

    pub fn apply(&mut self, a: PlayerAction) {
        debug_assert!(!self.terminal, "apply on terminal hand");
        let p = self.to_act;
        debug_assert!(!self.folded[p] && !self.allin[p]);

        match a {
            PlayerAction::Fold => {
                // Folding with nothing to call is treated as a check.
                if self.to_call() == 0 {
                    self.acted[p] = true;
                } else {
                    self.folded[p] = true;
                }
            }
            PlayerAction::CheckCall => {
                let pay = self.to_call().min(self.stacks[p]);
                self.commit(p, pay);
                self.acted[p] = true;
            }
            PlayerAction::RaiseTo(x) => match self.raise_bounds() {
                None => {
                    // Raising impossible: degrade to call.
                    let pay = self.to_call().min(self.stacks[p]);
                    self.commit(p, pay);
                    self.acted[p] = true;
                }
                Some((lo, hi)) => {
                    let x = x.clamp(lo, hi);
                    let inc = x - self.current_bet;
                    let pay = x - self.street_commit[p];
                    self.commit(p, pay);
                    if inc >= self.min_raise_inc {
                        // Full raise reopens action for everyone else.
                        self.min_raise_inc = inc;
                        for q in 0..self.n {
                            if q != p {
                                self.acted[q] = false;
                            }
                        }
                    }
                    self.current_bet = x;
                    self.n_raises = self.n_raises.saturating_add(1);
                    self.acted[p] = true;
                }
            },
        }

        self.advance(p);
    }

    /// Net chip result per seat. Only valid when terminal.
    pub fn utilities(&self) -> [i64; MAX_PLAYERS] {
        debug_assert!(self.terminal);
        let mut winnings = [0u64; MAX_PLAYERS];

        let live: Vec<usize> = (0..self.n).filter(|&p| !self.folded[p]).collect();
        if live.len() == 1 {
            winnings[live[0]] = self.pot() as u64;
        } else {
            // Showdown across side-pot layers.
            debug_assert_eq!(self.board_len, 5);
            let mut strength = [0u32; MAX_PLAYERS];
            for &p in &live {
                strength[p] = eval_hole_board(&self.holes[p], &self.board);
            }

            let mut levels: Vec<u32> = self.hand_commit[..self.n]
                .iter()
                .copied()
                .filter(|&c| c > 0)
                .collect();
            levels.sort_unstable();
            levels.dedup();

            let mut prev = 0u32;
            for &level in &levels {
                let layer: u64 = self.hand_commit[..self.n]
                    .iter()
                    .map(|&c| (c.min(level).saturating_sub(prev)) as u64)
                    .sum();
                let eligible: Vec<usize> = live
                    .iter()
                    .copied()
                    .filter(|&p| self.hand_commit[p] >= level)
                    .collect();
                debug_assert!(!eligible.is_empty(), "orphaned side-pot layer");
                let best = eligible.iter().map(|&p| strength[p]).max().unwrap();
                let winners: Vec<usize> = eligible
                    .into_iter()
                    .filter(|&p| strength[p] == best)
                    .collect();
                let share = layer / winners.len() as u64;
                let mut rem = layer - share * winners.len() as u64;
                for &w in &winners {
                    winnings[w] += share;
                    if rem > 0 {
                        winnings[w] += 1;
                        rem -= 1;
                    }
                }
                prev = level;
            }
        }

        let mut u = [0i64; MAX_PLAYERS];
        for p in 0..self.n {
            u[p] = winnings[p] as i64 - self.hand_commit[p] as i64;
        }
        u
    }

    /// Re-randomize hidden information from one player's point of view:
    /// all other players' hole cards and the undealt part of the board are
    /// redrawn from the unseen cards. With `keep = None`, every seat's holes
    /// are resampled (used by subgame resolving, where the solver trains all
    /// buckets and the hero's real bucket is queried afterwards).
    pub fn resample_hidden(&mut self, keep: Option<usize>, rng: &mut SmallRng) {
        let mut used = [false; 52];
        for &c in self.board() {
            used[c as usize] = true;
        }
        if let Some(p) = keep {
            used[self.holes[p][0] as usize] = true;
            used[self.holes[p][1] as usize] = true;
        }
        let mut unseen: Vec<Card> = (0..52u8).filter(|&c| !used[c as usize]).collect();
        // Partial Fisher-Yates over however many cards we need.
        let need = 2 * self.n + (5 - self.board_len);
        for k in 0..need.min(unseen.len()) {
            let j = rng.random_range(k..unseen.len());
            unseen.swap(k, j);
        }
        let mut next = 0;
        for p in 0..self.n {
            if Some(p) != keep {
                self.holes[p] = [unseen[next], unseen[next + 1]];
                next += 2;
            }
        }
        let base = 2 * self.n;
        for i in self.board_len..5 {
            self.deck[base + i] = unseen[next];
            next += 1;
        }
    }

    /// Like `resample_hidden`, but seats with `Some(combo)` receive exactly
    /// those hole cards (used by range-weighted subgame sampling); seats with
    /// `None` and the undealt board are drawn uniformly from the rest.
    /// Requested combos must be distinct and off the dealt board.
    pub fn resample_hidden_with(
        &mut self,
        want: &[Option<[Card; 2]>; MAX_PLAYERS],
        rng: &mut SmallRng,
    ) {
        let mut used = [false; 52];
        for &c in self.board() {
            used[c as usize] = true;
        }
        for w in want.iter().take(self.n).flatten() {
            for &c in w {
                debug_assert!(!used[c as usize], "requested card already in use");
                used[c as usize] = true;
            }
        }
        let mut unseen: Vec<Card> = (0..52u8).filter(|&c| !used[c as usize]).collect();
        let free_seats = (0..self.n).filter(|&p| want[p].is_none()).count();
        let need = 2 * free_seats + (5 - self.board_len);
        for k in 0..need.min(unseen.len()) {
            let j = rng.random_range(k..unseen.len());
            unseen.swap(k, j);
        }
        let mut next = 0;
        for (p, w) in want.iter().enumerate().take(self.n) {
            match w {
                Some(w) => self.holes[p] = *w,
                None => {
                    self.holes[p] = [unseen[next], unseen[next + 1]];
                    next += 2;
                }
            }
        }
        let base = 2 * self.n;
        for i in self.board_len..5 {
            self.deck[base + i] = unseen[next];
            next += 1;
        }
    }

    /// Overwrite the river card (board slot 4). For range-vector solvers
    /// that enumerate public river outcomes; holes and deck are untouched,
    /// so the hand must not be used for engine showdowns afterwards.
    pub fn force_river(&mut self, c: Card) {
        debug_assert_eq!(self.board_len, 5);
        self.board[4] = c;
    }

    // --- internals -------------------------------------------------------

    fn commit(&mut self, p: usize, amount: u32) {
        debug_assert!(amount <= self.stacks[p]);
        self.stacks[p] -= amount;
        self.street_commit[p] += amount;
        self.hand_commit[p] += amount;
        if self.stacks[p] == 0 {
            self.allin[p] = true;
        }
    }

    /// Next seat after `from` that still needs to act this street.
    fn next_actor(&self, from: usize) -> Option<usize> {
        for i in 1..=self.n {
            let q = (from + i) % self.n;
            if !self.folded[q]
                && !self.allin[q]
                && (!self.acted[q] || self.street_commit[q] != self.current_bet)
            {
                return Some(q);
            }
        }
        None
    }

    fn street_over(&self) -> bool {
        (0..self.n).all(|p| {
            self.folded[p]
                || self.allin[p]
                || (self.acted[p] && self.street_commit[p] == self.current_bet)
        })
    }

    fn advance(&mut self, just_acted: usize) {
        if self.live_count() == 1 {
            self.terminal = true;
            return;
        }
        if !self.street_over() {
            self.to_act = self
                .next_actor(just_acted)
                .expect("street not over but nobody to act");
            return;
        }
        if self.street == Street::River {
            self.terminal = true;
            return;
        }
        self.begin_street(self.street.next());
    }

    fn begin_street(&mut self, street: Street) {
        self.street = street;
        self.deal_board_for_street();
        self.street_commit = [0; MAX_PLAYERS];
        self.current_bet = 0;
        self.min_raise_inc = self.bb;
        self.n_raises = 0;
        self.acted = [false; MAX_PLAYERS];

        let can_act = (0..self.n)
            .filter(|&p| !self.folded[p] && !self.allin[p])
            .count();
        if can_act <= 1 {
            // No more betting possible: run the board out.
            self.fast_forward_to_showdown();
            return;
        }
        // First to act: left of the button (heads-up this is the non-button).
        self.to_act = self
            .next_actor(self.button)
            .expect("betting street with nobody to act");
    }

    fn deal_board_for_street(&mut self) {
        let base = 2 * self.n;
        let want = match self.street {
            Street::Preflop => 0,
            Street::Flop => 3,
            Street::Turn => 4,
            Street::River => 5,
        };
        while self.board_len < want {
            self.board[self.board_len] = self.deck[base + self.board_len];
            self.board_len += 1;
        }
    }

    fn fast_forward_to_showdown(&mut self) {
        while self.street != Street::River {
            self.street = self.street.next();
        }
        self.deal_board_for_street();
        self.terminal = true;
    }

    // --- accessors -------------------------------------------------------

    pub fn num_players(&self) -> usize {
        self.n
    }
    #[allow(dead_code)]
    pub fn button(&self) -> usize {
        self.button
    }
    pub fn to_act(&self) -> usize {
        self.to_act
    }
    pub fn street(&self) -> Street {
        self.street
    }
    pub fn board(&self) -> &[Card] {
        &self.board[..self.board_len]
    }
    pub fn hole(&self, p: usize) -> [Card; 2] {
        self.holes[p]
    }
    pub fn is_terminal(&self) -> bool {
        self.terminal
    }
    pub fn current_bet(&self) -> u32 {
        self.current_bet
    }
    pub fn street_commit(&self, p: usize) -> u32 {
        self.street_commit[p]
    }
    #[allow(dead_code)]
    pub fn hand_commit(&self, p: usize) -> u32 {
        self.hand_commit[p]
    }
    pub fn stack(&self, p: usize) -> u32 {
        self.stacks[p]
    }
    pub fn folded(&self, p: usize) -> bool {
        self.folded[p]
    }
    #[allow(dead_code)]
    pub fn all_in(&self, p: usize) -> bool {
        self.allin[p]
    }
    pub fn n_raises(&self) -> u8 {
        self.n_raises
    }
    #[allow(dead_code)]
    pub fn big_blind(&self) -> u32 {
        self.bb
    }

    /// Total chips in the pot (all commitments this hand).
    pub fn pot(&self) -> u32 {
        self.hand_commit[..self.n].iter().sum()
    }

    /// Amount the player to act must add to call.
    pub fn to_call(&self) -> u32 {
        self.current_bet - self.street_commit[self.to_act]
    }

    pub fn can_check(&self) -> bool {
        self.to_call() == 0
    }

    pub fn live_count(&self) -> usize {
        (0..self.n).filter(|&p| !self.folded[p]).count()
    }

    /// Legal (min_raise_to, max_raise_to) for the player to act, or None if
    /// raising is impossible (calling would already be all-in or match exactly).
    pub fn raise_bounds(&self) -> Option<(u32, u32)> {
        let p = self.to_act;
        let max_to = self.street_commit[p] + self.stacks[p];
        if max_to <= self.current_bet {
            return None;
        }
        let min_to = (self.current_bet + self.min_raise_inc).min(max_to);
        Some((min_to, max_to))
    }
}

// ---------------------------------------------------------------------------
// Tests (written first, TDD)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::{fresh_deck, parse_cards};
    use rand::seq::SliceRandom;
    use rand::{Rng, SeedableRng};
    use PlayerAction::{CheckCall, Fold, RaiseTo};

    /// Deck whose first cards are exactly `front` (parsed), rest arbitrary.
    fn deck_with_front(front: &str) -> [Card; 52] {
        let front = parse_cards(front).unwrap();
        let mut deck = [0u8; 52];
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

    fn cfg(n: usize) -> HandConfig {
        HandConfig {
            num_players: n,
            ..HandConfig::default()
        }
    }

    #[test]
    fn blinds_posted_and_utg_first() {
        let h = Hand::new(&cfg(6), 0, fresh_deck());
        assert_eq!(h.to_act(), 3); // UTG
        assert_eq!(h.pot(), 150);
        assert_eq!(h.current_bet(), 100);
        assert_eq!(h.stack(1), 9_950);
        assert_eq!(h.stack(2), 9_900);
        assert_eq!(h.street(), Street::Preflop);
        assert_eq!(h.to_call(), 100);
        assert!(!h.is_terminal());
    }

    #[test]
    fn fold_around_bb_wins_blinds() {
        let mut h = Hand::new(&cfg(6), 0, fresh_deck());
        for _ in 0..5 {
            h.apply(Fold); // 3, 4, 5, 0, SB
        }
        assert!(h.is_terminal());
        let u = h.utilities();
        assert_eq!(u[2], 50); // BB wins SB's 50
        assert_eq!(u[1], -50);
        assert_eq!(u.iter().sum::<i64>(), 0);
    }

    #[test]
    fn bb_gets_option_then_flop_order() {
        let mut h = Hand::new(&cfg(6), 0, fresh_deck());
        for _ in 0..4 {
            h.apply(Fold); // 3, 4, 5, 0
        }
        h.apply(CheckCall); // SB completes
        assert!(!h.is_terminal(), "BB must get the option");
        assert_eq!(h.to_act(), 2);
        assert!(h.can_check());
        h.apply(CheckCall); // BB checks
        assert_eq!(h.street(), Street::Flop);
        assert_eq!(h.board().len(), 3);
        assert_eq!(h.to_act(), 1); // SB first postflop
        assert_eq!(h.current_bet(), 0);
    }

    #[test]
    fn heads_up_blind_order() {
        let mut h = Hand::new(&cfg(2), 0, fresh_deck());
        // Button posts SB and acts first preflop.
        assert_eq!(h.to_act(), 0);
        assert_eq!(h.to_call(), 50);
        h.apply(CheckCall);
        assert_eq!(h.to_act(), 1); // BB option
        h.apply(CheckCall);
        assert_eq!(h.street(), Street::Flop);
        assert_eq!(h.to_act(), 1); // non-button first postflop
    }

    #[test]
    fn min_raise_rules_and_clamping() {
        let mut h = Hand::new(&cfg(6), 0, fresh_deck());
        assert_eq!(h.raise_bounds(), Some((200, 10_000)));
        h.apply(RaiseTo(300)); // open to 300, increment 200
        assert_eq!(h.current_bet(), 300);
        assert_eq!(h.raise_bounds(), Some((500, 10_000)));
        h.apply(RaiseTo(400)); // below min: clamped up to 500
        assert_eq!(h.current_bet(), 500);
        // A raise beyond stack clamps to all-in.
        h.apply(RaiseTo(50_000));
        assert_eq!(h.current_bet(), 10_000);
        assert!(h.all_in(5));
    }

    #[test]
    fn check_down_to_showdown_best_hand_wins() {
        // p0 has AA and flops top set; everyone calls preflop and checks down.
        let deck = deck_with_front(
            "As Ah 2c 7d 2d 7h 2h 7s 3c 8d 3d 8h \
             Ac Kc Qd Js 9h",
        );
        let mut h = Hand::new(&cfg(6), 0, deck);
        for _ in 0..5 {
            h.apply(CheckCall); // 3,4,5,0 call; SB completes
        }
        h.apply(CheckCall); // BB checks
        for _ in 0..3 {
            assert!(!h.is_terminal());
            for _ in 0..6 {
                h.apply(CheckCall); // check around
            }
        }
        assert!(h.is_terminal());
        let u = h.utilities();
        assert_eq!(u[0], 500);
        for &x in &u[1..6] {
            assert_eq!(x, -100);
        }
    }

    /// Regression for the Python bug: multi-way all-in preflop must terminate
    /// immediately with a full runout, not loop forever.
    #[test]
    fn multiway_allin_fast_forwards_to_showdown() {
        let mut h = Hand::new(&cfg(6), 0, fresh_deck());
        h.apply(RaiseTo(10_000)); // UTG shoves
        for _ in 0..5 {
            assert!(!h.is_terminal());
            assert_eq!(h.raise_bounds(), None); // equal stacks: only call/fold
            h.apply(CheckCall);
        }
        assert!(h.is_terminal());
        assert_eq!(h.board().len(), 5);
        let u = h.utilities();
        assert_eq!(u.iter().sum::<i64>(), 0);
        for &x in &u[..6] {
            assert!(x >= -10_000);
        }
    }

    #[test]
    fn side_pot_short_stack_only_wins_main_pot() {
        // p0 short with the best hand, p1 second best, p2 worst.
        let deck = deck_with_front("As Ah Ks Kh 2c 7d 3c 9d Jh 4s 8c");
        let mut h = Hand::new_with_stacks(&cfg(3), 0, deck, &[1_000, 5_000, 5_000]);
        assert_eq!(h.to_act(), 0);
        h.apply(RaiseTo(1_000)); // short stack all-in
        h.apply(RaiseTo(5_000)); // SB shoves over
        h.apply(CheckCall); // BB calls all-in
        assert!(h.is_terminal());
        let u = h.utilities();
        // Main pot 3000 to p0 (AA); side pot 8000 to p1 (KK > 72o).
        assert_eq!(u[0], 2_000);
        assert_eq!(u[1], 3_000);
        assert_eq!(u[2], -5_000);
        assert_eq!(u.iter().sum::<i64>(), 0);
    }

    #[test]
    fn uncalled_bet_is_returned() {
        let mut h = Hand::new(&cfg(6), 0, fresh_deck());
        h.apply(RaiseTo(1_000));
        for _ in 0..5 {
            h.apply(Fold);
        }
        assert!(h.is_terminal());
        let u = h.utilities();
        assert_eq!(u[3], 150); // wins only the blinds; own 1000 returned
        assert_eq!(u[1], -50);
        assert_eq!(u[2], -100);
        assert_eq!(u.iter().sum::<i64>(), 0);
    }

    /// A short all-in raise still forces the earlier bettor to respond
    /// (the exact situation that never resolved in the Python engine).
    #[test]
    fn short_allin_raise_must_be_called() {
        let deck = deck_with_front("2c 7d 2d 7h As Ah 3c 9d Jh 4s 8c");
        let mut h = Hand::new_with_stacks(&cfg(3), 0, deck, &[5_000, 5_000, 250]);
        h.apply(RaiseTo(200)); // p0 min-raises
        h.apply(Fold); // SB folds
        assert_eq!(h.raise_bounds(), Some((250, 250))); // BB can only shove short
        h.apply(RaiseTo(250)); // short all-in raise
        assert!(!h.is_terminal(), "p0 must get to call the extra 50");
        assert_eq!(h.to_act(), 0);
        assert_eq!(h.to_call(), 50);
        h.apply(CheckCall);
        assert!(h.is_terminal()); // runout: p2's aces win
        let u = h.utilities();
        assert_eq!(u[2], 300); // pot 550 (250+250+50) minus own 250
        assert_eq!(u[0], -250);
        assert_eq!(u[1], -50);
    }

    #[test]
    fn resample_hidden_keeps_hero_and_board_no_duplicates() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(5);
        let mut deck = fresh_deck();
        deck.shuffle(&mut rng);
        let mut h = Hand::new(&cfg(6), 0, deck);
        // Get to the flop: everyone calls, BB checks.
        for _ in 0..6 {
            h.apply(CheckCall);
        }
        assert_eq!(h.street(), Street::Flop);
        let hero_hole = h.hole(2);
        let board_before: Vec<Card> = h.board().to_vec();

        for _ in 0..50 {
            h.resample_hidden(Some(2), &mut rng);
            assert_eq!(h.hole(2), hero_hole);
            assert_eq!(h.board(), &board_before[..]);
            // All dealt cards distinct, and future board draws distinct too.
            let mut seen = std::collections::HashSet::new();
            for p in 0..6 {
                for c in h.hole(p) {
                    assert!(seen.insert(c), "duplicate card after resample");
                }
            }
            for &c in h.board() {
                assert!(seen.insert(c));
            }
            // Run a clone to the river: full board must stay collision-free.
            let mut probe = h.clone();
            while !probe.is_terminal() {
                probe.apply(CheckCall);
            }
            for &c in &probe.board()[3..] {
                assert!(seen.insert(c), "future board card collides");
            }
        }
    }

    #[test]
    fn resample_hidden_with_places_requested_holes() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(6);
        let mut deck = fresh_deck();
        deck.shuffle(&mut rng);
        let mut h = Hand::new(&cfg(6), 0, deck);
        for _ in 0..6 {
            h.apply(CheckCall);
        }
        assert_eq!(h.street(), Street::Flop);
        let board_before: Vec<Card> = h.board().to_vec();

        // Pick two combos that don't collide with the board.
        let free: Vec<Card> = (0..52u8)
            .filter(|c| !board_before.contains(c))
            .collect();
        let (w1, w2) = ([free[0], free[1]], [free[2], free[3]]);

        for _ in 0..50 {
            let mut want: [Option<[Card; 2]>; MAX_PLAYERS] = [None; MAX_PLAYERS];
            want[1] = Some(w1);
            want[4] = Some(w2);
            h.resample_hidden_with(&want, &mut rng);
            assert_eq!(h.hole(1), w1);
            assert_eq!(h.hole(4), w2);
            assert_eq!(h.board(), &board_before[..]);
            let mut seen = std::collections::HashSet::new();
            for p in 0..6 {
                for c in h.hole(p) {
                    assert!(seen.insert(c), "duplicate card after targeted resample");
                }
            }
            for &c in h.board() {
                assert!(seen.insert(c));
            }
            let mut probe = h.clone();
            while !probe.is_terminal() {
                probe.apply(CheckCall);
            }
            for &c in &probe.board()[3..] {
                assert!(seen.insert(c), "future board card collides");
            }
        }
    }

    /// Fuzz: random legal action sequences always terminate quickly,
    /// utilities are zero-sum, and nobody loses more than their stack.
    #[test]
    fn fuzz_random_playouts_terminate_zero_sum() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
        for n in 2..=6usize {
            for _ in 0..2_000 {
                let mut deck = fresh_deck();
                deck.shuffle(&mut rng);
                let stacks: Vec<u32> =
                    (0..n).map(|_| rng.random_range(200..20_000)).collect();
                let button = rng.random_range(0..n);
                let mut h = Hand::new_with_stacks(&cfg(n), button, deck, &stacks);
                let mut steps = 0;
                while !h.is_terminal() {
                    steps += 1;
                    assert!(steps < 500, "hand did not terminate");
                    let r = rng.random_range(0..100);
                    let a = if r < 50 {
                        CheckCall
                    } else if r < 75 && h.to_call() > 0 {
                        Fold
                    } else if let Some((lo, hi)) = h.raise_bounds() {
                        RaiseTo(rng.random_range(lo..=hi))
                    } else {
                        CheckCall
                    };
                    h.apply(a);
                }
                let u = h.utilities();
                assert_eq!(u[..n].iter().sum::<i64>(), 0, "not zero-sum");
                for p in 0..n {
                    assert!(u[p] >= -(stacks[p] as i64), "lost more than stack");
                }
            }
        }
    }
}
