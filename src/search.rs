//! Range tracking for online subgame resolving.
//!
//! Each seat gets a probability weight over all 1326 hole-card combos,
//! updated by Bayes' rule after every observed action: the weight of a combo
//! is multiplied by the blueprint probability of the observed action at that
//! combo's infoset (with a floor, so no hand is ever ruled out entirely —
//! humans do things the blueprint wouldn't). Board cards zero out conflicting
//! combos. The subgame resolver then samples opponents' hidden cards from
//! these tracked ranges instead of uniformly — Pluribus's key search
//! ingredient.

use crate::abstraction::{AbsAction, Abstraction};
use crate::cards::Card;
use crate::cfr::Blueprint;
use crate::engine::{Hand, MAX_PLAYERS};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

pub const NUM_COMBOS: usize = 1326;

/// Bayes-update floor: minimum per-action probability so observed actions
/// never fully exclude a combo.
const OBS_FLOOR: f64 = 0.02;

#[inline]
pub fn combo_index(a: Card, b: Card) -> usize {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    hi as usize * (hi as usize - 1) / 2 + lo as usize
}

/// All 1326 combos, indexed by `combo_index`.
pub fn all_combos() -> Vec<[Card; 2]> {
    let mut v = Vec::with_capacity(NUM_COMBOS);
    for hi in 1..52u8 {
        for lo in 0..hi {
            v.push([lo, hi]);
        }
    }
    v
}

pub struct RangeTracker {
    n: usize,
    combos: Vec<[Card; 2]>,
    /// weights[seat][combo]; unnormalized.
    weights: Vec<Vec<f64>>,
}

impl RangeTracker {
    pub fn new(n: usize) -> Self {
        RangeTracker {
            n,
            combos: all_combos(),
            weights: vec![vec![1.0; NUM_COMBOS]; n],
        }
    }

    /// Zero out combos containing any of `cards` (e.g. newly dealt board
    /// cards) for every seat.
    pub fn exclude(&mut self, cards: &[Card]) {
        for (i, combo) in self.combos.iter().enumerate() {
            if combo.iter().any(|c| cards.contains(c)) {
                for w in &mut self.weights {
                    w[i] = 0.0;
                }
            }
        }
    }

    /// Raw (unnormalized) weights over all 1326 combos for a seat, indexed
    /// by `combo_index`. Feeds the river range-vector solver.
    pub fn seat_weights(&self, seat: usize) -> &[f64] {
        &self.weights[seat]
    }

    #[allow(dead_code)]
    pub fn weight(&self, seat: usize, hole: [Card; 2]) -> f64 {
        self.weights[seat][combo_index(hole[0], hole[1])]
    }

    /// Normalized probability of `hole` in `seat`'s tracked range.
    #[allow(dead_code)]
    pub fn prob(&self, seat: usize, hole: [Card; 2]) -> f64 {
        let total: f64 = self.weights[seat].iter().sum();
        if total <= 0.0 {
            return 0.0;
        }
        self.weight(seat, hole) / total
    }

    /// Bayes-update `seat`'s range after observing it take `taken` at the
    /// decision point `h` (pre-action state) with history `hist`: for every
    /// combo, multiply by the blueprint's probability of that action.
    pub fn observe(
        &mut self,
        seat: usize,
        taken: AbsAction,
        h: &Hand,
        hist: &[u8],
        bp: &Blueprint,
        abs: &Abstraction,
    ) {
        let acts = abs.abstract_actions(h);
        let Some(idx) = acts.iter().position(|&a| a == taken) else {
            return; // off-menu action (shouldn't happen): no update
        };
        let board = h.board().to_vec();
        let combos = &self.combos;
        let updates: Vec<f64> = self.weights[seat]
            .par_iter()
            .enumerate()
            .map(|(ci, &w)| {
                if w <= 0.0 {
                    return w;
                }
                let mut rng = SmallRng::seed_from_u64(0xB0B0 ^ ci as u64);
                let bucket = abs.bucket(combos[ci], &board, &mut rng);
                let p = match bp.get(bucket, hist) {
                    Some(s) if s.len() == acts.len() => (s[idx] as f64).max(OBS_FLOOR),
                    _ => 1.0, // unseen infoset: no information
                };
                w * p
            })
            .collect();
        self.weights[seat] = updates;
    }

    /// Sample hole cards for every non-folded seat from the tracked ranges,
    /// avoiding the board and each other. Folded seats stay `None` (their
    /// dead cards are drawn uniformly by the engine). Falls back to `None`
    /// (uniform) for a seat whose range has no compatible combo left.
    pub fn sample_holes(
        &self,
        h: &Hand,
        rng: &mut SmallRng,
    ) -> [Option<[Card; 2]>; MAX_PLAYERS] {
        let mut out: [Option<[Card; 2]>; MAX_PLAYERS] = [None; MAX_PLAYERS];
        let mut used = [false; 52];
        for &c in h.board() {
            used[c as usize] = true;
        }
        for (seat, slot) in out.iter_mut().enumerate().take(self.n) {
            if h.folded(seat) {
                continue;
            }
            let w = &self.weights[seat];
            let mut total = 0.0;
            for (ci, combo) in self.combos.iter().enumerate() {
                if w[ci] > 0.0 && !used[combo[0] as usize] && !used[combo[1] as usize] {
                    total += w[ci];
                }
            }
            if total <= 0.0 {
                continue; // uniform fallback for this seat
            }
            let mut r = rng.random::<f64>() * total;
            for (ci, combo) in self.combos.iter().enumerate() {
                if w[ci] > 0.0 && !used[combo[0] as usize] && !used[combo[1] as usize] {
                    r -= w[ci];
                    if r <= 0.0 {
                        *slot = Some(*combo);
                        used[combo[0] as usize] = true;
                        used[combo[1] as usize] = true;
                        break;
                    }
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Tests (written first, TDD)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::{preflop_bucket, AbsConfig};
    use crate::cards::{fresh_deck, parse_cards};
    use crate::cfr::make_key;
    use crate::engine::HandConfig;
    use std::collections::{HashMap, HashSet};

    #[test]
    fn combo_index_is_a_bijection() {
        let combos = all_combos();
        assert_eq!(combos.len(), NUM_COMBOS);
        let mut seen = HashSet::new();
        for (i, c) in combos.iter().enumerate() {
            assert_eq!(combo_index(c[0], c[1]), i);
            assert_eq!(combo_index(c[1], c[0]), i);
            assert!(seen.insert(i));
        }
    }

    #[test]
    fn exclude_zeroes_conflicting_combos() {
        let mut t = RangeTracker::new(2);
        let board = parse_cards("As Kd 2c").unwrap();
        t.exclude(&board);
        let ak = parse_cards("As Qh").unwrap();
        assert_eq!(t.weight(0, [ak[0], ak[1]]), 0.0);
        let ok = parse_cards("Ah Qh").unwrap();
        assert!(t.weight(1, [ok[0], ok[1]]) > 0.0);
    }

    /// After watching a seat take the "raise" action under a blueprint where
    /// only AA raises, the tracked range must concentrate on AA.
    #[test]
    fn observe_concentrates_range_on_raising_hands() {
        let abs = Abstraction::new(AbsConfig::default());
        let h = Hand::new(&HandConfig::default(), 0, fresh_deck());
        let acts = abs.abstract_actions(&h); // UTG: [Fold, CheckCall, bets..., AllIn]
        let n_acts = acts.len();
        let aa = preflop_bucket([51, 50]); // any two aces
        let raise_idx = acts
            .iter()
            .position(|a| matches!(a, AbsAction::Bet(_)))
            .unwrap();

        let mut strategies = HashMap::new();
        for bucket in 0..169u16 {
            let mut s = vec![0.0f32; n_acts];
            if bucket == aa {
                s[raise_idx] = 1.0;
            } else {
                s[0] = 1.0; // everything else folds
            }
            strategies.insert(make_key(bucket, &[]).to_vec(), s);
        }
        let bp = Blueprint {
            strategies,
            iterations: 1,
            num_players: 6,
            abs_cfg: AbsConfig::default(),
            centroids: None,
        };

        let mut t = RangeTracker::new(6);
        let raise = acts[raise_idx];
        t.observe(3, raise, &h, &[], &bp, &abs);

        let aa_combo = parse_cards("As Ah").unwrap();
        let junk = parse_cards("7c 2d").unwrap();
        let p_aa = t.prob(3, [aa_combo[0], aa_combo[1]]);
        let p_junk = t.prob(3, [junk[0], junk[1]]);
        assert!(
            p_aa > 40.0 * p_junk,
            "AA must dominate after a raise: p_aa={p_aa:.5} p_junk={p_junk:.5}"
        );
        // Unobserved seat stays uniform.
        assert!((t.prob(4, [aa_combo[0], aa_combo[1]]) - 1.0 / NUM_COMBOS as f64).abs() < 1e-9);
    }

    #[test]
    fn sample_holes_respects_weights_and_conflicts() {
        let mut rng = SmallRng::seed_from_u64(31);
        let h = Hand::new(
            &HandConfig {
                num_players: 3,
                ..HandConfig::default()
            },
            0,
            fresh_deck(),
        );
        let mut t = RangeTracker::new(3);
        // Force seat 0 and seat 1 onto the SAME single combo; only one can get it.
        let combo = parse_cards("As Ah").unwrap();
        let target = combo_index(combo[0], combo[1]);
        for seat in 0..2 {
            for ci in 0..NUM_COMBOS {
                t.weights[seat][ci] = if ci == target { 1.0 } else { 0.0 };
            }
        }
        for _ in 0..20 {
            let holes = t.sample_holes(&h, &mut rng);
            // Exactly one of the two seats gets the combo; the other falls back.
            let got: Vec<bool> = (0..2)
                .map(|s| holes[s].map(|w| combo_index(w[0], w[1])) == Some(target))
                .collect();
            assert_eq!(got.iter().filter(|&&g| g).count(), 1);
            // Seat 2 (uniform) gets some combo not colliding with seat 0/1.
            if let (Some(a), Some(b)) = (holes[0].or(holes[1]), holes[2]) {
                assert!(a[0] != b[0] && a[0] != b[1] && a[1] != b[0] && a[1] != b[1]);
            }
        }
    }
}
