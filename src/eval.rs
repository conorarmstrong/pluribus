//! Fast 7-card hand evaluator.
//!
//! `eval7` returns a u32 where a higher value is a strictly better hand.
//! Encoding: category << 20 | up to five 4-bit rank values, most significant first.
//! Categories: 0 high card, 1 pair, 2 two pair, 3 trips, 4 straight,
//! 5 flush, 6 full house, 7 quads, 8 straight flush.

use crate::cards::{rank, suit, Card};

pub const CAT_HIGH: u32 = 0;
pub const CAT_PAIR: u32 = 1;
pub const CAT_TWO_PAIR: u32 = 2;
pub const CAT_TRIPS: u32 = 3;
pub const CAT_STRAIGHT: u32 = 4;
pub const CAT_FLUSH: u32 = 5;
pub const CAT_FULL_HOUSE: u32 = 6;
pub const CAT_QUADS: u32 = 7;
pub const CAT_STRAIGHT_FLUSH: u32 = 8;

#[inline(always)]
fn pack(cat: u32, ranks: &[u32]) -> u32 {
    let mut v = cat << 20;
    let mut shift = 16;
    for &r in ranks {
        v |= r << shift;
        shift -= 4;
    }
    v
}

/// Highest rank of a 5-in-a-row within a 13-bit rank mask, or None.
/// Handles the wheel (A-2-3-4-5) where the high card is the 5 (rank index 3).
#[inline(always)]
fn straight_high(mask: u16) -> Option<u32> {
    // Check from ace-high down to six-high.
    for high in (4..=12u32).rev() {
        let need = 0b11111u16 << (high - 4);
        if mask & need == need {
            return Some(high);
        }
    }
    // Wheel: A,2,3,4,5 -> bits 12,0,1,2,3
    let wheel = (1u16 << 12) | 0b1111;
    if mask & wheel == wheel {
        return Some(3);
    }
    None
}

/// Take the top `n` set bits of a 13-bit mask, highest first.
#[inline(always)]
fn top_ranks(mask: u16, n: usize, out: &mut [u32; 5]) -> usize {
    let mut cnt = 0;
    for r in (0..13u32).rev() {
        if mask & (1 << r) != 0 {
            out[cnt] = r;
            cnt += 1;
            if cnt == n {
                break;
            }
        }
    }
    cnt
}

/// Evaluate the best 5-card hand from exactly 7 cards.
pub fn eval7(cards: &[Card; 7]) -> u32 {
    let mut counts = [0u8; 13];
    let mut suit_masks = [0u16; 4];
    for &c in cards {
        counts[rank(c) as usize] += 1;
        suit_masks[suit(c) as usize] |= 1 << rank(c);
    }

    // Flush / straight flush
    for sm in suit_masks {
        if sm.count_ones() >= 5 {
            if let Some(h) = straight_high(sm) {
                return pack(CAT_STRAIGHT_FLUSH, &[h]);
            }
            let mut tops = [0u32; 5];
            top_ranks(sm, 5, &mut tops);
            return pack(CAT_FLUSH, &tops);
        }
    }

    let mut rank_mask: u16 = 0;
    let mut quad: i32 = -1;
    let mut trips_hi: i32 = -1;
    let mut trips_lo: i32 = -1;
    let mut pair_hi: i32 = -1;
    let mut pair_lo: i32 = -1;
    for r in (0..13usize).rev() {
        let n = counts[r];
        if n == 0 {
            continue;
        }
        rank_mask |= 1 << r;
        match n {
            4 => quad = r as i32,
            3 => {
                if trips_hi < 0 {
                    trips_hi = r as i32;
                } else if trips_lo < 0 {
                    trips_lo = r as i32;
                }
            }
            2 => {
                if pair_hi < 0 {
                    pair_hi = r as i32;
                } else if pair_lo < 0 {
                    pair_lo = r as i32;
                }
            }
            _ => {}
        }
    }

    if quad >= 0 {
        // Best kicker among remaining ranks
        let kicker_mask = rank_mask & !(1u16 << quad);
        let mut tops = [0u32; 5];
        top_ranks(kicker_mask, 1, &mut tops);
        return pack(CAT_QUADS, &[quad as u32, tops[0]]);
    }

    // Full house: trips + (second trips as pair, or best pair)
    if trips_hi >= 0 {
        let pair_part = if trips_lo >= 0 {
            trips_lo.max(pair_hi)
        } else {
            pair_hi
        };
        if pair_part >= 0 {
            return pack(CAT_FULL_HOUSE, &[trips_hi as u32, pair_part as u32]);
        }
    }

    if let Some(h) = straight_high(rank_mask) {
        return pack(CAT_STRAIGHT, &[h]);
    }

    if trips_hi >= 0 {
        let kicker_mask = rank_mask & !(1u16 << trips_hi);
        let mut tops = [0u32; 5];
        top_ranks(kicker_mask, 2, &mut tops);
        return pack(CAT_TRIPS, &[trips_hi as u32, tops[0], tops[1]]);
    }

    if pair_hi >= 0 && pair_lo >= 0 {
        let kicker_mask = rank_mask & !(1u16 << pair_hi) & !(1u16 << pair_lo);
        let mut tops = [0u32; 5];
        top_ranks(kicker_mask, 1, &mut tops);
        return pack(CAT_TWO_PAIR, &[pair_hi as u32, pair_lo as u32, tops[0]]);
    }

    if pair_hi >= 0 {
        let kicker_mask = rank_mask & !(1u16 << pair_hi);
        let mut tops = [0u32; 5];
        top_ranks(kicker_mask, 3, &mut tops);
        return pack(CAT_PAIR, &[pair_hi as u32, tops[0], tops[1], tops[2]]);
    }

    let mut tops = [0u32; 5];
    top_ranks(rank_mask, 5, &mut tops);
    pack(CAT_HIGH, &tops)
}

/// Convenience: evaluate 2 hole cards + 3..5 board cards (pads via best-of available
/// is NOT supported; board must have exactly 5 cards for showdown use, or use eval7 directly).
pub fn eval_hole_board(hole: &[Card; 2], board: &[Card]) -> u32 {
    debug_assert_eq!(board.len(), 5);
    let cards = [
        hole[0], hole[1], board[0], board[1], board[2], board[3], board[4],
    ];
    eval7(&cards)
}

pub fn category(value: u32) -> u32 {
    value >> 20
}

#[allow(dead_code)]
pub fn category_name(value: u32) -> &'static str {
    match category(value) {
        CAT_HIGH => "High Card",
        CAT_PAIR => "Pair",
        CAT_TWO_PAIR => "Two Pair",
        CAT_TRIPS => "Three of a Kind",
        CAT_STRAIGHT => "Straight",
        CAT_FLUSH => "Flush",
        CAT_FULL_HOUSE => "Full House",
        CAT_QUADS => "Four of a Kind",
        CAT_STRAIGHT_FLUSH => "Straight Flush",
        _ => "?",
    }
}

// ---------------------------------------------------------------------------
// Tests: spot checks + differential test against an independent naive evaluator.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::parse_cards;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    fn e(s: &str) -> u32 {
        let v = parse_cards(s).unwrap();
        let arr: [Card; 7] = v.try_into().unwrap();
        eval7(&arr)
    }

    #[test]
    fn categories() {
        assert_eq!(category(e("As Ks Qs Js Ts 2c 3d")), CAT_STRAIGHT_FLUSH);
        assert_eq!(category(e("As 2s 3s 4s 5s Kc Kd")), CAT_STRAIGHT_FLUSH); // steel wheel
        assert_eq!(category(e("Ac Ad Ah As Kc 2d 3h")), CAT_QUADS);
        assert_eq!(category(e("Ac Ad Ah Kc Kd 2d 3h")), CAT_FULL_HOUSE);
        assert_eq!(category(e("Ac Ad Ah Kc Kd Kh 3h")), CAT_FULL_HOUSE);
        assert_eq!(category(e("As Ks 9s 5s 2s 3c 4d")), CAT_FLUSH);
        assert_eq!(category(e("Ac Kd Qh Js Tc 2d 3h")), CAT_STRAIGHT);
        assert_eq!(category(e("Ac 2d 3h 4s 5c Kd 9h")), CAT_STRAIGHT); // wheel
        assert_eq!(category(e("Ac Ad Ah Kc Qd 2d 3h")), CAT_TRIPS);
        assert_eq!(category(e("Ac Ad Kh Kc Qd 2d 3h")), CAT_TWO_PAIR);
        assert_eq!(category(e("Ac Ad Kh Qc Jd 2d 3h")), CAT_PAIR);
        assert_eq!(category(e("Ac Kd Qh Js 9c 2d 3h")), CAT_HIGH);
    }

    #[test]
    fn orderings() {
        // Kings full over queens full
        assert!(e("Kc Kd Kh Qc Qd 2d 3h") > e("Qc Qd Qh Kc Kd 2d 3h"));
        // Wheel straight loses to six-high straight
        assert!(e("Ac 2d 3h 4s 5c 9d 8h") < e("2c 3d 4h 5s 6c 9d Kh"));
        // Ace-high flush beats king-high flush
        assert!(e("As 9s 5s 3s 2s Kc Qd") > e("Ks 9s 5s 3s 2s Ac Qd"));
        // Kicker matters on pair
        assert!(e("Ac Ad Kh Qc Jd 2d 3h") > e("Ac Ad Kh Qc Td 2d 3h"));
        // Trips with 3 pairs on board: best two pair uses third-pair kicker correctly
        assert!(e("Ac Ad Kh Kc Qd Qh 2s") > e("Ac Ad Kh Kc Jd Jh 2s"));
    }

    // Independent naive implementation: best of the 21 5-card combos,
    // each ranked with straightforward sort-and-count logic.
    fn naive5(cards: &[Card; 5]) -> u64 {
        let mut ranks: Vec<u8> = cards.iter().map(|&c| rank(c)).collect();
        ranks.sort_unstable_by(|a, b| b.cmp(a));
        let flush = cards.iter().all(|&c| suit(c) == suit(cards[0]));

        // count multiplicities
        let mut counts: Vec<(u8, u8)> = Vec::new(); // (count, rank), sorted desc
        for &r in &ranks {
            match counts.iter_mut().find(|(_, rr)| *rr == r) {
                Some((n, _)) => *n += 1,
                None => counts.push((1, r)),
            }
        }
        counts.sort_unstable_by(|a, b| b.cmp(a));

        // straight detection over 5 distinct sorted ranks
        let distinct = counts.len() == 5;
        let mut straight = false;
        let mut s_high = 0u8;
        if distinct {
            if ranks[0] - ranks[4] == 4 {
                straight = true;
                s_high = ranks[0];
            } else if ranks == [12, 3, 2, 1, 0] {
                straight = true;
                s_high = 3;
            }
        }

        let cat: u64 = if straight && flush {
            8
        } else if counts[0].0 == 4 {
            7
        } else if counts[0].0 == 3 && counts[1].0 == 2 {
            6
        } else if flush {
            5
        } else if straight {
            4
        } else if counts[0].0 == 3 {
            3
        } else if counts[0].0 == 2 && counts[1].0 == 2 {
            2
        } else if counts[0].0 == 2 {
            1
        } else {
            0
        };

        let mut v = cat << 40;
        if straight {
            v |= (s_high as u64) << 32;
        } else {
            let mut shift = 32i32;
            for (_, r) in &counts {
                v |= (*r as u64) << shift;
                shift -= 8;
            }
        }
        v
    }

    fn naive7(cards: &[Card; 7]) -> u64 {
        let mut best = 0u64;
        for i in 0..7 {
            for j in (i + 1)..7 {
                let five: Vec<Card> = (0..7)
                    .filter(|&k| k != i && k != j)
                    .map(|k| cards[k])
                    .collect();
                let arr: [Card; 5] = five.try_into().unwrap();
                best = best.max(naive5(&arr));
            }
        }
        best
    }

    #[test]
    fn differential_vs_naive() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let mut deck = crate::cards::fresh_deck();
        let mut hands: Vec<[Card; 7]> = Vec::new();
        for _ in 0..30_000 {
            deck.shuffle(&mut rng);
            hands.push(deck[..7].try_into().unwrap());
        }
        for i in 0..hands.len() {
            for j in (i + 1)..(i + 2).min(hands.len()) {
                let (a, b) = (&hands[i], &hands[j]);
                let (fa, fb) = (eval7(a), eval7(b));
                let (na, nb) = (naive7(a), naive7(b));
                assert_eq!(
                    fa.cmp(&fb),
                    na.cmp(&nb),
                    "ordering mismatch: {:?} vs {:?}",
                    a,
                    b
                );
                assert_eq!(category(fa) as u64, na >> 40, "category mismatch: {:?}", a);
            }
        }
    }
}
