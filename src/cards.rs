//! Card representation: a card is a u8 in 0..52, encoded rank*4 + suit.
//! Ranks: 0 = Two .. 12 = Ace. Suits: 0=c, 1=d, 2=h, 3=s.

pub type Card = u8;

pub const RANK_CHARS: [char; 13] = [
    '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A',
];
pub const SUIT_CHARS: [char; 4] = ['c', 'd', 'h', 's'];

#[inline(always)]
pub fn rank(c: Card) -> u8 {
    c >> 2
}

#[inline(always)]
pub fn suit(c: Card) -> u8 {
    c & 3
}

#[inline(always)]
pub fn make_card(rank: u8, suit: u8) -> Card {
    (rank << 2) | suit
}

pub fn card_str(c: Card) -> String {
    let mut s = String::with_capacity(2);
    s.push(RANK_CHARS[rank(c) as usize]);
    s.push(SUIT_CHARS[suit(c) as usize]);
    s
}

pub fn cards_str(cards: &[Card]) -> String {
    cards
        .iter()
        .map(|&c| card_str(c))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Parse a card like "As", "Td", "2c". Returns None on bad input.
pub fn parse_card(s: &str) -> Option<Card> {
    let mut it = s.chars();
    let r = it.next()?;
    let u = it.next()?;
    if it.next().is_some() {
        return None;
    }
    let r = RANK_CHARS
        .iter()
        .position(|&c| c == r.to_ascii_uppercase())? as u8;
    let u = SUIT_CHARS
        .iter()
        .position(|&c| c == u.to_ascii_lowercase())? as u8;
    Some(make_card(r, u))
}

/// Parse space-separated cards, e.g. "As Kd 7c".
#[allow(dead_code)]
pub fn parse_cards(s: &str) -> Option<Vec<Card>> {
    s.split_whitespace().map(parse_card).collect()
}

/// A fresh 52-card deck in order; shuffle with your own RNG.
pub fn fresh_deck() -> [Card; 52] {
    let mut d = [0u8; 52];
    for (i, c) in d.iter_mut().enumerate() {
        *c = i as u8;
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        for c in 0..52u8 {
            assert_eq!(parse_card(&card_str(c)), Some(c));
        }
    }

    #[test]
    fn parse_examples() {
        assert_eq!(parse_card("As"), Some(make_card(12, 3)));
        assert_eq!(parse_card("2c"), Some(make_card(0, 0)));
        assert_eq!(parse_card("Td"), Some(make_card(8, 1)));
        assert_eq!(parse_card("xx"), None);
    }
}
