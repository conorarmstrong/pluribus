//! Behavioral clone of Slumbot from logged hands (`slumbot --log`).
//!
//! Each JSONL line is a hand's final API response: the cumulative action
//! string, full board, both players' hole cards (Slumbot reveals its cards
//! every hand), and `client_pos`. Replaying the hand through the table
//! reconstructs the exact abstract infoset at each of Slumbot's decisions;
//! counting its chosen abstract actions per infoset and normalizing yields
//! a Blueprint-format strategy — a playable, exploitable model of the
//! static opponent (the target for restricted Nash response training).
//!
//! The abstraction must come from the blueprint the exploiter will use, so
//! buckets (and serialized centroids) line up at train and play time.

use crate::abstraction::Abstraction;
use crate::cards::{parse_card, Card};
use crate::cfr::{make_key, Blueprint, StrategyMap};
use crate::engine::{HandConfig, PlayerAction};
use crate::slumbot::{parse_incr, SlumAct, STACK};
use crate::table::Table;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use serde_json::Value;
use std::io::BufRead;

/// One of Slumbot's decisions: infoset key, chosen menu index, menu size.
struct Decision {
    key: Box<[u8]>,
    act_ix: usize,
    menu_len: usize,
}

#[derive(Default)]
pub struct CloneStats {
    pub hands: u64,
    pub skipped: u64,
    pub decisions: u64,
    pub infosets: usize,
    /// Held-out top-1 agreement per street [pre, flop, turn, river]:
    /// (matches, total).
    pub agree: [(u64, u64); 4],
}

/// Replay one logged hand; returns Slumbot's decisions as (key, action).
fn replay_hand(v: &Value, abs: &Abstraction) -> Result<Vec<Decision>, String> {
    let client_pos = v
        .get("client_pos")
        .and_then(|p| p.as_u64())
        .ok_or("no client_pos")?;
    let our: Vec<Card> = cards_field(v, "hole_cards")?;
    let bot: Vec<Card> = cards_field(v, "bot_hole_cards")?;
    let board: Vec<Card> = cards_field(v, "board").unwrap_or_default();
    if our.len() != 2 || bot.len() != 2 {
        return Err("bad hole cards".into());
    }
    let action = v.get("action").and_then(|a| a.as_str()).unwrap_or("");
    let (acts, _) = parse_incr(action, 0)?;

    let cfg = HandConfig {
        num_players: 2,
        stack: STACK,
        sb: 50,
        bb: 100,
    };
    // Seat 0 = us, seat 1 = Slumbot (mirrors the live adapter).
    let mut deck = crate::cards::fresh_deck();
    let mut used = [false; 52];
    for &c in our.iter().chain(bot.iter()).chain(board.iter()) {
        used[c as usize] = true;
    }
    deck[0] = our[0];
    deck[1] = our[1];
    deck[2] = bot[0];
    deck[3] = bot[1];
    let mut idx = 4;
    for c in 0..52u8 {
        if !used[c as usize] {
            deck[idx] = c;
            idx += 1;
            if idx == 52 {
                break;
            }
        }
    }
    let button = if client_pos == 1 { 0 } else { 1 };
    let mut table = Table::new(&cfg, button, deck);
    for (i, &c) in board.iter().enumerate() {
        table.real.force_board_card(i, c);
        table.shadow.force_board_card(i, c);
    }

    let mut rng = SmallRng::seed_from_u64(0); // bucket draws are key-seeded
    let mut out = Vec::new();
    for inc in acts {
        if matches!(inc, SlumAct::StreetSep) {
            continue; // the engine advances streets itself
        }
        if table.real.is_terminal() {
            return Err("action after terminal".into());
        }
        let p = table.real.to_act();
        let concrete = match inc {
            SlumAct::Check | SlumAct::Call => PlayerAction::CheckCall,
            SlumAct::Fold => PlayerAction::Fold,
            SlumAct::BetTo(x) => PlayerAction::RaiseTo(x),
            SlumAct::StreetSep => unreachable!(),
        };
        if p == 1 {
            let abs_a = table.map_concrete(concrete, abs);
            let menu = abs.abstract_actions(&table.shadow);
            let act_ix = menu
                .iter()
                .position(|&a| a == abs_a)
                .ok_or("mapped action not on menu")?;
            let bucket = abs.bucket([bot[0], bot[1]], table.shadow.board(), &mut rng);
            out.push(Decision {
                key: make_key(bucket, &table.hist),
                act_ix,
                menu_len: menu.len(),
            });
        }
        table.apply_concrete(concrete, abs);
    }
    Ok(out)
}

fn cards_field(v: &Value, field: &str) -> Result<Vec<Card>, String> {
    v.get(field)
        .and_then(|h| h.as_array())
        .ok_or_else(|| format!("no {field}"))?
        .iter()
        .map(|c| {
            c.as_str()
                .and_then(parse_card)
                .ok_or_else(|| format!("bad card in {field}"))
        })
        .collect()
}

/// Build a clone Blueprint from a JSONL log. `holdout` in [0,1) reserves
/// that fraction of hands (deterministic on hand index) for measuring
/// top-1 agreement of the trained clone. The abstraction (and the
/// centroids serialized into the output) must be the exploiter's.
pub fn build(
    path: &str,
    abs: &Abstraction,
    holdout: f64,
) -> Result<(Blueprint, CloneStats), String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut stats = CloneStats::default();
    let mut counts: StrategyMap = StrategyMap::default();
    let mut held: Vec<Decision> = Vec::new();

    for (i, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => {
                stats.skipped += 1;
                continue;
            }
        };
        let decisions = match replay_hand(&v, abs) {
            Ok(d) => d,
            Err(_) => {
                stats.skipped += 1;
                continue;
            }
        };
        stats.hands += 1;
        // Deterministic even split over the hand index.
        let is_held = holdout > 0.0
            && ((i + 1) as f64 * holdout).floor() > (i as f64 * holdout).floor();
        if is_held {
            held.extend(decisions);
            continue;
        }
        for d in decisions {
            stats.decisions += 1;
            let e = counts
                .entry(d.key.to_vec())
                .or_insert_with(|| vec![0.0f32; d.menu_len]);
            if e.len() == d.menu_len {
                e[d.act_ix] += 1.0;
            }
        }
    }
    stats.infosets = counts.len();

    // Normalize counts to probabilities (Blueprint::get renormalizes too,
    // but keep the stored form a proper distribution).
    for probs in counts.values_mut() {
        let total: f32 = probs.iter().sum();
        if total > 0.0 {
            probs.iter_mut().for_each(|p| *p /= total);
        }
    }

    // Held-out top-1 agreement, by street (street = '/'-count in the key's
    // history is not stored; recover from hist length is fragile — use the
    // key directly and bin by the decision order instead: preflop keys have
    // the shortest histories). Simpler and exact: recompute street from the
    // stored key by counting street-separator tokens in its hist suffix.
    for d in &held {
        let hist = &d.key[2..];
        let street = hist
            .iter()
            .filter(|&&t| t == crate::abstraction::TOKEN_STREET_SEP)
            .count()
            .min(3);
        let slot = &mut stats.agree[street];
        slot.1 += 1;
        if let Some(probs) = counts.get(&d.key[..].to_vec()) {
            let best = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i);
            if best == Some(d.act_ix) {
                slot.0 += 1;
            }
        }
    }

    let bp = Blueprint {
        strategies: counts,
        iterations: stats.decisions,
        num_players: 2,
        abs_cfg: abs.cfg.clone(),
        centroids: abs.centroids.clone(),
    };
    Ok((bp, stats))
}

// ---------------------------------------------------------------------------
// Tests (written first, TDD)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::{AbsConfig, TOKEN_STREET_SEP};
    use serde_json::json;

    fn abs_small() -> Abstraction {
        Abstraction::new(AbsConfig {
            postflop_buckets: 6,
            equity_rollouts: 40,
            dist_runouts: 8,
            runout_rollouts: 20,
            cache_cap: 200_000,
            menu: crate::abstraction::MenuShape::Wide,
        })
    }

    /// client_pos 0: Slumbot is the small blind and acts first preflop.
    /// "b200f" = Slumbot raises to 200, we fold: exactly one Slumbot
    /// decision, recorded at an empty history with a raise on the menu.
    #[test]
    fn replays_a_preflop_raise_fold() {
        let abs = abs_small();
        let v = json!({
            "action": "b200f",
            "client_pos": 0,
            "hole_cards": ["6c", "4d"],
            "bot_hole_cards": ["Ts", "2c"],
            "board": [],
            "winnings": -100,
        });
        let ds = replay_hand(&v, &abs).unwrap();
        assert_eq!(ds.len(), 1, "only Slumbot's raise is its decision");
        assert!(ds[0].menu_len >= 3);
        assert!(ds[0].act_ix > 1, "a raise is neither fold(0) nor call(1)");
        assert_eq!(&ds[0].key[2..], &[] as &[u8], "preflop first-to-act");
    }

    /// client_pos 1: we are the button; a checked-down hand records one
    /// Slumbot decision per street (BB checks first postflop... Slumbot
    /// checks last preflop and first on every later street).
    #[test]
    fn replays_a_checked_down_hand_across_streets() {
        let abs = abs_small();
        let v = json!({
            "action": "cc/kk/kk/kk",
            "client_pos": 1,
            "hole_cards": ["6c", "4d"],
            "bot_hole_cards": ["Ts", "2c"],
            "board": ["Ah", "Kd", "Qs", "Js", "3h"],
            "winnings": -100,
        });
        let ds = replay_hand(&v, &abs).unwrap();
        assert_eq!(ds.len(), 4, "one Slumbot decision per street");
        // Street separators accumulate in the history keys.
        let seps: Vec<usize> = ds
            .iter()
            .map(|d| {
                d.key[2..]
                    .iter()
                    .filter(|&&t| t == TOKEN_STREET_SEP)
                    .count()
            })
            .collect();
        assert_eq!(seps, vec![0, 1, 2, 3]);
    }

    /// build() over a small file: counts land in the blueprint and repeated
    /// identical decisions concentrate the distribution.
    #[test]
    fn builds_a_normalized_clone_from_a_log_file() {
        let abs = abs_small();
        let line = r#"{"action":"b200f","client_pos":0,"hole_cards":["6c","4d"],"bot_hole_cards":["Ts","2c"],"board":[],"winnings":-100}"#;
        let path = std::env::temp_dir().join("clone_build_test.jsonl");
        std::fs::write(&path, format!("{line}\n{line}\nnot json\n")).unwrap();
        let (bp, stats) = build(path.to_str().unwrap(), &abs, 0.0).unwrap();
        assert_eq!(stats.hands, 2);
        assert_eq!(stats.skipped, 1);
        assert_eq!(stats.decisions, 2);
        assert_eq!(stats.infosets, 1);
        let probs = bp.strategies.values().next().unwrap();
        let total: f32 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "stored dist must normalize");
        assert_eq!(probs.iter().cloned().fold(0.0, f32::max), 1.0);
        let _ = std::fs::remove_file(&path);
    }
}
