//! Benchmark the blueprint against the 10,000 hands the real Pluribus played
//! in the Brown & Sandholm 2019 Science experiment (PHH format, from
//! uoftcprg/phh-dataset).
//!
//! Every hand is replayed through the engine with the logged cards and
//! actions. At each decision the real Pluribus made, we look up our
//! blueprint's strategy for the same infoset and record (a) the probability
//! our blueprint assigns to the action Pluribus actually took (mapped onto
//! our action abstraction) and (b) whether it is our highest-probability
//! action. High agreement means the blueprint independently reproduces
//! Pluribus's play.

use crate::bot::Policy;
use crate::cards::{parse_card, Card};
use crate::engine::{HandConfig, PlayerAction, Street};
use crate::table::Table;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct PhhHand {
    pub sb: u32,
    pub bb: u32,
    pub stacks: Vec<u32>,
    pub players: Vec<String>,
    pub actions: Vec<String>,
    pub finishing: Vec<i64>,
}

/// Parse one .phh file (TOML).
pub fn parse_phh(text: &str) -> Option<PhhHand> {
    let doc: toml::Table = text.parse().ok()?;
    let arr_u32 = |k: &str| -> Option<Vec<u32>> {
        doc.get(k)?
            .as_array()?
            .iter()
            .map(|v| v.as_integer().map(|x| x as u32))
            .collect()
    };
    let arr_i64 = |k: &str| -> Option<Vec<i64>> {
        doc.get(k)?
            .as_array()?
            .iter()
            .map(|v| v.as_integer())
            .collect()
    };
    let arr_str = |k: &str| -> Option<Vec<String>> {
        doc.get(k)?
            .as_array()?
            .iter()
            .map(|v| v.as_str().map(str::to_string))
            .collect()
    };

    if doc.get("variant")?.as_str()? != "NT" {
        return None;
    }
    let antes = arr_u32("antes")?;
    if antes.iter().any(|&a| a != 0) {
        return None;
    }
    let blinds = arr_u32("blinds_or_straddles")?;
    if blinds.len() < 2 || blinds[2..].iter().any(|&b| b != 0) {
        return None; // straddles unsupported
    }
    Some(PhhHand {
        sb: blinds[0],
        bb: blinds[1],
        stacks: arr_u32("starting_stacks")?,
        players: arr_str("players")?,
        actions: arr_str("actions")?,
        finishing: arr_i64("finishing_stacks").unwrap_or_default(),
    })
}

/// One recorded Pluribus decision.
#[derive(Debug, Clone, Copy)]
pub struct Decision {
    pub street: Street,
    /// Probability our blueprint puts on Pluribus's (mapped) action, if the
    /// infoset was trained.
    pub prob: Option<f64>,
    /// Was it our argmax action?
    pub top1: bool,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct StreetStats {
    pub decisions: u64,
    pub covered: u64,
    pub top1: u64,
    pub prob_sum: f64,
}

#[derive(Debug, Default)]
pub struct BenchReport {
    pub hands: u64,
    pub skipped: u64,
    pub desynced: u64,
    pub chip_checked: u64,
    pub chip_mismatch: u64,
    pub by_street: [StreetStats; 4],
}

impl BenchReport {
    fn absorb_decision(&mut self, d: &Decision) {
        let s = &mut self.by_street[d.street as usize];
        s.decisions += 1;
        if let Some(p) = d.prob {
            s.covered += 1;
            s.prob_sum += p;
            if d.top1 {
                s.top1 += 1;
            }
        }
    }
}

pub enum ReplayError {
    Unsupported,
    Desync,
}

/// Replay one hand; returns the recorded Pluribus decisions and whether the
/// finishing-stack check passed (`None` when the file has no stacks to check).
pub fn replay_hand(
    phh: &PhhHand,
    policy: &Policy,
    seed: u64,
) -> Result<(Vec<Decision>, Option<bool>), ReplayError> {
    let n = phh.players.len();
    if !(2..=6).contains(&n) || phh.stacks.len() != n {
        return Err(ReplayError::Unsupported);
    }
    let hero = phh
        .players
        .iter()
        .position(|p| p == "Pluribus")
        .ok_or(ReplayError::Unsupported)?;
    // blinds_or_straddles[0] is the small blind, so the button is the last
    // seat (heads-up: the button posts the SB and is seat 0).
    let button = if n == 2 { 0 } else { n - 1 };

    // Collect dealt cards, then build a deck matching the engine's layout:
    // holes at deck[2p], deck[2p+1]; board at deck[2n..2n+5].
    let mut holes: Vec<Option<[Card; 2]>> = vec![None; n];
    let mut board: Vec<Card> = Vec::with_capacity(5);
    for a in &phh.actions {
        let mut w = a.split_whitespace();
        match (w.next(), w.next()) {
            (Some("d"), Some("dh")) => {
                let seat = seat_of(w.next().ok_or(ReplayError::Unsupported)?)?;
                let cards = w.next().ok_or(ReplayError::Unsupported)?;
                if seat < n && cards.len() == 4 {
                    let c1 = parse_card(&cards[0..2]);
                    let c2 = parse_card(&cards[2..4]);
                    if let (Some(c1), Some(c2)) = (c1, c2) {
                        holes[seat] = Some([c1, c2]);
                    }
                }
            }
            (Some("d"), Some("db")) => {
                let cards = w.next().ok_or(ReplayError::Unsupported)?;
                for i in (0..cards.len()).step_by(2) {
                    board.push(parse_card(&cards[i..i + 2]).ok_or(ReplayError::Unsupported)?);
                }
            }
            _ => {}
        }
    }

    let mut deck = [0u8; 52];
    let mut used = [false; 52];
    let mut place = |pos: usize, c: Card, used: &mut [bool; 52]| -> bool {
        if used[c as usize] {
            return false;
        }
        used[c as usize] = true;
        deck[pos] = c;
        true
    };
    for (p, h) in holes.iter().enumerate() {
        if let Some(h) = h {
            if !place(2 * p, h[0], &mut used) || !place(2 * p + 1, h[1], &mut used) {
                return Err(ReplayError::Unsupported); // duplicate card in log
            }
        }
    }
    for (i, &c) in board.iter().enumerate() {
        if !place(2 * n + i, c, &mut used) {
            return Err(ReplayError::Unsupported);
        }
    }
    // Fill unknown holes and the rest of the deck with unused cards.
    let mut filler = (0..52u8).filter(|&c| !used[c as usize]);
    for (p, h) in holes.iter().enumerate() {
        if h.is_none() {
            deck[2 * p] = filler.next().unwrap();
            deck[2 * p + 1] = filler.next().unwrap();
        }
    }
    for i in board.len()..5 {
        deck[2 * n + i] = filler.next().unwrap();
    }
    let tail: Vec<Card> = filler.collect();
    deck[2 * n + 5..].copy_from_slice(&tail);

    let cfg = HandConfig {
        num_players: n,
        stack: phh.stacks[0],
        sb: phh.sb,
        bb: phh.bb,
    };
    if phh.stacks.iter().any(|&s| s != phh.stacks[0]) {
        return Err(ReplayError::Unsupported); // engine table starts symmetric
    }

    let mut table = Table::new(&cfg, button, deck);
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut decisions = Vec::new();

    for a in &phh.actions {
        let mut w = a.split_whitespace();
        let actor = w.next().ok_or(ReplayError::Unsupported)?;
        if actor == "d" {
            continue;
        }
        let seat = seat_of(actor)?;
        let verb = w.next().ok_or(ReplayError::Unsupported)?;
        let action = match verb {
            "f" => PlayerAction::Fold,
            "cc" => PlayerAction::CheckCall,
            "cbr" => {
                let amt: u32 = w
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or(ReplayError::Unsupported)?;
                PlayerAction::RaiseTo(amt)
            }
            _ => continue, // sm (showdown) etc.
        };
        if table.real.is_terminal() || table.real.to_act() != seat {
            return Err(ReplayError::Desync);
        }

        if seat == hero {
            let acts = policy.abs.abstract_actions(&table.shadow);
            let mapped = table.map_concrete(action, &policy.abs);
            let idx = acts.iter().position(|&x| x == mapped);
            let bucket = policy
                .abs
                .bucket(table.shadow.hole(seat), table.shadow.board(), &mut rng);
            let strat = policy.blueprint.get(bucket, &table.hist);
            let (prob, top1) = match (idx, strat) {
                (Some(i), Some(s)) if s.len() == acts.len() => {
                    let argmax = s
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(j, _)| j);
                    (Some(s[i] as f64), argmax == Some(i))
                }
                _ => (None, false),
            };
            decisions.push(Decision {
                street: table.real.street(),
                prob,
                top1,
            });
        }
        table.apply_concrete(action, &policy.abs);
    }

    // Chip accounting check against the logged finishing stacks.
    let chips_ok = if table.real.is_terminal() && phh.finishing.len() == n {
        let u = table.real.utilities();
        Some((0..n).all(|p| phh.stacks[p] as i64 + u[p] == phh.finishing[p]))
    } else {
        None
    };
    Ok((decisions, chips_ok))
}

fn seat_of(token: &str) -> Result<usize, ReplayError> {
    token
        .strip_prefix('p')
        .and_then(|s| s.parse::<usize>().ok())
        .and_then(|i| i.checked_sub(1))
        .ok_or(ReplayError::Unsupported)
}

/// Run the benchmark over every .phh file under `dir` (recursively).
pub fn run(dir: &str, policy: &Policy) -> std::io::Result<BenchReport> {
    let mut files = Vec::new();
    collect_phh_files(std::path::Path::new(dir), &mut files)?;
    files.sort();

    let reports: Vec<BenchReport> = files
        .par_iter()
        .enumerate()
        .map(|(i, path)| {
            let mut r = BenchReport::default();
            let Ok(text) = std::fs::read_to_string(path) else {
                r.skipped = 1;
                return r;
            };
            let Some(phh) = parse_phh(&text) else {
                r.skipped = 1;
                return r;
            };
            match replay_hand(&phh, policy, 0xBEEF ^ i as u64) {
                Ok((decisions, chips_ok)) => {
                    r.hands = 1;
                    for d in &decisions {
                        r.absorb_decision(d);
                    }
                    if let Some(ok) = chips_ok {
                        r.chip_checked = 1;
                        r.chip_mismatch = u64::from(!ok);
                    }
                }
                Err(ReplayError::Desync) => r.desynced = 1,
                Err(ReplayError::Unsupported) => r.skipped = 1,
            }
            r
        })
        .collect();

    let mut total = BenchReport::default();
    for r in reports {
        total.hands += r.hands;
        total.skipped += r.skipped;
        total.desynced += r.desynced;
        total.chip_checked += r.chip_checked;
        total.chip_mismatch += r.chip_mismatch;
        for s in 0..4 {
            total.by_street[s].decisions += r.by_street[s].decisions;
            total.by_street[s].covered += r.by_street[s].covered;
            total.by_street[s].top1 += r.by_street[s].top1;
            total.by_street[s].prob_sum += r.by_street[s].prob_sum;
        }
    }
    Ok(total)
}

fn collect_phh_files(
    dir: &std::path::Path,
    out: &mut Vec<std::path::PathBuf>,
) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_phh_files(&path, out)?;
        } else if path.extension().is_some_and(|e| e == "phh") {
            out.push(path);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests (written first, TDD)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::{AbsConfig, Abstraction};
    use crate::cfr::Blueprint;
    use std::collections::HashMap;
    use std::sync::Arc;

    const SAMPLE: &str = r#"
variant = 'NT'
ante_trimming_status = true
antes = [0, 0, 0, 0, 0, 0]
blinds_or_straddles = [50, 100, 0, 0, 0, 0]
min_bet = 100
starting_stacks = [10000, 10000, 10000, 10000, 10000, 10000]
actions = ['d dh p1 ThQh', 'd dh p2 8s7h', 'd dh p3 Ad9h', 'd dh p4 3s5c', 'd dh p5 JhKs', 'd dh p6 4d2h', 'p3 f', 'p4 f', 'p5 cbr 225', 'p6 f', 'p1 cbr 950', 'p2 f', 'p5 f']
hand = 134
players = ['Bill', 'Eddie', 'Joe', 'Pluribus', 'MrBlue', 'MrPink']
finishing_stacks = [10325, 9900, 10000, 10000, 9775, 10000]
"#;

    fn empty_policy() -> Policy {
        Policy::new(
            Blueprint {
                strategies: HashMap::new(),
                iterations: 0,
                num_players: 6,
                abs_cfg: AbsConfig::default(),
                centroids: None,
            },
            Arc::new(Abstraction::new(AbsConfig {
                postflop_buckets: 6,
                equity_rollouts: 40,
                cache_cap: 100_000,
                ..AbsConfig::default()
            })),
        )
    }

    #[test]
    fn parses_the_sample_hand() {
        let phh = parse_phh(SAMPLE).expect("sample must parse");
        assert_eq!(phh.sb, 50);
        assert_eq!(phh.bb, 100);
        assert_eq!(phh.players[3], "Pluribus");
        assert_eq!(phh.actions.len(), 13);
        assert_eq!(phh.finishing[0], 10325);
    }

    /// Replaying the sample hand must (a) stay in sync with the engine,
    /// (b) record exactly one Pluribus decision (its preflop fold), and
    /// (c) reproduce the logged finishing stacks exactly.
    #[test]
    fn replays_the_sample_hand_with_exact_chip_accounting() {
        let phh = parse_phh(SAMPLE).unwrap();
        let policy = empty_policy();
        let (decisions, chips_ok) = match replay_hand(&phh, &policy, 1) {
            Ok(x) => x,
            Err(ReplayError::Desync) => panic!("desync"),
            Err(ReplayError::Unsupported) => panic!("unsupported"),
        };
        assert_eq!(decisions.len(), 1, "Pluribus acted once (folded p4)");
        assert_eq!(decisions[0].street, Street::Preflop);
        assert!(decisions[0].prob.is_none(), "empty blueprint: no coverage");
        assert_eq!(chips_ok, Some(true), "chip accounting must match the log");
    }

    /// With a blueprint that always folds, the fold decision is covered and
    /// counted as our top action.
    #[test]
    fn coverage_and_top1_with_synthetic_blueprint() {
        use crate::cfr::make_key;
        let phh = parse_phh(SAMPLE).unwrap();

        // Pluribus (p4, seat 3) folded facing no raise... it faced no bet
        // besides the blind: history [f, f] from p3? Seats act p3,p4 first.
        // Build the infoset key the replay will look up: UTG fold then hero.
        let abs = Abstraction::new(AbsConfig {
            postflop_buckets: 6,
            equity_rollouts: 40,
            cache_cap: 100_000,
            ..AbsConfig::default()
        });
        let cfg = HandConfig::default();
        let deck = crate::cards::fresh_deck();
        let h = crate::engine::Hand::new(&cfg, 5, deck);
        let n_acts = abs.abstract_actions(&h).len();

        let hole = [parse_card("3s").unwrap(), parse_card("5c").unwrap()];
        let bucket = crate::abstraction::preflop_bucket(hole);
        let mut strategies = HashMap::new();
        let mut s = vec![0.0f32; n_acts];
        s[0] = 1.0; // always fold
        strategies.insert(
            make_key(bucket, &[crate::abstraction::AbsAction::Fold.token()]).to_vec(),
            s,
        );
        let bp = Blueprint {
            strategies,
            iterations: 1,
            num_players: 6,
            abs_cfg: AbsConfig::default(),
            centroids: None,
        };
        let policy = Policy::new(bp, Arc::new(abs));

        let (decisions, _) = replay_hand(&phh, &policy, 1).ok().unwrap();
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].prob, Some(1.0));
        assert!(decisions[0].top1);
    }
}
