//! Slumbot API adapter: play the blueprint (with optional online search)
//! against Slumbot over its public HTTP API — heads-up NLHE, 200bb stacks
//! resetting per hand, 50/100 blinds (slumbot.com).
//!
//! Protocol (per slumbot.com/sample_api.py): POST JSON to
//! /slumbot/api/{login,new_hand,act}. The response's `action` string is the
//! cumulative hand action ("k" check, "c" call, "f" fold, "bNNN" = the
//! bettor has put NNN chips in ON THIS STREET, "/" street separator, e.g.
//! "b200c/kb400"); we parse incrementally past a cursor. `client_pos` 0
//! means we are the big blind, 1 the small blind. A `winnings` field means
//! the hand is over. Slumbot's bet sizes are arbitrary — the nested
//! re-solving path prices them at their real size while `map_raise` keeps
//! blueprint histories on-tree.
//!
//! The network sits behind a `Transport` trait so the whole protocol state
//! machine is testable against scripted responses.

use crate::abstraction::AbsAction;
use crate::bot::{Policy, SearchParams, SearchSession};
use crate::cards::{parse_card, Card};
use crate::cfr::TrainConfig;
use crate::engine::{HandConfig, PlayerAction};
use crate::search::RangeTracker;
use crate::table::Table;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use serde_json::Value;

pub const STACK: u32 = 20_000;

/// One parsed increment of Slumbot's action string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlumAct {
    Check,
    Call,
    Fold,
    /// Bet TO this many chips on the current street.
    BetTo(u32),
    StreetSep,
}

/// Parse the action-string suffix starting at byte `from`. Returns the
/// increments and the new cursor. Unknown characters error.
pub fn parse_incr(action: &str, from: usize) -> Result<(Vec<SlumAct>, usize), String> {
    let bytes = action.as_bytes();
    let mut out = Vec::new();
    let mut i = from;
    while i < bytes.len() {
        match bytes[i] {
            b'k' => {
                out.push(SlumAct::Check);
                i += 1;
            }
            b'c' => {
                out.push(SlumAct::Call);
                i += 1;
            }
            b'f' => {
                out.push(SlumAct::Fold);
                i += 1;
            }
            b'/' => {
                out.push(SlumAct::StreetSep);
                i += 1;
            }
            b'b' => {
                let j = i + 1;
                let mut k = j;
                while k < bytes.len() && bytes[k].is_ascii_digit() {
                    k += 1;
                }
                if k == j {
                    return Err(format!("missing bet size at {i} in '{action}'"));
                }
                let amt: u32 = action[j..k]
                    .parse()
                    .map_err(|e| format!("bad bet size in '{action}': {e}"))?;
                out.push(SlumAct::BetTo(amt));
                i = k;
            }
            other => {
                return Err(format!(
                    "unexpected character '{}' at {i} in '{action}'",
                    other as char
                ));
            }
        }
    }
    Ok((out, i))
}

/// Our concrete engine action → Slumbot incr string. The engine treats a
/// fold-when-free as a check; Slumbot rejects it, so map it to "k".
pub fn to_incr(a: PlayerAction, to_call: u32) -> String {
    match a {
        PlayerAction::Fold => {
            if to_call == 0 {
                "k".into()
            } else {
                "f".into()
            }
        }
        PlayerAction::CheckCall => {
            if to_call == 0 {
                "k".into()
            } else {
                "c".into()
            }
        }
        PlayerAction::RaiseTo(x) => format!("b{x}"),
    }
}

/// Network abstraction: the real client speaks HTTPS; tests script it.
pub trait Transport {
    fn new_hand(&mut self, token: Option<&str>) -> Result<Value, String>;
    fn act(&mut self, token: &str, incr: &str) -> Result<Value, String>;
}

pub struct HttpTransport;

impl HttpTransport {
    fn post(&self, endpoint: &str, body: Value) -> Result<Value, String> {
        let url = format!("https://slumbot.com/slumbot/api/{endpoint}");
        let resp = ureq::post(&url)
            .timeout(std::time::Duration::from_secs(30))
            .set("Content-Type", "application/json")
            .send_json(body)
            .map_err(|e| format!("{endpoint}: {e}"))?;
        let v: Value = resp
            .into_json()
            .map_err(|e| format!("{endpoint}: bad JSON: {e}"))?;
        if let Some(msg) = v.get("error_msg").and_then(|m| m.as_str()) {
            return Err(format!("{endpoint}: server error: {msg}"));
        }
        Ok(v)
    }

    pub fn login(&self, username: &str, password: &str) -> Result<String, String> {
        let v = self.post(
            "login",
            serde_json::json!({"username": username, "password": password}),
        )?;
        v.get("token")
            .and_then(|t| t.as_str())
            .map(String::from)
            .ok_or_else(|| "login: no token in response".into())
    }
}

impl Transport for HttpTransport {
    fn new_hand(&mut self, token: Option<&str>) -> Result<Value, String> {
        let body = match token {
            Some(t) => serde_json::json!({ "token": t }),
            None => serde_json::json!({}),
        };
        self.post("new_hand", body)
    }

    fn act(&mut self, token: &str, incr: &str) -> Result<Value, String> {
        self.post("act", serde_json::json!({"token": token, "incr": incr}))
    }
}

pub struct SlumbotCfg {
    pub hands: u64,
    pub search: Option<SearchParams>,
    pub seed: u64,
    pub token: Option<String>,
    pub verbose: bool,
    /// Append each completed hand's final API response as one JSON line
    /// (opponent-modeling data: action string, board, shown cards, winnings).
    pub log: Option<String>,
}

#[derive(Debug)]
pub struct SlumbotResult {
    pub hands: u64,
    /// Our winnings in mbb/hand (positive = beating Slumbot).
    pub mbb_per_hand: f64,
    pub ci95: f64,
    /// Hands aborted on protocol desync (excluded from the mean).
    pub desyncs: u64,
    /// Loss autopsy: aggregates by how each hand ended.
    pub autopsy: Autopsy,
}

/// Where the money goes: per ending-street × ending-kind totals, keyed by
/// the final action string's street count and whether the hand ended by
/// our fold, their fold, or showdown. mbb sums (not means) so cells are
/// additive; divide by count for per-hand rates.
#[derive(Debug, Default, Clone)]
pub struct Autopsy {
    /// [street 0..4][0 = we folded, 1 = they folded, 2 = showdown]
    pub count: [[u64; 3]; 4],
    pub mbb: [[f64; 3]; 4],
}

impl Autopsy {
    fn record(&mut self, action: &str, we_sent_fold: bool, mbb: f64) {
        // Trailing '/'s are dealt run-out streets after an all-in, not
        // streets with betting: bin by where betting actually ended.
        let betting = action.trim_end_matches('/');
        let st = betting.bytes().filter(|&b| b == b'/').count().min(3);
        let folded = betting.ends_with('f');
        let kind = if !folded {
            2
        } else if we_sent_fold {
            0
        } else {
            1
        };
        self.count[st][kind] += 1;
        self.mbb[st][kind] += mbb;
    }

    pub fn report(&self) -> String {
        let streets = ["preflop", "flop", "turn", "river"];
        let kinds = ["we fold", "they fold", "showdown"];
        let mut s = String::from(
            "loss autopsy (per ending street/kind: hands, total bb, bb/hand):\n",
        );
        for (si, street) in streets.iter().enumerate() {
            for (ki, kind) in kinds.iter().enumerate() {
                let n = self.count[si][ki];
                if n == 0 {
                    continue;
                }
                let bb = self.mbb[si][ki] / 1000.0;
                s += &format!(
                    "  {street:<7} {kind:<9} {n:>5}  {bb:>+9.1}  {:>+7.2}\n",
                    bb / n as f64
                );
            }
        }
        s
    }
}

/// Mirror of the table state for one Slumbot hand.
struct HandState {
    table: Table,
    tracker: RangeTracker,
    session: SearchSession,
    /// Cursor into the cumulative action string.
    cursor: usize,
    /// Board cards already forced into the engine.
    board_forced: usize,
    our_seat: usize,
    bot_seat: usize,
}

impl HandState {
    /// Build from the first response of a hand. Our seat is always engine
    /// seat 0; `client_pos` 1 means we are the small blind = button.
    fn build(r: &Value) -> Result<HandState, String> {
        let client_pos = r
            .get("client_pos")
            .and_then(|p| p.as_u64())
            .ok_or("no client_pos")?;
        let holes: Vec<Card> = r
            .get("hole_cards")
            .and_then(|h| h.as_array())
            .ok_or("no hole_cards")?
            .iter()
            .filter_map(|c| c.as_str().and_then(parse_card))
            .collect();
        if holes.len() != 2 {
            return Err("bad hole cards".into());
        }
        let cfg = HandConfig {
            num_players: 2,
            stack: STACK,
            sb: 50,
            bb: 100,
        };
        // Engine deal order: seat 0 gets deck[0..2], seat 1 deck[2..4].
        // Opponent cards are unknown: give seat 1 arbitrary non-colliding
        // cards (never read — ranges come from the tracker, showdowns are
        // scored by the API's `winnings`).
        let mut deck = crate::cards::fresh_deck();
        let mut used = [false; 52];
        for &c in &holes {
            used[c as usize] = true;
        }
        deck[0] = holes[0];
        deck[1] = holes[1];
        let mut idx = 2;
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
        let table = Table::new(&cfg, button, deck);
        let mut tracker = RangeTracker::new(2).with_widening();
        // The bot can't hold our cards.
        tracker.set_weight(1, [holes[0], holes[1]], 0.0);
        Ok(HandState {
            table,
            tracker,
            session: SearchSession::new(),
            cursor: 0,
            board_forced: 0,
            our_seat: 0,
            bot_seat: 1,
        })
    }

    /// Force newly revealed board cards into the engine.
    fn sync_board(&mut self, r: &Value) -> Result<(), String> {
        let board: Vec<Card> = r
            .get("board")
            .and_then(|b| b.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|c| c.as_str().and_then(parse_card))
                    .collect()
            })
            .unwrap_or_default();
        for (i, &c) in board.iter().enumerate().skip(self.board_forced) {
            self.table.real.force_board_card(i, c);
            self.table.shadow.force_board_card(i, c);
        }
        if board.len() > self.board_forced {
            let newly = &board[self.board_forced..];
            self.tracker.exclude(newly);
            self.board_forced = board.len();
        }
        Ok(())
    }

    /// Apply Slumbot's newly observed increments (everything past the
    /// cursor that happened before it is our turn again).
    fn apply_bot_increments(&mut self, r: &Value, policy: &Policy) -> Result<(), String> {
        let action = r.get("action").and_then(|a| a.as_str()).unwrap_or("");
        let (incrs, new_cursor) = parse_incr(action, self.cursor)?;
        self.cursor = new_cursor;
        for inc in incrs {
            match inc {
                SlumAct::StreetSep => continue, // engine advances itself
                _ if self.table.real.is_terminal() => {
                    return Err("action after terminal".into());
                }
                _ => {}
            }
            let p = self.table.real.to_act();
            let concrete = match inc {
                SlumAct::Check | SlumAct::Call => PlayerAction::CheckCall,
                SlumAct::Fold => PlayerAction::Fold,
                SlumAct::BetTo(x) => PlayerAction::RaiseTo(x),
                SlumAct::StreetSep => unreachable!(),
            };
            if p == self.bot_seat {
                let abs_a = self.table.map_concrete(concrete, &policy.abs);
                self.tracker.observe(
                    p,
                    abs_a,
                    &self.table.shadow,
                    &self.table.hist,
                    &policy.blueprint,
                    &policy.abs,
                );
            }
            // Our own echoed actions were already applied when we sent
            // them; only apply the bot's.
            if p == self.bot_seat {
                self.table.apply_concrete(concrete, &policy.abs);
            } else {
                return Err(format!(
                    "desync: server reports action for seat {p} (us) we did not send"
                ));
            }
        }
        Ok(())
    }
}

/// Play `cfg.hands` hands against Slumbot. Sequential (the API is a
/// stateful session).
pub fn run(
    policy: &Policy,
    transport: &mut dyn Transport,
    cfg: &SlumbotCfg,
    progress: &mut dyn FnMut(u64, f64),
) -> Result<SlumbotResult, String> {
    let train_cfg = TrainConfig {
        hand: HandConfig {
            num_players: 2,
            stack: STACK,
            sb: 50,
            bb: 100,
        },
        prune_after: u64::MAX,
        ..TrainConfig::default()
    };
    let mut rng = SmallRng::seed_from_u64(cfg.seed);
    let mut token = cfg.token.clone();
    let mut results: Vec<f64> = Vec::with_capacity(cfg.hands as usize);
    let mut desyncs = 0u64;
    let mut autopsy = Autopsy::default();
    let mut log = match &cfg.log {
        Some(p) => Some(std::io::BufWriter::new(
            std::fs::File::options()
                .create(true)
                .append(true)
                .open(p)
                .map_err(|e| format!("log {p}: {e}"))?,
        )),
        None => None,
    };

    'hands: for h in 0..cfg.hands {
        // A dropped connection must not abort a long session: retry
        // new_hand a few times before giving up on the whole run.
        let mut r = {
            let mut attempt = 0;
            loop {
                match transport.new_hand(token.as_deref()) {
                    Ok(v) => break v,
                    Err(e) if attempt < 2 => {
                        attempt += 1;
                        if cfg.verbose {
                            eprintln!("new_hand failed ({e}); retry {attempt}");
                        }
                        std::thread::sleep(std::time::Duration::from_secs(2));
                    }
                    Err(e) => return Err(e),
                }
            }
        };
        if let Some(t) = r.get("token").and_then(|t| t.as_str()) {
            token = Some(t.to_string());
        }
        let mut we_sent_fold = false;
        let mut st = match HandState::build(&r) {
            Ok(s) => s,
            Err(e) => {
                if cfg.verbose {
                    eprintln!("hand {h}: setup failed: {e}");
                }
                desyncs += 1;
                continue;
            }
        };
        loop {
            if let Some(t) = r.get("token").and_then(|t| t.as_str()) {
                token = Some(t.to_string());
            }
            if let Some(w) = r.get("winnings").and_then(|w| w.as_i64()) {
                let mbb = w as f64 / 100.0 * 1000.0;
                let action = r.get("action").and_then(|a| a.as_str()).unwrap_or("");
                autopsy.record(action, we_sent_fold, mbb);
                if let Some(f) = log.as_mut() {
                    use std::io::Write;
                    let _ = writeln!(f, "{r}");
                }
                results.push(mbb); // chips → mbb
                if h % 50 == 49 {
                    let mean = results.iter().sum::<f64>() / results.len() as f64;
                    progress(h + 1, mean);
                }
                continue 'hands;
            }
            if let Err(e) = st.sync_board(&r) {
                if cfg.verbose {
                    eprintln!("hand {h}: board sync failed: {e}");
                }
                desyncs += 1;
                continue 'hands;
            }
            if let Err(e) = st.apply_bot_increments(&r, policy) {
                if cfg.verbose {
                    eprintln!("hand {h}: {e}");
                }
                desyncs += 1;
                continue 'hands;
            }
            // Board can also change on the action that closed a street.
            if let Err(e) = st.sync_board(&r) {
                if cfg.verbose {
                    eprintln!("hand {h}: board sync failed: {e}");
                }
                desyncs += 1;
                continue 'hands;
            }
            if st.table.real.is_terminal() || st.table.real.to_act() != st.our_seat {
                // Server should have ended the hand or acted; ask again by
                // desync-abort (no polling endpoint exists).
                if cfg.verbose {
                    eprintln!("hand {h}: desync: not our turn after increments");
                }
                desyncs += 1;
                continue 'hands;
            }

            let a: AbsAction = match cfg.search {
                Some(params) => policy.act_with_search(
                    &st.table.real,
                    &st.table.shadow,
                    &st.table.hist,
                    params,
                    &train_cfg,
                    Some(&st.tracker),
                    Some(&mut st.session),
                    Some(&st.table.round),
                    &mut rng,
                ),
                None => policy.act_blueprint(&st.table.shadow, &st.table.hist, &mut rng),
            };
            let to_call = st.table.real.to_call();
            let concrete = st.table.apply_abs(a, &policy.abs);
            let incr = to_incr(concrete, to_call);
            we_sent_fold = incr == "f";
            st.cursor += incr.len();
            let tok = token.as_deref().ok_or("no token")?;
            // Mid-hand transport failure: the hand's outcome is unknown,
            // so score nothing and move on rather than kill the session.
            r = match transport.act(tok, &incr) {
                Ok(v) => v,
                Err(e) => {
                    if cfg.verbose {
                        eprintln!("hand {h}: act failed: {e}");
                    }
                    desyncs += 1;
                    continue 'hands;
                }
            };
        }
    }

    if let Some(f) = log.as_mut() {
        use std::io::Write;
        let _ = f.flush();
    }
    let n = results.len().max(1) as f64;
    let mean = results.iter().sum::<f64>() / n;
    let var = results.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>()
        / (results.len().saturating_sub(1).max(1)) as f64;
    Ok(SlumbotResult {
        hands: results.len() as u64,
        mbb_per_hand: mean,
        ci95: 1.96 * (var / n).sqrt(),
        desyncs,
        autopsy,
    })
}

// ---------------------------------------------------------------------------
// Tests (written first, TDD)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::{AbsConfig, Abstraction};
    use crate::cfr::Blueprint;
    use serde_json::json;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn parses_the_documented_action_strings() {
        let (a, cur) = parse_incr("b200c/kk/kk/kb200", 0).unwrap();
        assert_eq!(cur, 17);
        assert_eq!(
            a,
            vec![
                SlumAct::BetTo(200),
                SlumAct::Call,
                SlumAct::StreetSep,
                SlumAct::Check,
                SlumAct::Check,
                SlumAct::StreetSep,
                SlumAct::Check,
                SlumAct::Check,
                SlumAct::StreetSep,
                SlumAct::Check,
                SlumAct::BetTo(200),
            ]
        );

        // All-in call with empty streets.
        let (a, _) = parse_incr("b20000c///", 0).unwrap();
        assert_eq!(a[0], SlumAct::BetTo(20000));
        assert_eq!(&a[2..], &[SlumAct::StreetSep; 3]);

        // Incremental parse from a cursor.
        let (a, cur) = parse_incr("b200c/kb400", 6).unwrap();
        assert_eq!(a, vec![SlumAct::Check, SlumAct::BetTo(400)]);
        assert_eq!(cur, 11);

        assert!(parse_incr("b", 0).is_err());
        assert!(parse_incr("x", 0).is_err());
    }

    #[test]
    fn incr_encoding_matches_protocol() {
        assert_eq!(to_incr(PlayerAction::CheckCall, 0), "k");
        assert_eq!(to_incr(PlayerAction::CheckCall, 300), "c");
        assert_eq!(to_incr(PlayerAction::Fold, 300), "f");
        // Fold-when-free is a check both for the engine and the protocol.
        assert_eq!(to_incr(PlayerAction::Fold, 0), "k");
        assert_eq!(to_incr(PlayerAction::RaiseTo(450), 100), "b450");
    }

    struct Scripted {
        responses: Vec<Value>,
        sent: Vec<String>,
    }

    impl Transport for Scripted {
        fn new_hand(&mut self, _token: Option<&str>) -> Result<Value, String> {
            Ok(self.responses.remove(0))
        }
        fn act(&mut self, _token: &str, incr: &str) -> Result<Value, String> {
            self.sent.push(incr.to_string());
            Ok(self.responses.remove(0))
        }
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
            Arc::new(Abstraction::new(AbsConfig {
                postflop_buckets: 6,
                equity_rollouts: 40,
                dist_runouts: 8,
                runout_rollouts: 20,
                cache_cap: 200_000,
                menu: crate::abstraction::MenuShape::Wide,
            })),
        )
    }

    /// Scripted full hand: we are the big blind (client_pos 0), Slumbot
    /// (small blind) raises to 300, our check/call-fallback policy calls;
    /// streets check through; Slumbot's river bet gets called; the API
    /// reports winnings. The runner must track the whole hand without
    /// desync and score exactly the reported winnings.
    #[test]
    fn plays_a_scripted_hand_end_to_end() {
        let hand0 = json!({
            "token": "T1",
            "client_pos": 0,
            "hole_cards": ["Ac", "9d"],
            "board": [],
            "action": "b300",
        });
        // After our call: flop comes, Slumbot (BB=pos0? no — we are pos 0;
        // postflop WE act first). Server returns state when it's our turn:
        // flop dealt, no bot action yet.
        let after_call = json!({
            "token": "T1",
            "client_pos": 0,
            "hole_cards": ["Ac", "9d"],
            "board": ["Qs", "Js", "Ts"],
            "action": "b300c/",
        });
        // We check; bot checks behind; turn dealt; our turn again.
        let after_flop_check = json!({
            "token": "T1",
            "client_pos": 0,
            "hole_cards": ["Ac", "9d"],
            "board": ["Qs", "Js", "Ts", "3h"],
            "action": "b300c/kk/",
        });
        // We check; bot bets 500; back to us.
        let after_turn_check = json!({
            "token": "T1",
            "client_pos": 0,
            "hole_cards": ["Ac", "9d"],
            "board": ["Qs", "Js", "Ts", "3h"],
            "action": "b300c/kk/kb500",
        });
        // We call; river dealt; we check; bot checks; showdown: we win 800.
        let after_turn_call = json!({
            "token": "T1",
            "client_pos": 0,
            "hole_cards": ["Ac", "9d"],
            "board": ["Qs", "Js", "Ts", "3h", "4d"],
            "action": "b300c/kk/kb500c/",
        });
        let showdown = json!({
            "token": "T1",
            "client_pos": 0,
            "hole_cards": ["Ac", "9d"],
            "board": ["Qs", "Js", "Ts", "3h", "4d"],
            "action": "b300c/kk/kb500c/kk",
            "winnings": 800,
        });
        let mut t = Scripted {
            responses: vec![
                hand0,
                after_call,
                after_flop_check,
                after_turn_check,
                after_turn_call,
                showdown,
            ],
            sent: Vec::new(),
        };
        let policy = empty_policy();
        let cfg = SlumbotCfg {
            hands: 1,
            search: None,
            seed: 3,
            token: None,
            verbose: true,
            log: None,
        };
        let r = run(&policy, &mut t, &cfg, &mut |_, _| {}).unwrap();
        assert_eq!(r.hands, 1);
        assert_eq!(r.desyncs, 0, "scripted hand must not desync");
        // 800 chips = 8 bb = 8000 mbb.
        assert!((r.mbb_per_hand - 8_000.0).abs() < 1e-9);
        // The check/call policy's wire actions for the whole hand.
        assert_eq!(t.sent, vec!["c", "k", "k", "c", "k"]);
    }

    /// Transport whose calls can fail: network errors must not abort the
    /// run (an 8-hour session must survive one dead connection).
    struct Flaky {
        new_hands: Vec<Result<Value, String>>,
        acts: Vec<Result<Value, String>>,
    }

    impl Transport for Flaky {
        fn new_hand(&mut self, _token: Option<&str>) -> Result<Value, String> {
            self.new_hands.remove(0)
        }
        fn act(&mut self, _token: &str, _incr: &str) -> Result<Value, String> {
            self.acts.remove(0)
        }
    }

    fn sb_hand(token: &str) -> Value {
        json!({
            "token": token,
            "client_pos": 1,
            "hole_cards": ["7c", "2d"],
            "board": [],
            "action": "",
        })
    }

    fn sb_hand_won(token: &str) -> Value {
        json!({
            "token": token,
            "client_pos": 1,
            "hole_cards": ["7c", "2d"],
            "board": [],
            "action": "cf",
            "winnings": 100,
        })
    }

    /// With a log path, each completed hand's final API response (action
    /// string, board, both hole-card fields when shown, winnings) must be
    /// appended as one JSON line — the raw material for cloning Slumbot.
    #[test]
    fn logs_one_json_line_per_completed_hand() {
        let path = std::env::temp_dir().join("slumbot_log_test.jsonl");
        let _ = std::fs::remove_file(&path);
        let mut t = Flaky {
            new_hands: vec![Ok(sb_hand("T5")), Ok(sb_hand("T5"))],
            acts: vec![Ok(sb_hand_won("T5")), Ok(sb_hand_won("T5"))],
        };
        let policy = empty_policy();
        let cfg = SlumbotCfg {
            hands: 2,
            search: None,
            seed: 3,
            token: None,
            verbose: false,
            log: Some(path.to_string_lossy().into_owned()),
        };
        let r = run(&policy, &mut t, &cfg, &mut |_, _| {}).unwrap();
        assert_eq!(r.hands, 2);
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2, "one line per completed hand");
        for l in lines {
            let v: Value = serde_json::from_str(l).unwrap();
            assert_eq!(v.get("action").and_then(|a| a.as_str()), Some("cf"));
            assert_eq!(v.get("winnings").and_then(|w| w.as_i64()), Some(100));
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn act_transport_error_counts_desync_and_run_continues() {
        let mut t = Flaky {
            new_hands: vec![Ok(sb_hand("T3")), Ok(sb_hand("T3"))],
            acts: vec![Err("connection reset".into()), Ok(sb_hand_won("T3"))],
        };
        let policy = empty_policy();
        let cfg = SlumbotCfg {
            hands: 2,
            search: None,
            seed: 3,
            token: None,
            verbose: false,
            log: None,
        };
        let r = run(&policy, &mut t, &cfg, &mut |_, _| {}).unwrap();
        assert_eq!(r.desyncs, 1, "the failed hand is a desync, not a crash");
        assert_eq!(r.hands, 1, "the second hand still completes");
        assert!((r.mbb_per_hand - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn new_hand_transient_error_is_retried() {
        let mut t = Flaky {
            new_hands: vec![Err("timed out".into()), Ok(sb_hand("T4"))],
            acts: vec![Ok(sb_hand_won("T4"))],
        };
        let policy = empty_policy();
        let cfg = SlumbotCfg {
            hands: 1,
            search: None,
            seed: 3,
            token: None,
            verbose: false,
            log: None,
        };
        let r = run(&policy, &mut t, &cfg, &mut |_, _| {}).unwrap();
        assert_eq!(r.hands, 1);
        assert_eq!(r.desyncs, 0);
    }

    /// client_pos 1 (we are the small blind / button): we act first
    /// preflop, and our button fold must be scored from the API winnings.
    #[test]
    fn small_blind_position_maps_to_the_button() {
        let hand0 = json!({
            "token": "T2",
            "client_pos": 1,
            "hole_cards": ["7c", "2d"],
            "board": [],
            "action": "",
        });
        let folded = json!({
            "token": "T2",
            "client_pos": 1,
            "hole_cards": ["7c", "2d"],
            "board": [],
            "action": "f",
            "winnings": -50,
        });
        // The empty-policy fallback is check/call, so scripted Slumbot
        // folds to our limp... but the response above says WE folded —
        // instead script: we limp ("c"), bot raises... keep it simple:
        // respond with winnings right after our first action.
        let mut t = Scripted {
            responses: vec![hand0, folded],
            sent: Vec::new(),
        };
        let policy = empty_policy();
        let cfg = SlumbotCfg {
            hands: 1,
            search: None,
            seed: 3,
            token: None,
            verbose: true,
            log: None,
        };
        let r = run(&policy, &mut t, &cfg, &mut |_, _| {}).unwrap();
        assert_eq!(r.hands, 1);
        assert_eq!(t.sent, vec!["c"], "SB completes the blind with a call");
        assert!((r.mbb_per_hand - (-500.0)).abs() < 1e-9);
    }
}
