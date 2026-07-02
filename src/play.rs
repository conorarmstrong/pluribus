//! Interactive terminal play: you (seat 0) vs bots.
//! Stacks reset every hand (as in the Pluribus experiment); the button rotates.

use crate::abstraction::AbsAction;
use crate::bot::{Policy, SearchParams};
use crate::cards::{cards_str, fresh_deck};
use crate::cfr::TrainConfig;
use crate::engine::{HandConfig, PlayerAction, Street};
use crate::search::RangeTracker;
use crate::table::Table;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::io::Write;

pub struct PlayOpts {
    pub cfg: HandConfig,
    pub search: Option<SearchParams>,
    pub seed: u64,
}

const HUMAN: usize = 0;

pub fn play(policy: &Policy, opts: &PlayOpts) {
    let n = opts.cfg.num_players;
    let mut rng = SmallRng::seed_from_u64(opts.seed);
    let mut session_bb = 0.0f64;
    let mut hand_no = 0u64;
    let train_cfg = TrainConfig {
        hand: opts.cfg.clone(),
        ..TrainConfig::default()
    };

    println!("You are seat 0. {} bots at the table.", n - 1);
    println!("Actions: (f)old, (c)heck/call, r <amount> = raise TO <amount>, (a)ll-in, (q)uit\n");

    loop {
        hand_no += 1;
        let button = (hand_no as usize - 1) % n;
        let mut deck = fresh_deck();
        deck.shuffle(&mut rng);
        let mut table = Table::new(&opts.cfg, button, deck);

        println!("=== Hand #{hand_no} — button: {} ===", seat_name(button));
        println!("Your cards: {}", cards_str(&table.real.hole(HUMAN)));

        // Tracked ranges for every seat, Bayes-updated on each action; feeds
        // the bots' range-aware subgame search.
        let mut tracker = RangeTracker::new(n);
        let mut last_street = Street::Preflop;
        while !table.real.is_terminal() {
            if table.real.street() != last_street {
                last_street = table.real.street();
                println!(
                    "--- {:?}: [{}]  pot {} ---",
                    last_street,
                    cards_str(table.real.board()),
                    table.real.pot()
                );
            }
            let p = table.real.to_act();
            let board_before = table.real.board().len();
            if p == HUMAN {
                match prompt_human(&table) {
                    Some(act) => {
                        let abs_a = table.map_concrete(act, &policy.abs);
                        tracker.observe(
                            p,
                            abs_a,
                            &table.shadow,
                            &table.hist,
                            &policy.blueprint,
                            &policy.abs,
                        );
                        table.apply_concrete(act, &policy.abs);
                    }
                    None => {
                        print_session(session_bb, hand_no - 1);
                        return;
                    }
                }
            } else {
                let a = match opts.search {
                    Some(params) if table.real.street() != Street::Preflop => policy
                        .act_with_search(
                            &table.shadow,
                            &table.hist,
                            params,
                            &train_cfg,
                            Some(&tracker),
                            &mut rng,
                        ),
                    _ => policy.act_blueprint(&table.shadow, &table.hist, &mut rng),
                };
                let concrete = policy.abs.concrete(&table.shadow, a);
                describe_action(p, a, concrete, &table);
                tracker.observe(
                    p,
                    a,
                    &table.shadow,
                    &table.hist,
                    &policy.blueprint,
                    &policy.abs,
                );
                table.apply_abs(a, &policy.abs);
            }
            let board = table.real.board();
            if board.len() > board_before {
                tracker.exclude(&board[board_before..]);
            }
        }

        // Hand over: show results.
        let u = table.real.utilities();
        if table.real.board().is_empty() {
            println!("(no flop)");
        } else {
            println!("Board: [{}]", cards_str(table.real.board()));
        }
        for p in 0..n {
            if !table.real.folded(p) && table.real.live_count() > 1 && p != HUMAN {
                println!("  {} shows {}", seat_name(p), cards_str(&table.real.hole(p)));
            }
        }
        let bb = opts.cfg.bb as f64;
        session_bb += u[HUMAN] as f64 / bb;
        println!(
            "Result: you {}{} chips  |  session: {:+.1} BB\n",
            if u[HUMAN] >= 0 { "+" } else { "" },
            u[HUMAN],
            session_bb
        );
    }
}

/// Prompt for the human's action; `None` means quit. The caller applies it.
fn prompt_human(table: &Table) -> Option<PlayerAction> {
    let h = &table.real;
    let to_call = h.to_call();
    let stack = h.stack(HUMAN);
    println!(
        "Your turn. pot {} | to call {} | your stack {} | bet this street {}",
        h.pot(),
        to_call,
        stack,
        h.street_commit(HUMAN)
    );
    if let Some((lo, hi)) = h.raise_bounds() {
        println!("  raise TO between {lo} and {hi}");
    }
    loop {
        print!("> ");
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line).unwrap_or(0) == 0 {
            return None;
        }
        let line = line.trim().to_ascii_lowercase();
        let mut parts = line.split_whitespace();
        let action = match parts.next() {
            Some("q") | Some("quit") => return None,
            Some("f") | Some("fold") => Some(PlayerAction::Fold),
            Some("c") | Some("k") | Some("call") | Some("check") => {
                Some(PlayerAction::CheckCall)
            }
            Some("a") | Some("allin") | Some("all-in") => table
                .real
                .raise_bounds()
                .map(|(_, hi)| PlayerAction::RaiseTo(hi))
                .or(Some(PlayerAction::CheckCall)),
            Some("r") | Some("b") | Some("raise") | Some("bet") => {
                match parts.next().and_then(|s| s.parse::<u32>().ok()) {
                    Some(x) if table.real.raise_bounds().is_some() => {
                        Some(PlayerAction::RaiseTo(x))
                    }
                    Some(_) => {
                        println!("you can't raise here");
                        None
                    }
                    None => {
                        println!("usage: r <amount>  (raise TO that total)");
                        None
                    }
                }
            }
            _ => {
                println!("(f)old, (c)heck/call, r <amount>, (a)ll-in, (q)uit");
                None
            }
        };
        if let Some(a) = action {
            return Some(a);
        }
    }
}

fn describe_action(p: usize, a: AbsAction, concrete: PlayerAction, table: &Table) {
    let name = seat_name(p);
    match (a, concrete) {
        (AbsAction::Fold, _) => println!("{name}: folds"),
        (AbsAction::CheckCall, _) => {
            if table.real.to_call() == 0 {
                println!("{name}: checks");
            } else {
                println!("{name}: calls {}", table.real.to_call());
            }
        }
        (AbsAction::AllIn, PlayerAction::RaiseTo(x)) => println!("{name}: ALL-IN to {x}"),
        (AbsAction::Bet(_), PlayerAction::RaiseTo(x)) => println!("{name}: raises to {x}"),
        _ => println!("{name}: {:?}", concrete),
    }
}

fn seat_name(p: usize) -> String {
    if p == HUMAN {
        "You".to_string()
    } else {
        format!("Bot {p}")
    }
}

fn print_session(session_bb: f64, hands: u64) {
    println!("\nSession over: {hands} hands, net {session_bb:+.1} BB. Thanks for playing!");
}
