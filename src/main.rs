mod abstraction;
mod benchmark;
mod bot;
mod cards;
mod cfr;
mod engine;
mod eval;
mod lbr;
mod play;
mod river;
mod search;
mod table;

use abstraction::{AbsConfig, Abstraction, Centroids};
use bot::{Policy, SearchParams};
use cfr::{Blueprint, TrainConfig, Trainer};
use clap::{Parser, Subcommand};
use engine::HandConfig;
use indicatif::{ProgressBar, ProgressStyle};
use std::sync::Arc;
use table::Baseline;

#[derive(Parser)]
#[command(
    name = "pluribus",
    about = "Pluribus-style no-limit hold'em bot: Linear MCCFR blueprint + online subgame resolving"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Train a blueprint strategy with parallel Linear MCCFR.
    Train {
        /// MCCFR iterations (traversals).
        #[arg(long, default_value_t = 1_000_000)]
        iters: u64,
        /// Output blueprint file.
        #[arg(long, default_value = "blueprint.bin")]
        out: String,
        #[arg(long, default_value_t = 6)]
        players: usize,
        /// Resume from a training checkpoint.
        #[arg(long)]
        resume: Option<String>,
        /// Also write a training checkpoint here (enables later --resume).
        #[arg(long)]
        checkpoint: Option<String>,
        /// Postflop equity buckets per street.
        #[arg(long, default_value_t = 12)]
        buckets: u16,
        /// Monte Carlo rollouts per river equity estimate.
        #[arg(long, default_value_t = 200)]
        rollouts: u32,
        /// Sampled future boards per flop/turn distribution estimate.
        #[arg(long, default_value_t = 24)]
        runouts: u32,
        /// Situations sampled per street when training k-means centroids.
        #[arg(long, default_value_t = 30_000)]
        kmeans_samples: usize,
        /// Use raw-equity bucketing instead of EMD k-means clustering.
        #[arg(long)]
        raw_buckets: bool,
        /// Disable negative-regret pruning.
        #[arg(long)]
        no_prune: bool,
        /// Worker threads (default: all cores).
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Play interactively against the bot (you are seat 0).
    Play {
        #[arg(long, default_value = "blueprint.bin")]
        blueprint: String,
        #[arg(long, default_value_t = 6)]
        players: usize,
        /// Enable online subgame resolving for the bots' postflop decisions.
        #[arg(long)]
        search: bool,
        /// Time budget per searched decision, in milliseconds.
        #[arg(long, default_value_t = 2_000)]
        search_ms: u64,
        /// Model opponents as lambda-rational (logit QRE) during search:
        /// 0 = uniform random, higher = more rational; omit for equilibrium.
        #[arg(long)]
        qre_lambda: Option<f64>,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Measure blueprint winrate against baseline opponents (mbb/hand).
    Eval {
        #[arg(long, default_value = "blueprint.bin")]
        blueprint: String,
        #[arg(long, default_value_t = 100_000)]
        hands: u64,
        /// Opponent type in the other seats: random | caller.
        #[arg(long, default_value = "random")]
        baseline: Baseline,
        #[arg(long, default_value_t = 6)]
        players: usize,
        /// Duplicate mode: play each deal once per seat (hero rotating) and
        /// average within the deal — large variance reduction, same estimand.
        #[arg(long)]
        duplicate: bool,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Lower-bound the blueprint's exploitability with a Local Best Response
    /// agent (Lisý & Bowling 2017): heads-up blind vs blind, other seats fold.
    Lbr {
        #[arg(long, default_value = "blueprint.bin")]
        blueprint: String,
        #[arg(long, default_value_t = 20_000)]
        hands: u64,
        /// Board completions sampled per equity estimate.
        #[arg(long, default_value_t = 100)]
        runouts: u32,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Print blueprint statistics.
    Inspect {
        #[arg(long, default_value = "blueprint.bin")]
        blueprint: String,
    },
    /// Replay the 10,000 real Pluribus hands (PHH dataset) and measure how
    /// often the blueprint agrees with Pluribus's actual decisions.
    Benchmark {
        #[arg(long, default_value = "blueprint.bin")]
        blueprint: String,
        /// Directory of .phh files (searched recursively).
        #[arg(long, default_value = "data/pluribus")]
        dir: String,
    },
}

fn main() {
    match Cli::parse().cmd {
        Cmd::Train {
            iters,
            out,
            players,
            resume,
            checkpoint,
            buckets,
            rollouts,
            runouts,
            kmeans_samples,
            raw_buckets,
            no_prune,
            threads,
        } => {
            if let Some(t) = threads {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(t)
                    .build_global()
                    .expect("failed to build thread pool");
            }
            let abs_cfg = AbsConfig {
                postflop_buckets: buckets,
                equity_rollouts: rollouts,
                dist_runouts: runouts,
                ..AbsConfig::default()
            };
            let train_cfg = TrainConfig {
                hand: HandConfig {
                    num_players: players,
                    ..HandConfig::default()
                },
                prune_after: if no_prune { u64::MAX } else { 200_000 },
                ..TrainConfig::default()
            };
            let trainer = match &resume {
                Some(path) => {
                    let t = Trainer::load_checkpoint(path, train_cfg)
                        .unwrap_or_else(|e| die(&format!("cannot load checkpoint {path}: {e}")));
                    println!(
                        "resumed from {path}: {} iterations, {} infosets",
                        t.iterations(),
                        t.node_count()
                    );
                    t
                }
                None => {
                    let centroids = if raw_buckets {
                        None
                    } else {
                        println!(
                            "training EMD k-means centroids ({buckets} buckets, \
                             {kmeans_samples} samples/street)..."
                        );
                        let t0 = std::time::Instant::now();
                        let c = Centroids::train(&abs_cfg, kmeans_samples, 0xC1A5);
                        println!("centroids trained in {:.1}s", t0.elapsed().as_secs_f64());
                        Some(c)
                    };
                    Trainer::new(
                        Arc::new(Abstraction::with_centroids(abs_cfg, centroids)),
                        train_cfg,
                    )
                }
            };

            println!(
                "training {players}-max, {iters} iterations, pruning {}",
                if no_prune { "off" } else { "on" }
            );
            let pb = ProgressBar::new(iters);
            pb.set_style(
                ProgressStyle::with_template(
                    "{bar:40.cyan/blue} {pos}/{len} ({per_sec}, ETA {eta})",
                )
                .unwrap(),
            );
            let started = std::time::Instant::now();

            // Train in chunks so long runs checkpoint periodically.
            let chunk = if checkpoint.is_some() {
                iters.div_ceil(20).max(100_000).min(iters)
            } else {
                iters
            };
            let mut done_before = 0u64;
            while done_before < iters {
                let this = chunk.min(iters - done_before);
                let base = done_before;
                trainer.run(this, &|done| pb.set_position(base + done));
                done_before += this;
                if let Some(path) = &checkpoint {
                    trainer
                        .save_checkpoint(path)
                        .unwrap_or_else(|e| die(&format!("checkpoint save failed: {e}")));
                }
            }
            pb.finish();

            let secs = started.elapsed().as_secs_f64();
            println!(
                "trained {iters} iterations in {:.1}s ({:.0} iters/s), {} infosets",
                secs,
                iters as f64 / secs,
                trainer.node_count()
            );
            let bp = trainer.to_blueprint();
            bp.save(&out)
                .unwrap_or_else(|e| die(&format!("blueprint save failed: {e}")));
            println!(
                "blueprint saved to {out} ({} strategies)",
                bp.strategies.len()
            );
        }

        Cmd::Play {
            blueprint,
            players,
            search,
            search_ms,
            qre_lambda,
            seed,
        } => {
            let policy = load_policy(&blueprint);
            let opts = play::PlayOpts {
                cfg: HandConfig {
                    num_players: players,
                    ..HandConfig::default()
                },
                search: search.then_some(SearchParams {
                    time_ms: search_ms,
                    qre_lambda,
                    ..SearchParams::default()
                }),
                seed,
            };
            play::play(&policy, &opts);
        }

        Cmd::Eval {
            blueprint,
            hands,
            baseline,
            players,
            duplicate,
            seed,
        } => {
            let policy = load_policy(&blueprint);
            let cfg = HandConfig {
                num_players: players,
                ..HandConfig::default()
            };
            println!(
                "evaluating {} hands vs {:?} baselines ({}-max{})...",
                hands,
                baseline,
                players,
                if duplicate { ", duplicate deals" } else { "" }
            );
            let started = std::time::Instant::now();
            let r = if duplicate {
                table::run_eval_duplicate(&policy, &cfg, baseline, hands / players as u64, seed)
            } else {
                table::run_eval(&policy, &cfg, baseline, hands, seed)
            };
            println!(
                "winrate: {:+.1} mbb/hand (95% CI ±{:.1}) over {} hands in {:.1}s",
                r.mbb_per_hand,
                r.ci95,
                r.hands,
                started.elapsed().as_secs_f64()
            );
        }

        Cmd::Lbr {
            blueprint,
            hands,
            runouts,
            seed,
        } => {
            let policy = load_policy(&blueprint);
            let cfg = HandConfig {
                num_players: policy.blueprint.num_players,
                ..HandConfig::default()
            };
            println!(
                "LBR probe: {hands} hands blind-vs-blind ({}-max game, {runouts} runouts)...",
                cfg.num_players
            );
            let started = std::time::Instant::now();
            let r = lbr::run_lbr(&policy, &cfg, hands, runouts, seed);
            println!(
                "LBR wins {:+.1} mbb/hand (95% CI ±{:.1}) over {} hands in {:.1}s",
                r.mbb_per_hand,
                r.ci95,
                r.hands,
                started.elapsed().as_secs_f64()
            );
            println!("(lower bound on the blueprint's exploitability; 0 = unexploited)");
        }

        Cmd::Inspect { blueprint } => {
            let bp = Blueprint::load(&blueprint)
                .unwrap_or_else(|e| die(&format!("cannot load {blueprint}: {e}")));
            println!("blueprint: {blueprint}");
            println!("  trained iterations: {}", bp.iterations);
            println!("  players: {}", bp.num_players);
            println!("  infosets: {}", bp.strategies.len());
            // Street distribution: count separator tokens after the 2 bucket bytes.
            let mut by_street = [0u64; 4];
            for k in bp.strategies.keys() {
                let seps = k[2..]
                    .iter()
                    .filter(|&&t| t == abstraction::TOKEN_STREET_SEP)
                    .count()
                    .min(3);
                by_street[seps] += 1;
            }
            for (i, name) in ["preflop", "flop", "turn", "river"].iter().enumerate() {
                println!("  {name}: {}", by_street[i]);
            }
        }

        Cmd::Benchmark { blueprint, dir } => {
            let policy = load_policy(&blueprint);
            println!("replaying Pluribus hands from {dir}...");
            let started = std::time::Instant::now();
            let r = benchmark::run(&dir, &policy)
                .unwrap_or_else(|e| die(&format!("benchmark failed: {e}")));
            println!(
                "replayed {} hands in {:.1}s ({} skipped, {} desynced)",
                r.hands,
                started.elapsed().as_secs_f64(),
                r.skipped,
                r.desynced
            );
            if r.chip_checked > 0 {
                println!(
                    "chip accounting: {}/{} hands match the logged finishing stacks",
                    r.chip_checked - r.chip_mismatch,
                    r.chip_checked
                );
            }
            println!(
                "\n{:<8} {:>10} {:>10} {:>12} {:>12}",
                "street", "decisions", "covered", "top-1 agree", "mean prob"
            );
            let mut tot = benchmark::StreetStats::default();
            for (i, name) in ["preflop", "flop", "turn", "river"].iter().enumerate() {
                let s = &r.by_street[i];
                print_bench_row(name, s);
                tot.decisions += s.decisions;
                tot.covered += s.covered;
                tot.top1 += s.top1;
                tot.prob_sum += s.prob_sum;
            }
            print_bench_row("TOTAL", &tot);
        }
    }
}

fn print_bench_row(name: &str, s: &benchmark::StreetStats) {
    let pct = |num: u64, den: u64| {
        if den == 0 {
            "-".to_string()
        } else {
            format!("{:.1}%", 100.0 * num as f64 / den as f64)
        }
    };
    let mean_prob = if s.covered == 0 {
        "-".to_string()
    } else {
        format!("{:.3}", s.prob_sum / s.covered as f64)
    };
    println!(
        "{:<8} {:>10} {:>10} {:>12} {:>12}",
        name,
        s.decisions,
        pct(s.covered, s.decisions),
        pct(s.top1, s.covered),
        mean_prob
    );
}

fn load_policy(path: &str) -> Policy {
    let bp = Blueprint::load(path).unwrap_or_else(|e| {
        die(&format!(
            "cannot load blueprint '{path}': {e}\nrun `pluribus train --out {path}` first"
        ))
    });
    println!(
        "loaded blueprint: {} infosets from {} iterations ({} card buckets, {})",
        bp.strategies.len(),
        bp.iterations,
        bp.abs_cfg.postflop_buckets,
        if bp.centroids.is_some() {
            "EMD k-means"
        } else {
            "raw equity"
        }
    );
    let abs = Abstraction::with_centroids(bp.abs_cfg.clone(), bp.centroids.clone());
    Policy::new(bp, Arc::new(abs))
}

fn die(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1)
}
