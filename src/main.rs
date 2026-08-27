mod ablate;
mod abstraction;
mod aivat;
mod benchmark;
mod bot;
mod br;
mod cards;
mod cfr;
mod clone;
mod distill;
mod engine;
mod eval;
mod flop;
mod lbr;
mod net;
mod play;
mod portfolio;
mod river;
mod search;
mod slumbot;
mod table;
mod turn;
mod valuenet;

// Solver walks allocate range-vector scratch at every node visit; mimalloc
// is measurably faster than the system allocator on that churn.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use abstraction::{AbsConfig, Abstraction, Centroids};
use bot::{Policy, SearchParams};
use cfr::{Blueprint, TrainConfig, Trainer};
use clap::{Parser, Subcommand};
use abstraction::MenuShape;
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
        /// Starting stack in chips (blinds are 50/100, so 10,000 = 100bb
        /// — the Pluribus setup — and 20,000 = 200bb — the Slumbot / GTO
        /// Wizard benchmark setup).
        #[arg(long, default_value_t = 10_000)]
        stack: u32,
        /// Resume from a training checkpoint.
        #[arg(long)]
        resume: Option<String>,
        /// Also write a training checkpoint here (enables later --resume).
        #[arg(long)]
        checkpoint: Option<String>,
        /// Postflop equity buckets per street.
        #[arg(long, default_value_t = 12)]
        buckets: u16,
        /// Bet-size menu shape (stored in the blueprint): pluribus (coarse
        /// postflop, Pluribus's shape; the default, measured least
        /// exploitable), wide (2026-07 menu), or pluribus-fine-pre (also a
        /// fine preflop menu; worse at every budget tested).
        #[arg(long, value_enum, default_value_t = MenuShape::Pluribus)]
        menu: MenuShape,
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
        /// Cluster hands by the combined potential-aware feature (equity
        /// quantiles + opponent-cluster hand strength) instead of equity
        /// quantiles alone — a richer, opponent-relative card abstraction.
        /// Mutually exclusive with the other abstraction modes.
        #[arg(long, conflicts_with_all = ["raw_buckets", "strategic_from"])]
        ochs: bool,
        /// Strategy-aware co-training: cluster hands by how THIS previous
        /// blueprint plays and realizes them, instead of by equity
        /// distributions. The resulting blueprint must be loaded with
        /// --strat-prev pointing at the same file.
        #[arg(long)]
        strategic_from: Option<String>,
        /// Traversal RNG seed: distinct seeds give independent self-play
        /// runs (equilibrium-selection studies).
        #[arg(long, default_value_t = 0)]
        train_seed: u64,
        /// Restricted Nash response: opponent model (random | caller).
        #[arg(long)]
        rnr_model: Option<Baseline>,
        /// Restricted Nash response against a cloned opponent given as a
        /// blueprint file (e.g. slumbot_clone.bin). Overrides --rnr-model;
        /// the trainer adopts the clone's abstraction so keys align.
        #[arg(long, conflicts_with = "rnr_model")]
        rnr_opponent: Option<String>,
        /// RNR mixture weight p (0 = plain equilibrium, 1 = pure best
        /// response to the model). Requires --rnr-model or --rnr-opponent.
        #[arg(long, default_value_t = 0.5)]
        rnr_p: f64,
        /// Disable negative-regret pruning.
        #[arg(long)]
        no_prune: bool,
        /// VR-MCCFR: learned control-variate baselines at sampled opponent
        /// nodes (Schmid et al. 2019). Unbiased; cuts the variance of every
        /// regret update, so each visit is worth more.
        #[arg(long)]
        vr_baseline: bool,
        /// Legacy pruning: decide per action and prune everywhere, including
        /// the river and terminal-leading actions. The default is Pluribus's
        /// rules (per traversal, river and terminal actions exempt), which
        /// halved the BR exploitability bound at equal iterations.
        #[arg(long, conflicts_with = "no_prune")]
        per_action_prune: bool,
        /// Pluribus's postflop blueprint: mean of periodic snapshots of the
        /// current strategy (after a 10% warm-up, every 5% of iterations)
        /// instead of the linear running average; preflop keeps the average.
        #[arg(long)]
        snapshot_avg: bool,
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
        /// Belief-state value net enabling ReBeL-style flop solving.
        #[arg(long)]
        value_net: Option<String>,
        /// Safe (gadget) river resolving with rollout-estimated safety
        /// values — bounds exploitation when tracked ranges are wrong.
        #[arg(long)]
        safe_resolve: bool,
        /// Metareasoning: probe-solve first and exit early on decisions
        /// that are already near-pure.
        #[arg(long)]
        adaptive_search: bool,
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
        /// AIVAT mode: unbiased estimator with chance/decision correction
        /// terms (Burch et al. 2018) — strongest variance reduction.
        #[arg(long)]
        aivat: bool,
        /// Hero uses range-tracked online search per decision (slow;
        /// combine with --search-ms).
        #[arg(long)]
        search: bool,
        /// Paired search-gain mode: each deal played twice (hero searching
        /// vs hero on blueprint, everyone else blueprint); reports the mean
        /// per-deal difference — the value added by search itself.
        #[arg(long)]
        search_gain: bool,
        /// Paired net-gain mode: search WITH the value net vs search
        /// without it — isolates the value network's contribution.
        #[arg(long)]
        net_gain: bool,
        /// Per-decision search budget in milliseconds (with --search).
        #[arg(long, default_value_t = 100)]
        search_ms: u64,
        /// Per-decision search iteration cap (with --search).
        #[arg(long, default_value_t = 2_000_000)]
        search_iters: u64,
        /// Belief-state value net for ReBeL flop solving (with --search).
        #[arg(long)]
        value_net: Option<String>,
        /// Previous blueprint for strategic-abstraction lookups.
        #[arg(long)]
        strat_prev: Option<String>,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Tighter exploitability lower bound than `lbr`: same harness, but
    /// turn and river decisions play an exact best response of the whole
    /// remaining game (single expectimax pass, deterministic per seed).
    Br {
        #[arg(long, default_value = "blueprint.bin")]
        blueprint: String,
        #[arg(long, default_value_t = 2_000)]
        hands: u64,
        /// Board completions per equity estimate on the greedy (preflop/
        /// flop) streets.
        #[arg(long, default_value_t = 100)]
        runouts: u32,
        /// Previous blueprint for strategic-abstraction lookups.
        #[arg(long)]
        strat_prev: Option<String>,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// The bot resolves postflop decisions online (the probe then
        /// faces the bot as it actually plays, not the raw blueprint).
        #[arg(long)]
        search: bool,
        /// Per-decision search budget in milliseconds (with --search).
        #[arg(long, default_value_t = 2_000)]
        search_ms: u64,
        /// Per-decision search iteration cap (with --search); a cap makes
        /// the probe independent of machine load.
        #[arg(long, default_value_t = 2_000_000)]
        search_iters: u64,
        /// Root multiway resolves at the current decision instead of the
        /// start of the betting round (the pre-A3 behaviour; A/B arm).
        #[arg(long)]
        decision_root: bool,
        /// Postflop card buckets inside multiway resolves (equity
        /// quantiles; exact per board on the river). Default: the
        /// blueprint's.
        #[arg(long)]
        search_buckets: Option<u16>,
    },
    /// Lower-bound the blueprint's exploitability with a Local Best Response
    /// agent (Lisý & Bowling 2017): heads-up blind vs blind, other seats fold.
    Lbr {
        #[arg(long, default_value = "blueprint.bin")]
        blueprint: String,
        #[arg(long, default_value_t = 20_000)]
        hands: u64,
        /// Multiway probe: LBR in one rotating seat against the bot in
        /// every other seat (pots go multiway); the default probe is
        /// heads-up blind-vs-blind with the other seats folding.
        #[arg(long)]
        multiway: bool,
        /// Board completions sampled per equity estimate.
        #[arg(long, default_value_t = 100)]
        runouts: u32,
        /// Previous blueprint for strategic-abstraction lookups.
        #[arg(long)]
        strat_prev: Option<String>,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// The bot resolves postflop decisions online (the probe then
        /// faces the bot as it actually plays, not the raw blueprint).
        #[arg(long)]
        search: bool,
        /// Per-decision search budget in milliseconds (with --search).
        #[arg(long, default_value_t = 2_000)]
        search_ms: u64,
        /// Per-decision search iteration cap (with --search); a cap makes
        /// the probe independent of machine load.
        #[arg(long, default_value_t = 2_000_000)]
        search_iters: u64,
        /// Root multiway resolves at the current decision instead of the
        /// start of the betting round (the pre-A3 behaviour; A/B arm).
        #[arg(long)]
        decision_root: bool,
        /// Postflop card buckets inside multiway resolves (equity
        /// quantiles; exact per board on the river). Default: the
        /// blueprint's.
        #[arg(long)]
        search_buckets: Option<u16>,
    },
    /// Safety ablation: unsafe vs gadget river resolving under corrupted
    /// range beliefs — reports best-response margins beyond safety values.
    AblateSafety {
        #[arg(long, default_value_t = 40)]
        spots: usize,
        #[arg(long, default_value_t = 400)]
        iters: u64,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Generate exactly-solved turn spots (blueprint self-play + turn
    /// solver) as training data for the belief-state value network.
    GenTurnData {
        #[arg(long, default_value = "blueprint.bin")]
        blueprint: String,
        #[arg(long, default_value = "turn_data.bin")]
        out: String,
        #[arg(long, default_value_t = 10_000)]
        samples: usize,
        /// Vector-CFR iterations per exact turn solve.
        #[arg(long, default_value_t = 200)]
        solve_iters: u64,
        /// Per-solve wall-clock cap in milliseconds.
        #[arg(long, default_value_t = 30_000)]
        solve_ms: u64,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Train the belief-state value network on solved turn spots.
    TrainValueNet {
        /// Training data file(s); comma-separated files are concatenated.
        #[arg(long, default_value = "turn_data.bin")]
        data: String,
        #[arg(long, default_value = "value_net.bin")]
        out: String,
        /// Hidden layer sizes, comma-separated.
        #[arg(long, default_value = "512,512")]
        hidden: String,
        #[arg(long, default_value_t = 50)]
        epochs: usize,
        #[arg(long, default_value_t = 1e-3)]
        lr: f32,
        #[arg(long, default_value_t = 128)]
        batch: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Expert-iteration distillation: self-play with online search, then
    /// blend the resolved distributions back into the blueprint (the
    /// flywheel). Evaluate the output with br/crossplay/eval before use.
    Distill {
        #[arg(long, default_value = "blueprint.bin")]
        blueprint: String,
        #[arg(long, default_value = "distilled.bin")]
        out: String,
        /// Self-play hands (every seat searches).
        #[arg(long, default_value_t = 20_000)]
        hands: u64,
        /// Per-decision search budget in milliseconds.
        #[arg(long, default_value_t = 200)]
        search_ms: u64,
        /// Blend weight toward the search distribution (0 = no change,
        /// 1 = replace outright at recorded infosets).
        #[arg(long, default_value_t = 0.5)]
        alpha: f64,
        /// Belief-state value net for ReBeL flop solving during self-play.
        #[arg(long)]
        value_net: Option<String>,
        /// Gadget-safe resolves during self-play (slower).
        #[arg(long)]
        safe_resolve: bool,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Cross-play two blueprints: --focal in one rotating seat against a
    /// full table of --field. If self-play equilibria were interchangeable,
    /// the result would be ~0 — the multiplayer equilibrium-selection probe.
    Crossplay {
        #[arg(long)]
        focal: String,
        #[arg(long)]
        field: String,
        /// Previous blueprint for strategic-abstraction lookups (applied to
        /// whichever side is strategic).
        #[arg(long)]
        strat_prev: Option<String>,
        #[arg(long, default_value_t = 200_000)]
        hands: u64,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Play against Slumbot over its public API (heads-up NLHE, 200bb,
    /// 50/100) and report our winrate. Train the blueprint with
    /// `train --players 2 --stack 20000`.
    Slumbot {
        #[arg(long, default_value = "bp_hu200.bin")]
        blueprint: String,
        #[arg(long, default_value_t = 1_000)]
        hands: u64,
        /// Enable online subgame resolving (recommended).
        #[arg(long)]
        search: bool,
        #[arg(long, default_value_t = 800)]
        search_ms: u64,
        /// Safe (gadget) resolving.
        #[arg(long)]
        safe_resolve: bool,
        /// Belief-state value net for flop solving (must be trained on
        /// 200bb HU spots to help).
        #[arg(long)]
        value_net: Option<String>,
        /// Registered slumbot.com account (optional; anonymous otherwise).
        #[arg(long)]
        username: Option<String>,
        #[arg(long)]
        password: Option<String>,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Log per-hand protocol problems.
        #[arg(long)]
        verbose: bool,
        /// Append each hand's final API response (JSON lines) — data for
        /// opponent modeling.
        #[arg(long)]
        log: Option<String>,
    },
    /// Build a behavioral clone of Slumbot from a `slumbot --log` JSONL
    /// file: replay each hand, count Slumbot's abstract actions per
    /// infoset, and save the normalized strategy in Blueprint format.
    Clone {
        /// Hand log written by `slumbot --log`.
        #[arg(long, default_value = "slumbot_hands.jsonl")]
        log: String,
        /// Blueprint supplying the abstraction (buckets + centroids) the
        /// exploiter will use — must match at train and play time.
        #[arg(long, default_value = "bp_hu200_300m.bin")]
        blueprint: String,
        #[arg(long, default_value = "slumbot_clone.bin")]
        out: String,
        /// Fraction of hands held out for top-1 agreement measurement.
        #[arg(long, default_value_t = 0.1)]
        holdout: f64,
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
            stack,
            resume,
            checkpoint,
            buckets,
            menu,
            rollouts,
            runouts,
            kmeans_samples,
            raw_buckets,
            ochs,
            strategic_from,
            train_seed,
            rnr_model,
            rnr_opponent,
            rnr_p,
            no_prune,
            vr_baseline,
            per_action_prune,
            snapshot_avg,
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
                menu,
                equity_rollouts: rollouts,
                dist_runouts: runouts,
                ..AbsConfig::default()
            };
            let train_cfg = TrainConfig {
                hand: HandConfig {
                    num_players: players,
                    stack,
                    ..HandConfig::default()
                },
                prune_after: if no_prune { u64::MAX } else { 200_000 },
                seed: train_seed,
                pluribus_prune: !per_action_prune,
                snapshot_avg,
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
                None if rnr_opponent.is_some() => {
                    let path = rnr_opponent.as_ref().unwrap();
                    let clone = Blueprint::load(path)
                        .unwrap_or_else(|e| die(&format!("cannot load clone {path}: {e}")));
                    println!(
                        "RNR vs clone {path}: {} strategies, p = {rnr_p} \
                         (adopting the clone's abstraction)",
                        clone.strategies.len()
                    );
                    // Adopt the clone's abstraction so bucket/history keys align.
                    let abs =
                        Abstraction::with_centroids(clone.abs_cfg.clone(), clone.centroids.clone());
                    let rnr = Some(cfr::RnrCfg {
                        model: Baseline::Caller, // ignored when rnr_opp is set
                        p: rnr_p,
                    });
                    Trainer::new(Arc::new(abs), train_cfg)
                        .with_rnr(rnr)
                        .with_rnr_opponent(Some(Arc::new(clone)))
                }
                None => {
                    let strat_ctx = strategic_from.as_ref().map(|p| load_strat_ctx(p));
                    let centroids = if let Some(sc) = &strat_ctx {
                        println!(
                            "training STRATEGIC centroids from previous blueprint \
                             ({buckets} buckets, {kmeans_samples} samples/street)..."
                        );
                        let t0 = std::time::Instant::now();
                        let c = Centroids::train_strategic(&abs_cfg, kmeans_samples, 0xC1A5, sc);
                        println!("centroids trained in {:.1}s", t0.elapsed().as_secs_f64());
                        Some(c)
                    } else if raw_buckets {
                        None
                    } else if ochs {
                        println!(
                            "training OCHS k-means centroids ({buckets} buckets, \
                             {kmeans_samples} samples/street)..."
                        );
                        let t0 = std::time::Instant::now();
                        let c = Centroids::train_combined(&abs_cfg, kmeans_samples, 0xC1A5);
                        println!("centroids trained in {:.1}s", t0.elapsed().as_secs_f64());
                        Some(c)
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
                    let mut abs = Abstraction::with_centroids(abs_cfg, centroids);
                    if let Some(sc) = strat_ctx {
                        abs = abs.with_strat(sc);
                    }
                    let rnr = rnr_model.map(|model| {
                        println!("RNR training: model {model:?}, p = {rnr_p}");
                        cfr::RnrCfg { model, p: rnr_p }
                    });
                    Trainer::new(Arc::new(abs), train_cfg).with_rnr(rnr)
                }
            }
            .with_vr(vr_baseline);

            println!(
                "training {players}-max, {iters} iterations, menu {menu:?}, pruning {}, \
                 vr baselines {}, snapshot avg {}",
                if no_prune { "off" } else if per_action_prune { "per-action" } else { "pluribus" },
                if vr_baseline { "on" } else { "off" },
                if snapshot_avg { "on" } else { "off" }
            );
            let pb = ProgressBar::new(iters);
            pb.set_style(
                ProgressStyle::with_template(
                    "{bar:40.cyan/blue} {pos}/{len} ({per_sec}, ETA {eta})",
                )
                .unwrap(),
            );
            let started = std::time::Instant::now();

            // Train in chunks so long runs checkpoint periodically (and so
            // snapshot mode can sample the current strategy every 5%).
            let chunk = if checkpoint.is_some() || snapshot_avg {
                iters.div_ceil(20).max(100_000).min(iters)
            } else {
                iters
            };
            let mut done_before = 0u64;
            let mut snapshots = 0u32;
            while done_before < iters {
                let this = chunk.min(iters - done_before);
                let base = done_before;
                trainer.run(this, &|done| pb.set_position(base + done));
                done_before += this;
                if snapshot_avg && done_before * 10 > iters {
                    trainer.snapshot_strategy();
                    snapshots += 1;
                }
                if let Some(path) = &checkpoint {
                    trainer
                        .save_checkpoint(path)
                        .unwrap_or_else(|e| die(&format!("checkpoint save failed: {e}")));
                }
            }
            pb.finish();
            if snapshot_avg {
                println!("postflop blueprint = mean of {snapshots} strategy snapshots");
            }

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
            value_net,
            safe_resolve,
            adaptive_search,
            seed,
        } => {
            let net = value_net.map(|p| {
                let n = valuenet::ValueNet::load(&p)
                    .unwrap_or_else(|e| die(&format!("cannot load value net '{p}': {e}")));
                println!("loaded value net from {p}");
                Arc::new(n)
            });
            let policy = load_policy(&blueprint).with_value_net(net);
            let opts = play::PlayOpts {
                cfg: HandConfig {
                    num_players: players,
                    ..HandConfig::default()
                },
                search: search.then_some(SearchParams {
                    time_ms: search_ms,
                    qre_lambda,
                    safe_resolve,
                    adaptive: adaptive_search,
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
            aivat,
            search,
            search_gain,
            net_gain,
            search_ms,
            search_iters,
            value_net,
            strat_prev,
            seed,
        } => {
            let net = value_net.map(|p| {
                let n = valuenet::ValueNet::load(&p)
                    .unwrap_or_else(|e| die(&format!("cannot load value net '{p}': {e}")));
                println!("loaded value net from {p}");
                Arc::new(n)
            });
            let policy =
                load_policy_strat(&blueprint, strat_prev.as_deref()).with_value_net(net);
            let cfg = HandConfig {
                num_players: players,
                ..HandConfig::default()
            };
            println!(
                "evaluating {} hands vs {:?} baselines ({}-max{})...",
                hands,
                baseline,
                players,
                if net_gain {
                    ", paired net-gain"
                } else if search_gain {
                    ", paired search-gain"
                } else if search {
                    ", with search"
                } else if aivat {
                    ", AIVAT"
                } else if duplicate {
                    ", duplicate deals"
                } else {
                    ""
                }
            );
            let started = std::time::Instant::now();
            let params = SearchParams {
                time_ms: search_ms,
                max_iters: search_iters,
                adaptive: true,
                ..SearchParams::default()
            };
            let r = if net_gain {
                if policy.value_net.is_none() {
                    die("--net-gain needs --value-net");
                }
                let plain = policy.clone_without_net();
                table::run_eval_paired_policies(
                    &policy,
                    Some(params),
                    &plain,
                    Some(params),
                    &cfg,
                    hands,
                    seed,
                )
            } else if search_gain {
                table::run_eval_paired(&policy, &cfg, hands, params, seed)
            } else if search {
                table::run_eval_search(&policy, &cfg, baseline, hands, params, seed)
            } else if aivat {
                aivat::run_eval_aivat(&policy, &cfg, baseline, hands, seed)
            } else if duplicate {
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
            multiway,
            runouts,
            strat_prev,
            seed,
            search,
            search_ms,
            search_iters,
            decision_root,
            search_buckets,
        } => {
            let params = search.then_some(SearchParams {
                time_ms: search_ms,
                max_iters: search_iters,
                round_root: !decision_root,
                ..SearchParams::default()
            });
            let policy = load_policy_strat(&blueprint, strat_prev.as_deref())
                .with_search_buckets(search_buckets);
            let cfg = HandConfig {
                num_players: policy.blueprint.num_players,
                ..HandConfig::default()
            };
            println!(
                "LBR probe: {hands} hands {} ({}-max game, {runouts} runouts{})...",
                if multiway { "multiway, LBR seat rotating" } else { "blind-vs-blind" },
                cfg.num_players,
                if search {
                    format!(", bot searching {search_ms}ms/{search_iters} iters per decision, {} root, {} buckets", if decision_root { "decision" } else { "round" }, search_buckets.map_or("blueprint".to_string(), |b| b.to_string()))
                } else {
                    String::new()
                }
            );
            if search && !multiway {
                die("--search is only wired into the multiway LBR probe (use --multiway)");
            }
            let started = std::time::Instant::now();
            let r = if multiway {
                lbr::run_lbr_multiway(&policy, &cfg, hands, runouts, seed, params)
            } else {
                lbr::run_lbr(&policy, &cfg, hands, runouts, seed)
            };
            println!(
                "LBR wins {:+.1} mbb/hand (95% CI ±{:.1}) over {} hands in {:.1}s",
                r.mbb_per_hand,
                r.ci95,
                r.hands,
                started.elapsed().as_secs_f64()
            );
            println!("(lower bound on the blueprint's exploitability; 0 = unexploited)");
            if let Some(r) = bot::fallback_report() {
                println!("{r}");
            }
        }

        Cmd::Br {
            blueprint,
            hands,
            runouts,
            strat_prev,
            seed,
            search,
            search_ms,
            search_iters,
            decision_root,
            search_buckets,
        } => {
            let params = search.then_some(SearchParams {
                time_ms: search_ms,
                max_iters: search_iters,
                round_root: !decision_root,
                ..SearchParams::default()
            });
            let policy = load_policy_strat(&blueprint, strat_prev.as_deref())
                .with_search_buckets(search_buckets);
            let cfg = HandConfig {
                num_players: policy.blueprint.num_players,
                ..HandConfig::default()
            };
            println!(
                "BR probe: {hands} hands blind-vs-blind ({}-max game), exact \
                 turn/river subgame best response{}...",
                cfg.num_players,
                if search {
                    format!(", bot searching {search_ms}ms/{search_iters} iters per decision, {} root, {} buckets", if decision_root { "decision" } else { "round" }, search_buckets.map_or("blueprint".to_string(), |b| b.to_string()))
                } else {
                    String::new()
                }
            );
            let started = std::time::Instant::now();
            let r = br::run_br(&policy, &cfg, hands, runouts, seed, params);
            println!(
                "BR probe wins {:+.1} mbb/hand (95% CI ±{:.1}) over {} hands in {:.1}s",
                r.mbb_per_hand,
                r.ci95,
                r.hands,
                started.elapsed().as_secs_f64()
            );
            println!("(tighter lower bound on exploitability than lbr; 0 = unexploited)");
        }

        Cmd::AblateSafety { spots, iters, seed } => {
            println!(
                "safety ablation: {spots} random river spots, {iters} CFR iters/solve..."
            );
            let started = std::time::Instant::now();
            let rows = ablate::run(spots, iters, seed);
            println!(
                "\n{:<8} {:>13} {:>12} {:>11} {:>10} {:>9}",
                "epsilon", "unsafe mean", "unsafe max", "safe mean", "safe max", "pot"
            );
            for r in rows {
                println!(
                    "{:<8} {:>13.1} {:>12.1} {:>11.1} {:>10.1} {:>9.0}",
                    r.epsilon, r.unsafe_mean, r.unsafe_max, r.safe_mean, r.safe_max, r.pot
                );
            }
            println!(
                "\n(margins in chips: opponent best-response value beyond its safety \
                 value,\n true-range weighted; {:.1}s)",
                started.elapsed().as_secs_f64()
            );
        }

        Cmd::GenTurnData {
            blueprint,
            out,
            samples,
            solve_iters,
            solve_ms,
            seed,
        } => {
            let policy = load_policy(&blueprint);
            println!(
                "generating {samples} exactly-solved turn spots \
                 ({solve_iters} CFR iters or {solve_ms}ms per solve)..."
            );
            let started = std::time::Instant::now();
            let data = valuenet::generate(&policy, solve_iters, solve_ms, samples, seed, &|done| {
                println!(
                    "  {done}/{samples} samples ({:.1}s elapsed)",
                    started.elapsed().as_secs_f64()
                );
            });
            valuenet::save_samples(&out, &data)
                .unwrap_or_else(|e| die(&format!("cannot write {out}: {e}")));
            println!(
                "wrote {} samples to {out} in {:.1}s",
                data.len(),
                started.elapsed().as_secs_f64()
            );
        }

        Cmd::TrainValueNet {
            data,
            out,
            hidden,
            epochs,
            lr,
            batch,
            seed,
        } => {
            // Comma-separated data files are concatenated (e.g. combining
            // generation batches).
            let mut samples = Vec::new();
            for path in data.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                let mut s = valuenet::load_samples(path)
                    .unwrap_or_else(|e| die(&format!("cannot load {path}: {e}")));
                println!("loaded {} samples from {path}", s.len());
                samples.append(&mut s);
            }
            let hidden: Vec<usize> = hidden
                .split(',')
                .map(|s| s.trim().parse().unwrap_or_else(|_| die("bad --hidden")))
                .collect();
            println!(
                "training value net on {} samples (hidden {hidden:?}, {epochs} epochs)...",
                samples.len()
            );
            let started = std::time::Instant::now();
            let (net, val_loss) = valuenet::train(
                &samples,
                &hidden,
                epochs,
                lr,
                batch,
                seed,
                &mut |e, tr, va| {
                    println!(
                        "  epoch {:>3}: train {:.5}  val {:.5}  ({:.0}s)",
                        e + 1,
                        tr,
                        va,
                        started.elapsed().as_secs_f64()
                    );
                },
            );
            net.save(&out)
                .unwrap_or_else(|e| die(&format!("cannot write {out}: {e}")));
            println!(
                "value net saved to {out} (final val loss {val_loss:.5}, {:.0}s)",
                started.elapsed().as_secs_f64()
            );
        }

        Cmd::Distill {
            blueprint,
            out,
            hands,
            search_ms,
            alpha,
            value_net,
            safe_resolve,
            seed,
        } => {
            let net = value_net.map(|p| {
                let n = valuenet::ValueNet::load(&p)
                    .unwrap_or_else(|e| die(&format!("cannot load value net '{p}': {e}")));
                println!("loaded value net from {p}");
                Arc::new(n)
            });
            let policy = load_policy(&blueprint).with_value_net(net);
            let cfg = HandConfig {
                num_players: policy.blueprint.num_players,
                ..HandConfig::default()
            };
            let dcfg = distill::DistillCfg {
                hands,
                params: SearchParams {
                    time_ms: search_ms,
                    safe_resolve,
                    ..SearchParams::default()
                },
                alpha,
                seed,
            };
            println!(
                "distilling: {hands} self-play hands ({}-max, {search_ms}ms/decision, alpha {alpha})...",
                cfg.num_players
            );
            let started = std::time::Instant::now();
            let (records, samples) = distill::collect(&policy, &cfg, &dcfg);
            println!(
                "collected {samples} searched decisions at {} infosets in {:.1}s",
                records.len(),
                started.elapsed().as_secs_f64()
            );
            // Reclaim sole ownership of the blueprint so the merge mutates
            // in place instead of cloning a multi-GB map.
            let bot::Policy { blueprint: bp_arc, .. } = policy;
            let owned = Arc::try_unwrap(bp_arc)
                .unwrap_or_else(|arc| (*arc).clone());
            let (bp2, updated, skipped) = distill::merge(owned, &records, alpha);
            bp2.save(&out)
                .unwrap_or_else(|e| die(&format!("save failed: {e}")));
            println!(
                "updated {updated} infosets ({skipped} skipped on menu mismatch); \
                 distilled blueprint saved to {out} ({} strategies)",
                bp2.strategies.len()
            );
            println!("gate it: br/lbr, crossplay --focal {out} --field {blueprint}, eval");
        }

        Cmd::Crossplay {
            focal,
            field,
            strat_prev,
            hands,
            seed,
        } => {
            let focal_p = load_policy_strat(&focal, strat_prev.as_deref());
            let field_p = load_policy_strat(&field, strat_prev.as_deref());
            if focal_p.blueprint.num_players != field_p.blueprint.num_players {
                die("blueprints trained for different table sizes");
            }
            if focal_p.blueprint.abs_cfg.menu != field_p.blueprint.abs_cfg.menu {
                // One shared action history is tokenised by whichever policy
                // acts; a bet size the other menu lacks puts that policy
                // off-tree for the rest of the hand (check/call fallback).
                // Measured 2026-08-25: -908/-1334 in BOTH directions.
                die(&format!(
                    "cross-play needs matching bet menus: focal {:?}, field {:?}",
                    focal_p.blueprint.abs_cfg.menu, field_p.blueprint.abs_cfg.menu
                ));
            }
            let cfg = HandConfig {
                num_players: focal_p.blueprint.num_players,
                ..HandConfig::default()
            };
            println!("cross-play: {focal} (1 seat) vs {field} (rest), {hands} hands...");
            let r = table::run_crossplay(&focal_p, &field_p, &cfg, hands, seed);
            println!(
                "focal winrate: {:+.1} mbb/hand (95% CI ±{:.1}) over {} hands",
                r.mbb_per_hand, r.ci95, r.hands
            );
            if let Some(r) = bot::fallback_report() {
                println!("{r}");
            }
        }

        Cmd::Slumbot {
            blueprint,
            hands,
            search,
            search_ms,
            safe_resolve,
            value_net,
            username,
            password,
            seed,
            verbose,
            log,
        } => {
            let net = value_net.map(|p| {
                let n = valuenet::ValueNet::load(&p)
                    .unwrap_or_else(|e| die(&format!("cannot load value net '{p}': {e}")));
                println!("loaded value net from {p}");
                Arc::new(n)
            });
            let policy = load_policy(&blueprint).with_value_net(net);
            if policy.blueprint.num_players != 2 {
                die("Slumbot is heads-up: train with --players 2 --stack 20000");
            }
            let mut transport = slumbot::HttpTransport;
            let token = match (username, password) {
                (Some(u), Some(p)) => Some(
                    transport
                        .login(&u, &p)
                        .unwrap_or_else(|e| die(&format!("login failed: {e}"))),
                ),
                _ => None,
            };
            let cfg = slumbot::SlumbotCfg {
                hands,
                search: search.then_some(SearchParams {
                    time_ms: search_ms,
                    safe_resolve,
                    ..SearchParams::default()
                }),
                seed,
                token,
                verbose,
                log,
            };
            println!("playing {hands} hands vs Slumbot (search: {search})...");
            let started = std::time::Instant::now();
            let r = slumbot::run(&policy, &mut transport, &cfg, &mut |h, mean| {
                println!("  {h} hands, running mean {mean:+.0} mbb/hand");
            })
            .unwrap_or_else(|e| die(&format!("slumbot session failed: {e}")));
            println!(
                "vs Slumbot: {:+.1} mbb/hand (95% CI ±{:.1}) over {} hands \
                 ({} desyncs) in {:.0}s",
                r.mbb_per_hand,
                r.ci95,
                r.hands,
                r.desyncs,
                started.elapsed().as_secs_f64()
            );
            print!("{}", r.autopsy.report());
        }

        Cmd::Clone {
            log,
            blueprint,
            out,
            holdout,
        } => {
            let policy = load_policy(&blueprint);
            let (bp, stats) = clone::build(&log, &policy.abs, holdout)
                .unwrap_or_else(|e| die(&format!("clone build failed: {e}")));
            println!(
                "clone: {} hands ({} skipped), {} decisions, {} infosets",
                stats.hands, stats.skipped, stats.decisions, stats.infosets
            );
            let streets = ["preflop", "flop", "turn", "river"];
            for (s, (hit, n)) in streets.iter().zip(stats.agree) {
                if n > 0 {
                    println!(
                        "  held-out top-1 agreement {s}: {:.1}% ({n} decisions)",
                        100.0 * hit as f64 / n as f64
                    );
                }
            }
            bp.save(&out)
                .unwrap_or_else(|e| die(&format!("cannot save {out}: {e}")));
            println!("saved clone to {out}");
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

/// Previous-round context for strategic abstractions. The referenced
/// blueprint must itself use an equity abstraction (one co-training level).
fn load_strat_ctx(path: &str) -> abstraction::StratCtx {
    let bp = Blueprint::load(path)
        .unwrap_or_else(|e| die(&format!("cannot load previous blueprint '{path}': {e}")));
    if bp.centroids.as_ref().is_some_and(|c| c.is_strategic()) {
        die("chained strategic blueprints are not supported: the previous blueprint must use an equity abstraction");
    }
    println!(
        "previous blueprint for strategic fingerprints: {} infosets ({} buckets)",
        bp.strategies.len(),
        bp.abs_cfg.postflop_buckets
    );
    let abs = Abstraction::with_centroids(bp.abs_cfg.clone(), bp.centroids.clone());
    abstraction::StratCtx {
        bp: Arc::new(bp),
        abs: Arc::new(abs),
        rollouts: 16,
    }
}

fn load_policy(path: &str) -> Policy {
    load_policy_strat(path, None)
}

fn load_policy_strat(path: &str, strat_prev: Option<&str>) -> Policy {
    let bp = Blueprint::load(path).unwrap_or_else(|e| {
        die(&format!(
            "cannot load blueprint '{path}': {e}\nrun `pluribus train --out {path}` first"
        ))
    });
    let strategic = bp.centroids.as_ref().is_some_and(|c| c.is_strategic());
    println!(
        "loaded blueprint: {} infosets from {} iterations ({} card buckets, {:?} menu, {})",
        bp.strategies.len(),
        bp.iterations,
        bp.abs_cfg.postflop_buckets,
        bp.abs_cfg.menu,
        if strategic {
            "STRATEGIC clustering"
        } else if bp.centroids.is_some() {
            "EMD k-means"
        } else {
            "raw equity"
        }
    );
    let mut abs = Abstraction::with_centroids(bp.abs_cfg.clone(), bp.centroids.clone());
    if strategic {
        let prev = strat_prev.unwrap_or_else(|| {
            die("this blueprint uses strategic clustering: pass --strat-prev <previous blueprint>")
        });
        abs = abs.with_strat(load_strat_ctx(prev));
    }
    Policy::new(bp, Arc::new(abs))
}

fn die(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1)
}
