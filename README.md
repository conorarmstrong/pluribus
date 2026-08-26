# Pluribus-Style Poker Bot (Rust)

A Rust implementation of a Pluribus-style poker AI for 2-6 player no-limit
Texas Hold'em, based on "Superhuman AI for multiplayer poker" (Brown &
Sandholm, Science 2019). It trains a blueprint strategy with parallel
external-sampling **Linear MCCFR** (with Pluribus's negative-regret pruning)
over an **EMD k-means card abstraction**, and sharpens play at the table with
**range-tracked, depth-limited online subgame resolving**.

The bot plays any table size from 2 to 6; one blueprint covers every seat.
Six-max is the primary target because that is where most money games are
played. Heads-up is the 2-player special case of the same engine, trainer
and search stack, and it gets its own sections below only because it has
the one external benchmark (Slumbot) that a 6-max bot currently lacks.

This replaces an earlier Python implementation (preserved in `legacy/`),
which was ~4 orders of magnitude slower and had a game-engine bug that made
training impossible.

## Quick start

```bash
# Build
cargo build --release

# Train a blueprint (200M iterations ≈ 1 hour on 16 cores)
./target/release/pluribus train --iters 200000000 --out blueprint.bin

# Play against the bots (you are seat 0); --search enables online resolving
./target/release/pluribus play --blueprint blueprint.bin --search

# Measure winrate vs baseline opponents (--aivat / --duplicate reduce variance)
./target/release/pluribus eval --blueprint blueprint.bin --hands 200000 --baseline random
./target/release/pluribus eval --blueprint blueprint.bin --hands 100000 --baseline caller --aivat

# Lower-bound the blueprint's exploitability with a Local Best Response probe
./target/release/pluribus lbr --blueprint blueprint.bin --hands 20000

# Tighter bound: exact turn/river best response (deterministic per seed)
./target/release/pluribus br --blueprint blueprint.bin --hands 20000 --seed 1

# Replay the 10,000 hands the real Pluribus played (Science 2019) and
# measure how often this blueprint agrees with its decisions
./target/release/pluribus benchmark --blueprint blueprint.bin --dir data/pluribus

# Blueprint statistics
./target/release/pluribus inspect --blueprint blueprint.bin

# Heads-up 200bb: train, then play the external Slumbot benchmark
./target/release/pluribus train --players 2 --stack 20000 --iters 300000000 --out bp_hu200.bin
./target/release/pluribus slumbot --blueprint bp_hu200.bin --hands 2000 --search --safe-resolve
```

Every measured result, with the exact command, artifact hash and caveats,
is logged in [BASELINES.md](BASELINES.md). What is open and what comes next
is in [ROADMAP.md](ROADMAP.md).

## Commands

### `train`
Parallel Linear MCCFR self-play. Uses all cores by default. Before training,
it clusters flop/turn hands by their **equity distributions** (EMD k-medians)
and stores the centroids inside the blueprint, so play-time bucketing matches
training exactly.

| Flag | Default | Meaning |
|------|---------|---------|
| `--iters` | 1,000,000 | MCCFR traversals |
| `--out` | blueprint.bin | Output strategy file |
| `--players` | 6 | Players at the table (2-6) |
| `--stack` | 10,000 | Starting stack in chips; blinds are 50/100, so 10,000 = 100bb (Pluribus) and 20,000 = 200bb (Slumbot) |
| `--checkpoint <file>` | – | Write a resumable checkpoint every ~5% |
| `--resume <file>` | – | Continue from a checkpoint (restores abstraction too) |
| `--buckets` | 12 | Postflop card buckets per street |
| `--menu` | pluribus | Bet-size menu shape, stored in the blueprint: `pluribus` (Pluribus's coarse postflop: 50%/100%/all-in first-in, pot/all-in raise; the default, measured least exploitable), `wide` (2026-07 menu, 6-7 postflop sizes), `pluribus-fine-pre` (also a fine 7-size preflop menu; worse at every budget tested) |
| `--rollouts` | 200 | MC rollouts per river equity estimate |
| `--runouts` | 24 | Sampled future boards per flop/turn distribution |
| `--kmeans-samples` | 30,000 | Situations sampled per street for clustering |
| `--raw-buckets` | off | Plain equity quantization instead of k-means |
| `--ochs` | off | Potential-aware OCHS card abstraction: equity quantiles concatenated with the hand's equity against 8 preflop opponent tiers |
| `--strategic-from <file>` | – | Co-train: cluster by a previous blueprint's play instead of equity |
| `--train-seed` | 0 | Traversal RNG seed (distinct seeds → independent self-play runs) |
| `--rnr-model` | – | Restricted Nash response opponent model (`random`\|`caller`) |
| `--rnr-opponent <file>` | – | RNR against a cloned opponent given as a blueprint file (see `clone`); adopts its abstraction |
| `--rnr-p` | 0.5 | RNR mixture weight, 0=equilibrium, 1=pure best response; needs `--rnr-model` or `--rnr-opponent` |
| `--no-prune` | off | Disable negative-regret pruning |
| `--per-action-prune` | off | Legacy pruning: decided per action, applied everywhere. The default is Pluribus's rules (decided once per traversal, never on the river, never for actions leading straight to a terminal node), which halved the BR exploitability bound at equal iterations |
| `--snapshot-avg` | off | Postflop blueprint = mean of periodic snapshots of the current strategy (after a 10% warm-up, every 5%) instead of the linear running average; preflop keeps the average |
| `--vr-baseline` | off | VR-MCCFR learned control-variate baselines at sampled opponent nodes; unbiased variance reduction per visit |
| `--threads` | all cores | Worker threads |

### `play`
Interactive terminal game: you (seat 0) vs bots. Stacks reset every hand
(10,000 chips, 50/100 blinds — the same setup as the Pluribus experiment);
the button rotates.

Actions: `f` fold · `c`/`k` check/call · `r 500` raise TO 500 · `a` all-in · `q` quit.

`--search` enables online subgame resolving for the bots' postflop decisions
(`--search-ms` sets the per-decision budget, default 2000ms). The resolver:

- samples opponents' hidden cards from **tracked ranges** — every seat's
  range is Bayes-updated after every action using the blueprint's action
  probabilities (with a floor, so nothing is ever fully ruled out);
- roots every subgame at the **real table state**, so an opponent's
  off-tree bet is re-solved at its actual size (**nested re-solving**)
  instead of being priced at the nearest abstract size — the shadow hand
  is kept only for blueprint infoset lookups;
- with three or more live players, roots the subgame at the **start of
  the current betting round** (Pluribus): the actions taken so far form
  a spine, the bot's own are held fixed, every other seat's strategy in
  the round is re-solved from the ranges it held when the round began,
  and off-tree bets on the spine are solved at their real size.
  Traversals are steered onto the spine (eps 0.5 mixture,
  importance-weighted, so the solve stays unbiased). `--decision-root`
  on the probes restores the pre-A3 current-decision rooting;
- on the flop, solves **depth-limited** to the end of the street; at the
  leaves each player picks among four continuation strategies (blueprint
  as-is, fold-, call- or raise-biased) valued by blueprint rollouts —
  Pluribus's defense against exploitable leaf values;
- on the turn with two live players, solves **exactly** to the end of the
  hand: vector-form CFR+ over the turn tree, an explicit river chance
  node, and the full river tree beneath it (slim bet menu); multiway turn
  spots fall back to sampled MCCFR;
- on the river with two live players, solves **exactly**: vector-form CFR+
  over both players' full tracked ranges at once (all 1326 combos, ReBeL
  style), with O(N) sorted-sweep showdown evaluation and exact card-removal
  blocker effects.

`--safe-resolve` (opt-in) runs turn and river resolves as a **resolving
gadget game** (Burch et al. 2014): the opponent may take a per-combo safety
value instead of entering the subgame, which provably bounds how much a
resolve based on *wrong* tracked ranges can be exploited. Safety values
come from **continual resolving** (DeepStack-style) where possible: the
exact turn resolve records the opponent's counterfactual values at every
river-entry node, and the river resolve consumes them as its gadget
alternatives when the betting line stayed on the solve's tree and the bot
opens the river; otherwise they are rollout-estimated from the blueprint.
The `ablate-safety` command measures this: across random river spots with
beliefs corrupted at noise level ε, the opponent's best-response margin
beyond its safety values grows steadily for unsafe resolving (mean 2.7 →
8.9 chips, worst combo 205 chips ≈ 10% of pot at ε=1) while gadget
resolving stays at CFR convergence noise (~1.4 chips) **independent of ε**.

`--value-net <file>` (opt-in) enables **ReBeL-style flop solving**: flop
decisions against one live opponent are re-solved as a depth-limited
vector-CFR game over both tracked ranges, with end-of-flop leaves valued by
a trained belief-state value network (one query per candidate turn card,
refreshed as the leaf ranges evolve). Build the net with `gen-turn-data`
(exact turn solves as targets) and `train-value-net`.

`--qre-lambda <x>` (opt-in) models opponents during search as
**lambda-rational** (logit quantal response) instead of perfectly rational:
0 = uniform random, higher = more rational. The bot then best-responds to
that imperfect opponent — an exploitation mode that wins more against
weak opposition at the cost of theoretical balance. Omit for equilibrium
play.

`--adaptive-search` (opt-in metareasoning): spends 1/8 of the time budget
on a quick probe solve, then stops early if the root strategy is already
near-pure (max action probability ≥ 0.97) — more compute cannot change a
decision that is already settled. Otherwise it continues to the full
budget. Saves wall-clock on lopsided decisions without changing play.

### `eval`
Plays the blueprint (one rotating seat) against baseline opponents in every
other seat and reports the winrate in **mbb/hand** (milli-big-blinds per hand)
with a 95% confidence interval. Baselines: `random` (uniform over the action
menu), `caller` (always check/call). Two variance-reduction modes:

- `--duplicate` — ACPC-style duplicate deals: each sampled deal is played
  once per seat with the hero rotated through all of them, scored by the
  within-deal mean. Card luck partially cancels; the estimand is unchanged.
- `--aivat` — **AIVAT** (Burch et al., AAAI 2018): adds zero-mean correction
  terms at the hero's hole-card deal, every board reveal, and every
  known-distribution decision, using an omniscient value function (hero
  equity vs the opponents' actual hands × pot). Unbiased for *any* value
  function; ours is exact on the river, which cancels all showdown luck.
  Halves the CI at equal hands (≈4× fewer hands for equal precision).

`--search` (with `--search-ms`, `--value-net`) has the hero use the same
online subgame resolving as `play --search` instead of raw blueprint
lookups. `--search-gain` and `--net-gain` are paired modes for isolating
the value search and the value net each add (see Research experiments).
`--strat-prev <file>` points at the previous blueprint when evaluating a
strategic-abstraction blueprint (see `train --strategic-from`).

### `br`
A tighter exploitability lower bound than `lbr`: the same harness and
exact-Bayes range tracking, but every turn and river decision plays an
**exact best response of the entire remaining game** — a single expectimax
pass over the full-menu betting tree (turn decisions include the explicit
river chance node and the full river tree beneath it). The bot is a fixed,
known strategy, so one pass is exact; no CFR iterations, no convergence
error. Preflop and flop decisions fall back to the greedy LBR action (a
full-game exact BR is intractable: chance branching over boards puts it
at ~1e13 vector node-visits). Deterministic given a seed, so two code
versions can be compared on identical deals.

| Flag | Default | Meaning |
|------|---------|---------|
| `--blueprint` | blueprint.bin | Blueprint to probe |
| `--hands` | 2,000 | Hands played |
| `--runouts` | 100 | Equity runouts on the greedy (preflop/flop) streets |
| `--strat-prev` | – | Previous blueprint for strategic-abstraction lookups |
| `--seed` | 1 | RNG seed |
| `--search` | off | The bot resolves postflop decisions online, so the probe faces the bot as it plays |
| `--search-ms` / `--search-iters` | 2000 / 2M | Per-decision budget; the iteration cap makes the probe load-independent |
| `--decision-root` | off | Root multiway resolves at the current decision (pre-A3 A/B arm) |

### `lbr`
**Local Best Response** (Lisý & Bowling 2017): a lower bound on the
blueprint's exploitability. The LBR agent knows the bot's exact policy — it
tracks the bot's range with exact Bayes updates and greedily best-responds
using fold equity plus showdown equity against the tracked range under a
check/call-down assumption. Runs heads-up blind-vs-blind inside the
blueprint's native game (other seats fold), alternating blind seats. Reports
LBR's winnings in mbb/hand: 0 = unexploited by this probe. `--runouts`
(default 100) sets board completions sampled per equity estimate;
`--strat-prev` is the same strategic-abstraction lookup as above.

`--multiway` puts LBR in one rotating seat against the bot in *every*
other seat, so pots go multiway. It tracks each bot seat's range with exact
Bayes updates on the shared public history, values check/call by joint
showdown equity against all live ranges, and values a bet by the product
of the opponents' fold probabilities (each computed at the exact state it
would face) plus joint equity against the continuing parts of their
ranges. This is the only probe in the project that sees a three-way pot;
the default `lbr` and `br` are heads-up-line bounds, so quote both. Note on
reproducibility: `br` and both `lbr` modes are bit-for-bit reproducible per
seed only single-threaded (`RAYON_NUM_THREADS=1`); parallel runs vary by
roughly 1-3% of the bound between identical invocations (`eval` does not).
Paired-seed comparisons stay valid; the variation is far inside the CIs.

`--search` (with `--search-ms`, `--search-iters`, `--decision-root`, as on
`br`) has every bot seat resolve its postflop decisions online, sharing
one public-history range tracker per hand, so the LBR faces the bot as
it actually plays rather than the raw blueprint. The LBR's own range
model of the bot stays the blueprint (it cannot see the resolved
strategy), so with `--search` the number is a probe result, not a strict
lower bound. Multiway only.

### `benchmark`
Replays all 10,000 hands the real Pluribus played in the Science 2019
experiment (`data/pluribus`, PHH format from
[uoftcprg/phh-dataset](https://github.com/uoftcprg/phh-dataset)) through the
engine — the replay is validated chip-for-chip against the logged finishing
stacks — and, at every decision Pluribus made, reports how much probability
our blueprint puts on the action Pluribus chose and how often it is our
top action, per street.

### `ablate-safety`
The safety ablation behind the `--safe-resolve` numbers above: for
`--spots` random river spots, corrupts the tracked belief at a sweep of
noise levels ε and compares the opponent's best-response margin beyond its
safety value for unsafe vs. gadget (safe) resolving. `--iters` sets the
vector-CFR iterations per resolve.

| Flag | Default | Meaning |
|------|---------|---------|
| `--spots` | 40 | Random river spots sampled |
| `--iters` | 400 | Vector-CFR iterations per resolve |
| `--seed` | 1 | RNG seed |

### `gen-turn-data` / `train-value-net`
Build the belief-state value network consumed by `play --value-net`:
`gen-turn-data` plays the blueprint against itself, reaches turn-street
starts with Bayes-tracked ranges, and solves each one exactly with the turn
solver (`turn.rs`) to produce `(board, pot, stacks, ranges) → per-combo
values` training pairs; `train-value-net` fits an MLP (`net.rs`) to them.

| Flag (`gen-turn-data`) | Default | Meaning |
|------|---------|---------|
| `--blueprint` | blueprint.bin | Blueprint to self-play |
| `--out` | turn_data.bin | Output training data |
| `--samples` | 10,000 | Turn spots to solve |
| `--solve-iters` | 200 | Vector-CFR iterations per exact turn solve |
| `--solve-ms` | 30,000 | Per-solve wall-clock cap |
| `--seed` | 1 | RNG seed |

| Flag (`train-value-net`) | Default | Meaning |
|------|---------|---------|
| `--data` | turn_data.bin | Training data from `gen-turn-data` |
| `--out` | value_net.bin | Output network |
| `--hidden` | 512,512 | Hidden layer sizes, comma-separated |
| `--epochs` | 50 | Training epochs |
| `--lr` | 1e-3 | Adam learning rate |
| `--batch` | 128 | Minibatch size |
| `--seed` | 1 | RNG seed |

### `crossplay`
Cross-plays two blueprints: `--focal` occupies one rotating seat against a
full table of `--field`. The equilibrium-selection probe — if independent
self-play equilibria were interchangeable, every cross-play direction (and
the self-play sanity cell) should land near 0 mbb/hand.

| Flag | Default | Meaning |
|------|---------|---------|
| `--focal` | – | Blueprint in the rotating hero seat |
| `--field` | – | Blueprint filling the other seats |
| `--strat-prev` | – | Previous blueprint for strategic-abstraction lookups |
| `--hands` | 200,000 | Hands played |
| `--seed` | 1 | RNG seed |

### `distill`
Expert-iteration flywheel: self-play with online search, recording the
resolved action distribution at every searched postflop decision, then
blending those distributions back into the blueprint (`--alpha` sets the
blend weight). Measured as a negative — see Research experiments — and kept
for reproducibility. Gate any output with `br`, `crossplay` and `eval`
before using it.

| Flag | Default | Meaning |
|------|---------|---------|
| `--blueprint` | blueprint.bin | Teacher/parent blueprint |
| `--out` | distilled.bin | Output blueprint |
| `--hands` | 20,000 | Self-play hands (every seat searches) |
| `--search-ms` | 200 | Per-decision search budget |
| `--alpha` | 0.5 | Blend weight toward the search distribution |
| `--value-net` | – | Value net for ReBeL flop solving during self-play |
| `--safe-resolve` | off | Gadget-safe resolves during self-play |
| `--seed` | 1 | RNG seed |

### `slumbot`
Plays the blueprint against [Slumbot](https://www.slumbot.com) over its
public HTTP API — heads-up NLHE, 200bb stacks resetting per hand, 50/100
blinds — and reports our winrate in mbb/hand with a per-street loss
autopsy. This is the project's only *external* opponent: every other number
here is measured against the bot itself or against fixed baselines, so it is
the one benchmark that cannot be gamed by a shared abstraction. Train the
blueprint with `train --players 2 --stack 20000`.

Slumbot's bet sizes are arbitrary, so the nested re-solving path prices them
at their real size while the shadow hand keeps blueprint lookups on-tree.
Live play (`play` and `slumbot`) also enables **likelihood-calibrated belief
widening** in the range tracker: hard Bayes updates assume the opponent
plays our blueprint, and against a foreign opponent that posterior
concentrates on the wrong hands. Self-play harnesses keep exact Bayes.

| Flag | Default | Meaning |
|------|---------|---------|
| `--blueprint` | bp_hu200.bin | HU 200bb blueprint |
| `--hands` | 1,000 | Hands to play |
| `--search` | off | Online subgame resolving |
| `--search-ms` | 800 | Per-decision search budget |
| `--safe-resolve` | off | Gadget-safe turn/river resolving |
| `--value-net` | – | Value net (must be trained on 200bb HU spots to help) |
| `--username` / `--password` | – | Registered slumbot.com account; anonymous otherwise |
| `--verbose` | off | Log per-hand protocol problems |
| `--log <file>` | – | Append each hand's final API response as JSON lines |
| `--seed` | 1 | RNG seed |

### `clone`
Builds a behavioral clone of Slumbot from a `slumbot --log` JSONL file.
Slumbot reveals its hole cards every hand, so replaying a logged hand
reconstructs the exact abstract infoset at each of its decisions; counting
its chosen abstract actions per infoset and normalizing yields a playable,
exploitable model of the static opponent in Blueprint format. Feed it to
`train --rnr-opponent` to train a bounded exploiter.

| Flag | Default | Meaning |
|------|---------|---------|
| `--log` | slumbot_hands.jsonl | Hand log from `slumbot --log` |
| `--blueprint` | bp_hu200_300m.bin | Blueprint supplying the abstraction (must match at train and play time) |
| `--out` | slumbot_clone.bin | Output clone |
| `--holdout` | 0.1 | Fraction of hands held out for top-1 agreement measurement |

### `inspect`
Prints blueprint statistics: trained iteration count, player count, and
infoset counts overall and per street.

## Architecture

```
src/
├── cards.rs        Card representation (u8, rank*4+suit), parsing, decks
├── eval.rs         Fast 7-card evaluator (bitmask-based, ~10ns/hand),
│                   differentially tested against a naive best-of-21 evaluator
├── engine.rs       NLHE state machine: blinds, min-raise rules, big-blind
│                   option, heads-up order, all-in fast-forward, side pots,
│                   zero-sum net-chip utilities, targeted hidden-card resampling
├── abstraction.rs  Action abstraction (per-street pot-fraction menus + all-in,
│                   up to 7 first-in sizes spanning 25% through 200% overbets)
│                   and card abstraction (169 canonical preflop hands;
│                   flop/turn equity-distribution quantiles, optionally
│                   concatenated with potential-aware OCHS features, clustered
│                   by EMD k-medians; exact river equity buckets)
├── cfr.rs          External-sampling Linear MCCFR: parallel (rayon+dashmap),
│                   negative-regret pruning, CFR+ mode for subgames,
│                   depth-limited leaves with biased continuation strategies,
│                   logit-QRE opponent modeling, restricted Nash response,
│                   checkpoints, blueprint export
├── search.rs       RangeTracker: per-seat Bayes-updated weights over all 1326
│                   combos; range-weighted hidden-card sampling for resolving
├── river.rs        Exact river resolving: vector CFR+ over both tracked
│                   ranges, O(N) showdown sweep with blocker effects, and the
│                   adaptive (metareasoning) early-exit staged solve
├── turn.rs         Exact turn resolving: vector CFR+ over the turn tree, an
│                   explicit river-card chance node, and the river tree
│                   beneath each — one street deeper than river.rs
├── flop.rs         Depth-limited flop resolving: vector CFR+ over the flop
│                   tree, truncated at end-of-flop leaves valued by the
│                   belief-state value network (ReBeL search architecture)
├── net.rs          Dependency-free ReLU MLP (weighted-MSE loss, Adam),
│                   gradient-checked against finite differences
├── valuenet.rs     Belief-state value network (ReBeL-lite): turn-state
│                   leaf evaluator trained on turn.rs's exact solves
├── portfolio.rs    UCB1 bandit over a portfolio of blueprints (equilibrium
│                   + restricted-Nash-response exploiters)
├── distill.rs      Expert-iteration distillation: blend resolved search
│                   distributions back into the blueprint (measured null)
├── slumbot.rs      Slumbot HTTP API adapter: the external HU 200bb benchmark,
│                   incremental action-string parsing, per-street loss autopsy
├── clone.rs        Behavioral clone of a logged opponent from slumbot --log
│                   hands, in Blueprint format (the RNR exploit target)
├── ablate.rs       Safety ablation: unsafe vs gadget river resolving under
│                   corrupted range beliefs
├── bot.rs          Table policy: blueprint lookup + range-tracked
│                   depth-limited subgame resolving
├── table.rs        Real hand + abstract "shadow" hand + history tracking;
│                   off-tree bet mapping; eval harness (plain + duplicate)
├── aivat.rs        AIVAT variance-reduced unbiased winrate estimator
├── lbr.rs          Local Best Response exploitability lower bound
├── br.rs           Exact-subgame best response probe: LBR harness with
│                   true turn/river best response (tighter lower bound)
├── benchmark.rs    PHH parser + replay vs the real Pluribus's 10,000 hands
├── play.rs         Interactive terminal game (feeds the range tracker)
└── main.rs         CLI (clap)
```

### How it works

**Card abstraction (offline).** Preflop uses the lossless 169 canonical
hands. On the flop and turn, a hand is represented by the *distribution* of
its river equity over sampled runouts (8 quantiles): a naked flush draw and a
made middling pair may share a mean equity but have very different futures,
and this representation separates them where raw equity cannot. Those
quantile vectors are clustered into buckets with k-medians under the earth
mover's distance (for 1-D distributions, L1 between quantile vectors), the
Johanson/Ganzfried-Sandholm approach. `--ochs` extends the feature with
**opponent cluster hand strength**: the hand's equity against each of 8
preflop opponent strength tiers, concatenated with the quantiles, so hands
are separated by *who* they beat and not only by how much equity they hold.
The river uses quantized equity, computed exactly — one O(N) showdown sweep
per suit-canonical board covers all 1,326 combos at once, which made river
granularity free at any bucket count.
Centroids are trained once per blueprint and serialized with it. All cached
equity work is shared across **suit isomorphism** classes: on a cache miss
the (hole, board) pair is canonicalized to the lexicographic minimum over
the 24 suit relabelings, so up to 24 strategically identical hands share one
Monte Carlo computation. Monte Carlo draws are seeded from the canonical
key itself, so a given (hole, board) always produces the same estimate —
and the same bucket — in every process, thread interleaving, and cache
state.

**Blueprint (offline).** Each iteration deals a random hand and runs one
external-sampling MCCFR traversal for one player: the traverser explores its
whole action menu, opponents sample from their current regret-matched
strategy. Regret and average-strategy updates are weighted linearly by
iteration (Linear CFR). After a warm-up, 95% of traversals prune: actions
with very negative accumulated regret are skipped, except on the river and
except for actions that end the hand outright, which are always explored.
Those two exemptions are Pluribus's own rules (Science 2019 supplement,
Algorithm 1) and they matter: against the earlier per-action,
prune-everywhere scheme they halved this trainer's best-response
exploitability bound at equal iterations.
Training runs across all cores on a sharded concurrent hash map keyed by
`(card bucket, abstract betting history)`; positions are implicit in the
history, so one blueprint covers every seat.

**At the table (online).** Bots look up the blueprint strategy for the
current infoset and sample from it. Off-tree opponent bets map to the nearest
abstract action in log space on a "shadow" hand. With `--search`, postflop
decisions are re-solved in real time with hidden cards sampled from the
tracked ranges (see `play` above).

## Correctness

The project is TDD-built with 128 tests:

- evaluator: category spot checks, ordering checks, and a 30k-hand
  differential test against an independent naive evaluator
- engine: blind posting, big-blind option, heads-up order, min-raise and
  clamping rules, uncalled-bet return, side-pot splitting, multi-way all-in
  fast-forward, short-all-in reopening, targeted hidden-card resampling, plus
  a 10k-hand fuzz test asserting every random hand terminates zero-sum
- abstraction: menu shape (including turn/river overbets), quantile-vector
  properties (a flush draw's distribution is measurably wider than a made
  pair's), k-medians recovery of known clusters, monster-vs-air bucket
  separation under trained centroids, suit-isomorphism canonicalization
  (including the mirror cases first-appearance schemes miss), strategic
  (play-based) centroids bucket hands by realized action
- river solver: the O(N) showdown sweep is differentially tested against a
  naive O(N^2) evaluator; the nuts call a shove and air folds; strategies
  are proper distributions; a lambda=0 QRE solve bets far more than the
  equilibrium solve against the same ranges; the adaptive staged solve exits
  early only when the root strategy is already near-pure
- turn/flop solvers: the turn solver builds a multi-street tree with
  zero-sum, normalized root values (a made royal calls a shove for full
  pot); the gadget turn resolve still finds mandatory calls while combos
  paid more than the pot to leave stay out; the continual-resolving carry
  exposes river-entry CFVs by betting line (royal dominates junk,
  off-tree lines miss cleanly); the flop solver's leaf-sampled and
  full-solve root strategies both normalize
- table search routing: exact turn resolves fire through act_with_search
  (plain and gadget); off-tree bets are priced at their real size (a
  42%-equity bluff-catcher folds to a real 800-into-200 bet that maps to
  the callable 400 abstract size); the turn resolve's carry is consumed
  by the river gadget on the line actually played
- value network: the MLP's analytic gradients match central finite
  differences and it overfits a tiny nonlinear dataset (save/load
  round-trips); the belief-state encoding has the right shape and
  normalization, sample generation is valid, and training measurably
  reduces loss
- QRE: lambda=0 is uniform, large lambda approaches argmax; CFR+ subgame
  regrets stay nonnegative
- CFR: a 10bb heads-up push/fold training run must reproduce known-correct
  strategy (AA calls a shove, 32o folds, the button never folds AA);
  continuation-bias math; checkpoint/blueprint round-trips including
  centroids; restricted Nash response measurably exploits the modeled
  opponent; VR-MCCFR baselines are populated, finite and leave the fixed
  point alone; Pluribus pruning rules keep the fixed point under
  aggressive pruning; snapshot averaging fills postflop strategies only
  from snapshots while preflop keeps its running average
- search: range tracking concentrates on the hands that would take the
  observed action; sampling respects weights and card-removal conflicts; a
  rigged nuts-on-the-river resolve must call >90%; a range-tracked,
  depth-limited flop resolve trains a valid root strategy
- portfolio & safety ablation: against a calling station, the UCB1 bandit
  converges to the RNR arm and outscores the equilibrium arm alone; the
  safety ablation confirms gadget resolving bounds the opponent's
  best-response margin under corrupted beliefs while unsafe resolving does
  not
- BR probe: a made royal facing a turn shove must value exactly +2000
  (single exact pass — float-noise tolerance, not CFR convergence); the
  probe must plan multi-street stack extraction (check turn, shove river);
  net fold values are exact; results reproduce bit-for-bit under a fixed
  seed; and the walk rejects wrong streets and multiway spots
- Slumbot adapter: the documented action strings parse (street separators,
  per-street bet amounts, position mapping), our increment encoding matches
  the protocol, a scripted hand plays end-to-end against a mock API, a
  transport error counts a desync without aborting the run, a transient
  new_hand error is retried, and `--log` writes exactly one JSON line per
  completed hand
- opponent clone: a preflop raise-fold and a checked-down multi-street hand
  each replay from the log into the right infosets, and the built clone is a
  normalized strategy in Blueprint format
- distillation: blending inserts new distributions and skips menu
  mismatches, expansion aligns by action identity rather than index, and a
  smoke run produces a valid blueprint
- benchmark: PHH parsing, replay in lockstep with the engine, exact chip
  accounting on a real logged hand (and 9,992/9,992 checkable hands of the
  full dataset reproduce their logged finishing stacks exactly)
- table: shadow/real consistency under fuzz, off-tree bet mapping, and a
  symmetric-matchup eval that must come out statistically at zero
- evaluation science: duplicate deals of the deterministic caller-vs-caller
  matchup must score exactly 0 ± 0 (zero-sum cancellation) while plain
  evaluation is noisy; AIVAT must agree with the plain estimator on the mean
  and cut the CI by more than half; the AIVAT value function is exact at
  river states; LBR calls a shove with the nuts, folds air, exactly zeroes
  non-raising hands from an observed raiser's range, and crushes a calling
  station by four figures

Run them with `cargo test --release`. Release mode is required: four of
the solver tests are wall-clock budgeted, and an unoptimized build does not
reach enough CFR iterations inside the budget to satisfy their assertions.

## Results

Every number below is logged with its command, artifact hash and caveats in
[BASELINES.md](BASELINES.md), negatives included.

### 6-max blueprint (200M iterations, July 2026)

Blueprint: 12 EMD k-means buckets/street, full bet menus, 128.5M infosets
(101M exported strategies, 4.3GB), trained in 79 minutes on 16 cores.

- **vs baselines** (200k hands each): +4426 ±334 mbb/hand vs random,
  +3735 ±371 vs always-call. AIVAT agrees with half the CI at half the
  hands: +4285 ±273 vs random, +3704 ±260 vs always-call (100k hands).
- **exploitability lower bounds** (20k hands blind-vs-blind per seed,
  pooled over 8 seeds): **BR +472**, **LBR +354 mbb/hand**. The raw
  blueprint without search has real exploitable holes — the expected
  picture for an abstraction-level blueprint, and the reason Pluribus (and
  this bot) add real-time search on top. Single seeds swing by hundreds of
  mbb (the four recorded baseline BR seeds span +375..+570), so seed
  pooling and pairing are mandatory for any claim about a change.
- **value network** (20,000 exactly-solved turn spots, 512×512 MLP):
  validation loss 0.00089 weighted MSE ≈ 3% RMS of the maximum forward
  swing. Trained in 7.5 minutes on CPU.
- **search gains** (12,000 paired deals each, 800ms/decision, hero vs a
  table of blueprints; the paired design cancels deal luck):
  search with the value net beats search without it by **+313 ±198
  mbb/hand**, and beats the raw blueprint by **+331 ±213**. Search
  *without* the net measures ≈ 0 (−55 ±495; at 150ms budgets it is
  actively harmful, −520 ±399) — undertrained MCCFR resolves are worse
  than a 200M-iteration blueprint, and the learned leaf values are what
  make real-time flop solving pay. Re-measured after exact turn solving
  landed in *both* arms (40k paired deals, 800ms), the net's edge is
  **+174 ±100** — the no-net baseline itself got stronger.
- **vs the real Pluribus** (all 10,000 logged hands, 15,169 decisions,
  99.0% covered): our blueprint picks Pluribus's exact action as its own
  top choice **66.8%** of the time overall (75.6% preflop, ~45-50%
  postflop) and assigns Pluribus's action a mean probability of 0.60.
  Both strategies are mixed, so even a perfect clone would not reach 100%;
  for scale, uniform-random agreement would be ~20%. The replay is
  validated chip-for-chip: 9,992/9,992 checkable hands reproduce their
  logged finishing stacks exactly.

### 6-max modernization (400M iterations, blueprint_6max_v2.bin)

OCHS card abstraction + widened bet menus + 24 buckets, 400M iterations:
521.7M infosets, 391M exported strategies, 16GB, 1h56m at 57.2k iters/s.
Paired by seed against the 200M baseline over 8 seeds (BR is deterministic
per seed, so the deals are identical on both sides):

| Probe | baseline | v2 | paired diff | p |
|-------|----------|-----|-------------|---|
| BR | +471.8 | +391.1 | +80.7 ±165 | ~0.36 |
| LBR | +353.8 | +240.0 | +113.8 ±167 | ~0.21 |

Both probes favour v2 (less exploitable) by ~17-24%, but neither is
significant. Honest verdict: **no clear win**, and confounded by 5× visit
dilution (0.77 visits/infoset at this iteration budget). Earlier single-seed
and 4-seed reads of this experiment looked much stronger and were favourable
draws; they are corrected in BASELINES.md rather than deleted.

### Heads-up 200bb vs Slumbot (external benchmark)

The only opponent here that does not share our abstraction. bp_hu200_300m.bin
(300M iterations, HU, 200bb, exact river buckets) over Slumbot's public API:

| Config | mbb/hand | Hands |
|--------|----------|-------|
| **blueprint only** | **−714.5 ±331.5** | 10,000 |
| search 800ms, safe resolve | −1771.0 ±471.8 | 10,000 |

We lose. Online search made it *worse*, and the per-street autopsy says why:
range tracking assumes the opponent plays our blueprint, so against a foreign
opponent the posterior concentrates on the wrong hands and the resolver
best-responds to a fiction. Belief widening (+310 mbb) and carried-CFV gadget
alternatives on villain-led rivers (+660 mbb) recovered most of that, and the
remaining damage was traced to no-net flop MCCFR resolves stacking off at ~18%
equity (51 flop all-ins in 10k hands vs 2 for the blueprint) — now routed to
the blueprint instead. The post-fix 10k run is not yet recorded. Also null
along the way: 3× more training iterations and exact river bucketing moved
nothing (−782 → −719), and 36 buckets at equal iterations was worse (−1128).

### Trainer levers, paired 8-seed loop (August 2026)

Three sampling-compatible trainer changes, each an opt-in flag, decided by
the same loop: both arms trained fresh at 30M iterations, `br` on 8 seeds
paired by deal, 1M-hand crossplay, AIVAT eval vs caller.

| Lever | BR (plain +1755) | paired diff | speed | verdict |
|-------|------------------|-------------|-------|---------|
| Pluribus pruning rules (now default) | **+925** | **−830 ±139**, 8/8 seeds | same | **adopted** |
| Pluribus coarse postflop menu (now default) | **+672** | **−303 ±100**, 8/8 seeds | 1.3× faster, 2.5× smaller | **adopted** |
| Pluribus fine preflop menu (`--menu pluribus-fine-pre`) | +1282 | +308 worse, 8/8 | same | closed at this budget |
| VR-MCCFR baselines (`--vr-baseline`) | +1923 | +168 ±149 worse, 8/8 | 3.3× slower | closed |
| Snapshot blueprint (`--snapshot-avg`) | +2185 | +430 ±140 worse, 8/8 | 1.4× slower | closed |

Confirmed at reference scale: at 200M iterations the pruning change takes
the 8-seed BR bound from **+501 to +188 mbb/hand** (paired −314 ±190,
t=3.9, 8/8 seeds), and the coarse menu on top of it to **+116** (paired
−71 ±52, t=3.2, 7/8), with a 3.2× smaller tree and half the training
time. `blueprint_plu200.bin` is the standing 6-max blueprint: 75% below
the July reference (+472) on the heads-up line.

**The multiway caveat.** Every number above is a heads-up-line bound.
`lbr --multiway` (built 25 Aug) puts the same standing blueprint at
**+929 ±319 mbb/hand** in real multiway pots, with only 0.1% of decisions
on untrained infosets: trained, wrong multiway play, unchanged by either
fix. That is the open target (ROADMAP A3). Two side findings: the wide bet
menu on its own does nothing (plain 200M wide-menu +501 vs narrow-menu
+472), and Pluribus pruning costs 2.4× per iteration at 200M (67 vs 28
min; 402M vs 306M infosets), so the equal-wall-clock comparison is still
open.

The pruning fix is a two-line rule change with a t-statistic of 14. The
two published-technique negatives are instructive rather than surprising:
VR-MCCFR needs converged baselines and snapshot averaging needs a
converged current strategy, and at 0.2 visits per infoset this blueprint
has neither. Both are recorded in full in BASELINES.md.

### Opponent modeling from logged hands

`slumbot --log` banked 10,297 hands with Slumbot's hole cards shown every
hand; `clone` turns them into a Blueprint-format model — 24,976 decisions
over 5,107 infosets, held-out top-1 agreement **97.2% preflop / 51.8% flop /
44.1% turn / 29.3% river**. `train --rnr-opponent slumbot_clone.bin --rnr-p
<p>` then best-responds to that clone with a bounded-exploitability dial.
The exploiter has not yet been measured against the live opponent.

## Research experiments

Beyond the headline results above, the repo carries a set of controlled
experiments (all reproducible from the CLI):

- **Strategy-aware abstraction co-training** (`train --strategic-from`):
  cluster flop/turn hands by how a previously trained blueprint plays and
  realizes them (action distribution at a standardized spot + rollout value
  statistics) instead of by equity distributions. At an equal 30M-iteration
  budget and 12 buckets, strategic clustering scored **+220/+305 mbb/hand
  more vs caller/random baselines** (borderline significance) but showed
  **no LBR-exploitability improvement and no head-to-head edge** over
  equity clustering (both cross-play directions ≈ 0 ± 246) — at ~10× the
  per-iteration training cost. Verdict: an honest negative — equity-
  distribution clustering is hard to beat at this scale.
- **Restricted Nash response** (`train --rnr-model caller --rnr-p 0.9`):
  with probability p, a traversal's opponents all play a fixed model; the
  learned strategy maximally exploits the model while staying
  equilibrium-anchored with weight 1−p. Verified: RNR(caller, 0.9) beats a
  calling station by clearly more than the equal-budget equilibrium.
- **Portfolio bandit** (`portfolio.rs`): UCB1 over a set of blueprints
  (equilibrium + RNR exploiters), choosing per hand. Against a station it
  converges to the exploitative arm and outscores equilibrium-alone;
  against opponents that punish exploiters it retreats to the equilibrium
  arm — exploitation with bounded downside.
- **Expert-iteration distillation** (`distill`): self-play with online
  search, blending the resolved distributions back into the blueprint. At
  50,000 self-play hands and α=0.5 the distilled blueprint is
  indistinguishable from its parent: cross-play **−38.5 ±62.0 mbb/hand**
  over 1M hands, BR unchanged (+484.5 vs +475.4). A 73-minute generation
  touches 0.03% of infosets, and on exactly those high-traffic lines a
  200M-iteration blueprint is already near its best — search's edge lives
  in rare deep spots self-play rarely revisits. An honest negative; branch
  closed.
- **Value-net scaling** (`gen-turn-data` / `train-value-net`): 50,000
  exactly-solved turn spots and a 1024×1024 net cut validation loss 30%
  (0.00089 → 0.00062) and bought **+3 mbb/hand** at the table (+174.1 ±99.7
  vs +171.1 ±100.3 on the same 40k paired deals). Leaf-value accuracy is
  not the binding constraint at 800ms budgets; branch closed.
- **Safety ablation** (`ablate-safety`) and **paired search-gain
  measurement** (`eval --search-gain / --net-gain`): see above.
- **Equilibrium selection in 6-max** (`crossplay`): three independent
  30M-iteration self-play runs (`train --train-seed`), cross-played in all
  six directions at 200k hands each. Every cell (and the self-play sanity
  cell) is within ±80 mbb/hand of zero (CIs ±245): with the card
  abstraction held fixed, **independent MCCFR equilibria are
  interchangeable in practice** — the theoretical multiplayer
  equilibrium-selection worry does not materialize at this scale.

## Performance notes

- Hand evaluation: bitmask/rank-count based, no lookup tables, ~10ns per
  7-card hand.
- Distribution bucketing is ~6× the cost of raw equity per cache miss, but
  results are memoized in a collision-free packed-key cache shared between
  training, play, and search.
- The `eval` harness plays hundreds of thousands of hands per second; the
  PHH benchmark replays all 10,000 Pluribus hands in under a second.

## Remaining gaps vs the real Pluribus

- Range updates assume every player roughly follows the blueprint (with a
  2% floor per observed action). Pluribus made the same modeling assumption
  within its abstraction, but tracked exact reach probabilities. Against an
  opponent that is *not* our blueprint this is the dominant weakness: it is
  what made search a net loss against Slumbot. Live play mitigates it with
  likelihood-calibrated belief widening and with gadget resolving, which
  bounds how much a resolve on wrong beliefs can be exploited, but neither
  is a substitute for estimating the real opponent's ranges.
- The k-means abstraction clusters Monte-Carlo-sampled situations rather
  than exhaustively enumerating canonical boards, and uses 12 buckets/street
  vs Pluribus's ~200; `--buckets`/`--kmeans-samples` scale it up at the cost
  of blueprint size and training time. Scaling it is not free money: at a
  fixed iteration budget more buckets means fewer visits each, and both
  bucket-scaling experiments run so far (36 buckets HU, 24 buckets + OCHS
  6-max) failed to beat their baseline once seed noise was accounted for.
- Subgame roots trust the blueprint's action menu; there is no re-solving of
  earlier streets when an opponent's line goes far off-tree (the shadow-hand
  mapping absorbs it instead).
- Opponent modeling is offline and model-based, not adaptive: restricted
  Nash response (`--rnr-model`) and the portfolio bandit (`portfolio.rs`)
  exploit a small set of pre-specified, pre-trained opponent models (random,
  caller) chosen ahead of time. There is no online estimation of an unknown
  live opponent's tendencies from observed play (neither had Pluribus).

## References

- Brown & Sandholm, "Superhuman AI for multiplayer poker", Science 2019
- Brown, Amos & Sandholm, "Depth-Limited Solving for Imperfect-Information
  Games", NeurIPS 2018 (biased continuation strategies)
- Lanctot et al., "Monte Carlo Sampling for Regret Minimization in Extensive
  Games", NeurIPS 2009
- Brown & Sandholm, "Solving Imperfect-Information Games via Discounted
  Regret Minimization", AAAI 2019 (Linear CFR)
- Tammelin et al., "Solving Heads-Up Limit Texas Hold'em", IJCAI 2015 (CFR+)
- Burch, Johanson & Bowling, "Solving Imperfect Information Games Using
  Decomposition", AAAI 2014 (the resolving gadget behind --safe-resolve)
- Brown, Bakhtin, Lerer & Gong, "Combining Deep Reinforcement Learning and
  Search for Imperfect-Information Games", NeurIPS 2020 (ReBeL; the river
  solver's range-vector formulation)
- McKelvey & Palfrey, "Quantal Response Equilibria for Normal Form Games",
  GEB 1995
- Lisý & Bowling, "Equilibrium Approximation Quality of Current No-Limit
  Poker Bots", AAAI-17 Workshop (Local Best Response)
- Burch, Schmid, Moravčík, Morrill & Bowling, "AIVAT: A New Variance
  Reduction Technique for Agent Evaluation in Imperfect Information Games",
  AAAI 2018
- Ganzfried & Sandholm, "Potential-Aware Imperfect-Recall Abstraction with
  Earth Mover's Distance in Imperfect-Information Games", AAAI 2014
- Johanson, Burch, Valenzano & Bowling, "Evaluating State-Space Abstractions
  in Extensive-Form Games", AAMAS 2013 (opponent cluster hand strength,
  `train --ochs`)
- Johanson, Zinkevich & Bowling, "Computing Robust Counter-Strategies",
  NeurIPS 2007 (restricted Nash response, `--rnr-model`/`--rnr-p`)
- Auer, Cesa-Bianchi & Fischer, "Finite-time Analysis of the Multiarmed
  Bandit Problem", Machine Learning 47, 2002 (UCB1, the portfolio bandit)
- Schmid, Burch, Lanctot, Moravčík, Kadlec & Bowling, "Variance Reduction in
  Monte Carlo Counterfactual Regret Minimization (VR-MCCFR) for Extensive
  Form Games using Baselines", AAAI 2019 (`train --vr-baseline`)
- Brown & Sandholm, Supplementary Materials for "Superhuman AI for
  multiplayer poker", Science 2019 (Algorithm 1: the default pruning
  rules, and the snapshot blueprint behind `--snapshot-avg`)
- uoftcprg/phh-dataset — Poker Hand History format; the 10,000 Pluribus
  hands used by `benchmark`

## License

MIT. See [LICENSE](LICENSE).
