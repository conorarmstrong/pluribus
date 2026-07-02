# Pluribus-Style Poker Bot (Rust)

A Rust implementation of a Pluribus-style poker AI for 2-6 player no-limit
Texas Hold'em, based on "Superhuman AI for multiplayer poker" (Brown &
Sandholm, Science 2019). It trains a blueprint strategy with parallel
external-sampling **Linear MCCFR** (with Pluribus's negative-regret pruning)
over an **EMD k-means card abstraction**, and sharpens play at the table with
**range-tracked, depth-limited online subgame resolving**.

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

# Replay the 10,000 hands the real Pluribus played (Science 2019) and
# measure how often this blueprint agrees with its decisions
./target/release/pluribus benchmark --blueprint blueprint.bin --dir data/pluribus

# Blueprint statistics
./target/release/pluribus inspect --blueprint blueprint.bin
```

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
| `--checkpoint <file>` | – | Write a resumable checkpoint every ~5% |
| `--resume <file>` | – | Continue from a checkpoint (restores abstraction too) |
| `--buckets` | 12 | Postflop card buckets per street |
| `--rollouts` | 200 | MC rollouts per river equity estimate |
| `--runouts` | 24 | Sampled future boards per flop/turn distribution |
| `--kmeans-samples` | 30,000 | Situations sampled per street for clustering |
| `--raw-buckets` | off | Plain equity quantization instead of k-means |
| `--no-prune` | off | Disable negative-regret pruning |
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
- on the flop, solves **depth-limited** to the end of the street; at the
  leaves each player picks among four continuation strategies (blueprint
  as-is, fold-, call- or raise-biased) valued by blueprint rollouts —
  Pluribus's defense against exploitable leaf values;
- on the turn, solves to the end of the hand with MCCFR (CFR+ updates);
- on the river with two live players, solves **exactly**: vector-form CFR+
  over both players' full tracked ranges at once (all 1326 combos, ReBeL
  style), with O(N) sorted-sweep showdown evaluation and exact card-removal
  blocker effects.

`--qre-lambda <x>` (opt-in) models opponents during search as
**lambda-rational** (logit quantal response) instead of perfectly rational:
0 = uniform random, higher = more rational. The bot then best-responds to
that imperfect opponent — an exploitation mode that wins more against
weak opposition at the cost of theoretical balance. Omit for equilibrium
play.

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

### `lbr`
**Local Best Response** (Lisý & Bowling 2017): a lower bound on the
blueprint's exploitability. The LBR agent knows the bot's exact policy — it
tracks the bot's range with exact Bayes updates and greedily best-responds
using fold equity plus showdown equity against the tracked range under a
check/call-down assumption. Runs heads-up blind-vs-blind inside the
blueprint's native game (other seats fold), alternating blind seats. Reports
LBR's winnings in mbb/hand: 0 = unexploited by this probe.

### `benchmark`
Replays all 10,000 hands the real Pluribus played in the Science 2019
experiment (`data/pluribus`, PHH format from
[uoftcprg/phh-dataset](https://github.com/uoftcprg/phh-dataset)) through the
engine — the replay is validated chip-for-chip against the logged finishing
stacks — and, at every decision Pluribus made, reports how much probability
our blueprint puts on the action Pluribus chose and how often it is our
top action, per street.

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
│                   up to 5 sizes incl. overbets) and card abstraction
│                   (169 canonical preflop hands; flop/turn equity-distribution
│                   quantiles clustered by EMD k-medians; river equity buckets)
├── cfr.rs          External-sampling Linear MCCFR: parallel (rayon+dashmap),
│                   negative-regret pruning, CFR+ mode for subgames,
│                   depth-limited leaves with biased continuation strategies,
│                   logit-QRE opponent modeling, checkpoints, blueprint export
├── search.rs       RangeTracker: per-seat Bayes-updated weights over all 1326
│                   combos; range-weighted hidden-card sampling for resolving
├── river.rs        Exact river resolving: vector CFR+ over both tracked
│                   ranges, O(N) showdown sweep with blocker effects
├── bot.rs          Table policy: blueprint lookup + range-tracked
│                   depth-limited subgame resolving
├── table.rs        Real hand + abstract "shadow" hand + history tracking;
│                   off-tree bet mapping; eval harness (plain + duplicate)
├── aivat.rs        AIVAT variance-reduced unbiased winrate estimator
├── lbr.rs          Local Best Response exploitability lower bound
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
Johanson/Ganzfried-Sandholm approach. The river uses quantized MC equity.
Centroids are trained once per blueprint and serialized with it. All cached
equity work is shared across **suit isomorphism** classes: on a cache miss
the (hole, board) pair is canonicalized to the lexicographic minimum over
the 24 suit relabelings, so up to 24 strategically identical hands share one
Monte Carlo computation.

**Blueprint (offline).** Each iteration deals a random hand and runs one
external-sampling MCCFR traversal for one player: the traverser explores its
whole action menu, opponents sample from their current regret-matched
strategy. Regret and average-strategy updates are weighted linearly by
iteration (Linear CFR). After a warm-up, actions with very negative
accumulated regret are skipped with 95% probability (Pluribus's pruning).
Training runs across all cores on a sharded concurrent hash map keyed by
`(card bucket, abstract betting history)`; positions are implicit in the
history, so one blueprint covers every seat.

**At the table (online).** Bots look up the blueprint strategy for the
current infoset and sample from it. Off-tree opponent bets map to the nearest
abstract action in log space on a "shadow" hand. With `--search`, postflop
decisions are re-solved in real time with hidden cards sampled from the
tracked ranges (see `play` above).

## Correctness

The project is TDD-built with 68 tests:

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
  (including the mirror cases first-appearance schemes miss)
- river solver: the O(N) showdown sweep is differentially tested against a
  naive O(N^2) evaluator; the nuts call a shove and air folds; strategies
  are proper distributions; a lambda=0 QRE solve bets far more than the
  equilibrium solve against the same ranges
- QRE: lambda=0 is uniform, large lambda approaches argmax; CFR+ subgame
  regrets stay nonnegative
- CFR: a 10bb heads-up push/fold training run must reproduce known-correct
  strategy (AA calls a shove, 32o folds, the button never folds AA);
  continuation-bias math; checkpoint/blueprint round-trips including centroids
- search: range tracking concentrates on the hands that would take the
  observed action; sampling respects weights and card-removal conflicts; a
  rigged nuts-on-the-river resolve must call >90%; a range-tracked,
  depth-limited flop resolve trains a valid root strategy
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

Run them with `cargo test`.

## Results (200M-iteration blueprint, July 2026)

Blueprint: 12 EMD k-means buckets/street, full bet menus, 128.5M infosets
(101M exported strategies, 4.3GB), trained in 79 minutes on 16 cores.

- **vs baselines** (200k hands each): +4426 ±334 mbb/hand vs random,
  +3735 ±371 vs always-call. AIVAT agrees with half the CI at half the
  hands: +4285 ±273 vs random, +3704 ±260 vs always-call (100k hands).
- **exploitability lower bound** (LBR, 20k hands blind-vs-blind): the raw
  blueprint without search is exploitable by at least **+366 ±322
  mbb/hand** — the expected picture for an abstraction-level blueprint,
  and the reason Pluribus (and this bot) add real-time search on top.
- **vs the real Pluribus** (all 10,000 logged hands, 15,169 decisions,
  99.0% covered): our blueprint picks Pluribus's exact action as its own
  top choice **66.8%** of the time overall (75.6% preflop, ~45-50%
  postflop) and assigns Pluribus's action a mean probability of 0.60.
  Both strategies are mixed, so even a perfect clone would not reach 100%;
  for scale, uniform-random agreement would be ~20%. The replay is
  validated chip-for-chip: 9,992/9,992 checkable hands reproduce their
  logged finishing stacks exactly.

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
  within its abstraction, but tracked exact reach probabilities.
- The k-means abstraction clusters Monte-Carlo-sampled situations rather
  than exhaustively enumerating canonical boards, and uses 12 buckets/street
  vs Pluribus's ~200; `--buckets`/`--kmeans-samples` scale it up at the cost
  of blueprint size and training time.
- Subgame roots trust the blueprint's action menu; there is no re-solving of
  earlier streets when an opponent's line goes far off-tree (the shadow-hand
  mapping absorbs it instead).
- No opponent modeling / exploitation layer (neither had Pluribus).

## References

- Brown & Sandholm, "Superhuman AI for multiplayer poker", Science 2019
- Brown, Amos & Sandholm, "Depth-Limited Solving for Imperfect-Information
  Games", NeurIPS 2018 (biased continuation strategies)
- Lanctot et al., "Monte Carlo Sampling for Regret Minimization in Extensive
  Games", NeurIPS 2009
- Brown & Sandholm, "Solving Imperfect-Information Games via Discounted
  Regret Minimization", AAAI 2019 (Linear CFR)
- Tammelin et al., "Solving Heads-Up Limit Texas Hold'em", IJCAI 2015 (CFR+)
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
- uoftcprg/phh-dataset — Poker Hand History format; the 10,000 Pluribus
  hands used by `benchmark`
