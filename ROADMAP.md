# Roadmap

Where the project stands, why it is behind Pluribus at 6-max, and the plan
to get past it. Every claim here is backed by a run logged in
[BASELINES.md](BASELINES.md); this file is the plan, that file is the
evidence. Priority is 6-max: most money games have more than two players.
Heads-up work continues only where it feeds 6-max or keeps the one external
benchmark alive.

## Status (25 August 2026)

**6-max.** `blueprint_pprune200.bin` (200M iterations, 12 buckets, Pluribus
pruning rules) is the standing reference: BR exploitability lower bound
**+188 mbb/hand** pooled over 8 seeds, +2742 vs a calling station with
AIVAT. That is 60% below July's reference (+472) and came from one
two-line fix to the pruning rule. Against the real Pluribus's 10,000
logged hands the (older) blueprint picks Pluribus's action 66.8% of the
time. There is no external 6-max opponent to play, so "how far behind
Pluribus" is not a number we can currently produce.

**Heads-up 200bb.** The blueprint loses to Slumbot by -714.5 +-331.5
mbb/hand (10k hands); search made it worse (-1771). The frontier
(GTO Wizard AI) beats Slumbot by +194 mbb/hand. Gap of roughly 900-1400
mbb/hand. Deprioritised below; see "Heads-up, kept alive".

**Tooling.** A one-hour proof loop exists for trainer changes (below).
128 tests. 86GB of gitignored artifacts.

## Why we are behind Pluribus (2019)

In order of weight. Numbers from the Pluribus Science supplement and
BASELINES.md.

1. **Compute, ~800x.** Pluribus's blueprint: 8 days on 64 cores, about
   12,000 core-hours, 200 buckets per postflop street, ~413M action
   sequences visited. Ours: 200M iterations in 28-67 minutes on 16 cores,
   about 15 core-hours, 12 buckets. Every "abstraction wall" in the July
   records (36 buckets worse, 24 buckets a wash, <1 visit per infoset) is
   this compute gap wearing a disguise.
2. **Visits spent in the wrong places.** Pluribus's action abstraction is
   fine preflop (up to 14 raise sizes; it never searches preflop) and
   deliberately coarse postflop (2-3 first-in sizes, 2 for re-raises;
   search fixes sizing live). Ours is the reverse: 4 preflop opens, 7
   postflop first-in sizes. The wide postflop menu measured null at 200M
   (+501 vs +472 on the same seeds) while inflating the tree to 300-400M
   infosets.
3. **Search is Pluribus's main weapon; ours is partial.** Its supplement
   calls depth-limited search "the most important improvement that made
   six-player poker possible". It runs on every postflop decision,
   multiway, rooted at the start of the betting round with every opponent
   allowed to have deviated anywhere in the round, with 500 later-street
   buckets and four continuation strategies at the leaves. Ours: with two
   live players we have exact turn/river solvers and a net-backed flop
   solve; with three or more live players every postflop decision falls
   to a sampled MCCFR resolve rooted at the current decision point, which
   measured ~0 in the only setting it was ever measured (HU flop). No
   6-max search result has ever been checked against anything external.
4. **Unmatched-recipe bugs.** Per-action pruning cost 60% of
   exploitability and hid for two months behind abstraction experiments.
   Every other deviation from Algorithm 1 of the supplement is a suspect
   until measured.

## The plan

Three phases. A matches Pluribus's recipe where we deviate from it; B adds
what 2019 did not have; C builds the yardstick without which "better than
Pluribus" is not a claim we can make. Items are ordered by expected
exploitability gain per hour of work, and every one is gated by the loop.

### The proof loop (every item below uses it)

Both arms trained fresh with the same binary at `--train-seed 0`; `br
--hands 20000` on seeds 1-8, paired by seed (BR deals are deterministic per
seed, so the pair shares deals); 1M-hand `crossplay` both directions;
`eval --baseline caller --aivat --duplicate --hands 100000`. Two scales:

| Scale | train | probes | total | decides |
|-------|-------|--------|-------|---------|
| 30M iterations | 3-4 min/arm | 2 min/seed | ~1 h | t > 2.4 on paired BR |
| 200M iterations | 30-70 min/arm | 5 min/seed | ~4 h | confirmation only |

Adopt only what wins at 30M and confirms at 200M. Record nulls.

### Phase A: match Pluribus

**A1. Pluribus-shaped bet menu.** *Days.*
Fine preflop, coarse postflop. `bet_menu` in `src/abstraction.rs`:
preflop opens to something like 2x/2.5x/3x/4x/pot/all-in plus 3-bet and
4-bet ladders; flop/turn/river first-in to 50% / 100% / all-in, re-raises
to 100% / all-in. Prediction: 2-4x fewer infosets at equal iterations, so
2-4x the visits, so BR falls again. Both the trainer and the shadow-hand
mapping (`map_raise`) consume the menu, and every existing blueprint
becomes unloadable by the new binary (the stale-menu guard); that is
already true of July's blueprints, so the cost is one retrain.
Gate: paired BR at 30M, then 200M.
Risk: worse vs a calling station (fewer overbets). Acceptable; the
equilibrium target is exploitability, and search restores sizing at the
table (A3).

**A2. Equal-wall-clock check of the pruning default.** *One evening,
unattended.*
Pluribus pruning costs 2.4x per iteration at 200M. Train plain for ~480M
iterations (same wall-clock as pruned 200M) and probe. If plain-at-480M
beats pruned-at-200M, the default stays but the write-up changes: the
gain is partly "explore more", not only "prune smarter", and the right
setting depends on budget. Either answer is worth having before renting
hardware.

**A3. Search on every postflop decision, multiway.** *The biggest single
gap. Weeks.*
Today `search_dist` (`src/bot.rs`) hands any spot with three or more live
players to `resolve_subgame`: sampled MCCFR from the current decision
point, hidden cards sampled from the tracker, flop depth-limited with the
four-bias leaves, turn and river solved to the end of the hand with the
blueprint's full menu. Nothing about it is measured multiway. Bring it to
the Pluribus design:
- Root the subgame at the **start of the current betting round**, not the
  current decision. The hero's already-taken actions in the round are
  held fixed; every other player may have changed strategy anywhere in
  the round. This is the "nested unsafe search that is not very
  exploitable" the supplement describes, and it is what makes off-tree
  opponent bets solvable instead of shadow-mapped.
- Depth limit for multiway flop: end of the round, or immediately after
  the second raise, whichever is first (their rule). Turn and river: to
  the end of the hand.
- Finer card abstraction inside the subgame than in the blueprint
  (Pluribus: lossless current street, 500 buckets later). Ours can start
  with lossless current street and the blueprint's 12 later, then scale.
- Time budget per decision measured, not assumed; Pluribus averaged ~20
  seconds per hand on 28 cores.
Measurement: `eval --search-gain` is the paired harness (hero searching
vs hero on blueprint, everyone else blueprint). Extend it with a
`--min-live 3` filter so the multiway contribution is isolated. Then BR
against the searching bot, which does not exist yet: the probe currently
measures the raw blueprint only. Add `br --search`. Without it, search
improvements are unverifiable in the metric that matters.
Gate: search-gain > 0 at 40k paired deals with CI clear of zero, and
`br --search` at or below the blueprint's bound on 8 seeds.

**A4. Bucket scaling at fixed visits.** *Needs A1 and a rented box.*
With A1 and the pruning fix the tree shrinks enough that 50 buckets at
equal wall-clock may already pay on this machine. Beyond that: 200
buckets at Pluribus's visit density is one 3-day run on a 192-vCPU,
384GB+ instance, roughly $1,000 on-demand. Do not rent before A1-A3 are
measured; the run must be the final recipe, not a guess.
Gate: paired BR vs the 12-bucket blueprint at equal wall-clock.

**A5. Remaining recipe audit.** *Days each, cheap, run opportunistically.*
Each is a deviation from the supplement and each gets the 30M loop:
- Preflop blueprint from the average strategy, postflop from the current
  strategy: tested as snapshots and closed (worse). Not revisiting.
- Regret storage as i32 with the -310M floor: memory only; do it when a
  bigger run needs it.
- Strategy update interval: Pluribus accumulates the preflop average only
  every 10,000 iterations in a separate pass; we accumulate on every
  opponent-node visit. Cheaper per iteration if matched; unmeasured
  effect on quality.
- Continuation strategies at flop leaves: we use the four Pluribus biases
  with a fixed 5x multiplier; Pluribus's exact multipliers are not
  published, so sweep {2, 5, 10}.

### Phase B: exceed Pluribus

What Pluribus lacked, in the order it becomes usable.

**B1. Learned leaf values in multiway search.** *After A3.*
Pluribus valued flop leaves by rolling out four biased blueprints; it had
no neural network anywhere. We already have a ReBeL-shaped belief-state
value net (`src/valuenet.rs`, `src/net.rs`) and measured +313 mbb/hand
from it heads-up. Extend the encoding from two ranges to N ranges
(N <= 6, padded), generate multiway turn spots with A3's solver as the
target, and plug the net into A3's flop leaves. Value-net accuracy
scaling measured null heads-up (30% better loss, +3 at the table), so
the first net is enough; the win is in having one at all where Pluribus
had rollouts.
Gate: paired net-vs-rollout leaves via `eval --net-gain` multiway.

**B2. Opponent exploitation with bounded exploitability.** *The money
item. Parallel track once A3 exists.*
Pluribus modelled nobody. In money games with more than two players the
edge is in punishing the two weak seats, not in being unexploitable by
the strong one. Infrastructure that exists: restricted Nash response
(`--rnr-model`, `--rnr-opponent`), the UCB1 portfolio over
equilibrium-plus-exploiter blueprints (`src/portfolio.rs`), behavioural
cloning from logs (`clone`). Missing: online estimation of a live
opponent's action frequencies, folded into the range tracker as a prior
instead of the blueprint's own probabilities, and a per-seat exploiter
selection with a hard exploitability cap (May 2026's AlphaExploitem is
the published shape of this). Order: (1) per-seat frequency model in the
tracker, (2) per-seat portfolio arm selection, (3) online RNR retune.
Gate: winrate vs scripted weak archetypes (station, maniac, nit) with
AIVAT, and BR of the exploiting bot bounded within X of the equilibrium
bot's, X chosen up front. Both numbers must be reported together; an
exploiter that wins more and is 3x as exploitable has not passed.

**B3. Neural blueprint.** *Last, largest.*
Deep CFR or a ReBeL-style trained policy removes the bucket ceiling
outright. Only worth starting once A4 shows where tabular scaling stops
paying, and only with rented compute. Not before 2027 on current
evidence.

### Phase C: a benchmark, without which none of this is a claim

The heads-up world has Slumbot and the GTO Wizard API. 6-max has nothing
public. Options in cost order; do the first two regardless.

**C1. `br --search` and multiway search-gain.** Internal, covered in A3.
Necessary, not sufficient: the probe is a lower bound against our own
abstraction.

**C2. Pluribus behavioural clone from the logged hands.** *Days.*
`clone` already builds a Blueprint-format model from logged hands with
hole cards. The PHH dataset has Pluribus's cards on every hand. 10,000
6-max hands is thin postflop (heads-up cloning got 29% river agreement
from 10k), but preflop will be solid and it gives a fixed, external-ish
opponent to `crossplay` against and to exploit-test B2 on. Report it as
what it is: a preflop-accurate imitation.

**C3. A public 6-max API.** *Weeks; strategic.*
Publish the bot behind a Slumbot-style HTTP API with duplicate dealing
and AIVAT scoring. Nobody has one for 6-max; being the reference is worth
more than any single result, and every other bot that plays it becomes
our benchmark. Requires the search stack to be robust to arbitrary bet
sizes (A3) and a rate limit.

**C4. Humans.** *Money.*
Pluribus's own yardstick: paid sessions against strong players, AIVAT
scored, with a Pluribus-in-the-human-seat control. Only after C3 shows
the bot is not embarrassing.

### Sequencing

```
now      A1 menu -> A2 wall-clock -> A5 audit items (interleaved, cheap)
weeks    A3 multiway search + br --search   <- the centre of gravity
then     B1 net leaves, C2 Pluribus clone (parallel)
then     A4 rented 200-bucket run (final recipe only)
then     B2 exploitation track, C3 public API
2027     B3 neural blueprint, C4 humans
```

A1 and A2 can run this week on this machine. A3 is the item that decides
whether we are building a Pluribus or a blueprint with a nice probe.

## Heads-up, kept alive

Deprioritised, not abandoned: Slumbot is the only external opponent we
have, and A3/B1/B2 all get their first cheap external check there.

- **`--stack` on the internal probes.** `br`, `lbr`, `eval`, `crossplay`
  build a 10,000-chip `HandConfig` and cannot see a 200bb blueprint, so
  every HU result costs a 4-6 hour API run. One day of work; turns the HU
  loop into minutes and belongs before any further HU run.
- **Re-measure HU with the new defaults.** The HU blueprints predate the
  pruning fix and the flop-routing fix (`e539914`). Retrain
  `bp_hu200_300m` under the default, probe internally (needs `--stack`),
  then one 10k Slumbot run. Gate: beat -714.5 by +100.
- **River-showdown leak.** The exact river solver loses 2.4x more per
  showdown than blueprint play; suspect the 0.75 belief-widening cap.
  Sweep after the re-measure, not before.
- **RNR exploiter vs Slumbot.** `slumbot_exploiter_ckpt.bin` is partial.
  Finish, 2k diagnostic, 10k gate. This is B2's first data point.

## Closed branches

Do not redo these without a new reason. Full write-ups in BASELINES.md.

| Branch | Verdict |
|--------|---------|
| Policy distillation (`distill`) | Null. Cross-play -38.5 +-62.0 over 1M hands, BR unchanged. A generation touches 0.03% of infosets. |
| Value-net scaling | Null. 30% validation-loss improvement bought +3 mbb/hand at the table. |
| Strategy-aware abstraction (`--strategic-from`) | Null on exploitability and head-to-head, at ~10x per-iteration cost. |
| More HU iterations (100M to 300M) + exact river buckets | Null vs Slumbot (-782 to -719, well inside the interval). |
| 36 buckets at equal iterations (HU) | Worse (-1128). Granularity without visits does not pay. |
| OCHS + wide bets + 24 buckets (6-max v2) | Not significant at 8 seeds, and the wide menu alone is null at 200M. The July gain was pruning noise. |
| Wide postflop bet menu alone | Null at 200M (+501 vs +472 on the same seeds); costs 2x infosets and ~1600 mbb/hand vs a station. Reversed by A1. |
| Equilibrium selection worry (6-max) | Does not materialize: all cross-play cells within +-80 mbb of zero. |
| VR-MCCFR baselines (`--vr-baseline`) | Worse: +168 +-149 more exploitable, 8/8 seeds, 3.3x slower, 2x memory. Needs converged baselines; dilution starves it. |
| Snapshot-averaged postflop blueprint (`--snapshot-avg`) | Worse: +430 +-140 more exploitable, 8/8 seeds. Needs a converged current strategy. |

## Measurement discipline

Learned the hard way, twice each:

- **Pool seeds.** A single BR or LBR seed has misled this project more than
  once (v2 looked 5.4x better at seed 1, then 1.6x at 8 seeds). Pool at
  least 8, and pair by seed: BR is deterministic per seed, so paired
  comparison is free precision.
- **Small live samples do not decide.** A 2,000-hand Slumbot result carries
  a +-900 mbb interval. The -234 that appeared to clear a gate came back
  -1771 at 10k.
- **Cross-play does not see exploitability.** Every cross-play in the repo
  has been a wash, including between blueprints whose BR bounds differ by
  2x. Use it as a sanity check, never as the gate.
- **Record the negatives.** Most entries in BASELINES.md are nulls or
  regressions, and they are the reason the open list above is short.
- **Match the recipe before improving it.** The 60% exploitability cut
  came from reading the supplement's pseudocode line by line, not from a
  new idea. Do that for the rest of Algorithm 1 and the search section
  before inventing.
