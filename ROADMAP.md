# Roadmap

Where the project stands, why it is behind Pluribus at 6-max, and the plan
to get past it. Every claim here is backed by a run logged in
[BASELINES.md](BASELINES.md); this file is the plan, that file is the
evidence. The bot plays 2 to 6 players with one blueprint; 6-max is the
primary target because most money games have more than two players.
Heads-up is the 2-player case of the same system, not a separate bot, and
heads-up work continues only where it feeds 6-max or keeps the one external
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
131 tests. 86GB of gitignored artifacts.

**Since 24 Aug.** Pluribus pruning rules are the default (BR +501 to +188
at 200M, 8/8 seeds). `lbr --multiway` (A0) is built: LBR in one rotating
seat vs the bot in every other seat, exact Bayes per seat, joint-equity
valuation, deterministic per seed; first run on `blueprint_pprune200.bin`
queued. `--menu wide|pluribus|pluribus-fine-pre` (A1 infrastructure) is
stored in the blueprint and legacy blueprints load as Wide; the
three-arm 30M test is queued. A2 (plain 470M at the pruned run's
wall-clock) is in flight.

## Positioning: what "beat the best" means here

Two things we cannot win at, stated so nobody spends a year on them:
being closer to Nash than GTO Wizard AI heads-up (a compute race against
a funded team, on a fixed target where two near-Nash bots cannot beat
each other by more than their distance from it), and out-scaling
Pluribus's blueprint on a laptop.

What the frontier has deliberately left open, in its own words: GTO
Wizard AI "currently doesn't model and adjust to specific opponents";
Pluribus modelled nobody; AlphaExploitem (May 2026) is the first serious
attempt and it is heads-up. Nobody has a 6-max bot that adapts to the
table. In money games that is where the money is: Pluribus beat pros by
+48 mbb/hand at equilibrium, our raw blueprint beats a calling station by
+2742. The exploitation edge against real weak players is fifty times the
equilibrium edge, and the best bots choose not to collect it.

So the target is: the most money at a 6-max table of real opponents,
with exploitability held inside a hard cap. Phase A makes the base bot
sound enough to carry it; Phase B item B2 is the thesis; everything else
serves those two.

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

**A0. A multiway exploitability probe.** *Days. Before A3, not after.*
The `br` and `lbr` probes play heads-up blind-vs-blind inside the 6-max
game (the other seats fold). Every 6-max exploitability number in
BASELINES.md, including the +188, is therefore a heads-up-line number:
a three-way pot has never been probed. That is a keyhole, and A3 would
widen the part of the bot the keyhole cannot see. Build `lbr --multiway`
first: one best-responding seat with exact Bayes tracking of five fixed
blueprint seats, greedy LBR action selection (an exact multiway turn/
river best response is a later upgrade), rotating the responder through
all six seats, AIVAT-scored. Report it alongside the HU-line probe from
then on; adopt nothing that improves one and worsens the other.
Gate for the probe itself: crushes a table of callers by four figures,
reproduces bit-for-bit per seed, and returns a bound at or above the
HU-line bound on the same blueprint (it sees strictly more).

**A1. Pluribus-shaped bet menu.** *Days.*
Fine preflop, coarse postflop. `bet_menu` in `src/abstraction.rs`:
preflop opens to something like 2x/2.5x/3x/4x/pot/all-in plus 3-bet and
4-bet ladders; flop/turn/river first-in to 50% / 100% / all-in, re-raises
to 100% / all-in. Prediction, and it is only a prediction: fewer infosets
at equal iterations, so more visits, so BR falls again. A fine preflop
menu across six seats grows preflop sequences combinatorially, so the
net tree size after coarsening postflop could go either way; the first
30M run answers that before anything else. Both the trainer and the shadow-hand
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
`--min-live 3` filter so the multiway contribution is isolated. It is a
self-play number and self-play is blind to belief mismatch (the HU
lesson), so it is necessary, not sufficient. The sufficient instrument is
A0's multiway probe run against the *searching* bot: add `--search` to
both probes so the responder faces the bot as it actually plays, not the
raw blueprint.
Gate: search-gain > 0 at 40k paired deals with CI clear of zero, and both
probes with `--search` at or below their blueprint-only bounds on 8
seeds.

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

**B2. Few-shot safe exploitation at 6-max.** *The thesis. Starts the
moment A3 holds up; its first data point (below) does not wait.*

Pluribus modelled nobody. GTO Wizard AI names opponent modelling as
future work. Nothing published adapts to a 6-max table. The design,
settled 25 Aug:

*Three timescales, weights never updated at the table.*
1. **Per hand, seconds: explicit per-seat statistics.** Entry, raise,
   fold-to-bet, showdown and bet-size frequencies by street. Bayesian,
   exact, dozens of numbers. These become the range tracker's prior in
   place of the blueprint's own probabilities, which is the structural
   fix for the belief failure that made HU search a net loss.
2. **Per session, hundreds of hands: in-context inference, frozen
   weights.** A sequence model over a seat's public action history
   (tokenised board cards, amounts, actions) that outputs a style
   estimate and a belief prior. It adapts the way an LLM adapts to a
   prompt: through its context, never through gradient steps. It is
   pretrained to be a fast identifier, then never trained live.
3. **Per week, offline: retraining** on logged real hands folded into
   the population, validated, swapped in.

*Pretraining population, generated by us.* RNR exploiters at several p,
QRE-lambda variants, fold/call/raise-biased blueprints, clones of logged
opponents. Known ground truth per opponent, so the model learns
identification, then meets real tables. Real hand histories exist in
the millions online and only public actions are needed; that is the
second-stage data.

*Actions are never chosen by the model.* The solver plays one of a
portfolio of pre-solved strategies (equilibrium plus RNR exploiters),
each within a hard exploitability cap X of the equilibrium. The model
and the statistics choose which arm is in play for which seat (the UCB1
portfolio bandit exists; it becomes per-seat and model-informed). Frozen
weights plus a capped portfolio bound the damage a deceptive opponent
can do to "the wrong arm, within X".

*Why not online weight updates:* gradient steps from a few hundred hands
are noise the counts already extract; an opponent who knows you learn
live can poison you, and with live updates the worst case is unbounded;
and drifting weights break per-seed reproducibility, which is the only
reason any number in BASELINES.md can be trusted.

*Gate, always two numbers reported together:* winrate against scripted
archetypes (station, maniac, nit, mixed tables) with AIVAT, and the
multiway BR bound (A0) of the strategy actually played. An exploiter
that wins more and is more than X outside the equilibrium bound has
failed. X is chosen up front, before the first run.

*Order of work:* (1) the RNR exploiter vs Slumbot, already trained,
measured at 2k then 10k, as the first data point on the cap-vs-winrate
trade-off; (2) per-seat statistics into the tracker prior, measured by
`eval` against archetypes and by the multiway probe; (3) per-seat
portfolio arm selection; (4) the synthetic population and the frozen
profiler, on the GPU; (5) real-hand retraining.

*Risks:* a synthetic population may not cover real human styles (the
mitigation is stage 5); multiway convergence has no guarantees for
anyone, so every claim is empirical; and none of it is safe to deploy
before A3, because an exploiter still has to resolve rivers.

**B3. Neural blueprint.** *Last, largest.*
Deep CFR or a ReBeL-style trained policy removes the bucket ceiling
outright. Only worth starting once A4 shows where tabular scaling stops
paying, and only with rented compute. Not before 2027 on current
evidence.

### Phase C: a benchmark, without which none of this is a claim

The heads-up world has Slumbot and the GTO Wizard API. 6-max has nothing
public, and Pluribus itself cannot be played. "Better than Pluribus" will
therefore always be an indirect claim, and the honest form of it is:
the same design at comparable scale, lower internal bounds on both the
HU-line and multiway probes, and at least one external result. Nothing
in this file produces a single number that settles it. Options in cost
order; do the first two regardless.

**C1. Both probes with `--search`.** Internal, covered in A0 and A3.
Necessary, not sufficient: lower bounds against our own abstraction.

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
done     A0 probe built, A1 infra built
running  A2 wall-clock -> A0 first bound -> A1 three-arm menu test
next     B2 step 1 (RNR exploiter vs Slumbot), A5 audit items (cheap)
weeks    A3 multiway search, gated by A0 --search   <- makes B2 safe
then     B2 steps 2-3 (stats prior, per-seat arms), B1 net leaves,
         C2 Pluribus clone (parallel)
then     B2 steps 4-5 (population + frozen profiler, GPU), A4 rented run
then     C3 public API
2027     B3 neural blueprint, C4 humans
```

A0, A1 and A2 can run this week on this machine. A3 is the item that
decides whether we are building a Pluribus or a blueprint with a nice
probe. One ordering tension to be explicit about: the goal is money games
with weak players, which is B2, and B2 sits behind A3. The reason is that
exploitation on top of a search stack that collapses under off-tree play
is fragile; the HU search regression was exactly that failure. B2's first
data point (the RNR exploiter vs Slumbot, under "kept alive") is cheap
and does not wait.

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
- **The probes are heads-up-line probes.** Until A0 exists, no 6-max
  exploitability number in this repo has seen a multiway pot. Say so
  whenever quoting one.
- **Cross-play does not see exploitability.** Every cross-play in the repo
  has been a wash, including between blueprints whose BR bounds differ by
  2x. Use it as a sanity check, never as the gate.
- **Record the negatives.** Most entries in BASELINES.md are nulls or
  regressions, and they are the reason the open list above is short.
- **Two numbers or none.** Any exploitation result is reported with the
  exploitability bound of the strategy played, on the same line. One
  without the other is not a result.
- **Match the recipe before improving it.** The 60% exploitability cut
  came from reading the supplement's pseudocode line by line, not from a
  new idea. Do that for the rest of Algorithm 1 and the search section
  before inventing.
