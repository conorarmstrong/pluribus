# Roadmap

Where the project stands, what is open, and what is closed. Every claim
here is backed by a run logged in [BASELINES.md](BASELINES.md); this file
is the plan, that file is the evidence.

## Status (August 2026)

Two fronts, both measured, neither finished.

**6-max (the Pluribus setting).** The 200M-iteration blueprint
(`blueprint.bin`, 12 buckets/street) is the standing reference: BR +472,
LBR +354 mbb/hand pooled over 8 seeds, +4262 / +3705 mbb/hand vs
random / caller with AIVAT. A modernized blueprint (`blueprint_6max_v2.bin`:
OCHS features, widened bet menus, 24 buckets, 400M iterations) points the
right way on both probes but is not significant at 8 seeds (BR +81 ±165,
LBR +114 ±167), and is confounded by 5x visit dilution.

**Heads-up 200bb (the external benchmark).** Against Slumbot over its
public API the blueprint loses: -714.5 +-331.5 mbb/hand at 10,000 hands.
Online search made it worse (-1771 +-472), which traced to two real bugs
in the search stack, both now fixed, plus one open leak. This is the more
informative front, because Slumbot does not share our abstraction.

The repo carries 74GB of gitignored `.bin` artifacts (blueprints,
checkpoints, turn-solve datasets, value nets).

## Open now

### 1. Re-measure HU search after the flop-routing fix

The 10k search run predates commit `e539914`, which stopped no-net flop
decisions from being re-solved by an 800ms MCCFR pass. That pass was
producing 51 flop all-ins per 10k hands at roughly 18% equity, against 2
for the raw blueprint. Pre-river stack-offs cost -980 mbb/hand of the
total, and those 51 flop all-ins alone accounted for -6,600 bb. Flop now
falls back to the blueprint HU; the exact turn and river solvers are
unchanged.

- Run: `slumbot --blueprint bp_hu200_300m.bin --hands 2000 --search
  --search-ms 800 --safe-resolve --verbose` as a diagnostic, then 10,000
  hands as the gate.
- Gate: beat blueprint-only (-714.5) by +100 mbb/hand at 10k hands.
- Note: 2k-hand runs have +-0.8 to 1.0k mbb confidence intervals. They
  diagnose, they do not decide.

### 2. The river-showdown leak

Even the exact river solver loses 2.4x more per showdown than plain
blueprint river play (-7.68 vs -3.24 bb/hand, on 1,568 and 1,842
showdowns respectively). The solver is exact, so the input is wrong: the prime
suspect is the belief-widening beta cap (0.75) leaving ranges too wide by
the river, which inflates the estimated bluff frequency and turns marginal
folds into calls. Secondary suspect: gadget alternative quality on
turn-facing lines.

- Sweep the widening cap and re-run the per-street autopsy.
- Cheaper first check: compare river call frequency against the blueprint's
  on the same logged spots.

### 3. Finish and measure the RNR exploiter

The clone-and-exploit path is built end to end and untested against the
live opponent. `clone` turns 10,297 logged Slumbot hands into a
Blueprint-format model (24,976 decisions, 5,107 infosets; held-out top-1
agreement 97.2% preflop, 51.8% flop, 44.1% turn, 29.3% river), and
`train --rnr-opponent slumbot_clone.bin --rnr-p <p>` trains a bounded
best response to it. `slumbot_exploiter_ckpt.bin` is a partial run.

- Finish training, then measure at 2k hands, then 10k.
- Gate: beat -714.5 mbb/hand.
- Known risk: clone agreement is weakest exactly on turn and river, where
  the money is lost. If the exploiter underperforms, more logged hands is
  the first lever, not a bigger `--rnr-p`.

### 4. Settle the 6-max modernization question, or drop it

The v2 result mixes three changes (OCHS, wide bets, 24 buckets) with a 5x
dilution confound. Two ways to resolve it, in cost order:

- Ablate: OCHS alone at 12 buckets and the baseline menu, 200M iterations,
  8 seeds paired. Isolates the feature from the dilution.
- Or remove the confound: v2's abstraction at enough iterations to restore
  the baseline's 1.56 visits per infoset, which means roughly doubling the
  run to 800M iterations. Expensive.

If neither shows a significant BR improvement, the abstraction-refinement
direction is closed and the effort belongs on search and beliefs.

## Next, once the above resolves

### Internal probes cannot see 200bb

`br`, `lbr`, `eval` and `crossplay` all build a `HandConfig` with the
default 10,000-chip stack and override only the player count, and a
`Blueprint` does not record the stack it was trained at. So an HU 200bb
blueprint has no internal exploitability number at all, and every HU
result costs a 4 to 6 hour live API run. Adding `--stack` to the probes
(and ideally recording the training stack in the blueprint) turns that
loop from hours into minutes, and would have caught the search regression
without burning a string of multi-hour live runs.

### Opponent modeling is the structural fix

Range tracking assumes every opponent plays our blueprint. Against
Slumbot that assumption is what made search a net negative. Belief
widening and gadget resolving bound the damage; neither estimates the real
opponent. The candidates, roughly in order of ambition:

- Online estimation of the live opponent's action frequencies, folded into
  the tracker as a prior (the `clone` pipeline already proves the offline
  version works).
- A population belief network trained across a portfolio of opponents,
  queried at the table instead of the blueprint's own probabilities.

### A 200bb HU value net

None exists, so `--value-net` is inactive heads-up and flop search there
has no learned leaf values. Building one means `gen-turn-data` against a
200bb HU blueprint. Worth doing only if item 1 shows turn and river search
is finally net positive; otherwise it is optimizing a losing branch.

### Solver menus

Turn and flop resolves use slim bet menus, which restrict the hero more
than the opponent. Widening them is the remaining untested strength lever
inside the search stack, now that leaf-value accuracy and policy
distillation have both measured null.

## Longer horizon

- **Neural blueprint.** Tabular buckets are the wall: every experiment
  that pushed on granularity ran into visit dilution instead. Deep CFR or
  a ReBeL-style trained policy attacks that directly.
- **Exhaustive canonical abstraction.** Cluster enumerated canonical
  boards rather than Monte-Carlo samples of them.
- **Multiway search.** Nothing learned heads-up transfers: multiway turn
  spots still fall back to sampled MCCFR, and 6-max has no exact solver.
- **Earlier-street re-solving.** Subgame roots trust the blueprint's action
  menu; far off-tree lines are absorbed by shadow-hand mapping rather than
  re-solved.

## Closed branches

Do not redo these without a new reason. Full write-ups in BASELINES.md.

| Branch | Verdict |
|--------|---------|
| Policy distillation (`distill`) | Null. Cross-play -38.5 +-62.0 over 1M hands, BR unchanged. A generation touches 0.03% of infosets. |
| Value-net scaling | Null. 30% validation-loss improvement bought +3 mbb/hand at the table. |
| Strategy-aware abstraction (`--strategic-from`) | Null on exploitability and head-to-head, at ~10x per-iteration cost. |
| More HU iterations (100M to 300M) + exact river buckets | Null vs Slumbot (-782 to -719, well inside the interval). |
| 36 buckets at equal iterations (HU) | Worse (-1128). Granularity without visits does not pay. |
| Equilibrium selection worry (6-max) | Does not materialize: all cross-play cells within +-80 mbb of zero. |

## Measurement discipline

Learned the hard way, twice each:

- **Pool seeds.** A single BR or LBR seed has misled this project more than
  once (v2 looked 5.4x better at seed 1, then 1.6x at 8 seeds). Pool at
  least 8, and pair by seed: BR is deterministic per seed, so paired
  comparison is free precision.
- **Small live samples do not decide.** A 2,000-hand Slumbot result carries
  a +-900 mbb interval. The -234 that appeared to clear a gate came back
  -1771 at 10k.
- **Record the negatives.** Most entries in BASELINES.md are nulls or
  regressions, and they are the reason the open list above is short.
