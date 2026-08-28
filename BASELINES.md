# Baselines

Frozen measurement records for later comparison. Append a new section per
baseline; never edit an old one. Artifacts are gitignored, so the SHA-256
hashes below are the only durable link between these numbers and the
binaries that produced them — verify with `shasum -a 256 <file>` before
trusting a comparison.

## 2026-07 — 200M-iteration blueprint (pre-research-programme reference)

- Code: commit `fae2399404422d000814c2b7eae2eab02877d496` (2026-07-03)
- Toolchain: rustc 1.90.0, `--release` (lto, codegen-units=1)
- Hardware: 16 cores (Apple Silicon, Darwin 25.5.0)
- Tests at time of record: 90/90 passing

### Artifacts

| File | SHA-256 | Size (bytes) | Produced by |
|------|---------|--------------|-------------|
| blueprint.bin | `ff74852ca3aaf4cefda307605569f8eeb868b0bd78eb752f97c791537e9f3eb1` | 4,343,417,226 | `train --iters 200000000 --out blueprint.bin` |
| bp_equity30.bin | `e1fab58dd8864e62f0dded858ed25f1d19afaec44cdd2c4a3abdfb74913cda58` | 2,213,011,970 | `train --iters 30000000 --out bp_equity30.bin` |
| bp_seed1.bin | `0f730ec06427a84d05402fb3516087f06af191d598446d6d113832ab1ba4c67f` | 2,204,175,907 | `train --iters 30000000 --train-seed 1 --out bp_seed1.bin` |
| bp_seed2.bin | `85c969b9553f7d4fdb88e215c5c1b524953b503ae119412c42615ad383ee4249` | 2,186,197,567 | `train --iters 30000000 --train-seed 2 --out bp_seed2.bin` |
| bp_strat30.bin | `44598bf6f6b4bb07bf3e9d5bf22f479775ecf37f4e068f20919f023bd2b1504b` | 2,141,672,873 | `train --iters 30000000 --strategic-from bp_equity30.bin --out bp_strat30.bin` |
| turn_data.bin | `6036f24b42e5e6c4c67517fa8dbc03a7b0db1caed32245405f3b6acbe18d2cfa` | 425,440,008 | `gen-turn-data --blueprint blueprint.bin --samples 20000` |
| value_net.bin | `0e59296c2a3a5f22e62f1e0ad542463912a8d079dc7e3878699b945f32efcabf` | 12,038,616 | `train-value-net --data turn_data.bin --hidden 512,512` |
| value_net_256.bin | `0031ab712a878422421c73a9fc8238c34abce555966d6060157e5df5b37ebd9c` | 5,762,520 | `train-value-net --data turn_data.bin --hidden 256,256` |

blueprint.bin: 101,022,969 exported strategies (preflop 19,912,089 /
flop 20,169,606 / turn 27,164,266 / river 33,777,008), 12 EMD k-means
buckets/street, 6 players, trained in 79 minutes.

### Metrics

| Metric | Value | Command |
|--------|-------|---------|
| vs random, plain | +4426 ±334 mbb/hand | `eval --blueprint blueprint.bin --hands 200000 --baseline random` |
| vs caller, plain | +3735 ±371 mbb/hand | `eval --blueprint blueprint.bin --hands 200000 --baseline caller` |
| vs random, AIVAT | +4285 ±273 mbb/hand | `eval --blueprint blueprint.bin --hands 100000 --baseline random --aivat` |
| vs caller, AIVAT | +3704 ±260 mbb/hand | `eval --blueprint blueprint.bin --hands 100000 --baseline caller --aivat` |
| LBR exploitability lower bound | +366 ±322 mbb/hand | `lbr --blueprint blueprint.bin --hands 20000` |
| Search gain (net leaves vs rollout leaves) | +313 ±198 mbb/hand | `eval --net-gain --value-net value_net.bin --search-ms 800 --hands 12000` |
| Search+net gain vs raw blueprint | +331 ±213 mbb/hand | `eval --search-gain --value-net value_net.bin --search-ms 800 --hands 12000` |
| Search gain without net, 800ms | −55 ±495 mbb/hand | `eval --search-gain --search-ms 800 --hands 12000` |
| Search gain without net, 150ms | −520 ±399 mbb/hand | `eval --search-gain --search-ms 150 --hands 12000` |
| Value net validation loss | 0.00089 weighted MSE (≈3% RMS of max swing) | `train-value-net --data turn_data.bin --hidden 512,512` |
| Pluribus top-1 agreement (overall) | 66.8% (15,169 decisions, 99.0% covered) | `benchmark --blueprint blueprint.bin --dir data/pluribus` |
| Pluribus agreement by street | 75.6% / 49.9% / 46.0% / 44.3% (pre/flop/turn/river) | same |
| Pluribus mean action probability | 0.599 | same |
| Replay chip accounting | 9,992/9,992 hands exact | same |
| Crossplay (3 seeds × 6 directions, 200k hands each) | all cells within ±80 mbb/hand of 0 (CIs ±245) | `crossplay --focal ... --field ...` |
| Strategic vs equity abstraction, head-to-head | ≈ 0 ±246 both directions | `crossplay --focal bp_strat30.bin --field bp_equity30.bin --strat-prev bp_equity30.bin` |

Benchmark and inspect rows re-verified 2026-07-03 against
`blueprint.bin` (`ff74852c…`): identical output.

### Known caveats frozen into this baseline

- LBR CI (±322) is nearly as large as the measurement; treat it as
  order-of-magnitude only until an exact best-response metric exists.
- Eval baselines are `random`/`caller` only — winrates are not comparable
  to human-pool or cross-bot winrates.
- Search-gain rows used 800ms/decision on this hardware; budget and
  hardware changes shift them.

## 2026-07-03 — Phase 0: BR probe (exact turn/river subgame best response)

- Code: working tree after `fae2399` (br probe + canonical-key-seeded
  equity draws; not yet committed at time of record)
- Tests at time of record: 96/96 passing
- Blueprint: blueprint.bin `ff74852c…` (the 200M baseline above)

New measurement command: `br` — LBR's harness, but every turn and river
decision plays an exact best response of the entire remaining game (single
expectimax pass, full menus, no CFR convergence error). Sound lower bound,
strictly tighter than LBR in expectation, bit-for-bit reproducible per
seed. Preflop/flop still use the greedy LBR action (full-game exact BR
≈ 1e13 vector node-visits — intractable).

Also in this change: Monte Carlo equity draws are now seeded from the
canonical (hole, board) key instead of the caller's rng — bucketing is
identical across processes, thread interleavings, and cache evictions.
LBR re-measured under the new code for a same-code comparison.

| Metric | Value | Command |
|--------|-------|---------|
| BR probe exploitability lower bound | **+475.4 ±320.6 mbb/hand** (20k hands, 314s) | `br --blueprint blueprint.bin --hands 20000 --seed 1` |
| LBR (same code, for comparison) | +301.3 ±321.3 mbb/hand (20k hands, 76s) | `lbr --blueprint blueprint.bin --hands 20000 --seed 1` |

Reading: the stronger probe tightens the lower bound by ~+174 mbb/hand at
the point estimate (CIs overlap; the probes are not per-deal paired — they
draw different deal sequences). This is the Phase 0 reference number for
gating Phase 1: nested safe re-solving and continual resolving should push
the BR probe number down measurably on this same seed.

### Caveats

- The probe is exact only from the turn; preflop/flop still greedy, so
  the true exploitability is higher than this bound.
- CI is still deal-noise dominated (±320 at 20k hands); scale `--hands`
  or add duplicate dealing to the probe before reading small deltas.
- The probes measure the RAW blueprint. Online search is not probed yet;
  Phase 1 should add a `br`-vs-search mode.

## 2026-07-03 — Phase 1a/1b: nested re-solving + continual resolving

- Code: commits `266ae46` (nested re-solving), `7558f04` (continual
  resolving), `b49bd3a`/`0f274d8` (distill infra)
- Tests at time of record: 103/103 passing

Changes to table play: subgames root at the REAL state (off-tree bets
priced at actual size); turn decisions with two live players use the
exact turn+river vector solver (slim menu) instead of sampled MCCFR;
turn resolves run inside the Burch gadget under `--safe-resolve` and
carry opponent river-entry CFVs into the river gadget (SearchSession).

| Metric | Value | Command |
|--------|-------|---------|
| Search gain w/o net, exact turn solving | +30.1 ±128.8 mbb/hand (40k paired deals, 800ms, 530s) | `eval --blueprint blueprint.bin --search-gain --hands 40000 --search-ms 800 --seed 3` |

Reading: no-net search stays statistically zero against blueprint tables
even with exact turn solving (prior record: −55 ±495 at 12k deals). The
learned leaf values remain what makes search pay; flywheel distillation
therefore uses net-search as the teacher. Note the off-tree pricing fix
(1a) cannot show in this metric — all self-play actions are on-menu; the
off-tree probe (task list) is the instrument for that.

## 2026-07-03 — Phase 2 gen1: POLICY distillation is a clear negative

- `distill --blueprint blueprint.bin --out bp_gen1.bin --hands 10000
  --search-ms 800 --alpha 0.5 --value-net value_net.bin --seed 1`
- 15,730 searched decisions distilled into 9,442 infosets (868s)

| Gate | Result | Reference |
|------|--------|-----------|
| BR probe on gen1 (`--hands 20000 --seed 1`) | **+1306.3 ±348.9 mbb/hand** | gen0: +475.4 ±320.6 |
| Crossplay gen1 vs gen0 (200k hands) | +82.2 ±137.9 (≈0) | — |

Verdict: FAIL — do not adopt. Resolved distributions are near-pure best
responses to tracked ranges; blending them at α=0.5 (often one sample
per infoset) plants deterministic, exploitable strategies on the most
common lines. Exploitability lower bound ~2.8×; no head-to-head gain.
This is the known theoretical failure mode of policy-space expert
iteration in imperfect-information games, caught by the Phase 0 metric.
ReBeL distills VALUES for exactly this reason. bp_gen1.bin kept on disk
for forensics; not hashed as a baseline artifact.

Follow-ups worth testing before abandoning policy space entirely:
low α (≤0.1) + minimum-sample thresholds per key. Primary path forward:
value-space flywheel (regenerate turn data under current search
conditions, retrain/scale the net).

### Correction (same day): gen1's failure was a distill BUG, not (yet)
### evidence against policy distillation

The α=0.1 falsification run (bp_gen1b) came back **α-independent**:
BR +1387.8 ±350.2, crossplay +82.6 ±137.9 — statistically identical to
α=0.5. A 5× smaller blend should have produced ~5× less damage; it
produced none less. Diagnosis: the merge's overwrite-on-length-mismatch
rule. The slim-menu solvers (turn: 4 actions, flop-net: 5) returned
distributions shorter than the blueprint's full-menu entries, so nearly
every turn/flop distillation REPLACED a full-menu strategy with a short
vector — and a length-mismatched entry falls back to pure check/call at
play time. The distillation had been quietly converting the bot's most
common postflop infosets into a calling station; α never mattered.

Fixed in `afbb08e` (collect-time re-expression of solver distributions
onto the full blueprint menu by action identity; merge skips mismatches,
never overwrites). gen1c (α=0.1, corrected) queued with the same gates.
The original policy-purity concern remains open, now actually testable.

### gen1c (corrected distill, α=0.1): safe, gain unproven

- `distill --hands 10000 --search-ms 800 --alpha 0.1 --value-net
  value_net.bin --seed 1` → 15,801 decisions, 9,505 infosets, 0 menu
  mismatches

| Gate | Result | Reference |
|------|--------|-----------|
| BR probe (`--hands 20000 --seed 1`) | **+484.5 ±321.6** | gen0: +475.4 ±320.6 |
| Crossplay vs gen0 (200k hands) | +87.5 ±137.6 (not significant) | — |

Reading: exploitability restored to baseline exactly — the corrupted
gens' +1300 was entirely the menu bug, and no purity damage is visible
at α=0.1. Head-to-head is positive but underpowered at this coverage.
gen1d queued as the definitive test: 50k hands, α=0.5, 1M-hand
crossplay (±60 CI).

### gen1d (definitive): policy distillation is safe and WORTHLESS —
### Phase 2 policy branch closed

- `distill --hands 50000 --search-ms 800 --alpha 0.5 --value-net
  value_net.bin --seed 1` → 80,097 decisions, 33,373 infosets, 0
  mismatches, 73 min

| Gate | Result | Reference |
|------|--------|-----------|
| BR probe (`--hands 20000 --seed 1`) | +484.5 ±328.5 | gen0: +475.4 ±320.6 |
| Crossplay vs gen0, **1M hands** | **−38.5 ±62.0** | — |

Verdict: at 5× coverage and α=0.5, with a ±62 CI: no head-to-head
effect (gen1c's +87.5 ±138 and gen1d's −38.5 ±62 bracket zero), no
exploitability change. A 73-min generation touches 0.03% of infosets,
and on exactly those high-traffic lines the 200M-iteration blueprint is
already near its best; search's edge lives in rare deep spots self-play
rarely revisits. Policy-space expert iteration at this scale: honest
negative. Pivot to value space: more/bigger turn data + larger net (the
net is the measured +313 search edge). turn_data2.bin (30k spots,
seed 2) generation launched.

Experiment artifacts bp_gen1{,b,c,d}.bin (4 × 4.3GB) are candidates for
deletion — none passed gates; diagnosis recorded here. (Deleted 3 Jul
with user approval.)

## 2026-07-04 — Phase 2 value branch: better nets don't help at the
## table — flywheel closed

30k new exactly-solved turn spots (turn_data2.bin, seed 2, 8.4h)
combined with the original 20k; two nets trained on the 50k total:

| Net | Val loss | Net-gain, same 40k paired deals (seed 7, 800ms) |
|-----|----------|--------------------------------------------------|
| value_net.bin (20k, 512×512) | 0.00089 | +171.1 ±100.3 mbb/hand |
| value_net50k_512.bin | 0.00066 | not run (dominated by 1024 on loss) |
| value_net50k_1024.bin | 0.00062 | **+174.1 ±99.7** |

Verdict: a 30% val-loss improvement translates to +3 mbb at the table —
nothing. Leaf-value accuracy is no longer the binding constraint at
800ms budgets. (Both net-gain arms now include exact turn solving,
which also explains the lower absolute net-gain vs the historical
+313 ±198: the no-net baseline got stronger in Phase 1b.)

Phase 2 conclusion: policy distillation null, value scaling null — the
search stack's remaining strength levers are solver menus (slim
TURN/FLOP menus restrict the hero), iteration budgets, leaf-query
sampling (16 of 49 turns), and belief quality (Phase 3's target).
value_net50k_512.bin equals the old net's speed with better loss —
harmless default swap if desired; 1024 costs 2× per query for nothing
measured. Artifact hashes:
- turn_data2.bin `bed4dcc33f88d7fe602275fabc3aff69939369ba45b6d9144519be38fd6ad8ab` (638,160,008 B)
- value_net50k_512.bin `a6655a55baafaf1c353cc0d1db1f5fbfc2335876f7e4ebd57b62c12761b2e158` (12,038,616 B)
- value_net50k_1024.bin `2cfa690afe61f6692355f37e75d4b19b1558c07bee92f7b4f8822397e5f641e6` (26,163,672 B)

## 2026-07-04 — HU 200bb blueprint + Slumbot adapter (external benchmark)

- Code: commit `134b43f` (Slumbot adapter), `--stack` train flag
- bp_hu200.bin `5f4ac04c2ca7af9ff0427fe2731bd130b28a037ec845f22ace79f7a335a47568`
  (178,501,229 B): `train --iters 100000000 --players 2 --stack 20000
  --out bp_hu200.bin` — 4.1h, 4,530,427 infosets, 4,203,000 strategies
- Live protocol smoke test: 20 hands, 0 desyncs, ~1.5s/hand
- Caveats: internal probes (br/lbr/eval) have no --stack flag yet, so
  this blueprint has no internal exploitability number; no 200bb value
  net exists (flop-net search inactive).

### First Slumbot results (2,000 hands each, sequential API sessions)

| Config | mbb/hand | Command seed |
|--------|----------|--------------|
| search 800ms, unsafe | −1649.1 ±1130.5 | 2 |
| search 800ms, --safe-resolve | −1207.2 ±983.4 | 4 |
| **blueprint only** | **−782.0 ±809.4** | 3 |

Readings (CIs overlap; ordering consistent across configs):
1. Search HURTS vs a non-blueprint opponent — the tracker models
   Slumbot as playing our blueprint, and resolves best-respond to
   fictional ranges. Internal paired evals never see this (self-play
   opponents ARE the blueprint). Phase 3's population belief net is the
   structural fix.
2. The gadget recovers part of the damage but not all — note its
   safety values are rollout-estimated FROM the tracker, so wrong
   beliefs contaminate the alternatives too.
3. Blueprint-only ≈ always-fold level: the 100M 200bb core is
   undertrained. bp_hu200_300m.bin (300M iters, ~12h) training.
4. Live play found a real crash (fixed, `0744d3c`): off-tree drift can
   leave shadow all-in while real can raise; apply_abs now degrades.
   0 desyncs across all 6,000+ live hands.

## 2026-07-05 — Exact river bucketing (10× training) + 300M null

Training profiling found 83% of CPU in per-(hole,board) MC river equity
(200 rollouts/miss, hit rate collapsing — the 300M run tracked 2+
days). Fix: one exact O(N) sweep per suit-canonical board (~134k
total), all 1,326 combos at once, verified bucket-for-bucket vs naive;
plus read-before-write locking and a thread-local board memo
(commits `b5be9ec`..`27cc811`). Sustained training rate 6,762 →
28,846 iters/s (cold-cache probe 69k); 300M = 2.9h.

- bp_hu200_300m.bin
  `1f285e9057c117b6a29870ea7577221e9189ec4831a361f345aa65d48c39c800`
  (187,029,939 B): 300M iters, 200bb HU, exact river buckets, 4.57M
  infosets.

| Blueprint | vs Slumbot (2k hands, blueprint-only, seed 3) |
|-----------|------------------------------------------------|
| 100M, MC river buckets | −782.0 ±809.4 |
| 300M, exact river buckets | **−719.0 ±867.3** |

Verdict: NULL — 3× training + exact buckets moved nothing. Iteration
count is not the binding constraint; the 12-bucket abstraction is.
Next: bucket scaling (`--buckets 36`, running) — river granularity is
now free (exact tables cost the same at any bucket count).

### Bucket scaling, first point: 36 buckets @ equal iterations is WORSE

- bp_hu200_b36.bin: `--buckets 36`, 300M iters, 2.5h (33,930 iters/s),
  13.5M infosets (3× the 12-bucket count)

| Config | vs Slumbot (2k hands, blueprint-only, seed 3) |
|--------|------------------------------------------------|
| 12 buckets @ 300M | −719.0 ±867.3 |
| 36 buckets @ 300M | −1128.2 ±1014.9 |

Reading: not a refutation of bucket scaling — 3× infosets at equal
iterations means 1/3 the visits each (granularity/convergence
tradeoff). Visit-parity test running: 36 buckets @ 1B iterations
(bp_hu200_b36_1b). If THAT is also ≤ the 12-bucket line, the 200bb
wall is the bet-menu abstraction, not card buckets. Note all 2k-hand
CIs are ±0.8-1.0k mbb — differences under ~400 are suggestive only.

### Stage 0+1a: loss autopsy + belief widening (commits `e33db42`..`510b990`)

Root-caused the search regression: hard-Bayes range tracking assumes
the opponent plays the blueprint; against Slumbot the posterior
concentrates on wrong hands and search optimizes against a fiction.
Fix: likelihood-calibrated belief widening (opt-in, live play only —
self-play harnesses keep exact Bayes). Plus per-street loss autopsy.

Command: `slumbot --blueprint bp_hu200_300m.bin --hands 2000 --search
--search-ms 800 --safe-resolve --seed 5 --verbose` (80 min, 0 desyncs)

| Config (2k hands each) | mbb/hand |
|------------------------|----------|
| search, hard-Bayes beliefs | −1649.2 ±925.3 |
| safe search, hard-Bayes | −1206.7 ±961.2 |
| blueprint-only | −719.0 ±867.3 |
| **safe search, widened beliefs** | **−894.0 ±838.2** |

Verdict: widening recovers ~310 mbb over pre-widening safe search and
lands at blueprint parity (all gaps within CI). Search no longer
actively hurts; it does not yet help. Stage 1 gate (+100 over
blueprint) NOT met.

Autopsy (bb/hand by ending street/kind): the river is the entire
loss — river showdowns −5.18 × 316 hands (−1636 bb) and river folds
−10.88 × 94 hands (−1023 bb) sum to −1713 of the −1788 bb total net.
Every "they fold" row is profitable. 6 of 9 pre-river all-in
stack-offs lost (−200bb each; n tiny). Directs Stage 1b at river
resolves: villain-led rivers still use tracker-contaminated rollout
alts instead of carried CFVs.

### Stage 1b: carried-CFV gadget alts on villain-led rivers (commit `569993a`)

The river resolve previously consumed the turn carry only when we
opened river betting; villain-led rivers fell back to
tracker-contaminated rollout alts. Fix: match the carry on the turn
line alone (opponent constraint values are unchanged by the
opponent's own actions — continual resolving).

Command: `slumbot --blueprint bp_hu200_300m.bin --hands 2000 --search
--search-ms 800 --safe-resolve --seed 6 --verbose` (81 min, 0 desyncs)

| Config (2k hands each) | mbb/hand |
|------------------------|----------|
| safe search, hard-Bayes | −1206.7 ±961.2 |
| blueprint-only | −719.0 ±867.3 |
| safe search, widened (1a) | −894.0 ±838.2 |
| **+ carried river alts (1b)** | **−234.2 ±826.9** |

Autopsy delta vs 1a run: river showdowns −5.18 → −1.84 bb/hand
(≈ +530 mbb/hand of the +660 total move — the predicted leak, fixed);
river we-fold −10.88 → −8.83. River total −1713 → −770 bb. Turn
we-fold slightly worse (−5.30 → −6.50, n 91→112). Point estimate
clears the Stage 1 gate (+100 over blueprint) by ~500 but CIs
overlap; 10k-hand confirmation run is the formal gate.

### Stage 1 gate: FAILED at 10k hands (run of commit `569993a` code)

Command: `slumbot --blueprint bp_hu200_300m.bin --hands 10000 --search
--search-ms 800 --safe-resolve --seed 7 --verbose` (6.2h, 0 desyncs)

**−1771.0 ±471.8 mbb/hand.** The 2k-hand −234 (seed 6) was a lucky
draw; pooled post-widening search (14k hands) ≈ −1426 — worse than
blueprint-only (−719/−782). Gate decisively not met.

Autopsy at 10k (reliable cell counts): pre-river all-in stack-offs
−980 mbb/hand of the total — 84 all-in pots averaging −117 bb each
(≈21% equity at stack-off); the 51 flop all-ins alone −6600 bb (≈18%
equity). River showdowns −7.68 bb/hand × 1568. Every "they fold" row
profitable (+1636 mbb/hand summed).

Diagnosis + fix (commit `e539914`): with no value net, flop search
decisions came from an 800ms MCCFR resolve of a 200bb tree — already
measured ≈0 vs blueprint in self-play, and the source of the flop
stack-offs. No-net HU flop now plays the blueprint; exact solvers
keep turn/river. Diagnostic 2k run (seed 9) queued. River-showdown
loss under the exact solver remains the open question — suspects:
over-widened beliefs by the river (0.75 beta cap), gadget alt quality
on turn-facing lines.

### Blueprint-only at 10k hands + first blueprint autopsy (data-collection run)

Command: `slumbot --blueprint bp_hu200_300m.bin --hands 10000 --seed 8
--verbose --log slumbot_hands.jsonl` (4.0h, 0 desyncs; also banked
10,297 hands of clone data incl. Slumbot's hole cards every hand)

**−714.5 ±331.5 mbb/hand** — confirms the 2k estimates (−719/−782)
at 3× precision. THE baseline to beat.

Autopsy vs the search gate run (both 10k):

| Cell | blueprint | search (569993a) |
|------|-----------|------------------|
| pre-river all-ins | 25 hands, −1600 bb | 84 hands, −9800 bb |
| flop showdowns | 2 hands, ±0 | 51 hands, −6600 bb |
| river showdown bb/hand | −3.24 × 1842 | −7.68 × 1568 |
| river total | −4992 bb | −12305 bb |

Reading: the flop-MCCFR stack-off indictment is airtight (2 vs 51
flop all-ins), validating fix e539914. Second finding: even the EXACT
river solver loses 2.4× more per showdown than blueprint river play —
the widened-belief river solve calls too loosely. Suspect: WIDEN
beta cap keeps ranges too wide by the river → solver overestimates
bluff frequency. Diagnostic seed 9 (flop fix) will isolate how much
of the −1056 gap the flop fix recovers.

### 6-max standing: 200M blueprint (blueprint.bin), pivot baseline (6 Jul)

The number to beat for the 6-max push. Exploitability lower bounds +
variance-reduced winrate vs baselines, all seed 1.

| Probe | Result | Hands |
|-------|--------|-------|
| BR (exact turn/river best response) | +469.5 ±322.9 mbb/hand | 20k |
| LBR (local best response) | +331.2 ±320.0 mbb/hand | 20k |
| eval vs caller (AIVAT+dup) | +3704.8 ±259.8 mbb/hand | 100k |
| eval vs random (AIVAT+dup) | +4262.0 ±271.9 mbb/hand | 100k |

BR/LBR are exploitability *lower bounds* (mbb/hand a best-responder
wins blind-vs-blind; 0 = unexploitable). +469 confirms real
exploitable holes — the 12-bucket abstraction wall again. Consistent
with the stale build-time numbers (+475/+301/+3735/+4426), so the
blueprint is intact. Next 6-max lever = deeper/finer blueprint
(neural or many-bucket abstraction), since HU online-search doesn't
transfer multiway.

### 6-max modernization: OCHS abstraction + wide bets (in training, 6 Jul)

Three frontier-gap fixes (commits: bets `HEAD~2`, OCHS `HEAD~1`):
- Bet abstraction: first-in menus now 25-33% through 200% overbets
  (was 3-5 Pluribus-era sizes); raise/re-raise gain a 200% size.
- Card abstraction: OCHS potential-aware features (8 preflop-tier
  opponent clusters, hand's equity vs each) concatenated with runout
  quantiles (dim 16), + 24 buckets (2x baseline). `train --ochs`.
- Value net (item 2): already ReBeL-shaped (per-hand CFV output, raw
  ranges/board in) — retrained for 6-max in the post-train chain.

Retrain: `train --players 6 --stack 10000 --ochs --buckets 24
--kmeans-samples 40000 --iters 400000000` → blueprint_6max_v2.bin.
27.5k iters/s at start (smoke). RISK: wide bets + finer buckets
explode infosets (4.7M at 200k iters vs baseline 101M at 200M) →
visit dilution, the same effect that made HU 36-buckets worse. BR/LBR
on the finished blueprint (baseline +469.5/+331.2) is the verdict;
lower = the modernization beat dilution. Post-train chain also builds
+ measures the 6-max value net (net-vs-no-net paired search gain).

### 6-max modernization VERDICT: exploitability dropped (7 Jul, blueprint_6max_v2)

`train --players 6 --stack 10000 --ochs --buckets 24 --kmeans-samples
40000 --iters 400000000` → 521.7M infosets, 391.3M stored strategies,
16GB, 57.2k iters/s (1h56m). 5x the baseline's infoset count — heavy
visit dilution (<1 visit/infoset avg), yet:

| Probe (20k hands, seed 1) | baseline | v2 (OCHS+wide bets) |
|---------------------------|----------|---------------------|
| BR (exploitability LB)    | +469.5 ±322.9 | **+86.9 ±352.2** |
| LBR                       | +331.2 ±320.0 | **+224.9 ±348.1** |

BR fell 5.4x (lower = closer to unexploitable). The richer/wider
abstraction more than paid for the dilution: wider sizing leaves the
best-responder fewer gaps, OCHS separates hands the old features
lumped. CAVEAT: single-seed diff is ~1.5 sigma unpaired (CIs wide at
20k); BR is deterministic-per-seed so likely stronger paired.
Confirmation sweep (BR+LBR seeds 2-4) running before building the
value net (item 2) on this blueprint.

### CORRECTION: v2 exploitability pooled over 4 seeds (seed-1 was a lucky draw)

v2 (OCHS+wide bets) BR by seed: +86.9, +318.1, +508.2, +243.4 →
**pooled +289 ±168** (80k hands). LBR: +224.9, +169.5, +304.2,
+499.9 → **pooled +300 ±168**. The dramatic seed-1 BR (+86.9) was a
favorable draw — same single-seed noise that faked the Slumbot −234.
Seed spread is huge (BR +87..+508). Baseline was only seed 1
(+469.5/+331.2), so the comparison isn't yet valid. Measuring the
baseline (blueprint.bin) across seeds 1-4 with the pre-bet-change
binary (git worktree @73ad29a — the new binary's wide menu would trip
the stale-menu guard and corrupt it) for a fair pooled-vs-pooled test.

### Fair pooled exploitability: v2 vs baseline, same 4 seeds (paired)

Baseline (blueprint.bin, narrow menu, @73ad29a binary) seeds 1-4:
BR +457.2/+374.9/+569.6/+517.0 (mean +479.7); LBR
+344.5/+152.0/+423.1/+279.5 (mean +299.8).
v2 seeds 1-4: BR mean +289.2; LBR mean +299.6.

BR is deterministic per seed → PAIRED by seed (same deals). Paired BR
diffs (base-v2): +370.3/+56.8/+61.4/+273.6, mean +190.5, t≈2.43 df=3
→ p≈0.09 (~90%, not yet 95%). LBR paired diff mean ≈ +0.2 (no
change). Reading: wider bets + OCHS probably cut TRUE exploitability
(BR) ~190 mbb/hand but the local probe (LBR) is flat — a strong
best-responder finds the closed gaps a local one never exploited.
Extending to 8 seeds (both blueprints) to settle significance.

### 8-seed verdict: modernization is a SMALL, NON-SIGNIFICANT improvement

Extended both pools to 8 seeds (paired by seed; br deterministic).

| Probe | baseline (8-seed) | v2 (8-seed) | paired diff | t (df7) | p |
|-------|-------------------|-------------|-------------|---------|---|
| BR    | +471.8 | +391.1 | +80.7 ±165 | 0.98 | ~0.36 |
| LBR   | +353.8 | +240.0 | +113.8 ±167 | 1.36 | ~0.21 |

Both point estimates favor v2 (less exploitable) by ~80-114 mbb/hand
(~17-24%), SAME direction on both probes — mildly suggestive of a
small real effect, but NEITHER is significant at 8 seeds. The seed-1
BR +86.9 and the 4-seed +190 were both favorable draws; v2 BR by seed
ranges +87..+706. HONEST CONCLUSION: OCHS + wide bets + 24 buckets at
400M iters did NOT clearly beat the baseline; any gain is small and
confounded by 5x visit dilution (0.77 visits/infoset). To confirm a
~20% effect against this variance needs many more seeds OR removing
the dilution confound (far more iters, or ablate wide-bets to isolate
OCHS). Code (items 1-3) is sound and committed; the ABSTRACTION-refine
direction shows limited blueprint payoff so far.

## 2026-08-24 — Trainer levers, paired 8-seed loop: VR-MCCFR is a clear NEGATIVE

- Code: commit `b342db6` (flags + default flip; the 30M/200M arms below
  ran on the same source, built before the default flip, with the flag
  set explicitly)
- Toolchain/hardware: as the July records (rustc 1.90.0 `--release`,
  16-core Apple Silicon, 128GB)
- Tests at time of record: 128/128 passing
- 30M-arm blueprints (bp_plain30/vr30/pprune30/snap30) were scratch
  artifacts, probed and deleted; not hashed. The 200M arms are hashed
  below.

Research pass over 2024-26 CFR work (PDCFR+/dynamic discounting, correlated
chance sampling, VR-MCCFR, neural/parallel CFR, the Pluribus supplement)
picked three cheap, sampling-compatible levers to test as opt-in `train`
flags. Proof loop, chosen so a change is decided in ~1h: both arms trained
FRESH with the same binary (the 5-action blueprint.bin cannot be probed by
the 6-action binary), `train --iters 30000000 --train-seed 0` 6-max
default abstraction, then `br --hands 20000` seeds 1-8 PAIRED by seed
(deterministic deals), 1M-hand crossplay both directions, and
`eval --baseline caller --aivat --duplicate` 100k.

Regime note: at 30M iterations the wide menu already yields 139.6M
infosets (more than the 200M-iteration narrow-menu reference had), i.e.
~0.2 visits/infoset — deep inside the dilution wall. Absolute BR numbers
here (~+1750) are therefore far above the 200M reference (+472) and are
only meaningful paired within this table.

### Arm A: plain (reference for this loop)

`train --iters 30000000 --train-seed 0`: 193.7s, 154,866 iters/s,
139,648,570 infosets, 106.4M strategies.

### Arm B: `--vr-baseline` (VR-MCCFR, Schmid et al. AAAI 2019)

Control-variate baselines at sampled opponent nodes, keyed by opponent
infoset + traverser seat + traverser bucket, EMA alpha 0.5; unbiased by
construction (unit test: fixed point unchanged on 10bb push/fold). On the
push/fold toy it was already a null (mean L1 to a 4M-iteration reference:
plain 0.725/0.480/0.314 vs vr 0.713/0.506/0.321 at 40k/160k/640k iters).

Training: 638.6s, 46,974 iters/s (3.3x SLOWER), 142.4M infosets, ~50GB RSS.

| Seed | plain BR | vr BR | vr − plain |
|------|----------|-------|------------|
| 1 | +1708.5 | +1938.7 | +230.2 |
| 2 | +1570.5 | +1688.4 | +117.9 |
| 3 | +1700.1 | +2141.1 | +441.0 |
| 4 | +1806.7 | +1829.2 | +22.5 |
| 5 | +1788.6 | +1829.4 | +40.8 |
| 6 | +1582.2 | +1628.6 | +46.4 |
| 7 | +2009.6 | +2032.2 | +22.6 |
| 8 | +1872.7 | +2298.7 | +426.0 |
| **mean** | **+1754.9** | **+1923.3** | **+168.4 ±148.5** (t=2.68, df=7, p≈0.03) |

Crossplay 1M hands: vr focal vs plain field −149.1 ±126.4; plain focal vs
vr field −37.6 ±126.7. Eval vs caller (AIVAT+dup, 100k): plain
+2667.3 ±242.5, vr +1941.4 ±229.2.

VERDICT: significantly MORE exploitable (8/8 seeds), loses head-to-head,
726 mbb/hand worse vs a calling station, 3.3x slower, 2x memory. Closed.
Reading: VR-MCCFR's gains need baselines that have converged, i.e. many
visits per (infoset, action). At 0.2 visits/infoset most baseline entries
are 1-2-sample estimates, so the correction term
sum_a sigma(a) b(a) − b(a_sampled) adds noise instead of removing it. The
lever is aimed at exactly the wrong regime for this blueprint: the
dilution wall starves it. (In the paper the 250x speedup is on Leduc with
millions of visits per infoset.) Flag kept as documented opt-in.

### Arm C: `--pluribus-prune` — exploitability HALVED (the win)

Pluribus's own pruning rules from the Science supplement (Algorithm 1,
p.14): (1) the prune/no-prune decision is made once per traversal (95%
prune, 5% explore all) instead of per action; (2) no pruning on the final
betting round; (3) actions leading directly to a terminal node are never
pruned. This trainer had been pruning per action, everywhere.

Training: 195.6s, 153,409 iters/s (same speed as plain), 167.7M infosets
(+20%: river and terminal-leading actions are now always explored).

| Seed | plain BR | pluribus-prune BR | improvement |
|------|----------|-------------------|-------------|
| 1 | +1708.5 | +853.9 | +854.6 |
| 2 | +1570.5 | +613.5 | +957.0 |
| 3 | +1700.1 | +1114.4 | +585.7 |
| 4 | +1806.7 | +849.1 | +957.6 |
| 5 | +1788.6 | +981.2 | +807.4 |
| 6 | +1582.2 | +1011.9 | +570.3 |
| 7 | +2009.6 | +1071.4 | +938.2 |
| 8 | +1872.7 | +901.4 | +971.3 |
| **mean** | **+1754.9** | **+924.6** | **+830.3 ±138.6** (t=14.2, df=7, p<1e-5) |

Eval vs caller (AIVAT+dup, 100k): +2879.8 ±245.5 vs plain +2667.3 ±242.5.
Crossplay 1M: pprune focal vs plain field −129.7 ±126.0; plain focal vs
pprune field −45.3 ±125.5 (both directions negative for the focal seat —
a wash within CI, as every cross-play in this repo has been).

VERDICT: BR exploitability lower bound cut by 47%, 8/8 seeds, every seed
by 570-970 mbb/hand, no speed cost, +213 vs a calling station. Head-to-
head is a wash, which is the usual pattern here (cross-play does not see
exploitability). Reading: per-action pruning on the river is pure loss —
there is no later street whose abstraction it could sharpen — and
pruning a terminal-leading action (a fold, a call that closes the action)
removes exactly the cheap exact payoffs that anchor the regret. Pruning
"everywhere" was leaving whole river subtrees with regrets frozen at
their early, wrong values. 200M-iteration confirmation queued (see below).

### Arm D: `--snapshot-avg` — NEGATIVE

Pluribus's postflop blueprint construction (supplement p.15): no running
average after the first betting round; the blueprint is the mean of
snapshots of the current regret-matching strategy (here: after a 10%
warm-up, every 5% → 18 snapshots). Preflop keeps the linear average.

Training: 280.5s incl. 18 full-map snapshot passes (~5s each), 130.0M
infosets.

BR by seed: +2313.7, +2072.7, +2152.9, +2103.2, +2168.7, +2201.4,
+2120.6, +2342.5 → mean **+2184.5**; paired vs plain **+429.6 ±139.9
worse** (t=7.3, 8/8 seeds). Eval vs caller +1879.2 ±227.0 (plain
+2667.3). Crossplay: snap focal vs plain −50.3 ±124.1, plain focal vs
snap −88.7 ±123.6.

VERDICT: significantly more exploitable, and 790 mbb/hand worse vs a
station. Pluribus chose snapshots for memory, and after ~11,000 minutes
of training, when 18+ snapshots of a nearly converged current strategy
average to something close to the true average. At 30M iterations the
current strategy of a 0.2-visits/infoset blueprint is mostly regret
noise, and 18 snapshots of noise average to noise; the linearly weighted
running average at least integrates every visit. Do not use below full
convergence; probably not worth revisiting at all on this hardware.

### 200M confirmation: Pluribus pruning at reference scale — CONFIRMED, new 6-max standing

Same loop at the reference iteration count. Both arms `train --iters
200000000 --train-seed 0` (6-max, 12 buckets, wide menu), current binary.

| Arm | train time | iters/s | infosets | strategies |
|-----|-----------|---------|----------|------------|
| plain (per-action pruning) | 1706s (28 min) | 117,218 | 306.1M | 232.6M |
| pluribus-prune | 4013s (67 min) | 49,834 | 402.1M | 304.2M |

Speed caveat (new at this scale): with 2.4x the per-iteration cost the
"no speed cost" seen at 30M does not hold at 200M — the exemptions add
31% more infosets and the bigger map costs contention. At equal
wall-clock, plain could run ~480M iterations; not tested.

| Seed | plain 200M | pluribus-prune 200M | improvement | July ref (narrow menu) | July v2 (OCHS, 400M) |
|------|-----------|---------------------|-------------|------------------------|----------------------|
| 1 | +295.7 | +183.9 | +111.8 | +457.2 | +86.9 |
| 2 | +333.0 | +169.8 | +163.2 | +374.9 | +318.1 |
| 3 | +594.1 | +103.4 | +490.7 | +569.6 | +508.2 |
| 4 | +641.3 | +239.9 | +401.4 | +517.0 | +243.4 |
| 5 | +335.4 | +211.8 | +123.6 | | |
| 6 | +625.3 | +260.0 | +365.3 | | |
| 7 | +246.7 | +133.3 | +113.4 | | |
| 8 | +937.1 | +197.5 | +739.6 | | |
| **mean** | **+501.1** | **+187.5** | **+313.6 ±190.3** (t=3.90, df=7, p≈0.006) | +471.8 (8-seed) | +391.1 (8-seed) |

VERDICT: confirmed. 8/8 seeds, 63% cut in the BR exploitability lower
bound at reference scale, and the new default blueprint (+187.5) is the
least exploitable 6-max blueprint this project has produced: 60% below
the July reference (+471.8) and 52% below the 400M OCHS/wide-bets v2
(+391.1), at 12 buckets and half v2's iterations. Two side findings:
(a) the wide bet menu on its own does nothing — plain 200M with the wide
menu (+501.1) matches the narrow-menu July reference (+471.8; seeds 1-4
paired: 466.0 vs 479.7); (b) which means July's v2 result (+391) was
never about OCHS or bet sizes either: with hindsight the whole
"abstraction refinement" branch was fighting a pruning bug. Pluribus
pruning rules are now the trainer default (`--per-action-prune`
restores the old behaviour for comparisons).

Artifact: blueprint_pprune200.bin, hash and tail metrics below.

| File | SHA-256 | Size (bytes) | Produced by |
|------|---------|--------------|-------------|
| blueprint_pprune200.bin | `198c7aa2804c89797a23e14ff7bb2fbc37a8ff6911bc87669b091a503a8da880` | 13,376,525,660 | `train --iters 200000000 --train-seed 0` (Pluribus pruning, the new default) |
| bp_plain200.bin (scratch, not kept) | `f7e52b34613b2347303892e6310d87381679bbb7b60486bb9854d1d3408bade2` | 10,085,728,352 | `train --iters 200000000 --train-seed 0 --per-action-prune` |

Tail metrics (200M arms):

| Metric | plain 200M | pluribus-prune 200M |
|--------|-----------|---------------------|
| eval vs caller (AIVAT+dup, 100k) | +2108.9 ±210.2 | **+2742.4 ±227.8** |
| LBR seed 1 (20k) | +156.6 ±353.5 | +42.5 ±322.4 |
| crossplay 1M, focal vs other's table | −49.3 ±72.4 (pprune focal) | +40.5 ±72.4 (plain focal) |

Vs-caller note: both wide-menu 200M blueprints win far less against a
calling station than July's narrow-menu reference (+3704.8). Not a
regression in exploitability (BR says the opposite); the wide menu's
overbet-heavy lines simply extract less from a station. Worth an ablation
if station-crushing matters; it does not for the equilibrium goal.

## 2026-08-25 — A2: pruning default at equal wall-clock, plain gets 1.4x and still loses (7/8)

ROADMAP A2. Question: is the pruning gain "prune smarter" or just "explore
more per iteration"? Plain (per-action pruning) trained for 470M
iterations, sized to the pruned 200M run's 4013s at plain's 200M rate.
Plain's rate collapsed as its map grew (82,151 iters/s average), so it
actually took **5721s = 1.43x** the pruned arm's wall-clock. 412.3M
infosets (pruned 200M: 402.1M — same tree, more visits).

| Seed | plain 200M | plain 470M | pruned 200M | pruned − plain470 |
|------|-----------|-----------|-------------|-------------------|
| 1 | +295.7 | +77.2 | +183.9 | +106.7 (plain better) |
| 2 | +333.0 | +191.1 | +169.8 | −21.3 |
| 3 | +594.1 | +533.5 | +103.4 | −430.1 |
| 4 | +641.3 | +294.2 | +239.9 | −54.3 |
| 5 | +335.4 | +325.4 | +211.8 | −113.6 |
| 6 | +625.3 | +280.3 | +260.0 | −20.3 |
| 7 | +246.7 | +273.2 | +133.3 | −139.9 |
| 8 | +937.1 | +390.1 | +197.5 | −192.6 |
| **mean** | **+501.1** | **+295.6** | **+187.5** | pruned better by **+108.2 ±132.6** (t=1.93, df=7, p≈0.10) |

Plain 200M → 470M (2.35x iterations): +205.4 ±165.2 (t=2.94) — more
visits help a lot. Pruned 200M vs plain 470M: pruned still ahead on 7/8
seeds with 30% LESS wall-clock, but the margin is not significant at 95%.

VERDICT: default stands. Honest decomposition of the 200M gap of +313:
roughly +200 is visit-equivalent compute that the exemptions buy per
iteration (river and terminal-leading actions always explored) and
roughly +100 is the smarter placement of those visits, the latter
suggestive (p≈0.10) rather than proven. Practical reading: at any fixed
wall-clock budget on this machine, pruned-N beats plain-2.35N. Not worth
more seeds; the decision does not change either way.

Eval vs caller (AIVAT+dup, 100k): plain 470M +2384.2 ±211.1 (plain 200M
+2108.9; pruned 200M +2742.4). Same ordering as BR.

## 2026-08-25 — A0: first multiway exploitability bound — the keyhole was hiding 1300 mbb/hand

`lbr --multiway` (commit `f8b955f` era, `src/lbr.rs`): LBR in one rotating
seat, the bot in every other seat, exact Bayes range per seat on the
public history, calls valued by joint showdown equity vs all live ranges,
bets by the product of responders' fold probabilities plus joint equity vs
the continuing ranges. Deterministic per seed. Both probes below on
`blueprint_pprune200.bin` (`198c7aa2…`), seed 1.

| Probe | hands | LBR wins (mbb/hand) | time |
|-------|-------|---------------------|------|
| HU-line `lbr` (other seats fold) | 20,000 | +42.5 ±322.4 | 60s |
| HU-line `lbr` | 5,000 | −10.2 ±666.4 | 17s |
| **multiway `lbr --multiway`** | 5,000 | **+1380.4 ±714.3** | 12s |

Reading: the blueprint that is statistically unexploitable on the
heads-up line (every 6-max number in this file until today) gives up
more than a big blind per hand once a best-responder plays it in real
multiway pots. The two bounds are for different games (six live seats
vs two, LBR rotating through all positions vs blinds only) and are not
comparable as numbers; what is comparable is the claim "a best-responder
who knows the policy wins X per hand at this table", and X is ~30x
larger multiway. Every 6-max verdict recorded before this section was
measured through the HU-line keyhole and must be re-read with that in
mind; the pruning fix's multiway effect is not yet known (measured next,
in the A1 run, one line per arm).

Suspects, to be separated by instrumentation before anything is fixed:
(a) unvisited multiway infosets — the bot's fallback on an unseen key is
check/call, and multiway lines are the rarest in self-play, so part of
this may be LBR farming a calling station that only exists off the
trained tree; (b) genuinely wrong multiway frequencies — Pluribus notes
that 6-max equilibria fold most hands early, and the blueprint's
multiway continuation ranges have never been probed. Next: log the
fallback rate per street during the multiway probe, and run 20k hands on
8 seeds so the bound has a CI worth quoting.

Operational note: the 13GB blueprint loads took 40 min to 3 h during this
run instead of 3.5 min, from machine-wide memory pressure (compressor
~48GB, a 28GB `node` process). Probe timings above are the runs
themselves.

## 2026-08-25 — A1: Pluribus-shaped bet menu — coarse postflop WINS, fine preflop LOSES

ROADMAP A1. `--menu` stored in the blueprint (commit `f8b955f`). Three
arms, `train --iters 30000000 --train-seed 0`, everything else default
(Pluribus pruning). Paired BR seeds 1-8.

| Arm | menu | train | infosets |
|-----|------|-------|----------|
| wide | 4 preflop opens; 6-7 postflop first-in sizes (25-200%), 4-size raise, 2-size re-raise | 177s, 169k it/s | 166.7M |
| pluribus | preflop as wide; postflop 50%/100%/all-in first-in, pot/all-in raise, call/fold/all-in beyond | 137s, 220k it/s | **67.3M** |
| pluribus-fine-pre | as pluribus + 7 opens / 5 three-bets / 3 four-bets preflop | 193s, 155k it/s | 171.3M |

| Seed | wide | pluribus | fine-pre | wide − pluribus |
|------|------|----------|----------|-----------------|
| 1 | +1312.5 | +827.1 | +1101.5 | +485.4 |
| 2 | +682.1 | +320.9 | +1521.6 | +361.2 |
| 3 | +1082.3 | +675.0 | +1325.4 | +407.3 |
| 4 | +719.2 | +533.4 | +898.3 | +185.8 |
| 5 | +972.9 | +737.0 | +1198.8 | +235.9 |
| 6 | +1044.6 | +903.5 | +1795.2 | +141.1 |
| 7 | +901.8 | +538.6 | +1307.3 | +363.2 |
| 8 | +1078.7 | +838.4 | (below) | +240.3 |
| **mean** | **+974.3** | **+671.7** | **+1306.9** (7) | **+302.5 ±99.7** (t=7.18, df=7, p<1e-4) |

Fine-pre vs wide, 7 seeds: +347.5 ±332.2 worse (t=2.56). The fine
preflop menu adds 104M infosets on its own (preflop sequences across six
seats grow combinatorially) and at 30M iterations those visits are not
there; it gives back more than the coarse postflop menu gained.

Training-run noise, measured for the first time: today's `wide` arm is
the same configuration as yesterday's `pluribus-prune` 30M arm (BR
+924.6). Paired by seed: +49.7 ±165.6 (t=0.71). So two trainings of one
config differ by ~±165 at 8 seeds; the paired design removes deal noise
but not this. Effects under ~150 mbb/hand at 30M need a replication
before they count. Nothing adopted so far is that small.

Cross-play between the arms is INVALID and is not reported as a
measurement: `run_crossplay` shares one action history that each policy
tokenises with its own menu, so a bet size the other menu lacks throws
that policy off-tree (check/call fallback) for the rest of the hand.
Observed: pluribus focal vs wide −117.0 ±125.3 / reverse −47.5 ±124.9;
fine-pre focal vs wide −908.2 ±153.7 / reverse −1334.5 ±184.1 — both
directions losing is the artefact's signature. `crossplay` now refuses
mismatched menus. Every earlier cross-play in this file compared
same-menu blueprints and stands.

VERDICT: coarse Pluribus postflop menu adopted pending the 200M
confirmation (queued): 2.5x smaller tree, 30% faster, 31% less
exploitable at equal iterations, 8/8 seeds. Fine preflop closed at this
budget; it is the right shape only once preflop visits are plentiful
(Pluribus's regime), and it should be revisited with the rented run
(A4), not before. Crossplay, eval and multiway lines below.

A1 tail (seed 1 unless noted): fine-pre seed 8 BR +1109.5 → 8-seed mean
+1281.9 (worse than wide on 8/8). Eval vs caller (AIVAT+dup, 100k):
wide +2346.1 ±237.8, **pluribus +2942.4 ±276.0**, fine-pre +2183.6 ±242.4.
Multiway LBR, 5k hands: wide +2506.2 ±935.5, pluribus +2362.9 ±973.5,
fine-pre +4555.4 ±1265.8. At 5k the coarse menu's HU-line gain does not
show multiway (CIs ±950); 20k runs follow in A0b. The 30M arms are all
far more exploitable multiway than the 200M standing blueprint (+1380).

### A0b: 20k multiway bounds with the unseen-infoset fallback split

`lbr` now reports how many blueprint lookups fell back to check/call on
an infoset the blueprint never stored (commit `999239a`). Seed 1, 20k
hands each.

| Blueprint | HU-line LBR | fallback (river) | multiway LBR | fallback (turn / river) |
|-----------|-------------|------------------|--------------|-------------------------|
| wide 30M | +774.9 ±467.9 | 0.2% (10.7%) | +2102.7 ±482.4 | 1.8% (22.5% / 60.9%) |
| pluribus-menu 30M | +355.7 ±443.9 | 0.1% (2.3%) | +2405.9 ±490.6 | 1.3% (12.1% / 31.9%) |
| fine-pre 30M | +589.8 ±544.7 | 0.1% (5.0%) | +3663.6 ±623.8 | 3.5% (36.3% / 69.0%) |
| **blueprint_pprune200 (200M)** | **+23.1 ±322.1** | 0.0% (0.2%) | **+934.1 ±332.6** | **0.2% (1.8% / 5.8%)** |

Readings:
1. At 200M the multiway leak is NOT coverage. The standing blueprint
   falls back on 0.2% of multiway decisions and still gives up +934
   mbb/hand to a best-responder in multiway pots, against +23 (zero
   within CI) on the heads-up line. That is trained, wrong multiway
   play: the fix is A3 (search on every postflop decision) and the
   abstraction/compute axis, not more visits to the same tree.
2. At 30M, coverage is a large part of it: 30M blueprints reach
   multiway rivers on untrained lines 32-69% of the time and play a
   calling station there. 30M-scale multiway numbers are therefore
   dominated by dilution and should not be used to rank multiway play.
3. The coarse Pluribus menu's HU-line gain (+775 → +356 here; −303 ±100
   over 8 seeds in A1) does not appear multiway at 30M (+2103 vs +2406,
   same seed, unpaired CIs ±490) despite halving the river fallback
   rate. Per the roadmap rule (adopt nothing that improves one probe and
   worsens the other) the coarse menu is NOT adopted on this evidence;
   paired 8-seed multiway probes on both arms (A1m) and the 200M
   confirmation (A1c) decide.
4. The multiway probe is cheap (20-30s per 20k hands once the blueprint
   is loaded) and should run on 8 seeds as standard from now on.

### A1m: paired 8-seed multiway probe, wide vs pluribus menu (30M arms) — NULL

| Seed | wide | pluribus | pluribus − wide |
|------|------|----------|-----------------|
| 1 | +2171.6 | +2437.2 | +265.6 |
| 2 | +2794.1 | +2465.8 | −328.3 |
| 3 | +2731.4 | +2920.8 | +189.4 |
| 4 | +2750.7 | +2581.2 | −169.5 |
| 5 | +2820.9 | +2495.8 | −325.1 |
| 6 | +2231.9 | +2386.1 | +154.2 |
| 7 | +2078.3 | +2290.1 | +211.8 |
| 8 | +2213.0 | +2013.4 | −199.6 |
| **mean** | **+2474.0** | **+2448.8** | **−25.2 ±212.5** (t=−0.28) |

The coarse menu is neutral multiway at 30M (4/8 seeds each way) while
−303 ±100 better on the heads-up line (A1). Improves one probe, worsens
neither: adoptable if A1c (200M) confirms. Multiway bounds at 30M are
dilution-dominated (A0b) and would not be expected to separate the arms.

### Probe reproducibility (found during A1m)

Identical invocations, same blueprint (wide 30M), seed 1:
`lbr --multiway` 5k: +2610.5 / +2537.6; 1k parallel +824.9 / +603.0;
1k with RAYON_NUM_THREADS=1: +554.9 / +554.9 (identical).
`lbr` HU-line 5k: +1332.3 / +1276.0. `br` 2k: +928.7 / +920.7.
`eval --baseline caller` 20k: +2025.0 / +2025.0 (identical).

So `br` and both `lbr` modes are reproducible only single-threaded; in
parallel they vary by ~1-3% of the bound. The bucket cache values are
deterministic by construction (key-seeded MC, collision-free packed keys)
and `eval` shares that cache and does not vary, so the cause is on the
LBR/BR path (range tracking / exact-solver code under rayon), not yet
isolated. Consequence: the "bit-for-bit" claims in this file for `br`
were true as of July and are not now; every paired comparison in this
file remains valid because the variation (±10-60) is an order of
magnitude inside the CIs (±300-500). Fix or isolate before any claim
that depends on effects under ~100 mbb/hand.

### A1c: coarse Pluribus menu at 200M — CONFIRMED, new 6-max standing (+116)

`train --iters 200000000 --train-seed 0 --menu pluribus`: 2090.8s (35 min,
95,657 iters/s), **124.1M infosets** (wide/pruned 200M: 402.1M, 67 min).

| Seed | wide 200M (prev. default) | pluribus-menu 200M | improvement |
|------|---------------------------|--------------------|-------------|
| 1 | +183.9 | +38.9 | +145.0 |
| 2 | +169.8 | +40.0 | +129.8 |
| 3 | +103.4 | +132.3 | −28.9 |
| 4 | +239.9 | +174.1 | +65.8 |
| 5 | +211.8 | +80.1 | +131.7 |
| 6 | +260.0 | +247.2 | +12.8 |
| 7 | +133.3 | +98.9 | +34.4 |
| 8 | +197.5 | +119.8 | +77.7 |
| **mean** | **+187.5** | **+116.4** | **+71.0 ±52.3** (t=3.21, df=7, p≈0.015), 7/8 |

Multiway `lbr --multiway` 20k seed 1: **+928.8 ±318.8** (wide 200M:
+934.1 ±332.6) — neutral, fallbacks 0.1%. Eval vs caller (AIVAT+dup,
100k): +2913.5 ±258.5 (wide 200M: +2742.4 ±227.8).

VERDICT: adopted as the default (`--menu wide` restores the 2026-07
menu). The 6-max standing is now `blueprint_plu200.bin`: heads-up-line
BR **+116**, multiway LBR **+929**, at a third of the tree and half the
training time of the previous standing. Since 24 Aug the heads-up-line
bound has gone +472 → +188 → +116 from two Pluribus-recipe corrections;
the multiway bound (+929) has not moved and is the target from here
(ROADMAP A3).

| File | SHA-256 | Size (bytes) | Produced by |
|------|---------|--------------|-------------|
| blueprint_plu200.bin | `086b84958641cef6cb56396efa2c2ff8625d5f5a23f0010ba1fd9527ed2ba3ff` | 4,090,176,388 | `train --iters 200000000 --train-seed 0 --menu pluribus` (now the default) |

Test note: `cfr::tests::push_fold_convergence` (32o fold > 0.6 at 80k
iterations) failed once in a full-suite run under machine load and passes
3/3 in isolation; a pre-existing threshold sensitivity, not the menu.

### A3 gate: online search in multiway pots (26-27 Aug 2026)

`lbr --multiway --hands 10000` seeds 1-8, standing blueprint
`blueprint_plu200.bin`, three arms per seed: raw blueprint; the bot
searching every postflop decision rooted at the current decision
(`--search --search-iters 20000 --decision-root`, the pre-A3 resolver);
the same rooted at the start of the betting round (`--search
--search-iters 20000`, commit 66d272b: spine, fixed hero actions,
round-start ranges, eps-0.5 importance-weighted spine sampling). The
bot's card abstraction inside the resolve is the blueprint's 12 buckets.
Iteration-capped, so the numbers are independent of machine load;
~36 min per search arm per seed. The LBR's range model of the bot stays
the blueprint, so the search arms are probe results, not strict bounds.

| Seed | blueprint | search, decision-rooted | search, round-rooted |
|------|-----------|-------------------------|----------------------|
| 1 | +926.2 | +776.4 | +803.0 |
| 2 | +846.6 | +762.0 | +742.3 |
| 3 | +1063.8 | +1035.8 | +1090.5 |
| 4 | +1315.7 | +1176.5 | +1131.7 |
| 5 | +1546.5 | +1226.8 | +1218.2 |
| 6 | +785.3 | +625.2 | +571.8 |
| 7 | +1047.6 | +672.0 | +581.2 |
| 8 | +1158.8 | +927.7 | +954.7 |
| **mean** | **+1086.3** | **+900.3** | **+886.7** |

Paired by seed (95% CI, df=7):
- decision-rooted search − blueprint: **−186.0 ±97.6** (t=4.5), 8/8
- round-rooted search − blueprint: **−199.6 ±124.2** (t=3.8), 7/8
- round-rooted − decision-rooted: −13.6 ±40.6 (t=0.8), 5/8

VERDICT: searching multiway decisions cuts the multiway leak by about
17% and does so on every seed; that is the first movement of the
multiway bound. Round-start rooting is not distinguishable from
decision rooting at this budget and abstraction (it stays the default
as Pluribus's design and because it is not worse, but it is not the
gain). Note the per-seed blueprint bound varies +785..+1547 with
10k-hand CIs of ±430-470; the 20k seed-1 figure (+929) remains the
quoted standing number. Roughly +890 of leak remains with search on.
Next lever, queued: finer card abstraction inside the resolve
(`--search-buckets 200`, commit b47d280), which the unit test showed
matters (a 6-bucket "nuts" class folds 18% against the re-solved
shove range).

### A3 lever: finer card abstraction inside the resolve — NEGATIVE (27 Aug 2026)

Same 8 seeds and budget as the gate above, round-rooted search with
`--search-buckets 200` (equity quantiles, exact per-board river tables;
blueprint lookups keep 12), commit b47d280.

| Seed | round-rooted, 12 buckets | round-rooted, 200 buckets |
|------|--------------------------|---------------------------|
| 1 | +803.0 | +847.8 |
| 2 | +742.3 | +795.8 |
| 3 | +1090.5 | +1138.3 |
| 4 | +1131.7 | +1311.4 |
| 5 | +1218.2 | +1265.7 |
| 6 | +571.8 | +746.4 |
| 7 | +581.2 | +712.9 |
| 8 | +954.7 | +1005.9 |
| **mean** | **+886.7** | **+978.0** |

Paired: 200 − 12 buckets **+91.3 ±50.4** (t=4.3), worse on 8/8; 200
buckets − blueprint −108.3 ±116.7 (7/8), so it keeps roughly half of
the search gain. VERDICT: closed at this budget. 20k traversals spread
over ~17x more infosets leave each one under-trained; the unit-test
effect (coarse "nuts" class hedging) is real but smaller than the
sample-starvation cost. Revisit only with a much larger per-decision
budget or a different way of sharing statistics across buckets. Search
timing was 30 min per 10k hands at both bucket counts once the machine
was uncontended.

### A3 heads-up check: `br --search` (27 Aug 2026)

Paired `br --hands 20000` seeds 1-8, blueprint vs the bot searching
(`--search --search-iters 20000`; heads-up turns/rivers use the exact
solvers, flop plays the blueprint). The probe's best response is computed
against the *blueprint's* strategy, so against the searching bot it is
no longer a best response: the search column is a probe result, not an
exploitability bound.

| Seed | blueprint | search |
|------|-----------|--------|
| 1 | +40.1 | −359.1 |
| 2 | +78.6 | −321.9 |
| 3 | +154.8 | −203.2 |
| 4 | +192.8 | −198.6 |
| 5 | +106.8 | −311.0 |
| 6 | +256.3 | −108.2 |
| 7 | +91.0 | −276.9 |
| 8 | +125.0 | −213.7 |
| **mean** | **+130.7** | **−249.1** |

Paired: search − blueprint **−379.8 ±22.2** (t=40), 8/8. The
blueprint-modelled exploiter goes from winning +131 to losing −249
against the searching bot. VERDICT: heads-up play does not worsen with
search on (the ROADMAP A3 gate's "neither probe worsens" condition
holds); the tightness is the paired-deal design, not evidence about
the true exploitability of the searching bot, which needs an exploiter
that models the searched strategy (open item). ~18 min per 20k-hand
seed.

### A3 search-gain and verdict (27 Aug 2026)

`eval --search-gain --search-iters 20000` (hero searching vs hero on the
blueprint, everyone else blueprint, paired deals): 40k deals **+58.7
±99.7** (17 min), 200k deals **+125.1 ±40.2** (83 min). CI clear of
zero.

A3 GATE: multiway LBR −186 ±98 (8/8) and −200 ±124 (7/8); heads-up `br`
not worse (8/8); search-gain +125 ±40. All three conditions hold.
ADOPTED: online search is the `play` default (`--no-search` restores
the raw blueprint). The probes keep search opt-in (`--search`) so the
blueprint's own bounds stay the quoted standing numbers. Closed: round
vs decision rooting (neutral), 200 in-subgame buckets (worse).

### A3b: is the multiway leak compute or abstraction? COMPUTE (27 Aug 2026)

One training run (`--train-seed 0`, defaults) checkpointed at 25/50/100/
200M via `--checkpoint`/`--resume` (`--iters` is additional on resume),
probed at each stage: `lbr --multiway --hands 20000` seeds 1-2 and
`br --hands 20000` seed 1.

| Iterations | infosets | multiway LBR s1 | s2 | mean | heads-up BR s1 |
|-----------:|---------:|----------------:|---:|-----:|---------------:|
| 25M | 60.5M | +2526.8 | +2433.1 | +2480 | +723.5 |
| 50M | 79.1M | +1780.9 | +1900.4 | +1841 | +225.9 |
| 100M | 99.8M | +1464.0 | +1536.3 | +1500 | +34.5 |
| 200M | 121.1M | +1003.8 | +854.8 | +929 | +48.3 |

The heads-up line converges by 100M (+35, +48; standing +39) and stops.
The multiway bound is still falling at every doubling (−640, −340, −570
per doubling) with no sign of a floor at 200M. Training time 37 min for
the 200M total. VERDICT: the remaining multiway leak is under-training
of multiway lines (most traversals fold to heads-up before a multiway
pot), not the card abstraction. Next: multiway-focused training
(steer traversals toward multiway lines, importance-weighted so the
fixed point is unchanged) on this machine; fallback, a longer run on
rented hardware (A4).

### Multiway-focused training — NEGATIVE (27 Aug 2026)

`train --iters 30000000 --train-seed 0 --multiway-focus 0.3` (commit
5f9ac2c: opponents' preflop folds suppressed at sampling time, every
update importance-weighted by sigma/q, fixed point unchanged; unit test
`multiway_focus_keeps_the_fixed_point`) vs the plain 30M arm, paired
`lbr --multiway` and `br` at 20k hands, seeds 1-8.

| Probe | plain mean | focus 0.3 mean | paired focus − plain |
|-------|-----------:|---------------:|---------------------:|
| multiway LBR | +2638 | +2528 | **−111 ±156** (t=1.7), 6/8 |
| heads-up BR | +698 | +1232 | **+534 ±213** (t=5.9), worse 8/8 |

Per seed (plain/focus), multiway: +2650/+2513, +2762/+2666, +2772/+2573,
+2415/+2597, +2549/+2566, +2767/+2298, +2583/+2450, +2607/+2557.
Heads-up: +469/+1380, +403/+1056, +609/+804, +414/+1240, +802/+1272,
+814/+1268, +1218/+1459, +859/+1380.

VERDICT: closed. At a fixed traversal budget, steering samples from
fold-to-the-blinds lines toward multiway lines buys a small, unproven
multiway gain (−4%) and costs the heads-up line 76%: four suppressed
folds sample the heads-up lines ~4x less often. Two earlier variants
were worse still: a uniform-over-non-fold mixture wasted ~30% of samples
on zero-probability actions, and an implementation that weighted
descendants but did not scale the value returned to ancestors was
biased (fixed in 55625c4; the same flaw was in the round-rooted search's
spine sampling, whose gate arm is being re-measured). Implication for
the multiway leak: it is not a matter of *redistributing* a fixed
budget; the budget itself has to grow (A4, rented compute), or the
per-traversal cost of multiway lines has to fall.

### A3 addendum: round-rooted search re-measured after the off-policy fix (28 Aug 2026)

Same 8 seeds, 10k hands, 20k iters as the A3 gate, round-rooted search
with the corrected spine sampling (55625c4: returned values scaled by
sigma/q). Per seed: +893.8, +709.0, +1247.9, +1103.7, +1244.3, +560.7,
+652.4, +948.7; mean **+920.1**.

Paired: corrected round − blueprint **−166.2 ±148.1** (t=2.7), 7/8;
corrected round − decision-rooted +19.9 ±83.0 (4/8); corrected − biased
round +33.5 ±56.4 (4/8). VERDICT unchanged: search beats the blueprint;
round-start rooting is indistinguishable from decision rooting, before
and after the fix. The A3 record stands with search as the gain.

### Trainer speed: bucket table (28 Aug 2026)

macOS `sample` profile of `train` (10 s, 16 threads): 68% of samples in
`eval::eval_hole_board` under `Abstraction::bucket -> equity_quantiles`
(flop/turn bucket-cache misses); `Trainer::traverse` 3%, DashMap < 1%.
Removing per-visit heap allocations in `traverse`: 92,630 -> 92,028
iters/s (no change). `bucket-table` (15,246,842 canonical flop/turn
pairs, 173 s to build, 152 MB) preloaded via `train --bucket-table`:
**248,660 iters/s vs 92,630** on 10M iterations, same machine load
(2.7x). Bucket values identical to the lazy path by construction.

### A4 local: 2B iterations on this machine (28 Aug 2026)

One run (`--train-seed 0 --bucket-table buckets.bin`, defaults),
checkpointed and probed at 500M/1B/2B; `lbr --multiway --hands 20000`
seeds 1-2, `br --hands 20000` seed 1. Continues the A3b curve.

| Iterations | infosets | train time | multiway s1 | s2 | mean | heads-up s1 |
|-----------:|---------:|-----------:|------------:|---:|-----:|------------:|
| 200M (A3b) | 121M | 37 min | +1003.8 | +854.8 | +929 | +48.3 |
| 500M | 150M | 64 min | +677.7 | +492.0 | +585 | +239.3 |
| 1B | 177M | +86 min | +490.7 | +788.1 | +639 | +372.5 |
| 2B | 209M | +170 min | +634.5 | +673.2 | +654 | +392.4 |

Rate fell from 248k iters/s (10M) to ~100k (map at 150-210M infosets,
36-45 GB RSS): memory-bound. VERDICT: the multiway bound drops 37% from
200M to 500M then plateaus at ~+600-650; the heads-up line drifts WORSE
from +48 (200M) to +392 (2B), outside its ±260 CI. Under the current
recipe more iterations stop paying at ~500M and start costing heads-up.
Suspected cause: Linear CFR weights applied for the whole run (regrets
grow ~t^2, the fixed prune threshold −3e8 becomes over-aggressive late);
Pluribus stopped linear weighting after 400 minutes. Next arm: a cap
on the linear weight, paired at 500M. Blueprints kept in the scratchpad
(a4_bp_{500M,1B,2B}.bin; 500M = 5.0 GB) pending that result.

### Linear-weight cap at 500M — NEGATIVE (28 Aug 2026)

`train --iters 500000000 --linear-cap 100000000 --bucket-table buckets.bin`
vs the uncapped 500M blueprint above, paired `lbr --multiway` and `br`
at 20k hands, seeds 1-8.

| Probe | uncapped mean | cap 100M mean | paired cap − uncapped |
|-------|--------------:|--------------:|----------------------:|
| multiway LBR | +649.0 | +730.5 | **+81.4 ±96.4** (t=2.0), worse 6/8 |
| heads-up BR | +271.0 | +336.2 | +65.2 ±128.1 (t=1.2), worse 5/8 |

Per seed uncapped multiway: +702 +595 +592 +636 +680 +613 +686 +688;
capped: +635 +674 +777 +688 +910 +820 +701 +639. Uncapped heads-up:
+230 +181 +431 +270 +214 +455 +123 +263; capped: +304 +290 +326 +214
+72 +740 +273 +471.

VERDICT: closed; the linear weight is not the cause of the late
heads-up drift. Also recorded: the uncapped 500M blueprint on all 8
seeds vs the standing 200M `blueprint_plu200.bin`: multiway **+649 vs
+1086** (better on 8/8, −40%), heads-up **+271 vs +116** (worse on 8/8:
+230/+39, +181/+40, +431/+132, +270/+174, +214/+80, +455/+247,
+123/+99, +263/+120). It fails the "worsens neither" rule, so it is
NOT adopted; it is kept as `a4_bp_500000000.bin` (scratchpad). The
pattern (heads-up converged by 100M in the abstract game, real-game
exploitability rising afterwards while multiway still improves) is the
abstraction-pathology signature (Waugh et al. 2009): the 12-bucket card
abstraction is the suspect. Probe-variation note: re-probing the same
500M blueprint gave +678/+702 (seed 1) and +492/+595 (seed 2), so
parallel multiway probes vary more than the 1-3% measured on the 200M
blueprint; 8-seed paired means remain the unit of evidence.

### 50 card buckets at 500M (28 Aug 2026)

`bucket-table --buckets 50` (194 s) then `train --iters 500000000
--buckets 50 --bucket-table buckets50.bin`: 71 min, **243M infosets**
(12 buckets: 150M). Paired probes vs the 12-bucket 500M blueprint,
seeds 1-8, 20k hands.

| Probe | 12 buckets | 50 buckets | paired 50 − 12 |
|-------|-----------:|-----------:|---------------:|
| multiway LBR | +649.0 | +796.9 | **+147.8 ±120.5** (t=2.9), worse 6/8 |
| heads-up BR | +271.0 | +175.8 | **−95.1 ±99.5** (t=2.3), better 6/8 |

Per seed 50-bucket multiway: +935 +846 +811 +784 +972 +804 +601 +622;
heads-up: +222 +35 +218 +231 +97 +159 +157 +287.

VERDICT: the finer abstraction reverses the heads-up drift (supporting
the abstraction-pathology reading of the 2B curve) but at equal
iterations its 1.6x larger tree is under-trained multiway. Not adopted
at 500M. Next: 50 buckets at 2B with a 1B checkpoint, where the
12-bucket run had already plateaued.
