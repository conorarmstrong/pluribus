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
