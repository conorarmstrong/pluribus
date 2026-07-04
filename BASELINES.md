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
