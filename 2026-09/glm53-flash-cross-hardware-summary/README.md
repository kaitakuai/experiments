# GLM-5.3-Flash — cross-hardware summary: six arms, one matrix, and what the chain should do with it

**Date:** 2026-09-01
**Model:** [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) and its two
fraud variants — `LibertAIDAI/GLM-5.3-Flash-NVFP4` (quantisation) and
`patrickbdevaney/GLM-5.3-Flash-REAP50-FP8` (expert pruning).
**Hardware:** every arm measured so far — 2×B300 TP=2, 4×H200 TP=4, 4×B200 TP=4.
**Images:** `vllm-poc:glm53-poc-v4-ed8873884` (August, FlashInfer 0.6.17) and
`mlnode-*-glm-5-3-flash:0.2.14-vllm0.28-glm53-test-k3` (September, FlashInfer 0.6.18).
**PoC:** seq_len 1024, k_dim 12, seeds s1/s2/s3 from the fixed set.

## Summary

**Nothing was measured for this folder.** `scripts/matrix.py` reads the committed nonce sets
from the six arm folders, so the matrix cannot drift from the reports it summarises.

| comparison | past the 0.40 gate |
|---|---:|
| honest vs itself, same box | **0 %** (B200) · 0.1 % (H200) |
| **honest vs honest, different GPU generation** | **16–17 %** |
| fraud NVFP4 vs honest, same box | 43 % (B200) · 97 % (B300) |
| fraud REAP50 vs honest, same box | **90 %** |

The middle row is the constraint. Both sides are honest, and at the chain's `p_mis = 0.001` a
healthy mixed fleet is called fraudulent. **The distance threshold is fine; the mismatch
tolerance is not.**

## Environment

Each arm's environment is documented in its own folder and is not restated here. What matters
for reading the matrix is which dimensions differ per pair:

| arm | hardware | image | TP | collection batch |
|---|---|---|---:|---:|
| b300 honest / nvfp4 (Aug) | 2×B300 | 0.6.17 | 2 | 32 |
| h200 honest / reap50 | 4×H200 | `k3` / 0.6.18 | 4 | 16 |
| b200 honest / nvfp4 | 4×B200 | `k3` / 0.6.18 | 4 | 16 |

## Config

No engine is launched here. The analysis config is fixed in `scripts/matrix.py`:
threshold `0.40`, seeds `s1/s2/s3`, batch-position split at 16, and L2 computed the way the
chain does it (`vllm/poc/data.py`: fp16 LE → fp32, fp64 norm, strict `>`).

### What changed vs the default

Nothing is re-configured. The one deliberate choice is that the matrix reports the **median of
per-seed medians** rather than pooling all three seeds, so a single anomalous seed cannot
dominate a cell.

## Validation

### L2

`varies` lists what actually differs between the two arms — several pairs change more than one
thing at once and must not be read as clean comparisons.

| A | B | median L2 | past 0.40 | varies |
|---|---|---:|---:|---|
| h200 honest | b200 honest | 0.2606 | 16.5 % | **hardware only** |
| b300 honest (aug) | b200 honest | 0.2666 | 16.5 % | hardware, image, TP |
| b300 honest (aug) | h200 honest | 0.2658 | 16.7 % | hardware, image, TP |
| b200 honest | b200 nvfp4 | 0.3791 | 43.2 % | **checkpoint only** |
| h200 honest | b200 nvfp4 | 0.3814 | 45.0 % | hardware, checkpoint |
| b300 honest (aug) | b200 nvfp4 | 0.3846 | 44.8 % | hardware, image, TP, checkpoint |
| h200 reap50 | b200 honest | 0.5953 | 90.6 % | hardware, checkpoint |
| h200 honest | h200 reap50 | 0.6047 | 90.6 % | **checkpoint only** |
| b300 honest (aug) | h200 reap50 | 0.6064 | 90.3 % | hardware, image, TP, checkpoint |
| h200 reap50 | b200 nvfp4 | 0.6264 | 93.3 % | hardware, checkpoint |
| b300 nvfp4 (aug) | h200 reap50 | 0.6832 | 96.2 % | hardware, image, TP, checkpoint |
| b300 honest (aug) | b300 nvfp4 (aug) | 0.7132 | 97.3 % | **checkpoint only** |
| b300 nvfp4 (aug) | h200 honest | 0.7164 | 97.2 % | hardware, image, TP, checkpoint |
| b300 nvfp4 (aug) | b200 honest | 0.7188 | 97.1 % | hardware, image, TP, checkpoint |
| **b300 nvfp4 (aug)** | **b200 nvfp4** | **0.7368** | **98.0 %** | hardware, image, TP |

### Cross-hardware L2

The one pair in the matrix that varies **only** hardware — 4×H200 ↔ 4×B200, both on `k3`, both
TP=4, same seeds. This is the honest cross-generation floor, and the number the chain has to be
calibrated against.

| seed | mean | median | p25 | p75 | p95 | max | past 0.40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| s1 | 0.3338 | 0.2604 | 0.2003 | 0.3490 | 1.0850 | 1.8511 | 164 (16.4 %) |
| s2 | 0.3359 | 0.2606 | 0.2024 | 0.3480 | 1.1310 | 1.7987 | 170 (17.0 %) |
| s3 | 0.3297 | 0.2626 | 0.2012 | 0.3484 | 1.0883 | 1.6494 | 162 (16.2 %) |
| **all three, 3000 nonces** | **0.3331** | **0.2609** | | | | | **496 (16.5 %)** |

What the matrix settles, given that number:

**The cross-generation gap is architecture, not build.** The confounded pairings against the
August 2×B300 arm — which also change image and TP — land on 16.5–16.7 %, indistinguishable
from the 16.5 % above. Image and TP contribute nothing measurable.

**A fraud fingerprint is not portable.** Two runs of the *same* NVFP4 checkpoint on different
platforms sit at 0.741 / 98.0 % — further apart than fraud is from honest on either box (0.379
and 0.713). Detection must be framed as *far from honest*, never as *close to a known
signature*.

**Structural fraud is the stable case.** REAP50 reads 0.60 / 90.6 % against its own honest
baseline, 0.60 / 90.6 % against honest B200 and 0.61 / 90.3 % against honest B300 — changing
the validator's hardware moves the verdict by 0.3 points.

### The batch-boundary artifact

Nonces at `index % batch == 0` are 100 % past the gate in every cross-hardware pair in this
matrix — on all three architectures, both images, at batch 16 and at batch 32. They contribute
about 6 of the 17 points in the honest cross-generation row. Cause unknown; a candidate is
documented in [`../glm53-flash-fp8-4xh200/`](../glm53-flash-fp8-4xh200/).

### Throughput

Best sweep result per arm, 120 s windows, taken from the arm folders:

| arm | batch | nonces/min | Δ vs its honest baseline |
|---|---:|---:|---:|
| b300 honest (TP=2) | 32 | 2030 | — |
| b300 nvfp4 (TP=2) | 32 | 2767 | **+36 %** |
| h200 honest (TP=4) | 8/16 | 1439 | — |
| h200 reap50 (TP=4) | 16 | ~1425 | −1 % |
| b200 honest (TP=4) | 32 | 2727 | — |
| b200 nvfp4 (TP=4) | 16 | 3431 | **+23 %** (per-seed, equal budget) |

Batch 32 is the highest usable value everywhere; batch 48 collapses on both B300 (with ample
budget) and B200 (Triton illegal access).

### Serving

Not aggregated here — serving was measured only on the September arms, with a single pass per
concurrency level, and the numbers live in those folders. Pooling them would imply a
comparability across images and topologies that the measurements do not support.

### Integrity checks

`scripts/matrix.py` asserts that every arm's `block_hash` matches per seed before comparing, so
a mismatched-seed comparison cannot silently produce the ~1.41 asymptote. It keys sets by
(folder, arm) rather than filename, avoiding the `compare_nonces.py` basename trap.

## What this does not settle

- **8-GPU topologies.** Only relevant to 80 GB cards. On H200 (141 GB) and B200 (183 GB) the
  weights fit at TP=4 with room for KV, so a 4-card node is a full production topology, not a
  reduced one — the image's baked `TP=8` targets H100-class boxes, where TP=4 leaves 76 GB of
  weights on an 80 GB card and no room for cache. **8×H100 is therefore the one unmeasured
  topology**, and no H100 arm of any width exists in this matrix.
- **The B300 ↔ B200 NVFP4 disagreement is not decomposed** into architecture, image and TP.
- **Only two fraud classes**: expert pruning at 50 % and one NVFP4 producer.
- **No honest floor for the B300 arm** — the August run collected one pass per seed.

## Files

| path | what |
|---|---|
| [`artifacts/matrix.json`](artifacts/matrix.json) | the matrix above, machine-readable, with `varies` per pair |
| [`scripts/matrix.py`](scripts/matrix.py) | builds it from the other folders' committed sets |
| [`scripts/poc_seeds.json`](scripts/poc_seeds.json) | the fixed seed set |

No nonce sets are duplicated into this folder by design.

## Reproduce

```bash
python3 scripts/matrix.py > artifacts/matrix.json
```

Success criteria: the hardware-only pair (h200 ↔ b200) at ≈ 0.26 / 16.5 %; the NVFP4 ↔ NVFP4
cross-platform pair at ≈ 0.737 / 98 %; the script exiting without a seed-mismatch error.

**Recommended calibration, which is the point of this folder:**

1. Keep the distance threshold at **0.40** — it separates every fraud arm from every honest arm.
2. Raise `p_mis` to **≈ 0.20** wherever prover and validator may sit on different GPU
   generations. At 0.001 the chain rejects honest nodes.
3. Do not calibrate against a published fraud fingerprint; calibrate the honest floor per
   architecture pair and flag distance from it.
4. Either fix or exclude the `index % batch == 0` nonces — they carry no information about
   honesty and consume a third of the cross-generation budget.

## Gotchas

- **Never read a multi-`varies` row as a clean comparison.** Three of the four rows involving
  the August arm change hardware, image and TP simultaneously.
- **Give nonce files distinct basenames** when comparing by hand with `compare_nonces.py`; it
  labels pairs by basename and reports a false `L2 = 0.0000` for two identically named files.
- **Arms differ in collection batch** (32 for B300, 16 for the rest), so the batch-boundary
  artifact lands on different indices per arm.

## Related

| arm | folder |
|---|---|
| 2×B300 honest / NVFP4 (Aug data) | [`../glm53-flash-fp8-2xb300/`](../glm53-flash-fp8-2xb300/) · [`../glm53-flash-nvfp4-libertai-2xb300/`](../glm53-flash-nvfp4-libertai-2xb300/) |
| 4×H200 honest / REAP50 | [`../glm53-flash-fp8-4xh200/`](../glm53-flash-fp8-4xh200/) · [`../glm53-flash-reap50-patrickbdevaney-4xh200/`](../glm53-flash-reap50-patrickbdevaney-4xh200/) |
| 4×B200 honest / NVFP4 | [`../glm53-flash-fp8-4xb200/`](../glm53-flash-fp8-4xb200/) · [`../glm53-flash-nvfp4-libertai-4xb200/`](../glm53-flash-nvfp4-libertai-4xb200/) |

## Reproducibility checklist

- [x] A reader with only this repo can regenerate the matrix with one command.
- [x] Every source arm is linked; no data is duplicated into this folder.
- [x] Confounds are explicit per pair (`varies`), not left for the reader to infer.
- [x] Seed identity is asserted in code before any comparison.
- [x] No links to `.claude/...` and no paths into sibling repos.
- [x] The recommendation states what to change and by how much.
- [x] Sections absent for a reason say so instead of being omitted.
