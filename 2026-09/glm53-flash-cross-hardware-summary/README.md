# GLM-5.3-Flash — cross-hardware summary: six arms, one matrix, and what the chain should do with it

**Date:** 2026-09-01
**Scope:** every GLM-5.3-Flash PoC arm measured so far — 2×B300 (August, previous image),
4×H200 and 4×B200 (September, `k3` image), honest and fraud on each.
**Nothing was measured for this folder.** `scripts/matrix.py` reads the committed nonce sets
from the other experiment folders, so the matrix cannot drift from the reports it summarises.

## The three levels

| comparison | past the 0.40 gate |
|---|---:|
| honest vs itself, same box | **0 %** (B200) · 0.1 % (H200) |
| **honest vs honest, different GPU generation** | **16–17 %** |
| fraud NVFP4 vs honest, same box | 43 % (B200) · 97 % (B300) |
| fraud REAP50 vs honest, same box | **90 %** |

The middle row is the constraint. Both sides are honest, and at the chain's `p_mis = 0.001`
a healthy mixed fleet is called fraudulent. **The distance threshold is fine; the mismatch
tolerance is not.** For cross-generation validation it has to be ≈ 0.20.

## Full pairwise matrix

Median of the per-seed medians over s1/s2/s3, and the mean mismatch rate. `varies` lists what
actually differs between the two arms — several pairs change more than one thing at once and
must not be read as clean comparisons.

| A | B | median L2 | past 0.40 | varies |
|---|---|---:|---:|---|
| h200 honest | b200 honest | 0.2606 | 16.5 % | **hardware only** |
| b300 honest (aug) | b200 honest | 0.2666 | 16.5 % | hardware, image, TP |
| b300 honest (aug) | h200 honest | 0.2658 | 16.7 % | hardware, image, TP |
| b200 honest | b200 nvfp4 | 0.3791 | 43.2 % | **checkpoint only** |
| h200 honest | b200 nvfp4 | 0.3814 | 45.0 % | hardware, checkpoint |
| b300 honest (aug) | b200 nvfp4 | 0.3846 | 44.8 % | hardware, image, TP, checkpoint |
| h200 honest | h200 reap50 | 0.6047 | 90.6 % | **checkpoint only** |
| b300 honest (aug) | h200 reap50 | 0.6064 | 90.3 % | hardware, image, TP, checkpoint |
| h200 reap50 | b200 honest | 0.5953 | 90.6 % | hardware, checkpoint |
| h200 reap50 | b200 nvfp4 | 0.6264 | 93.3 % | hardware, checkpoint |
| b300 nvfp4 (aug) | h200 reap50 | 0.6832 | 96.2 % | hardware, image, TP, checkpoint |
| b300 honest (aug) | b300 nvfp4 (aug) | 0.7132 | 97.3 % | **checkpoint only** |
| b300 nvfp4 (aug) | h200 honest | 0.7164 | 97.2 % | hardware, image, TP, checkpoint |
| b300 nvfp4 (aug) | b200 honest | 0.7188 | 97.1 % | hardware, image, TP, checkpoint |
| **b300 nvfp4 (aug) | b200 nvfp4** | **0.7368** | **98.0 %** | hardware, image, TP |

Regenerate with `python3 scripts/matrix.py`.

## What the matrix settles

**The cross-generation gap is architecture, not build.** The one pair that varies *only*
hardware (h200 ↔ b200, both on `k3`, both TP=4) gives 16.5 % — indistinguishable from the
confounded pairings that also change image and TP (16.5–16.7 %). Image and TP contribute
nothing measurable.

**A fraud fingerprint is not portable.** The last row is two runs of the *same* NVFP4
checkpoint on different platforms: **0.737, 98 % past the gate** — further apart than fraud is
from honest on either box (0.379 and 0.713). Detection must be framed as *far from honest*,
never as *close to a known signature*.

**Structural fraud is the stable case.** Expert pruning (REAP50) reads 0.60 / 90 % against
honest on its own box and 0.60 / 90 % against a *different-generation* honest arm — the signal
barely moves with hardware, because the model computes a different function rather than the
same one less precisely.

**The batch-boundary defect is universal.** Nonces at `index % batch == 0` are 100 % past the
gate in every cross-hardware pair in this matrix, on all three architectures and both images,
at batch 16 and at batch 32. They are ~6 of the 17 points in the honest cross-generation row.
Cause unknown; a candidate is documented in
[`../glm53-flash-fp8-4xh200/`](../glm53-flash-fp8-4xh200/).

## Recommended calibration

1. Keep the distance threshold at **0.40**. It separates every fraud arm measured from every
   honest arm measured.
2. Raise `p_mis` to **≈ 0.20** wherever prover and validator may sit on different GPU
   generations. At 0.001 the chain rejects honest nodes.
3. Do not calibrate against a published fraud fingerprint. Calibrate the honest floor per
   architecture pair, and flag distance from it.
4. Either fix or exclude the `index % batch == 0` nonces. They carry no information about
   honesty and consume a third of the cross-generation budget.

## Files

| path | what |
|---|---|
| [`artifacts/matrix.json`](artifacts/matrix.json) | the matrix above, machine-readable, with `varies` per pair |
| [`scripts/matrix.py`](scripts/matrix.py) | builds it from the other folders' committed sets |
| [`scripts/poc_seeds.json`](scripts/poc_seeds.json) | the fixed seed set |

`matrix.py` asserts that every arm's `block_hash` matches per seed before comparing, so a
mismatched-seed comparison cannot silently produce the ~1.41 asymptote.

## Sources

| arm | folder |
|---|---|
| 2×B300 honest / NVFP4 (Aug data) | [`../glm53-flash-fp8-2xb300/`](../glm53-flash-fp8-2xb300/) · [`../glm53-flash-nvfp4-libertai-2xb300/`](../glm53-flash-nvfp4-libertai-2xb300/) |
| 4×H200 honest / REAP50 | [`../glm53-flash-fp8-4xh200/`](../glm53-flash-fp8-4xh200/) · [`../glm53-flash-reap50-patrickbdevaney-4xh200/`](../glm53-flash-reap50-patrickbdevaney-4xh200/) |
| 4×B200 honest / NVFP4 | [`../glm53-flash-fp8-4xb200/`](../glm53-flash-fp8-4xb200/) · [`../glm53-flash-nvfp4-libertai-4xb200/`](../glm53-flash-nvfp4-libertai-4xb200/) |

## What is not covered

- **8-GPU topologies.** The image ships `TP=8`; every arm here is TP=2 or TP=4.
- **The B300↔B200 NVFP4 disagreement is not decomposed** into architecture, image and TP.
- **Only two fraud classes**: expert pruning at 50 % and one NVFP4 producer.

## Reproducibility checklist

- [x] A reader with only this repo can regenerate the matrix with one command.
- [x] Every source arm is linked; no data is duplicated into this folder.
- [x] Confounds are explicit per pair (`varies`), not left for the reader to infer.
- [x] Seed identity is asserted in code before any comparison.
- [x] No links to `.claude/...` and no paths into sibling repos.
- [x] The recommendation states what to change and by how much.
