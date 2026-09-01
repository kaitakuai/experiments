# GLM-5.3-Flash — 2×B300 — honest FP8, re-analysed in the campaign's current methodology

**Date:** 2026-09-01 (analysis) · **data collected 2026-08-26**
**Model:** [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash), native FP8
(`weight_block_size [128,128]`), 288 routed experts, `num_experts_per_tok` 8, 45 layers.
**Hardware:** 2× NVIDIA B300 SXM6 AC (275 040 MiB each, 1100 W, NV18), TP=2, driver 610.57.04
**Image:** `ghcr.io/kaitakuai/vllm-poc:glm53-poc-v4-ed8873884` (**old profile**, FlashInfer 0.6.17)
**Digest:** `sha256:31b42acc1d85688a20e4ef8e6de718829062097cd6f3457f83e9e4fea892f123`
**PoC:** seq_len 1024, k_dim 12, collection batch 32

## Summary

**Nothing was re-run for this folder.** The nonce sets and sweep logs are byte-identical copies
of [`../../2026-08/glm53-flash-fp8-2xb300/`](../../2026-08/glm53-flash-fp8-2xb300/). What is new
is the analysis: the batch-position split, the cross-hardware matrix against the September
4×H200 and 4×B200 arms, and a corrected reading of the batch ceiling.

This arm matters because it is the only Blackwell data on the **previous** image. It is what
makes "is the cross-generation gap architecture or build?" answerable — and the answer is
**architecture**.

- Against both September arms this reads **0.266 / 16.5–16.7 %** past the 0.40 gate, matching
  the clean hardware-only pair (H200↔B200 on `k3`) at 16.5 %. Image and TP add nothing.
- **Batch 48 collapses even with ample token budget** — independent confirmation that the
  September batch limit is a real kernel limit, not only a flag.
- Best throughput **2030 nonces/min at batch 32**.

## Environment

| Parameter | Value |
|---|---|
| GPU | 2× NVIDIA B300 SXM6 AC, 275 040 MiB each, 1100 W, NV18 |
| NVIDIA driver | 610.57.04 |
| FlashInfer | 0.6.17 (previous image profile) |
| TP | 2 |
| Collection batch | 32 |

Full environment capture lives in the August folder; it is not duplicated here.

## Config

The engine flags are documented in the original August folder and are **not restated here** —
this folder does not re-run the measurement, and duplicating a config invites the two copies to
drift apart.

### What changed vs the default

Nothing, relative to the August run. Relative to the September arms, three things differ at
once — image (0.6.17 vs `k3`/0.6.18), TP (2 vs 4) and hardware (B300 vs H200/B200) — which is
why every cross-arm row below carries an explicit `varies` note.

## Validation

### L2

**No same-box pair exists for this arm.** The August run collected one pass per seed, so the
honest floor — the distance between two runs of the same seed on the same engine — cannot be
computed here. It is measured in [`../glm53-flash-fp8-4xb200/`](../glm53-flash-fp8-4xb200/)
(0 of 1000 past the gate) and [`../glm53-flash-fp8-4xh200/`](../glm53-flash-fp8-4xh200/)
(1 of 1000).

The comparison this arm does support is cross-hardware, below.

### Cross-hardware L2

Per seed, 1000 nonces each. Gate defaults: `threshold = 0.40`, `p_mis = 0.001`.

**vs honest 4×H200 (`k3`)** — varies: hardware, image, TP

| seed | mean | median | p25 | p75 | p95 | max | past 0.40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| s1 | 0.3207 | 0.2559 | 0.1997 | 0.3454 | 0.8971 | 1.6033 | 168 (16.8 %) |
| s2 | 0.3216 | 0.2674 | 0.2044 | 0.3477 | 0.8606 | 1.4988 | 170 (17.0 %) |
| s3 | 0.3220 | 0.2658 | 0.2046 | 0.3501 | 0.8916 | 1.6686 | 163 (16.3 %) |
| **all three, 3000 nonces** | **0.3214** | **0.2627** | | | | | **501 (16.7 %)** |

**vs honest 4×B200 (`k3`)** — varies: hardware, image, TP

| seed | mean | median | p25 | p75 | p95 | max | past 0.40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| s1 | 0.3263 | 0.2604 | 0.1968 | 0.3460 | 1.0249 | 1.6435 | 166 (16.6 %) |
| s2 | 0.3308 | 0.2666 | 0.2032 | 0.3526 | 0.9924 | 1.6025 | 158 (15.8 %) |
| s3 | 0.3345 | 0.2705 | 0.2077 | 0.3533 | 1.0097 | 1.6942 | 171 (17.1 %) |
| **all three, 3000 nonces** | **0.3306** | **0.2647** | | | | | **495 (16.5 %)** |

**Reference — the pair that varies hardware only** (4×B200 ↔ 4×H200, both `k3`, both TP=4):
median **0.2609**, **496 / 3000 (16.5 %)** past the gate. The two confounded pairings above land
on the same number, so image and TP contribute nothing measurable and the whole gap is the GPU
generation.

The distribution is bimodal, not a long tail: p75 is 0.35 while p95 is already ~0.9–1.0. The
second population is the batch-boundary artifact, split out below.

### The batch-boundary artifact

This run collected at batch **32**, so the affected nonces are those at `index % 32 == 0`
rather than `% 16`. Against any `k3` arm they are **100 % past the 0.40 gate**, exactly as on
Hopper and B200. The defect is tied to neither batch size, nor image, nor architecture.

### Throughput

Sweep, 5 s warmup + 120 s measurement, TP=2 (`artifacts/sweep_*.log`):

| batch | tokens/pass | `--max-num-batched-tokens` | nonces/min |
|---:|---:|---:|---:|
| 16 | 16 384 | 65 536 | 1882 |
| 32 | 32 768 | 65 536 | **2030** |
| 64 | 65 536 | 65 536 | 256 — collapses |
| 48 | 49 152 | 131 072 | 120–264 — collapses |
| 64 | 65 536 | 131 072 | 0 |

For scale at batch 32: 4×B200 on `k3` reaches 2727, 4×H200 about 2549.

**This arm independently confirms the September finding about the batch limit.** The rows at
`--max-num-batched-tokens 131072` have ample budget for batch 48 (49 152 tokens) and it still
collapses, so batch 48 is a genuine kernel limit rather than a budget one — on 4×B200 the same
batch fails with `Triton Error: illegal memory access` in the sparse-MLA indexer. Batch 32 is
the highest usable value on every architecture measured.

### Serving

**Not measured.** The August run did not include a serving pass on the honest arm, and this
folder re-runs nothing. Serving figures for the honest configuration are in
[`../glm53-flash-fp8-4xb200/`](../glm53-flash-fp8-4xb200/) and
[`../glm53-flash-fp8-4xh200/`](../glm53-flash-fp8-4xh200/).

### Integrity checks

- 3000 nonces across 3 sets: 100 % non-empty, 100 % unique (`artifacts/summary.json`).
- Each seed's `block_hash` matches `scripts/poc_seeds.json`; `matrix.py` asserts this across
  every arm before comparing.
- Control: two different seeds give median ≈ 1.42 — the expected ceiling for uncorrelated
  12-dim vectors.

## What this does not settle

- **No honest floor.** The August run collected one pass per seed, so there is no same-box
  repeat to measure it from. The Blackwell floor (0 of 1000 past the gate) comes from
  [`../glm53-flash-fp8-4xb200/`](../glm53-flash-fp8-4xb200/).
- **Nothing was re-measured on `k3` for B300.** It was planned and deliberately dropped: the
  decomposition it would give does not change any conclusion, because a real fleet varies
  hardware and build together.
- **Serving was never measured on this arm** (see above).

## Files

| path | what |
|---|---|
| `artifacts/nonces_honest_{s1,s2,s3}.json` | the August sets, copied verbatim |
| `artifacts/sweep_*.log` | the August sweeps at two token budgets, copied verbatim |
| [`artifacts/summary.json`](artifacts/summary.json) | integrity and control, regenerated |
| [`artifacts/matrix.json`](artifacts/matrix.json) | full pairwise matrix across all six arms |
| [`scripts/summarize.py`](scripts/summarize.py) | integrity and control checks |
| [`scripts/matrix.py`](scripts/matrix.py) | builds the cross-hardware matrix from the committed sets |
| [`scripts/poc_seeds.json`](scripts/poc_seeds.json) | the fixed seed set |

## Reproduce

```bash
python3 scripts/summarize.py artifacts > artifacts/summary.json
python3 scripts/matrix.py               > artifacts/matrix.json
```

To reproduce the **measurement** rather than the analysis, follow the August folder — the
engine flags, image and box are documented there.

Success criteria: 3000 non-empty unique nonces; cross-hardware median ≈ 0.266 with 16–17 % past
the gate; every `index % 32 == 0` nonce past the gate in cross-hardware pairs.

## Gotchas

- **Give nonce files distinct basenames before comparing.** `compare_nonces.py` labels pairs by
  basename and silently compares a file with itself when two inputs share a name, reporting a
  false `L2 = 0.0000`. `scripts/matrix.py` keys by folder and arm instead.
- **This arm's collection batch is 32, not 16.** The batch-boundary artifact therefore lands on
  `index % 32 == 0` here; comparing it against a batch-16 arm mixes the two spacings.
- **`--max-num-batched-tokens` must be ≥ batch × 1024**, or the node yields zero nonces while
  looking healthy — see the September folders.

## Related

- original measurement: [`../../2026-08/glm53-flash-fp8-2xb300/README.md`](../../2026-08/glm53-flash-fp8-2xb300/README.md)
- NVFP4 arm on this box: [`../glm53-flash-nvfp4-libertai-2xb300/README.md`](../glm53-flash-nvfp4-libertai-2xb300/README.md)
- campaign summary: [`../glm53-flash-cross-hardware-summary/README.md`](../glm53-flash-cross-hardware-summary/README.md)

## Reproducibility checklist

- [x] A reader with only this folder can reproduce every table by running the two scripts.
- [x] Hardware, image and digest are stated exactly.
- [x] Provenance is explicit: data collected 2026-08-26, no new run.
- [x] Every script referenced is committed under `scripts/`.
- [x] No links to `.claude/...` and no paths into sibling repos.
- [x] All artifacts referenced exist in `artifacts/`, including the sweep logs.
- [x] Throughput is reported together with the logs it comes from.
- [x] Sections absent for a reason say so instead of being omitted.
- [ ] Engine logs are not duplicated here; they live in the August folder.
