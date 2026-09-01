# GLM-5.3-Flash — 2×B300 — honest FP8, re-analysed in the campaign's current methodology

**Data collected:** 2026-08-26 — **nothing was re-run for this folder.** The nonce sets are
byte-identical copies of [`../../2026-08/glm53-flash-fp8-2xb300/`](../../2026-08/glm53-flash-fp8-2xb300/).
What is new is the analysis: the batch-position split, the cross-hardware matrix against the
4×H200 and 4×B200 arms, and the corrected reading of the batch ceiling.

**Model:** [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash), native FP8
**Hardware:** 2× NVIDIA B300 SXM6 AC (275 040 MiB each, 1100 W, NV18), TP=2, driver 610.57.04
**Image:** `ghcr.io/kaitakuai/vllm-poc:glm53-poc-v4-ed8873884` (**old profile**, FlashInfer 0.6.17)
**Digest:** `sha256:31b42acc1d85688a20e4ef8e6de718829062097cd6f3457f83e9e4fea892f123`
**PoC:** seq_len 1024, k_dim 12, collection batch 32

## Why this folder exists

The September runs ([`../glm53-flash-fp8-4xh200/`](../glm53-flash-fp8-4xh200/),
[`../glm53-flash-fp8-4xb200/`](../glm53-flash-fp8-4xb200/)) are all on the `k3` image. This arm
is the only Blackwell data on the **previous** image, so it is what makes the
"is the cross-generation gap architecture or build?" question answerable at all. The answer,
from the matrix below, is **architecture**.

## What the re-analysis shows

| pair | median L2 | past 0.40 | what differs |
|---|---:|---:|---|
| this arm ↔ 4×H200 honest (k3) | 0.2658 | 16.7 % | hardware, image, TP |
| **4×B200 honest (k3) ↔ 4×H200 honest (k3)** | **0.2606** | **16.5 %** | **hardware only** |
| this arm ↔ 4×B200 honest (k3) | 0.2666 | 16.5 % | hardware, image, TP |

All three agree within 0.2 points. The confounded pairings (which also change image and TP)
land on the same number as the clean one, so **image and TP contribute nothing measurable** and
the 16–17 % is the GPU generation.

### The batch-boundary artifact is present here too

This run collected at batch **32**, so the affected nonces are those at `index % 32 == 0`
rather than `% 16`. Against any k3 arm they are **100 % past the 0.40 gate**, exactly as on
Hopper and B200 — the defect is not tied to a batch size, an image, or an architecture.

### Batch ceiling — the earlier reading was wrong

The August report ran sweeps at batch 16/32/64 with `--max-num-batched-tokens 65536` and found
batch 32 healthy. Later Hopper work concluded a "ceiling of 16" and blamed DeepGEMM; that was
an artifact of running with `--max-num-batched-tokens 16384`, which cannot fit
batch × 1024 tokens above batch 16. This arm is the counter-example that was available all
along: same model, batch 32, working — because its budget was 65536.

## Files

| path | what |
|---|---|
| `artifacts/nonces_honest_{s1,s2,s3}.json` | the August sets, copied verbatim |
| [`artifacts/summary.json`](artifacts/summary.json) | integrity + control, regenerated |
| [`artifacts/matrix.json`](artifacts/matrix.json) | full pairwise matrix across all six arms |
| [`scripts/summarize.py`](scripts/summarize.py) | integrity and control checks |
| [`scripts/matrix.py`](scripts/matrix.py) | builds the cross-hardware matrix from the committed sets |
| [`scripts/poc_seeds.json`](scripts/poc_seeds.json) | the fixed seed set |

## Reproduce the analysis

```bash
python3 scripts/summarize.py artifacts > artifacts/summary.json
python3 scripts/matrix.py               > artifacts/matrix.json
```

To reproduce the **measurement**, follow the original August folder — the engine flags, image
and box are documented there, and this folder deliberately does not restate them.

## What this does not settle

- **No honest floor.** The August run collected one pass per seed, so there is no same-box
  repeat here. The Blackwell floor (0 of 1000 past the gate) comes from the 4×B200 folder.
- **Nothing was re-measured on `k3` for B300.** It was planned and deliberately dropped: the
  decomposition it would give does not change any conclusion, because a real fleet varies
  hardware and build together.

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
- [x] All artifacts referenced exist in `artifacts/`.
- [ ] Engine logs are not duplicated here; they live in the August folder.
