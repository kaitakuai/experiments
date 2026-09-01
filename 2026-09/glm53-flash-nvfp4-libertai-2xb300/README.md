# GLM-5.3-Flash — 2×B300 — NVFP4 fraud, re-analysed: the 0.711 figure does not transfer

**Date:** 2026-09-01 (analysis) · **data collected 2026-08-26**
**Model (fraud):** [`LibertAIDAI/GLM-5.3-Flash-NVFP4`](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4)
— NVFP4 via ModelOpt (`quant_algo: NVFP4`), snapshot `11d73216cd636238e82e1d77fe1042ffab36e7fa`,
183 GB, 120 shards.
**Model (reference):** `zai-org/GLM-5.3-Flash`, native FP8, measured on this same box.
**Hardware:** 2× NVIDIA B300 SXM6 AC (275 040 MiB each, 1100 W, NV18), TP=2, driver 610.57.04
**Image:** `ghcr.io/kaitakuai/vllm-poc:glm53-poc-v4-ed8873884` (**old profile**, FlashInfer 0.6.17)
**Digest:** `sha256:31b42acc1d85688a20e4ef8e6de718829062097cd6f3457f83e9e4fea892f123`
**PoC:** seq_len 1024, k_dim 12, collection batch 32

## Summary

**Nothing was re-run for this folder.** The nonce sets and sweep logs are byte-identical copies
of [`../../2026-08/glm53-flash-nvfp4-libertai-2xb300/`](../../2026-08/glm53-flash-nvfp4-libertai-2xb300/).
What is new is the comparison against the September 4×B200 run of the **same checkpoint**.

The August report called this "the loudest fraud we have measured" — median **0.711**, 97.3 %
past the 0.40 gate. That reproduces here. **But it is not a property of the attack:** the
identical checkpoint on 4×B200 with the `k3` image reads **0.379 / 43 %**, and the two fraud
runs disagree with *each other* (0.737) more than either disagrees with its own honest baseline.

A published NVFP4 signature is therefore a property of the platform it was measured on.
Detection must be framed as *far from honest*, never as *close to a known fingerprint*.

## Environment

| Parameter | Value |
|---|---|
| GPU | 2× NVIDIA B300 SXM6 AC, 275 040 MiB each, 1100 W, NV18 |
| NVIDIA driver | 610.57.04 |
| FlashInfer | 0.6.17 (previous image profile) |
| TP | 2 (TP=1 impossible — the model crashes on a single card) |
| Collection batch | 32 |

Identical to the honest arm on this box; only the checkpoint differs.

## Config

Engine flags are documented in the original August folder and are **not restated here** — this
folder re-runs nothing, and a duplicated config only invites the copies to drift.

### What changed vs the default

Relative to the honest arm on the same box: the checkpoint, and nothing else. Relative to the
September NVFP4 arm: hardware, image and TP all differ at once, which is exactly why the
comparison below cannot be decomposed.

## Validation

### L2

Gate defaults: `threshold = 0.40`, `p_mis = 0.001`.

| seed | median L2 | past 0.40 |
|---|---:|---:|
| s1 | 0.7138 | 96.7 % |
| s2 | 0.7132 | 97.3 % |
| s3 | 0.7050 | 97.8 % |

### Cross-hardware L2

| pair | median L2 | past 0.40 | what differs |
|---|---:|---:|---|
| this NVFP4 ↔ its own honest arm | 0.7132 | 97.3 % | checkpoint |
| **this NVFP4 ↔ September NVFP4 on 4×B200** | **0.7368** | **98.0 %** | hardware, image, TP |
| September NVFP4 ↔ its own honest arm | 0.3791 | 43.2 % | checkpoint |
| *reference:* honest ↔ honest across the same platform move | 0.2666 | 16.5 % | hardware, image, TP |

Two runs of one quantised checkpoint sit further apart (0.737) than fraud sits from honest on
either box. Quantised numerics depend on which kernels execute them far more than FP8 numerics
do — the honest arms move only 0.267 across the same platform change.

### The batch-boundary artifact

Collected at batch 32, so the affected nonces are at `index % 32 == 0`. Against the honest arm
on this same box they are 98.9 % past the gate, against `k3` arms 100 % — but since this arm's
overall rate is already 97 %, removing them changes the verdict by well under a point. The
detection here does not rest on the artifact.

### Throughput

Sweep, 5 s warmup + 120 s measurement (`artifacts/sweep_*.log`):

| batch | TP | nonces/min | note |
|---:|---:|---:|---|
| 16 | 2 | 2623 | |
| 32 | 2 | **2767** | best; **+36 %** over the honest arm's 2030 |
| 16 | 1 | 216 | TP=1 unusable, eager |
| 16 | 1 | 56 | TP=1 unusable, CUDA graphs |

The attack pays **+36 %** at identical topology. It cannot additionally drop to TP=1 — the
model does not run on a single card regardless of graph mode — so the attacker's ceiling here
is +36 %. For comparison, the same checkpoint on 4×B200 buys +23 %.

### Serving

**Not measured.** The August run did not include a serving pass. Serving comparisons for
quantisation fraud are in [`../glm53-flash-nvfp4-libertai-4xb200/`](../glm53-flash-nvfp4-libertai-4xb200/).

### Integrity checks

- 6000 nonces across 6 sets (3 fraud + 3 honest reference): 100 % non-empty, 100 % unique
  (`artifacts/summary.json`).
- Each seed's `block_hash` matches `scripts/poc_seeds.json`; `matrix.py` asserts this across
  every arm before comparing.
- Control: two different seeds give median 1.4210 — the expected ceiling.

## What this does not settle

- **The B300↔B200 disagreement is not decomposed.** Architecture, image and TP all differ. A
  2×B300 run on `k3` would separate them; it was planned and dropped, because the actionable
  conclusion — the quantity is not stable — does not depend on the split.
- **One quantisation producer only** (ModelOpt via LibertAI).
- **Serving was never measured on this arm** (see above).

## Files

| path | what |
|---|---|
| `artifacts/nonces_nvfp4_{s1,s2,s3}.json` | the August fraud sets, copied verbatim |
| `artifacts/ref_nonces_honest_{s1,s2,s3}.json` | the August honest sets, same box |
| `artifacts/sweep_*.log` | the August sweeps, TP=1 and TP=2, copied verbatim |
| [`artifacts/summary.json`](artifacts/summary.json) | L2 tables and the batch split, regenerated |
| [`artifacts/matrix.json`](artifacts/matrix.json) | full pairwise matrix across all six arms |
| [`scripts/summarize.py`](scripts/summarize.py) | L2, batch split, integrity, control |
| [`scripts/matrix.py`](scripts/matrix.py) | builds the cross-hardware matrix from the committed sets |
| [`scripts/poc_seeds.json`](scripts/poc_seeds.json) | the fixed seed set |

## Reproduce

```bash
python3 scripts/summarize.py artifacts > artifacts/summary.json
python3 scripts/matrix.py               > artifacts/matrix.json
```

To reproduce the **measurement** rather than the analysis, follow the August folder.

Success criteria: median L2 0.705–0.714 with ~97 % past the gate on all three seeds; the
cross-platform NVFP4↔NVFP4 pair at ≈ 0.737.

## Gotchas

- **Give nonce files distinct basenames before comparing.** `compare_nonces.py` labels pairs by
  basename and silently compares a file with itself when two inputs share a name, reporting a
  false `L2 = 0.0000`. That happened during this analysis and was caught only by checking the
  `PAIR:` / `vs:` lines. `scripts/matrix.py` keys by folder and arm instead of by filename.
- **This arm's collection batch is 32**, so the batch-boundary artifact lands on
  `index % 32 == 0`, not `% 16`.
- **Do not quote 0.711 as the NVFP4 signature.** It is this platform's number; the same
  checkpoint reads 0.379 elsewhere.

## Related

- the same checkpoint on 4×B200, `k3` image: [`../glm53-flash-nvfp4-libertai-4xb200/README.md`](../glm53-flash-nvfp4-libertai-4xb200/README.md)
- honest arm on this box: [`../glm53-flash-fp8-2xb300/README.md`](../glm53-flash-fp8-2xb300/README.md)
- original measurement: [`../../2026-08/glm53-flash-nvfp4-libertai-2xb300/README.md`](../../2026-08/glm53-flash-nvfp4-libertai-2xb300/README.md)
- campaign summary: [`../glm53-flash-cross-hardware-summary/README.md`](../glm53-flash-cross-hardware-summary/README.md)

## Reproducibility checklist

- [x] A reader with only this folder can reproduce every table by running the two scripts.
- [x] Hardware, image and digest are stated exactly.
- [x] Provenance is explicit: data collected 2026-08-26, no new run.
- [x] Every script referenced is committed under `scripts/`.
- [x] No links to `.claude/...` and no paths into sibling repos.
- [x] All artifacts referenced exist in `artifacts/`, including the honest reference sets and
      the sweep logs.
- [x] Throughput is reported together with the logs it comes from.
- [x] Sections absent for a reason say so instead of being omitted.
- [x] The known analysis trap (identical basenames) is documented.
- [ ] Engine logs are not duplicated here; they live in the August folder.
