# Hy3 fraud arm — INT4 W4A16 (compressed-tensors) on 4×H200 — detectable only in aggregate, 38 nonces for p < 1e-6

**Date:** 2026-08-19
**Fraud model:** `cyankiwi/Hy3-AWQ-INT4` — 182 GB, 34 shards.
**Despite the repo name this is not AWQ.** From `config.json`:
`quant_method: compressed-tensors`, format `pack-quantized`, INT4 **W4A16** asymmetric,
`group_size 32`, observer `mse`, `input_activations: null` (activations stay BF16),
`kv_cache_scheme: null`. The `ignore` list has 915 entries — **588 of them are the whole
MTP layer 80**, plus layer 0, every router gate, `lm_head` and all dense MLPs.
Kernels chosen by vLLM: `MarlinLinearKernel` + `CompressedTensorsWNA16MarlinMoEMethod`.
Refer to it as **INT4 W4A16 (compressed-tensors)**, not AWQ.

**Honest reference:** `tencent/Hy3-FP8` on the same host — see
[`../hy3-fp8-honest-baseline/`](../hy3-fp8-honest-baseline/). The reference nonce sets are
duplicated here as `ref_nonces_*` so this folder is self-contained.

**Hardware:** 4× NVIDIA **H200 SXM** (700 W, 143 GB, NV18, driver 610.57.04, sm_90),
Vast.ai instance 48115352.
**Image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash-0731:3.0.16-overlay-k5`
**Digest:** `sha256:8ce8830b4182b3dfd642c5e00f548f50a830611e4bc64ec4fbd84fe91070e3aa`

## Summary

On Hopper neither the honest nor the fraudulent arm is bit-reproducible, so detection is
necessarily statistical. The fraud separates cleanly **in aggregate** (10× more nonces over
the gate than honest noise) but **not per nonce** (ROC AUC 0.88, best-case FPR 17 %).

| Comparison | L2 median | >0.40 |
|---|---:|---:|
| honest FP8 ↔ honest FP8 (repeat) | 0.2025 | **4.1 %** |
| INT4 ↔ INT4 (repeat) | 0.1611 | 1.8 % |
| **honest FP8 ↔ INT4** | **0.3741** | **41.8 %** |

This arm is also **slower** than honest: 896 vs 1408 nonces/min. Its only advantage is
memory — 164 GiB of weights against 276 GiB — i.e. it fits where the honest model does not.

## Environment

| Parameter | Value |
|---|---|
| CUDA | 13.0.2 (image), driver 610.57.04 |
| vLLM | 0.25.1, build `752a3a5` |
| Python | 3.12.13 |
| mlnode | 3.0.16 |

## Config

Identical to the honest baseline (`scripts/patch_hy3.py`, `TP=4`), only the model id
changes:

```bash
curl -X POST http://127.0.0.1:8081/api/v1/inference/up/async \
  -H 'Content-Type: application/json' \
  -d '{"model":"cyankiwi/Hy3-AWQ-INT4","dtype":"auto","additional_args":[]}'
```

### What changed vs the honest arm

| Parameter | Honest | This arm |
|---|---|---|
| model | `tencent/Hy3-FP8` | `cyankiwi/Hy3-AWQ-INT4` |
| KV scales | static, baked in the checkpoint | `kv_cache_scheme: null` → computed at runtime under the forced `--kv-cache-dtype fp8` |
| everything else | — | unchanged |

The KV-scale difference is part of the measured distance and should be kept in mind when
attributing the divergence purely to weight quantisation.

## Validation

### Throughput ⚠️

| batch | 8 | 16 | 32 | 64 |
|---:|---:|---:|---:|---:|
| INT4 | 816 | 864 | **896** | 896 |
| honest FP8 | 1248 | 1376 | **1408** | 1408 |

Both columns were taken with a 30 s window and the pre-fix accounting bug
(`kaitakuai/experiments` PR #7), so the absolute values are inflated by an unknown
0–40 %. The **relative** statement (INT4 is ~36 % slower, prefill costs 28–30 % of a round
against 19 % on other hardware) survives because both arms were measured identically, but
**do not cite the absolute numbers**.

### Fingerprint

```bash
python3 scripts/l2_matrix.py artifacts ref_nonces_h200_fp8_s1.json nonces_h200_int4_s1.json
```

| Pair | bit-identical | L2 median | p95 | max | >0.40 |
|---|---:|---:|---:|---:|---:|
| honest ↔ honest (floor) | 0.0 % | 0.2025 | 0.3808 | 0.9540 | 4.1 % |
| INT4 ↔ INT4 (repeat) | 0.0 % | 0.1611 | 0.3163 | 0.8625 | 1.8 % |
| **honest ↔ INT4, s1** | 0.0 % | 0.3741 | 0.6363 | 1.0990 | 41.8 % |
| **honest ↔ INT4, s2** | 0.0 % | 0.3641 | 0.6432 | 1.0432 | 39.1 % |
| **honest ↔ INT4, s3** | 0.0 % | 0.3835 | 0.6442 | 1.0743 | 45.3 % |

### Separability analysis

Per-nonce ROC AUC (honest floor vs fraud): **0.881 / 0.882 / 0.891** for s1/s2/s3.
The best single-nonce threshold, 0.284, yields TPR 0.80 at **FPR 0.172** — one honest
prover in six would be falsely accused. A per-nonce verdict is therefore impossible.

Aggregate decisions work well. Share of nonces above a gate, and the set size needed to
separate honest from fraud at p < 1e-6:

| gate | honest | fraud | nonces needed |
|---:|---:|---:|---:|
| 0.28 | 18.4 % | 79.5–81.5 % | **38** |
| 0.30 | 14.3 % | 73.3–77.2 % | 41 |
| 0.40 | 4.1 % | 39.1–45.3 % | 87 |
| 0.50 | 1.0 % | 17.4–19.5 % | 193 |
| 0.60 | 0.4 % | 7.1–7.5 % | 515 |

**Raising the gate makes detection worse, not better.** At 0.60 honest false positives
nearly vanish, but the signal vanishes with them and 515 nonces are needed instead of 38.
The optimum sits *inside* the honest distribution — the gate is a statistic parameter, not
an accusation criterion. Using the set mean instead of a share gives ~10σ at 25 nonces.

### Inference

| Scenario | honest TTFT | honest out tok/s | INT4 TTFT | INT4 out tok/s |
|---|---:|---:|---:|---:|
| s1 long, sequential | 0.278 | 93.5 | 0.473 | 82.3 |
| s2 short, 30 runners | 0.188 | 1224.2 | 0.315 | 1282.1 |
| s3 very long, sequential | 0.625 | 79.1 | 1.058 | 67.8 |
| s4 very long, 20 runners | 4.405 | 275.8 | 7.667 | 201.1 |

INT4 loses almost everywhere; the single exception is s2 (+5 %), where decode is
memory-bound. TTFT degrades badly (s4: 7.67 s vs 4.41 s).

### Resources

| Arm | weights/rank | total | KV tokens | bringup |
|---|---:|---:|---:|---:|
| honest FP8 | 69.28 GiB | 276 GiB | 1 049 888 | 225 s |
| **INT4 W4A16** | **41.03 GiB** | **164 GiB** | 1 793 792 (+71 %) | 144 s |

## Findings

1. **Detection is aggregate-only on Hopper.** 4.1 % honest vs 39–45 % fraud over the 0.40
   gate — a 10× ratio — but per-nonce AUC is 0.88 and the honest maximum (0.954) overlaps
   the fraud median (0.374).
2. **The gate should be tuned inside the honest distribution** (≈0.28), not above it.
3. **This fraud is not a speed attack.** It is 36 % slower in PoC and worse at serving. Its
   value is fitting into fewer cards: 164 GiB against 276 GiB.
4. **The repo name is misleading** — compressed-tensors W4A16, not AWQ, and it leaves the
   MTP layer unquantised.
5. **Marlin dequantisation is expensive in prefill** — 28–30 % of a PoC round here.

## Files

```
artifacts/
  nonces_h200_int4_{s1,s1_r2,s2,s3}.json   fraud arm (s1_r2 = repeat of s1)
  ref_nonces_h200_fp8_{s1,s1_r2,s2,s3}.json  honest reference, same host
  sweep_h200_int4_30s_BUGGY.log            sweep (invalid timing, see above)
  serving_h200_int4.sqlite                 compressa-perf database
scripts/
  patch_hy3.py  run_pow_generation.py  collect_artifacts.py  l2_matrix.py  poc_seeds.json
```

## Reproducibility checklist

- [x] Image pinned by digest; quantisation scheme read from `config.json`, not the repo name
- [x] Honest reference committed alongside the fraud sets — folder is self-contained
- [x] All scripts committed under `scripts/`
- [x] L2 and separability tables reproducible via `scripts/l2_matrix.py`
- [x] 3 seeds behind every fingerprint claim
- [x] Invalid throughput numbers labelled in place
- [x] No internal-tooling links, absolute paths or sibling-repo references
