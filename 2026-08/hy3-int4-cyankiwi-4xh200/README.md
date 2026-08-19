# Hy3 INT4 W4A16 by `cyankiwi` (compressed-tensors) — 4×H200 — fraud arm, detectable only in aggregate

**Date:** 2026-08-19
**Model:** `cyankiwi/Hy3-AWQ-INT4` — 182 GB, 34 shards. **Despite the repo name this is not
AWQ.** From `config.json`: `quant_method: compressed-tensors`, format `pack-quantized`,
INT4 **W4A16** asymmetric, `group_size 32`, observer `mse`, `input_activations: null`
(activations stay BF16), `kv_cache_scheme: null`. The `ignore` list holds 915 entries — of
which **588 are the whole MTP layer 80** — plus layer 0, every router gate, `lm_head` and all
dense MLPs. Kernels: `MarlinLinearKernel` + `CompressedTensorsWNA16MarlinMoEMethod`.
Weights in VRAM: **164 GiB** (41.03 GiB/rank).

**Honest reference:** [`../hy3-fp8-4xh200/`](../hy3-fp8-4xh200/) — same host, same session.
Its nonce sets are duplicated here as `ref_nonces_*` so this folder is self-contained.

**Hardware:** 4× NVIDIA **H200 SXM** (700 W, 143 GB, NV18, driver **610.57.04**, sm_90).
Vast.ai instance 48115352.
**Image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash-0731:3.0.16-overlay-k5`
**Digest:** `sha256:8ce8830b4182b3dfd642c5e00f548f50a830611e4bc64ec4fbd84fe91070e3aa`

> The DeepSeek-V4-Flash foundry image, reused only as a vLLM 0.25.1 + PoC-plugin runtime.
> Its `runner.py` hardcodes V4-specific flags that must be replaced — `scripts/patch_hy3.py`.

## Summary

Neither arm is bit-reproducible on Hopper, so detection is necessarily statistical. The
fraud separates cleanly **in aggregate** — 10× more nonces over the gate than honest noise —
but **not per nonce**: ROC AUC is 0.88 and the best achievable single-nonce threshold still
misclassifies one honest prover in six.

| Comparison | L2 median | >0.40 |
|---|---:|---:|
| honest ↔ honest (repeat) | 0.2025 | **4.1 %** |
| INT4 ↔ INT4 (repeat) | 0.1611 | 1.8 % |
| **honest ↔ INT4** | **0.3741** | **41.8 %** |

This arm is also **slower** than honest and worse at serving. Its only advantage is memory:
164 GiB against 276 GiB, i.e. it fits where the honest model does not.

## Environment

| Parameter | Value |
|---|---|
| CUDA | 13.0.2 (image), driver 610.57.04 |
| vLLM | 0.25.1, build `752a3a504485790a2e8491cacbb35c137339ad34` |
| Python | 3.12.13 |
| mlnode | 3.0.16, `gonka_poc.entrypoint.api_router` |

## Config

Identical to the honest baseline; only the model id changes.

```bash
TP=4 python3 scripts/patch_hy3.py
curl -X POST http://127.0.0.1:8081/api/v1/inference/up/async \
  -H 'Content-Type: application/json' \
  -d '{"model":"cyankiwi/Hy3-AWQ-INT4","dtype":"auto","additional_args":[]}'
```

```
--tensor-parallel-size 4   --gpu-memory-utilization 0.90
--max-model-len 262144        --max-num-batched-tokens 65536
--kv-cache-dtype fp8          --logprobs-mode processed_logprobs
--worker-extension-cls gonka_poc.worker.PoCWorkerExtension
--tool-call-parser hy_v3      --reasoning-parser hy_v3
--trust-remote-code --enable-auto-tool-choice --enable-expert-parallel
```

### What changed vs the honest arm

| Parameter | Honest | This arm |
|---|---|---|
| model | `tencent/Hy3-FP8` (276 GiB) | `cyankiwi/Hy3-AWQ-INT4` (164 GiB) |
| MoE kernel | FP8 path | MARLIN WNA16 |
| KV scales | static, baked into the checkpoint | `kv_cache_scheme: null` → computed at runtime under the forced `--kv-cache-dtype fp8` |
| everything else | — | unchanged |

The KV-scale difference is part of the measured distance and should be kept in mind before
attributing all of it to weight quantisation.

## Validation

### Throughput

| batch | 8 | 16 | 32 | 64 |
|---:|---:|---:|---:|---:|
| INT4 | 816 | 864 | **896** | 896 |
| honest FP8 | 1248 | 1376 | **1408** | 1408 |

> ⚠️ **These throughput numbers are invalid.** They were taken with a 30 s measurement
> window *and* the pre-fix boundary-accounting bug (`kaitakuai/experiments` PR #7), which
> inflates by an unknown 0–40 %. On top of that, callbacks arrive in ~5 s bulks, so a 30 s
> window carries ±17 % noise on its own. They are published rather than deleted because the
> relative comparisons on this host were taken identically. For valid numbers see the
> Blackwell runs, which use the corrected script and a 120 s window.

The relative statement survives — both arms were measured identically, INT4 is ~36 % slower,
and prefill takes 28–30 % of a PoC round here against 19 % on other hardware, which is the
cost of MARLIN dequantisation.

### Fingerprint

```bash
python3 scripts/l2_matrix.py artifacts ref_nonces_fp8_s1.json nonces_int4_s1.json
```

| Pair | bit-identical | L2 median | p95 | max | >0.40 |
|---|---:|---:|---:|---:|---:|
| honest ↔ honest (floor) | 0.0 % | 0.2025 | 0.3808 | 0.9540 | 4.1 % |
| INT4 ↔ INT4 (repeat) | 0.0 % | 0.1611 | 0.3163 | 0.8625 | 1.8 % |
| **honest ↔ INT4, s1** | 0.0 % | 0.3741 | 0.6363 | 1.0990 | 41.8 % |
| **honest ↔ INT4, s2** | 0.0 % | 0.3641 | 0.6432 | 1.0432 | 39.1 % |
| **honest ↔ INT4, s3** | 0.0 % | 0.3835 | 0.6442 | 1.0743 | 45.3 % |

### Separability

Per-nonce ROC AUC (honest floor vs fraud): **0.881 / 0.882 / 0.891** across s1/s2/s3. The
best single-nonce threshold, 0.284, gives TPR 0.80 at **FPR 0.172**. A per-nonce verdict is
therefore impossible.

Aggregate decisions work. Share of nonces above a gate, and the set size needed to separate
honest from fraud at p < 1e-6:

| gate | honest | fraud | nonces needed |
|---:|---:|---:|---:|
| 0.28 | 18.4 % | 79.5–81.5 % | **38** |
| 0.30 | 14.3 % | 73.3–77.2 % | 41 |
| 0.40 | 4.1 % | 39.1–45.3 % | 87 |
| 0.50 | 1.0 % | 17.4–19.5 % | 193 |
| 0.60 | 0.4 % | 7.1–7.5 % | 515 |

**Raising the gate makes detection worse.** At 0.60 honest false positives nearly vanish,
but so does the signal, and 515 nonces are needed instead of 38. The optimum sits *inside*
the honest distribution: the gate is a parameter of a statistic, not an accusation criterion.
Using the set mean instead of a share gives ~10σ separation at 25 nonces.

### Inference

| Scenario | honest TTFT | honest out tok/s | INT4 TTFT | INT4 out tok/s |
|---|---:|---:|---:|---:|
| s1 long, sequential | 0.278 | 93.5 | 0.473 | 82.3 |
| s2 short, 30 runners | 0.188 | 1224.2 | 0.315 | 1282.1 |
| s3 very long, sequential | 0.625 | 79.1 | 1.058 | 67.8 |
| s4 very long, 20 runners | 4.405 | 275.8 | 7.667 | 201.1 |

The only win is s2 (+5 %), where decode is memory-bound. TTFT degrades badly (s4: 7.67 s vs
4.41 s).

### Resources

| | honest | this arm |
|---|---:|---:|
| weights / rank | 69.28 GiB | **41.03 GiB** |
| total weights | 276 GiB | **164 GiB** |
| KV cache | 1 049 888 | 1 793 792 (+71 %) |
| bring-up | 225 s | 144 s |

## Findings

1. **Aggregate-only detection on Hopper**: 4.1 % honest vs 39–45 % fraud over the 0.40 gate,
   but per-nonce AUC 0.88 and an honest maximum (0.954) that overlaps the fraud median.
2. **Tune the gate inside the honest distribution** (≈0.28), not above it.
3. **Not a speed attack** — 36 % slower in PoC, worse at serving. The value is fitting into
   fewer cards.
4. **The repo name is misleading** — compressed-tensors W4A16, not AWQ, MTP left unquantised.
5. **MARLIN dequantisation is expensive in prefill** — 28–30 % of a round.

## Files

```
artifacts/
  nonces_int4_{s1,s1_r2,s2,s3}.json    fraud arm (s1_r2 = repeat of s1)
  ref_nonces_fp8_{s1,s1_r2,s2,s3}.json honest reference, same host
  sweep_30s_BUGGY.log                   invalid timing, see above
  serving.sqlite                        compressa-perf database
scripts/
  patch_hy3.py  run_pow_generation.py  collect_artifacts.py  l2_matrix.py  poc_seeds.json
```

Related: [honest FP8 on this host](../hy3-fp8-4xh200/) ·
[NVFP4 ModelOpt](../hy3-nvfp4-r0b0tlab-2xb300/) · [NVFP4 llm-compressor](../hy3-nvfp4-redhatai-4xb200/)

## Reproducibility checklist

- [x] Image pinned by digest; quantisation described from `config.json` / loader output
- [x] Every script referenced above committed under `scripts/`
- [x] L2 tables reproducible from committed artifacts via `scripts/l2_matrix.py`
- [x] 3 seeds behind every fingerprint claim
- [x] Invalid or superseded measurements labelled in place, not dropped
- [x] No internal-tooling links, absolute paths, or sibling-repo references
