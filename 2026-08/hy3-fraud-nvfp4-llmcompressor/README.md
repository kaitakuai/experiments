# Hy3 fraud arm — NVFP4 + FP8 (llm-compressor) on 4×B200 — the quietest fraud measured, 0.367 vs 0.491 for the same scheme

**Date:** 2026-08-19
**Fraud model:** `RedHatAI/Hy3-NVFP4-FP8` — 178 GB, 5 shards, built with **llm-compressor**.
From `config.json`: `quant_method: compressed-tensors`, `format: mixed-precision`, two
config groups —

| group | targets | weights | input activations |
|---|---|---|---|
| 0 | `re:.*self_attn\..*` | FP8, `strategy: block`, `block_structure [128,128]`, static | FP8, `strategy: group`, `group_size 128`, **dynamic** |
| 1 | `re:.*mlp.*` | **NVFP4**, `tensor_group`, `group_size 16`, scale dtype `float8_e4m3` | **NVFP4**, `tensor_group 16`, `dynamic: "local"` |

`kv_cache_scheme: null`. The `ignore` list has 81 entries: router gates of layers 1–79,
`lm_head`, and `^model.layers.80.*` — i.e. **the MTP layer is left unquantised**, as in
every other Hy3 fraud build measured.

**Honest reference:** `tencent/Hy3-FP8` on the same host — see
[`../hy3-fp8-honest-baseline/`](../hy3-fp8-honest-baseline/); reference sets duplicated here.
A sample of the other NVFP4 build ([ModelOpt](../hy3-fraud-nvfp4-modelopt/)) is included as
`other_fraud_*` for the fraud-vs-fraud comparison.

**Hardware:** 4× NVIDIA **B200 SXM** (1000 W, 183 GB, NV18, driver 580.126.20, sm_100),
Vast.ai instance 48135501.
**Image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash-0731:3.0.16-overlay-k5`
**Digest:** `sha256:8ce8830b4182b3dfd642c5e00f548f50a830611e4bc64ec4fbd84fe91070e3aa`

## Summary

This arm exists to answer one question: **is the PoC fingerprint a property of the
quantisation scheme or of the specific build?** It is a second, independently produced
NVFP4 checkpoint of the same shape as the ModelOpt one — 4-bit MoE, everything sensitive in
FP8 — but from a different toolchain.

The answer is *the build*. On the same host, against the same honest reference:

| fraud build | toolchain | L2 median vs honest (s1/s2/s3) | >0.40 |
|---|---|---|---:|
| **this arm** | llm-compressor | **0.3670 / 0.3653 / 0.3844** | 38.5–45.9 % |
| ModelOpt build | NVIDIA ModelOpt | 0.4909 / 0.4890 / 0.5002 | 70.7–75.8 % |

A third lower, for nominally the same scheme. Worse, the two frauds are **further from each
other (0.52) than either is from honest** (0.37 and 0.49), so "resembles a known fraud" is
not a usable detector — only "far from honest" is.

Consequence: thresholds cannot be calibrated per quantisation scheme. They must be
calibrated against the **quietest known build**, which today is this one.

It is also the fastest arm measured on this host: **1952 nonces/min against the honest
1888** (+3.4 %), with serving gains up to +53 %.

## Environment

| Parameter | Value |
|---|---|
| CUDA | 13.0.2 (image), driver 580.126.20 |
| vLLM | 0.25.1, build `752a3a5` |
| compressed-tensors | 0.17.0 (pinned by vLLM 0.25.1) |
| Python | 3.12.13 |
| mlnode | 3.0.16 |

### Does it load on vLLM 0.25.1?

Yes, unmodified — worth stating because `format: mixed-precision` is not referenced
anywhere in vLLM's `compressed_tensors.py`. The format enum lives in the pinned
`compressed-tensors 0.17.0` library, which knows both `mixed_precision` and
`DynamicType.LOCAL` ("local is only currently supported for NVFP4", validated to require
`tensor_group`). vLLM dispatches on quant *arguments*, not on the format string:
`_is_nvfp4_format(weight_quant)` selects `CompressedTensorsW4A4Nvfp4MoEMethod`, and the FP8
group passes `_is_fp8_w8a8` through its "dynamic activations are always supported" branch.

Observed kernel selection:

```
Selected DeepGemmFp8BlockScaledMMKernel for CompressedTensorsW8A8Fp8
Using FlashInferCuteDslNvFp4LinearKernel for NVFP4 GEMM
Using 'FLASHINFER_TRTLLM' NvFp4 MoE backend
```

## Config

Identical to the honest baseline (`scripts/patch_hy3.py`, `TP=4`); only the model id changes:

```bash
curl -X POST http://127.0.0.1:8081/api/v1/inference/up/async \
  -H 'Content-Type: application/json' \
  -d '{"model":"RedHatAI/Hy3-NVFP4-FP8","dtype":"auto","additional_args":[]}'
```

Measurement window **120 s**, batches 16/32/64.

## Validation

### Throughput (120 s window, corrected accounting)

| batch | honest FP8 | **this arm** | ModelOpt build |
|---:|---:|---:|---:|
| 16 | 1736 | 1768 | 1776 |
| 32 | 1840 | 1904 | 1872 |
| 64 | **1888** | **1952 (+3.4 %)** | 1920 |

### Fingerprint

```bash
python3 scripts/l2_matrix.py artifacts ref_nonces_b200_fp8_s1.json nonces_b200_rh_s1.json
```

| Pair | bit-identical | L2 median | p95 | >0.40 |
|---|---:|---:|---:|---:|
| this arm ↔ itself (repeat) | **100.0 %** | 0.0000 | 0.0000 | 0.0 % |
| honest ↔ this arm, s1 | 0.0 % | 0.3670 | 0.6319 | 40.9 % |
| honest ↔ this arm, s2 | 0.0 % | 0.3653 | 0.6333 | 38.5 % |
| honest ↔ this arm, s3 | 0.0 % | 0.3844 | 0.6388 | 45.9 % |
| **this arm ↔ ModelOpt build, s1** | 0.0 % | **0.5243** | — | 77.2 % |

Note the last row: the two fraudulent builds are further apart than either is from the
honest model. "Fraudulence" is not a direction in this space — each build drifts its own way.

For scale, the INT4 W4A16 arm on Hopper — a completely different quantisation (4-bit weights,
BF16 activations, MARLIN) — measures 0.374, i.e. almost exactly this arm's 0.372 average.
Distance does not identify the scheme either.

### Inference

Warm-engine numbers, throughput recomputed from `measurements`.

| Scenario | honest FP8 | this arm | Δ |
|---|---:|---:|---:|
| s1 long, sequential | 84.2 | 97.1 | +15 % |
| s2 short, 30 runners | 1183.2 | 1264.9 | +7 % |
| s3 very long, sequential | 84.8 | 107.3 | **+27 %** |
| s4 very long, 20 runners | 369.3 | **566.8** | **+53 %** |

### Resources and bring-up

| Arm | weights/rank | KV tokens | bringup |
|---|---:|---:|---:|
| honest FP8 | 69.28 GiB | 2 046 704 | 293 s |
| **this arm** | **39.99 GiB** (160 GiB total) | 2 868 080 | **819 s** |
| ModelOpt build (after this one) | — | 2 816 208 | 159 s |

**819 s with only 75 s of compilation** — the remainder is FlashInfer FP4 autotuning
(`enable_flashinfer_autotune=True`). The next FP4 arm on the same box booted in 159 s
because the JIT cache was warm. Practical rule: **allow ≥20 minutes** before declaring an
FP4 bring-up hung. During the wait GPU utilisation is 0 % while all workers sit at ~95 %
CPU — that is compilation, not a deadlock.

## Findings

1. **The fingerprint belongs to the build, not the scheme.** 0.367 here vs 0.491 for the
   ModelOpt build of the same shape, both against the same honest reference on the same host.
2. **Detectors must measure distance from honest, not similarity to known frauds** — the two
   frauds are 0.52 apart, further than either is from honest.
3. **Calibrate thresholds against the quietest known build.** Today that is this one; a gate
   tuned on the ModelOpt build would let it through.
4. **Distance does not identify the quantisation either** — INT4 W4A16 on Hopper lands at
   0.374, indistinguishable in magnitude from this NVFP4 build.
5. **`mixed-precision` loads on vLLM 0.25.1 unmodified**, contrary to what a reading of
   vLLM's own source alone suggests — the format enum lives in `compressed-tensors 0.17.0`.
6. **FP4 bring-up is dominated by autotuning**, not compilation: 819 s cold, 159 s warm.

## Files

```
artifacts/
  nonces_b200_rh_{s1,s1_r2,s2,s3}.json                this arm (s1_r2 = repeat of s1)
  ref_nonces_b200_fp8_{s1,s1_r2,s2,s3}.json           honest reference, same host
  other_fraud_nonces_b200_nvfp4_modelopt_s1.json      the other NVFP4 build, for fraud-vs-fraud
  sweep_b200_redhat_120s.log
  serving_b200_redhat.sqlite
scripts/
  patch_hy3.py  run_pow_generation.py  collect_artifacts.py  l2_matrix.py  poc_seeds.json
```

## Reproducibility checklist

- [x] Image pinned by digest; quantisation transcribed from `config.json` group by group
- [x] Honest reference and a competing fraud build committed here — folder is self-contained
- [x] Loader/kernel selection quoted from the engine log, not inferred
- [x] All scripts committed under `scripts/`
- [x] 120 s window for every quoted throughput number
- [x] L2 tables reproducible via `scripts/l2_matrix.py`
- [x] 3 seeds behind every fingerprint claim
- [x] No internal-tooling links, absolute paths or sibling-repo references
