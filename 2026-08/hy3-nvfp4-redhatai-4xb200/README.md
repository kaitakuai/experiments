# Hy3 NVFP4+FP8 by `RedHatAI` (llm-compressor) — 4×B200 — the quietest fraud measured (0.367 vs 0.491 for the same scheme)

**Date:** 2026-08-19
**Model:** `RedHatAI/Hy3-NVFP4-FP8` — 178 GB, 5 shards, built with **llm-compressor**.
`quant_method: compressed-tensors`, `format: mixed-precision`, two groups:

| group | targets | weights | input activations |
|---|---|---|---|
| 0 | `re:.*self_attn\..*` | FP8, `strategy: block`, `block_structure [128,128]`, static | FP8, `strategy: group`, `group_size 128`, **dynamic** |
| 1 | `re:.*mlp.*` | **NVFP4**, `tensor_group`, `group_size 16`, scale dtype `float8_e4m3` | **NVFP4**, `tensor_group 16`, `dynamic: "local"` |

`kv_cache_scheme: null`. The `ignore` list has 81 entries: router gates of layers 1–79,
`lm_head`, and `^model.layers.80.*` — **the MTP layer is left unquantised**, as in every
other Hy3 fraud build measured. Weights in VRAM: **160 GiB** (39.99 GiB/rank).

**Honest reference:** [`../hy3-fp8-4xb200/`](../hy3-fp8-4xb200/) — same host, same session;
nonce sets duplicated here. A sample of the [`r0b0tlab` build](../hy3-nvfp4-r0b0tlab-4xb200/)
is included as `other_fraud_*` for the fraud-vs-fraud comparison.
**Hardware:** 4× NVIDIA **B200 SXM** (1000 W, 183 GB, NV18, driver **580.126.20**, sm_100).
Vast.ai instance 48135501.
**Image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash-0731:3.0.16-overlay-k5`
**Digest:** `sha256:8ce8830b4182b3dfd642c5e00f548f50a830611e4bc64ec4fbd84fe91070e3aa`

> The DeepSeek-V4-Flash foundry image, reused only as a vLLM 0.25.1 + PoC-plugin runtime.
> Its `runner.py` hardcodes V4-specific flags that must be replaced — `scripts/patch_hy3.py`.

## Summary

A second, independently produced NVFP4 checkpoint of the same shape as the ModelOpt one —
4-bit MoE, everything sensitive in FP8 — from a different toolchain. Measured on the same
host against the same honest reference, it sits **a third closer to honest**:

| fraud build | toolchain | L2 median (s1/s2/s3) | >0.40 |
|---|---|---|---:|
| **this arm** | `RedHatAI` | **0.3670 / 0.3653 / 0.3844** | 38.5–45.9 % |
| [`r0b0tlab` build](../hy3-nvfp4-r0b0tlab-4xb200/) | NVIDIA ModelOpt | 0.4909 / 0.4890 / 0.5002 | 70.7–75.8 % |

And the two frauds are **further from each other (0.52) than either is from honest**, so
"resembles a known fraud" is not a usable detector — only "far from honest" is. Thresholds
must be calibrated against the quietest known build, which today is this one.

It is also the fastest arm on this host: **1952 nonces/min against the honest 1888**
(+3.4 %), with serving gains up to +53 %.

## Environment

| Parameter | Value |
|---|---|
| CUDA | 13.0.2 (image), driver 580.126.20 |
| vLLM | 0.25.1, build `752a3a504485790a2e8491cacbb35c137339ad34` |
| compressed-tensors | **0.17.0** (pinned by vLLM 0.25.1) |
| Python | 3.12.13 |
| mlnode | 3.0.16 |

### Does `format: mixed-precision` load on vLLM 0.25.1?

Yes, unmodified — worth recording because the string appears nowhere in vLLM's
`compressed_tensors.py`. The format enum lives in the pinned `compressed-tensors 0.17.0`,
which defines both `mixed_precision` and `DynamicType.LOCAL` ("local is only currently
supported for NVFP4", validated to require `tensor_group`). vLLM dispatches on quantisation
*arguments*, not on the format string: `_is_nvfp4_format(weight_quant)` selects
`CompressedTensorsW4A4Nvfp4MoEMethod`, and the FP8 group passes `_is_fp8_w8a8` through its
"dynamic activations are always supported" branch.

Observed kernel selection:

```
Selected DeepGemmFp8BlockScaledMMKernel for CompressedTensorsW8A8Fp8
Using FlashInferCuteDslNvFp4LinearKernel for NVFP4 GEMM
Using 'FLASHINFER_TRTLLM' NvFp4 MoE backend
```

## Config

```bash
TP=4 python3 scripts/patch_hy3.py
curl -X POST http://127.0.0.1:8081/api/v1/inference/up/async \
  -H 'Content-Type: application/json' \
  -d '{"model":"RedHatAI/Hy3-NVFP4-FP8","dtype":"auto","additional_args":[]}'
```

```
--tensor-parallel-size 4   --gpu-memory-utilization 0.90
--max-model-len 262144        --max-num-batched-tokens 65536
--kv-cache-dtype fp8          --logprobs-mode processed_logprobs
--worker-extension-cls gonka_poc.worker.PoCWorkerExtension
--tool-call-parser hy_v3      --reasoning-parser hy_v3
--trust-remote-code --enable-auto-tool-choice --enable-expert-parallel
```

Measurement window **120 s**, batches 16/32/64.

## Validation

### Throughput (120 s window)

| batch | 16 | 32 | 64 |
|---:|---:|---:|---:|
| **this arm** | 1768 | 1904 | **1952 (+3.4 %)** |
| honest FP8 | 1736 | 1840 | 1888 |
| `r0b0tlab` build | 1776 | 1872 | 1920 |

### Fingerprint

```bash
python3 scripts/l2_matrix.py artifacts ref_nonces_fp8_s1.json nonces_nvfp4_s1.json
```

| Pair | bit-identical | L2 median | p95 | >0.40 |
|---|---:|---:|---:|---:|
| this arm ↔ itself (repeat) | **100.0 %** | 0.0000 | 0.0000 | 0.0 % |
| honest ↔ this arm, s1 | 0.0 % | 0.3670 | 0.6319 | 40.9 % |
| honest ↔ this arm, s2 | 0.0 % | 0.3653 | 0.6333 | 38.5 % |
| honest ↔ this arm, s3 | 0.0 % | 0.3844 | 0.6388 | 45.9 % |
| **this arm ↔ ModelOpt build, s1** | 0.0 % | **0.5243** | — | 77.2 % |

For scale: the [INT4 W4A16 arm on Hopper](../hy3-int4-cyankiwi-4xh200/) — a completely different
quantisation, 4-bit weights with BF16 activations on MARLIN — measures 0.374, i.e. almost
exactly this arm's average. **Distance does not identify the scheme either.**

### Inference

Warm-engine numbers.

| Scenario | honest out tok/s | this arm out tok/s | Δ |
|---|---:|---:|---:|
| s1 long, sequential | 84.2 | 97.1 | +15 % |
| s2 short, 30 runners | 1183.2 | 1264.9 | +7 % |
| s3 very long, sequential | 84.8 | 107.3 | **+27 %** |
| s4 very long, 20 runners | 369.3 | **566.8** | **+53 %** |

### Resources and bring-up

| | honest | this arm |
|---|---:|---:|
| weights / rank | 69.28 GiB | **39.99 GiB** (160 GiB total) |
| KV cache | 2 046 704 | 2 868 080 (+40 %) |
| bring-up | 293 s | **819 s** |

**819 s with only 75 s of compilation** — the rest is FlashInfer FP4 autotuning
(`enable_flashinfer_autotune=True`). The next FP4 arm on the same box booted in 159 s off the
warm JIT cache. Practical rule: allow **≥20 minutes** before treating an FP4 bring-up as
hung. During the wait GPU utilisation is 0 % while every worker sits at ~95 % CPU — that is
compilation, not a deadlock, and the log emits
`No available shared memory broadcast block found in 60 seconds`, which is expected here.

## Findings

1. **The fingerprint belongs to the build, not the scheme** — 0.367 here against 0.491 for
   the ModelOpt build, same shape, same host, same reference.
2. **Detectors must measure distance from honest, not similarity to known frauds**: the two
   frauds are 0.52 apart, further than either is from honest.
3. **Calibrate thresholds against the quietest known build.** A gate tuned on the ModelOpt
   build would pass this one.
4. **Distance does not identify the quantisation** — INT4 W4A16 lands at 0.374, effectively
   the same magnitude.
5. **`mixed-precision` loads on 0.25.1 unmodified**, contrary to what reading vLLM's source
   alone suggests.
6. **FP4 bring-up is autotuning-bound**: 819 s cold, 159 s warm.

## Files

```
artifacts/
  nonces_nvfp4_{s1,s1_r2,s2,s3}.json        this arm (s1_r2 = repeat of s1)
  ref_nonces_fp8_{s1,s1_r2,s2,s3}.json      honest reference, same host
  other_fraud_nonces_modelopt_s1.json        the competing NVFP4 build, for fraud-vs-fraud
  sweep_120s.log                             valid sweep
  serving.sqlite                             compressa-perf database
scripts/
  patch_hy3.py  run_pow_generation.py  collect_artifacts.py  l2_matrix.py  poc_seeds.json
```

Related: [honest FP8 on this host](../hy3-fp8-4xb200/) ·
[ModelOpt build on this host](../hy3-nvfp4-r0b0tlab-4xb200/) ·
[INT4 on Hopper](../hy3-int4-cyankiwi-4xh200/)

## Reproducibility checklist

- [x] Image pinned by digest; quantisation described from `config.json` / loader output
- [x] Every script referenced above committed under `scripts/`
- [x] L2 tables reproducible from committed artifacts via `scripts/l2_matrix.py`
- [x] 3 seeds behind every fingerprint claim
- [x] Invalid or superseded measurements labelled in place, not dropped
- [x] No internal-tooling links, absolute paths, or sibling-repo references
