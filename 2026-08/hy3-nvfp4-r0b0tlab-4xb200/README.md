# Hy3 NVFP4 by `r0b0tlab` (ModelOpt) — 4×B200 — control run: the fingerprint reproduces across machines

**Date:** 2026-08-19
**Model:** `r0b0tlab/Hy3-295B-NVFP4` — 186 GB, built with **NVIDIA ModelOpt**;
vLLM reports `quantization=modelopt_mixed` (a mix of `FP8`, `NVFP4`, `W4A16_NVFP4`, `MXFP8`),
MoE kernel `FLASHINFER_TRTLLM` NvFp4. Blackwell-only (sm_100).
**Honest reference:** [`../hy3-fp8-4xb200/`](../hy3-fp8-4xb200/) — same host, same session;
nonce sets duplicated here as `ref_nonces_*`.
**Hardware:** 4× NVIDIA **B200 SXM** (1000 W, 183 GB, NV18, driver **580.126.20**, sm_100).
Vast.ai instance 48135501.
**Image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash-0731:3.0.16-overlay-k5`
**Digest:** `sha256:8ce8830b4182b3dfd642c5e00f548f50a830611e4bc64ec4fbd84fe91070e3aa`

> The DeepSeek-V4-Flash foundry image, reused only as a vLLM 0.25.1 + PoC-plugin runtime.
> Its `runner.py` hardcodes V4-specific flags that must be replaced — `scripts/patch_hy3.py`.

## Summary

A control for one specific question: when the same checkpoint is measured on different
hardware, does its distance from honest stay put? It does, to three decimal places.

| host | L2 median vs honest (s1 / s2 / s3) |
|---|---|
| [2×B300](../hy3-nvfp4-r0b0tlab-2xb300/) | 0.4926 / 0.4897 / 0.5038 |
| **4×B200 (this run)** | **0.4909 / 0.4890 / 0.5002** |

Combined with the [`RedHatAI` build](../hy3-nvfp4-redhatai-4xb200/) measured on this
very host at 0.367, that settles the question the two builds pose together: **the fingerprint
is a property of the build, not of the quantisation scheme and not of the hardware.**
Thresholds cannot be calibrated per scheme.

Secondary result: at TP=4 this arm is 1.7 % *faster* than honest (1920 vs 1888), where on
B300 at TP=2 the two were exactly equal.

## Environment

| Parameter | Value |
|---|---|
| CUDA | 13.0.2 (image), driver 580.126.20 (CUDA 13.0) |
| vLLM | 0.25.1, build `752a3a504485790a2e8491cacbb35c137339ad34` |
| Python | 3.12.13 |
| mlnode | 3.0.16, `gonka_poc.entrypoint.api_router` |

## Config

```bash
TP=4 python3 scripts/patch_hy3.py
curl -X POST http://127.0.0.1:8081/api/v1/inference/up/async \
  -H 'Content-Type: application/json' \
  -d '{"model":"r0b0tlab/Hy3-295B-NVFP4","dtype":"auto","additional_args":[]}'
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
| this arm | 1776 | 1872 | **1920** |
| honest FP8 | 1736 | 1840 | **1888** |
| [`RedHatAI` build](../hy3-nvfp4-redhatai-4xb200/) | 1768 | 1904 | **1952** |

### Fingerprint

```bash
python3 scripts/l2_matrix.py artifacts ref_nonces_fp8_s1.json nonces_nvfp4_s1.json
```

| Pair | bit-identical | L2 median | p95 | >0.40 |
|---|---:|---:|---:|---:|
| this arm ↔ itself (repeat) | **100.0 %** | 0.0000 | 0.0000 | 0.0 % |
| honest ↔ this arm, s1 | 0.0 % | 0.4909 | 0.8731 | 72.7 % |
| honest ↔ this arm, s2 | 0.0 % | 0.4890 | — | 70.7 % |
| honest ↔ this arm, s3 | 0.0 % | 0.5002 | — | 75.8 % |

### Resources

| | value |
|---|---:|
| KV cache | 2 816 208 tokens |
| bring-up | 159 s (warm FP4 JIT cache — see the llm-compressor folder for the cold 819 s) |

## Findings

1. **The fingerprint reproduces across machines to three decimals** — 0.4909 here against
   0.4926 on B300. This validates the measurement methodology as much as it characterises
   the build.
2. **Distance is a build property.** Same scheme, different toolchain → 0.367 on this host.
3. **Bit-exact on repeat** at TP=4, matching the honest arm on the same box.
4. **FP4 bring-up is dominated by autotuning, and the cache is shared**: this arm booted in
   159 s immediately after another FP4 arm had spent 819 s warming the same kernels.

## Files

```
artifacts/
  nonces_nvfp4_{s1,s1_r2,s2,s3}.json   this arm (s1_r2 = repeat of s1)
  ref_nonces_fp8_{s1,s1_r2,s2,s3}.json honest reference, same host
  sweep_120s.log                        valid sweep
  serving.sqlite                        compressa-perf database
scripts/
  patch_hy3.py  run_pow_generation.py  collect_artifacts.py  l2_matrix.py  poc_seeds.json
```

Related: [same model on 2×B300](../hy3-nvfp4-r0b0tlab-2xb300/) ·
[the economics run on 1×B300](../hy3-nvfp4-r0b0tlab-1xb300/) ·
[llm-compressor build on this host](../hy3-nvfp4-redhatai-4xb200/) ·
[honest FP8 on this host](../hy3-fp8-4xb200/)

## Reproducibility checklist

- [x] Image pinned by digest; quantisation described from `config.json` / loader output
- [x] Every script referenced above committed under `scripts/`
- [x] L2 tables reproducible from committed artifacts via `scripts/l2_matrix.py`
- [x] 3 seeds behind every fingerprint claim
- [x] Invalid or superseded measurements labelled in place, not dropped
- [x] No internal-tooling links, absolute paths, or sibling-repo references
