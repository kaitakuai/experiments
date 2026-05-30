# Qwen3-235B-A22B-Instruct-2507-FP8 — vLLM 0.20.0 @ 240k context — PoC v2 benchmark

**Date:** 2026-05-30
**Model:** `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` (FP8 block-scaled, 94 layers, GQA 64q/4kv, 262144 native ctx)
**Hardware:** 4× H100 80GB SXM (Vast.ai, instance 38587372, machine 136133)
**vLLM:** 0.20.0 (torch 2.11.0+cu130, driver 580.159.04)

## Summary

Investigated why this model at `--max-model-len 240000` fits on vLLM 0.15.1 but fails to start on 0.20.0 with identical prod flags, found the root cause, restored a working 240k config without reducing context, and ran the PoC v2 nonce benchmark. On 0.20.0 the model **refuses to start** at 240k (`Available KV cache 4.15 GiB < 10.76 GiB needed`). Root cause: the **prefill MoE activation buffer scales with `--max-num-batched-tokens`** and at the prod value `65536` consumes ~12.8 GiB/GPU, leaving too little for KV. Fix without touching context: **lower `--max-num-batched-tokens` and/or raise `--gpu-memory-utilization`**. Best PoC throughput at 240k: **batch 32 → 1216 nonces/min** (TRITON MoE).

## Hardware

| Item | Value |
| --- | --- |
| GPUs | 4× NVIDIA H100 80GB HBM3 (SXM) |
| Driver | 580.159.04 (CUDA 13 native) |
| CPU / RAM | 104 vCPU / 885 GB |
| /dev/shm | 442 GB (model loaded from TMPFS) |
| Net (down) | 13.3 Gbps |

## Software

| Item | Value |
| --- | --- |
| Image | `ghcr.io/kaitakuai/mlnode-h100-minimax-m2-7:0.2.13-vllm0.20.0-k1` |
| vLLM | 0.20.0 |
| torch | 2.11.0+cu130 |
| Python | 3.12.13 |
| MoE backend | TRITON (forced: `VLLM_USE_FLASHINFER_MOE_FP8=0`) |
| Attention backend | FLASHINFER (Hopper default in 0.20) |

## The 240k memory regression (0.15.1 → 0.20.0)

With identical prod flags (`-tp 4 --gpu-memory-utilization 0.92 --max-num-seqs 128 --max-model-len 240000 --max-num-batched-tokens 65536 --enable-expert-parallel --disable-custom-all-reduce --num-gpu-blocks-override 15000`), 0.20.0 refuses to start:

```
ValueError: To serve at least one request with the model's max seq len (240000),
10.76 GiB KV cache is needed, which is larger than the available KV cache memory (4.15 GiB).
```

The strict V1 check `_check_enough_kv_cache_memory` fires **before** `--num-gpu-blocks-override` is applied, so the override is moot.

### Measured per-GPU memory (gpu-mem-util 0.92, budget ≈ 73.3 GiB)

| Consumer | `maxnbt 65536` | `maxnbt 8192` |
| --- | --- | --- |
| Model weights | 55.22 GiB | 55.22 GiB |
| Prefill activation + fixed workspace | **12.78 GiB** | **4.08 GiB** |
| CUDA-graph reserve (#38284, default-on) | 1.10 GiB | (disabled in test) |
| **Available for KV** | **4.15 / 5.25 GiB** | **13.95 GiB** |
| 240k fits? | ❌ (need 10.76) | ✅ |

**Root cause:** lowering `--max-num-batched-tokens` 65536→8192 frees **8.7 GiB** — i.e. the full 65536-token MoE prefill activation. On 0.15.1 the MoE forward was internally chunked (PR #34086 removed FusedMoE kernel chunking in 0.18; #39107 removed MoE-DP chunking in 0.20), so the per-pass peak stayed small even at maxnbt=65536. 0.20 materializes the whole pass. Minor co-contributors: CUDA-graph memory profiling on-by-default (+1.10 GiB, PR #38284) and FlashInfer/DeepGEMM/TRTLLM-allreduce fixed workspaces (~2.8 GiB). These are deliberate trade-offs (safer memory accounting, faster kernels, simpler MoE code) that only bite this exact-fit case.

### Working 240k configs (context preserved)

| Goal | Change vs prod | Result |
| --- | --- | --- |
| Just fit 240k | `--max-num-batched-tokens 8192` (+ optional `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`) | Available KV 13.95 GiB, fits; **caps PoC batch at 8** (8×1024=8192) |
| 240k **and** PoC batch 32 | `--max-num-batched-tokens 32768` + `--gpu-memory-utilization 0.96` | Available KV **12.15 GiB**, fits; batch 32 works |

Batch 64 (65536 tokens/forward) cannot coexist with 240k unless `--kv-cache-dtype fp8` (halves KV need) — but that changes nonce numerics, so not used for L2.

## Startup profile (240k + maxnbt 32768 + gmu 0.96, TRITON)

```
Using AttentionBackendEnum.FLASHINFER backend
Using TRITON Fp8 MoE backend
Model loading took 55.22 GiB memory and 20.3s
Estimated CUDA graph memory: 1.10 GiB total
Available KV cache memory: 12.15 GiB
GPU KV cache size: 240,000 tokens
Maximum concurrency for 240,000 tokens per request: 1.00x
Graph capturing finished in 18 secs, took 0.96 GiB
init engine took 70.62 s (compilation: 42.18 s)
```

## PoC v2 benchmark results (240k, TRITON MoE)

| Batch | Nonces/min |
| ---: | ---: |
| 2 | 636 |
| 8 | 1120 |
| 16 | 1184 |
| 32 | **1216** ★ |

- Best batch size: **32 → 1216 nonces/min** (script default is 8; recommend updating `PHASE1_BATCH_SIZE`).
- `max-num-batched-tokens` is the PoC batch ceiling: batch N needs `maxnbt ≥ N×1024` in one `execute_poc_forward` (no chunking). With `maxnbt 8192`, batch ≥ 16 times out (`RPC call to execute_poc_forward timed out`) — not an OOM.

## Nonce collection

- **1000 PoC nonces** collected at batch 32 in 51s (~1181/min), TRITON MoE → `artifacts/nonces_1000.json`.
- 5-language inference probe (sp/en/zh/ar/hi) all correct ("capital of France → Paris") → `artifacts/inference_5langs.json`.
- Config → `artifacts/config.json`.

## Reproduction

```bash
# model already in /dev/shm; mlnode running on 8081 with VLLM_USE_FLASHINFER_MOE_FP8=0
SNAP=/dev/shm/hf/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/<hash>
curl -s -X POST http://127.0.0.1:8081/api/v1/inference/up/async -H 'Content-Type: application/json' -d '{
  "model":"'$SNAP'","dtype":"auto","additional_args":[
    "--served-model-name","Qwen3-235B-A22B-Instruct-2507-FP8",
    "--tensor-parallel-size","4","--max-num-batched-tokens","32768",
    "--gpu-memory-utilization","0.96","--max-num-seqs","128",
    "--max-model-len","240000","--enable-expert-parallel",
    "--disable-custom-all-reduce","--num-gpu-blocks-override","15000","--trust-remote-code"]}'
# then: HOST_IP=127.0.0.1 python3 run_pow_generation.py --phase 3 --skip-check   (MLNODE_URL=8081, BATCH_SIZES=[2,8,16,32])
```

## Key observations

1. The 240k OOM on 0.20 is the **prefill MoE activation** at `maxnbt 65536`, not weights or KV math. Measured, not inferred.
2. The fix does **not** reduce capability — context (240k), quality, and decode are unchanged; only prefill chunk size changes (re-creating 0.15.1's implicit MoE chunking).
3. `maxnbt` is simultaneously the **240k-fit lever** and the **PoC-batch ceiling** — they trade off. `maxnbt 32768 + gmu 0.96` satisfies both (240k + batch 32).
4. `VLLM_ATTENTION_BACKEND` is an **unknown env var in 0.20** (renamed) — forcing FA3 that way is a silent no-op.
5. FP8 MoE oracle picks **DEEPGEMM** by default here (not FlashInfer-CUTLASS); forced to **TRITON** for cross-validatable nonces.
