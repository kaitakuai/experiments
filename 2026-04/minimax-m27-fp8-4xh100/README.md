# PoC Benchmark: MiniMax-M2.7 (FP8) on 4×H100

**Date:** 2026-04-15
**Model:** `MiniMaxAI/MiniMax-M2.7`
**Quantization:** FP8 block-wise 128×128 (E4M3, dynamic activations)
**Hardware:** 4×NVIDIA H100 80GB HBM3
**vLLM:** 0.19.0

## Summary

PoC v2 benchmark on 4×H100 (TP=4). Default backends: FLASH_ATTN attention + FLASHINFER_CUTLASS FP8 MoE. H100 is Hopper architecture (SM 90) with native FP8 support.

## Hardware

| Parameter | Value |
|-----------|-------|
| GPU | 4× NVIDIA H100 80GB HBM3 (81,559 MiB each) |
| vLLM | 0.19.0 |
| `/dev/shm` | 442 GB |

## Backend

- **Attention:** FLASH_ATTN
- **MoE:** FLASHINFER_CUTLASS Fp8 MoE

## Results

| Batch Size | Nonces (30s) | Nonces/min |
|-----------:|-------------:|-----------:|
| 8 | 744 | 1,488 |
| 16 | 800 | 1,600 |
| **32** ★ | **832** | **1,664** |
| 64 | 0 | 0 (OOM) |
| 128 | 0 | 0 (OOM) |

**Best: batch=32 → 1,664 nonces/min**

## Nonce collection

Collected **1,056 nonces** at **1,508 nonces/min** (batch_size=16).

## Cross-GPU comparison

| Pair | L2 mean |
|------|---------|
| H100 vs A100 (FP8) | 0.391 |
| H100 vs H200 (FP8) | 0.503 |
| H100 vs B200 (FP8) | **0.361** (closest) |
| H100 fraud (FP8 vs AWQ) | **0.752** |

## Artifacts

- `artifacts/nonces_1000.json` — 1000 PoC nonce vectors
- `artifacts/config.json` — benchmark configuration
