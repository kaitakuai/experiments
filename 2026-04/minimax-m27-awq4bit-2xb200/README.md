# PoC Benchmark: MiniMax-M2.7 AWQ-4bit (fraud) on 2×B200

**Date:** 2026-04-15
**Model:** `demon-zombie/MiniMax-M2.7-AWQ-4bit`
**Quantization:** AWQ INT4 (compressed-tensors, W4A16)
**Hardware:** 2×NVIDIA B200 183GB (TP=2)
**vLLM:** 0.19.0
**Purpose:** Fraud model on B200.

## Nonce collection

| Config | Nonces/min |
|--------|-----------|
| Default | 1,181 |
| TRITON forced | 1,145 |

## Fraud detection: FP8 vs AWQ on B200

| Config | L2 mean |
|--------|---------|
| Default (FLASHINFER_TRTLLM) | **0.740** |
| TRITON forced | **0.738** |

Fraud clearly detectable on both backends.

## Artifacts

- `artifacts/nonces_1000.json` — 1000 fraud nonces (default backend)
- `artifacts/nonces_triton.json` — 1000 fraud nonces (TRITON forced)
- `artifacts/config.json` — configuration
