# PoC Benchmark: MiniMax-M2.7 AWQ-4bit (fraud) on 2×H100

**Date:** 2026-04-15
**Model:** `demon-zombie/MiniMax-M2.7-AWQ-4bit`
**Quantization:** AWQ INT4 (compressed-tensors, W4A16)
**Hardware:** 2×NVIDIA H100 80GB HBM3 (TP=2)
**vLLM:** 0.19.0
**Purpose:** Fraud model on H100.

## Notes

- `--max-model-len 32768` required (131072 exceeds KV cache at TP=2 on H100 80GB)

## Nonce collection

Collected **1,040 nonces** at **1,095 nonces/min** (batch_size=16).

## Fraud detection

| Metric | Value |
|--------|-------|
| H100 FP8 vs AWQ L2 | **0.752** |

## Artifacts

- `artifacts/nonces_1000.json` — 1000 fraud nonce vectors
- `artifacts/config.json` — configuration
