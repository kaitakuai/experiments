# PoC Benchmark: MiniMax-M2.7 AWQ-4bit (fraud model) on 2×A100 SXM4

**Date:** 2026-04-14
**Model:** `demon-zombie/MiniMax-M2.7-AWQ-4bit`
**Quantization:** AWQ INT4 (compressed-tensors, W4A16)
**Hardware:** 2×NVIDIA A100-SXM4-80GB (TP=2)
**vLLM:** 0.19.0
**Purpose:** Fraud detection — frauder running cheaper quantization on fewer GPUs.

## Summary

Fraud scenario: attacker uses AWQ-4bit quantization on 2×A100 instead of honest FP8 on 4×A100. Saves ~75% on hardware (2 GPUs vs 4, cheaper quant). PoC fraud detection validates by comparing nonce vectors.

## Hardware

| Parameter | Value |
|-----------|-------|
| GPU | 2× NVIDIA A100-SXM4-80GB |
| NVIDIA Driver | 590.48.01 |
| CPU | 256 vCPUs |
| RAM | 1.0 TiB |

## Reproduction

### 1. Download model

```bash
HF_HOME=/dev/shm/hf python3 -c \
  'from huggingface_hub import snapshot_download; snapshot_download("demon-zombie/MiniMax-M2.7-AWQ-4bit", max_workers=16)'
```

Model: 120 GB (compressed-tensors AWQ).

### 2. Start vLLM (TP=2)

```bash
# Limit to 2 GPUs
CUDA_VISIBLE_DEVICES=0,1

curl -X POST http://127.0.0.1:8081/api/v1/inference/up/async \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "'"$MODEL_PATH"'",
    "dtype": "auto",
    "additional_args": [
      "--served-model-name", "demon-zombie/MiniMax-M2.7-AWQ-4bit",
      "--tensor-parallel-size", "2",
      "--gpu-memory-utilization", "0.92",
      "--max-num-seqs", "128",
      "--max-model-len", "32768",
      "--trust-remote-code"
    ]
  }'
```

Note: `--max-model-len 32768` required — 131072 exceeds KV cache at TP=2 (needs 15.5 GiB, only 13.49 available).

### 3. Collect nonces

```bash
python3 -u collect_artifacts.py \
  --url http://127.0.0.1:5001 \
  --model 'demon-zombie/MiniMax-M2.7-AWQ-4bit' \
  --output-dir /tmp/artifacts \
  --nonces 1000 --batch-size 16 --logprobs-count 0
```

## Nonce collection

Collected **1,008 nonces** at **593 nonces/min** (batch_size=16, 102 seconds).

## Fraud detection: FP8 (TP=4) vs AWQ-4bit (TP=2)

### L2 Distances

| Pair | L2 mean | Identical | Detectable? |
|------|---------|-----------|-------------|
| **FP8 TP=4 vs AWQ TP=2** | **0.7542** | 0/1000 | **Yes (100%)** |
| Cross-GPU honest (reference) | ~0.07 | — | N/A |

All 1,000 nonces above threshold 0.2 — **100% fraud detection rate**.

### Fraud economics

| | Honest (FP8 TP=4) | Fraud (AWQ TP=2) |
|--|---|---|
| GPUs | 4× A100 | 2× A100 |
| Model memory/GPU | 54.58 GiB | ~56 GiB |
| Nonces/min | 864 | 593 |
| Hardware cost | 4× | **2×** (50% cheaper) |
| PoC detected? | — | **Yes (L2=0.75)** |

Frauder saves 50% on GPUs but is clearly detectable by PoC.

## Artifacts

- `artifacts/nonces_1000.json` — 1000 fraud nonce vectors (AWQ-4bit, TP=2)

## Key observations

- **Fraud clearly detectable** — L2 = 0.75, all nonces above threshold
- **TP mismatch also contributes** — same AWQ model at TP=4 vs TP=2 gives L2 = 0.30 (different float reduction order)
- **max-model-len reduced** to 32768 at TP=2 due to limited KV cache (13.49 GiB)
- **593 nonces/min** — slower than honest (864) despite cheaper model, because TP=2 has less parallelism
