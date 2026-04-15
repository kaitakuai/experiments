# PoC Benchmark: MiniMax-M2.7 AWQ-4bit (fraud model) on 2×H200

**Date:** 2026-04-15
**Model:** `demon-zombie/MiniMax-M2.7-AWQ-4bit`
**Quantization:** AWQ INT4 (compressed-tensors, W4A16)
**Hardware:** 2×NVIDIA H200 141GB (TP=2)
**vLLM:** 0.19.0
**Purpose:** Fraud detection — cheaper quantization on same GPU type as honest.

## Summary

Fraud scenario on H200: attacker uses AWQ-4bit instead of honest FP8, same TP=2, same GPU. Saves on model quality/accuracy but not on hardware.

## Hardware

| Parameter | Value |
|-----------|-------|
| GPU | 2× NVIDIA H200 (143,771 MiB / ~141 GiB each) |
| `/dev/shm` | 503 GB |

## Reproduction

### 1. Download model

```bash
HF_HOME=/dev/shm/hf python3 -c \
  'from huggingface_hub import snapshot_download; snapshot_download("demon-zombie/MiniMax-M2.7-AWQ-4bit", max_workers=16)'
```

Model: 120 GB.

### 2. Start vLLM (TP=2)

```bash
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
      "--max-model-len", "131072",
      "--trust-remote-code"
    ]
  }'
```

Note: AWQ at TP=2 on H200 fits with `--max-model-len 131072` (unlike A100 which needed 32768).

### 3. Collect nonces

```bash
python3 -u collect_artifacts.py \
  --url http://127.0.0.1:5001 \
  --model 'demon-zombie/MiniMax-M2.7-AWQ-4bit' \
  --output-dir /tmp/artifacts \
  --nonces 1000 --batch-size 16 --logprobs-count 0
```

## Nonce collection

Collected **1,088 nonces** at **1,053 nonces/min** (batch_size=16, 62 seconds).

## Fraud detection

### L2 Distances

| Pair | L2 mean | L2 min | L2 max | Detectable? |
|------|---------|--------|--------|-------------|
| **FP8 H200 vs AWQ H200** (fraud) | **0.7285** | 0.1433 | 1.6808 | **Yes (100%)** |
| FP8 H200 vs FP8 A100 (cross-GPU) | 0.3829 | 0.0189 | 1.5359 | ⚠️ Needs careful threshold |

### Cross-GPU honest distance is high

FP8 on H200 (TP=2) vs FP8 on A100 (TP=4) gives L2 = 0.38. This is because:
- **Different TP** (2 vs 4) — different allreduce/split order → different rounding
- **Different MoE backend** (FLASHINFER_CUTLASS on H200 vs MARLIN on A100)
- **Different GPU architecture** (Hopper vs Ampere) → different float behavior

Cross-validation between H200 and A100 running the same FP8 model would need a high threshold (~0.5), which reduces fraud detection sensitivity.

### Comparison across all MiniMax experiments

| Pair | L2 mean | What it measures |
|------|---------|-----------------|
| FP8 H200 vs AWQ H200 | **0.73** | Fraud (same GPU, different quant) |
| FP8 A100 vs AWQ A100 (TP=4) | **0.73** | Fraud (same GPU, different quant) |
| FP8 A100 vs AWQ A100 (TP=2) | **0.75** | Fraud (same GPU, different quant+TP) |
| FP8 H200 vs FP8 A100 | **0.38** | Cross-GPU honest (different GPU+TP+backend) |

Fraud L2 (~0.73) is consistently ~2× higher than cross-GPU honest L2 (~0.38), providing separation for fraud detection.

## Artifacts

- `artifacts/nonces_1000.json` — 1000 fraud nonce vectors
- `artifacts/config.json` — benchmark configuration

## Key observations

- **Fraud clearly detectable** on same GPU — L2 = 0.73 (FP8 vs AWQ on H200)
- **Cross-GPU honest L2 = 0.38** — high due to TP mismatch (2 vs 4) and different MoE backends
- **AWQ on H200 is fast** — 1,053 nonces/min (vs 593 on A100 TP=2), thanks to more VRAM headroom
- **max-model-len 131072 works** on H200 (unlike A100 which needed 32768 at TP=2)
