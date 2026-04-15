# PoC Benchmark: MiniMax-M2.7 (FP8) on 2×H200

**Date:** 2026-04-15
**Model:** `MiniMaxAI/MiniMax-M2.7`
**Quantization:** FP8 block-wise 128×128 (E4M3, dynamic activations)
**Hardware:** 2×NVIDIA H200 141GB
**vLLM:** 0.19.0

## Summary

PoC v2 benchmark for MiniMax-M2.7 FP8 on 2×H200 (TP=2). H200 has 141 GiB VRAM — enough for this 230 GB model at TP=2 (~107 GiB/GPU). FLASHINFER_CUTLASS FP8 MoE backend (not available on A100).

## Hardware

| Parameter | Value |
|-----------|-------|
| GPU | 2× NVIDIA H200 (143,771 MiB / ~141 GiB each) |
| NVIDIA Driver | (Vast.ai managed) |
| CPU | (Vast.ai) |
| RAM | (Vast.ai) |
| `/dev/shm` | 503 GB |

## Software

| Component | Version |
|-----------|---------|
| vLLM | 0.19.0 |
| torch | 2.10.0+cu129 |
| Python | 3.12.13 |
| OS | Ubuntu 22.04.5 LTS |

## Reproduction

### 1. Download model

```bash
mkdir -p /dev/shm/hf
HF_HOME=/dev/shm/hf python3 -c \
  'from huggingface_hub import snapshot_download; snapshot_download("MiniMaxAI/MiniMax-M2.7", max_workers=16)'
```

### 2. Start vLLM (TP=2)

```bash
# Limit to 2 GPUs in runner.py:
# env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0,1")

curl -X POST http://127.0.0.1:8081/api/v1/inference/up/async \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "'"$MODEL_PATH"'",
    "dtype": "auto",
    "additional_args": [
      "--served-model-name", "MiniMaxAI/MiniMax-M2.7",
      "--tensor-parallel-size", "2",
      "--gpu-memory-utilization", "0.92",
      "--max-num-seqs", "128",
      "--max-model-len", "131072",
      "--trust-remote-code"
    ]
  }'
```

### 3. Run benchmark & collect nonces

```bash
export HOST_IP=127.0.0.1
python3 -u run_pow_generation.py --phase 3 --skip-check

python3 -u collect_artifacts.py \
  --url http://127.0.0.1:5001 \
  --model 'MiniMaxAI/MiniMax-M2.7' \
  --output-dir /tmp/artifacts \
  --nonces 1000 --batch-size 16 --logprobs-count 0
```

## Startup profile

| Phase | Duration |
|-------|----------|
| Model loading (from `/dev/shm`) | 40.68 s |
| Model memory/GPU | 107.3 GiB |
| Available KV cache | 16.58 GiB (140,192 tokens) |
| **Total cold start** | **~6 min 45 s** |

## Backend

- **FP8 Linear:** (auto-selected for H200)
- **Attention:** FLASH_ATTN
- **MoE:** FLASHINFER_CUTLASS Fp8 MoE
- **Custom fusions:** norm_quant, act_quant

## Results

| Batch Size | Nonces (30s) | Nonces/min |
|-----------:|-------------:|-----------:|
| 8 | 592 | 1,184 |
| **16** ★ | **640** | **1,279** |
| 32 | 640 | 1,279 |
| 64 | 0 | 0 (OOM) |
| 128 | 0 | 0 (OOM) |

**Best: batch=16/32 → 1,279 nonces/min**

## Nonce collection

Collected **1,056 nonces** at **1,218 nonces/min** (batch_size=16, 52 seconds).

## Comparison: MiniMax-M2.7 FP8 across GPUs

| Hardware | TP | Nonces/min | Model/GPU | KV cache | MoE backend |
|----------|---:|------------|-----------|----------|-------------|
| 4×A100 SXM4 80GB | 4 | 864 | 54.58 GiB | 13.66 GiB | MARLIN |
| **2×H200 141GB** | **2** | **1,279** | **107.3 GiB** | **16.58 GiB** | **FLASHINFER_CUTLASS** |
| 4×RTX PRO 6000 96GB | 4 | 848 | — | — | TRITON |

H200 at TP=2 is **48% faster** than A100 at TP=4, using **half the GPUs**. Key advantage: FLASHINFER_CUTLASS FP8 MoE backend (not available on A100, which falls back to MARLIN).

## Artifacts

- `artifacts/nonces_1000.json` — 1000 PoC nonce vectors
- `artifacts/config.json` — benchmark configuration

## Key observations

- **1,279 nonces/min on 2×H200** — fastest MiniMax-M2.7 result, beating 4×A100 (864/min) with half the GPUs
- **FLASHINFER_CUTLASS** MoE backend on H200 vs MARLIN on A100 — significant throughput difference
- **107.3 GiB/GPU** at TP=2 — tight fit on H200 (141 GiB), only 16.58 GiB KV cache remaining
- **OOM at batch≥64** — same as A100, limited by KV cache headroom for PoC scratch tensors
- Batch=16 and batch=32 yield identical throughput (1,279/min)
