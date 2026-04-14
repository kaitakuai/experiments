# PoC Benchmark: MiniMax-M2.7 (FP8) on 4×A100 SXM4

**Date:** 2026-04-14
**Model:** `MiniMaxAI/MiniMax-M2.7`
**Quantization:** FP8 block-wise 128×128 (E4M3, dynamic activations)
**Hardware:** 4×NVIDIA A100-SXM4-80GB
**vLLM:** 0.19.0

## Summary

PoC v2 nonce generation benchmark for MiniMax-M2.7 FP8 on A100. The server has 8×A100 but model was run on 4 GPUs (TP=4) using `CUDA_VISIBLE_DEVICES=0,1,2,3`. FP8 block-wise quantization with MARLIN MoE backend, FLASH_ATTN attention.

Note: TP=8 failed on A100 with `ValueError: output_size of gate's and up's weight = 192 is not divisible by weight quantization block_n = 128`. This is an A100-specific limitation — FP8 block-wise (128×128) requires dimensions divisible by 128, but MiniMax-M2.7 MoE experts have size 192 at TP=8. TP=4 works because expert dimensions are larger.

## Hardware

| Parameter | Value |
|-----------|-------|
| GPU | 4× NVIDIA A100-SXM4-80GB (81,920 MiB each) |
| NVIDIA Driver | 590.48.01 |
| CPU | 256 vCPUs |
| RAM | 1.0 TiB |
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

Model: 230 GB, 125 safetensors shards.

### 2. Install MLNode

Standard setup from `kaitakuai/rtx-pro-6000` gonka-source submodule.

Additional patches for 8-GPU server with TP=4:
```bash
# Limit vLLM to 4 GPUs (prevents MLNode from spawning 2 instances)
# In runner.py after env = os.environ.copy():
env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")

# A100 doesn't support FlashInfer FP8 MoE
env["VLLM_USE_FLASHINFER_MOE_FP8"] = "0"
```

### 3. Start vLLM

```bash
curl -X POST http://127.0.0.1:8081/api/v1/inference/up/async \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "'"$MODEL_PATH"'",
    "dtype": "auto",
    "additional_args": [
      "--served-model-name", "MiniMaxAI/MiniMax-M2.7",
      "--tensor-parallel-size", "4",
      "--gpu-memory-utilization", "0.92",
      "--max-num-seqs", "128",
      "--max-model-len", "131072",
      "--trust-remote-code"
    ]
  }'
```

### 4. Run benchmark & collect nonces

```bash
# Benchmark (via vLLM direct, not MLNode proxy)
export HOST_IP=127.0.0.1
python3 -u run_pow_generation.py --phase 3 --skip-check

# Collect nonces
python3 -u collect_artifacts.py \
  --url http://127.0.0.1:5001 \
  --model 'MiniMaxAI/MiniMax-M2.7' \
  --output-dir /tmp/artifacts \
  --nonces 1000 --batch-size 16 --logprobs-count 0
```

## Startup profile

| Phase | Duration |
|-------|----------|
| Model loading (from `/dev/shm`) | 50.44 s |
| Model memory/GPU | 54.58 GiB |
| Dynamo bytecode transform | 2.97 s |
| torch.compile total | 11.08 s |
| CUDA graph capture (PIECEWISE=35, FULL=19) | ~2 s |
| Available KV cache | 13.66 GiB (231,088 tokens) |
| **Total cold start** | **~2 min 5 s** |

## Backend

- **FP8 Linear:** MarlinFP8ScaledMMLinearKernel
- **Attention:** FLASH_ATTN
- **MoE:** MARLIN Fp8 MoE backend
- **Custom fusions:** norm_quant, act_quant

## Results

| Batch Size | Nonces (30s) | Nonces/min |
|-----------:|-------------:|-----------:|
| 8 | 408 | 816 |
| **16** ★ | **432** | **864** |
| 32 | 416 | 832 |
| 64 | 0 | 0 (OOM) |
| 128 | 0 | 0 (OOM) |

**Best: batch=16 → 864 nonces/min**

## Nonce collection

Collected **1,008 nonces** at **840 nonces/min** (batch_size=16, 72 seconds).

## A100-specific issues

1. **TP=8 fails** — `ValueError: output_size of gate's and up's weight = 192 is not divisible by weight quantization block_n = 128`. A100 doesn't support fine-grained FP8 block sizes for MiniMax MoE dimensions at TP=8.

2. **FlashInfer FP8 MoE not supported** — A100 (SM 80) doesn't have native FP8. Need `VLLM_USE_FLASHINFER_MOE_FP8=0`, falls back to MARLIN backend.

3. **CUDA_VISIBLE_DEVICES required** — On 8-GPU servers with TP=4, MLNode spawns 2 instances. Must limit GPUs via `CUDA_VISIBLE_DEVICES=0,1,2,3` in runner.py.

## Artifacts

- `artifacts/nonces_1000.json` — 1000 PoC nonce vectors (seq_len=1024, k_dim=12)

## Key observations

- MiniMax-M2.7 FP8 on 4×A100: **864 nonces/min** at batch=16
- MARLIN FP8 MoE backend works well on A100 (alternative to FlashInfer/DeepGEMM)
- 54.58 GiB/GPU model footprint leaves 13.66 GiB for KV cache
- OOM at batch≥64 — consistent with tight memory headroom
