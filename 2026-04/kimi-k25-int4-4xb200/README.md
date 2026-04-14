# PoC Benchmark: Kimi-K2.5 (INT4) on 4×B200

**Date:** 2026-04-11
**Model:** `moonshotai/Kimi-K2.5`
**Quantization:** INT4 (compressed-tensors, W4A16, MoE experts only, group_size=32)
**Hardware:** 4×NVIDIA B200 183GB
**vLLM:** 0.19.0

## Summary

Standard PoC v2 nonce generation benchmark. The `moonshotai/Kimi-K2.5` model is distributed as selective INT4 — only routed MoE expert weights (384 experts × 61 layers) are INT4-quantized; attention, shared experts, and dense MLP layers remain in BF16. Model size: 555 GB.

## Hardware

| Parameter | Value |
|-----------|-------|
| GPU | 4× NVIDIA B200 (183,359 MiB each) |
| NVIDIA Driver | 595.58.03 |
| CPU | AMD EPYC 9575F 64-Core (256 vCPUs) |
| RAM | 1,548 GB |
| Disk | Samsung MZQL27T6HBLA-00A07, PCIe 5.0/16x |

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
  'from huggingface_hub import snapshot_download; snapshot_download("moonshotai/Kimi-K2.5", max_workers=16)'
```

### 2. Install MLNode

```bash
# Copy packages from kaitakuai/rtx-pro-6000 gonka-source submodule
tar xzf mlnode_packages.tar.gz -C /app/packages/
pip install toml accelerate fire fastrlock h2 termcolor typer-slim setuptools-scm tenacity

# Patches
sed -i 's/MAX_UNHEALTHY_COUNT = 3/MAX_UNHEALTHY_COUNT = 9999/' /app/packages/api/src/api/watcher.py
sed -i "s|VLLM_PYTHON_PATH = \"/usr/bin/python3.12\"|VLLM_PYTHON_PATH = \"$(which python3)\"|" /app/packages/api/src/api/inference/vllm/runner.py
sed -i 's/env\["VLLM_USE_V1"\] = "0"/env["VLLM_USE_V1"] = "1"/' /app/packages/api/src/api/inference/vllm/runner.py

# Start MLNode
PYTHONPATH="/app:/app/packages/api/src:/app/packages/pow/src:/app/packages/train/src:/app/packages/common/src" \
  python3 -m uvicorn api.app:app --host 0.0.0.0 --port 8081 --app-dir /app/packages/api/src
```

### 3. Start vLLM

```bash
MODEL_PATH=/dev/shm/hf/hub/models--moonshotai--Kimi-K2.5/snapshots/<hash>

curl -X POST http://127.0.0.1:8081/api/v1/inference/up/async \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "'"$MODEL_PATH"'",
    "dtype": "auto",
    "additional_args": [
      "--served-model-name", "moonshotai/Kimi-K2.5",
      "--tensor-parallel-size", "4",
      "--gpu-memory-utilization", "0.92",
      "--max-num-seqs", "128",
      "--max-model-len", "131072",
      "--trust-remote-code"
    ]
  }'
```

No special flags needed for INT4 — works out of the box.

### 4. Run benchmark

```bash
export HOST_IP=127.0.0.1
python3 -u run_pow_generation.py --phase 3 --skip-check
```

### 5. Collect nonces

```bash
python3 -u collect_artifacts.py \
  --url http://127.0.0.1:5001 \
  --model 'moonshotai/Kimi-K2.5' \
  --output-dir /tmp/artifacts \
  --nonces 1000 --batch-size 32 --logprobs-count 0
```

## Startup profile

| Phase | Duration |
|-------|----------|
| Model loading (from `/dev/shm`) | 161.92 s |
| Model memory/GPU | 140.49 GiB |
| Dynamo bytecode transform | 4.03 s |
| torch.compile total | 21.05 s |
| CUDA graph capture (PIECEWISE=35, FULL=19) | ~7 s |
| Available KV cache | 14.44 GiB (220,704 tokens) |
| **Total cold start** | **~5 min 32 s** |

## Results

| Batch Size | Nonces (30s) | Nonces/min |
|-----------:|-------------:|-----------:|
| 8 | 416 | 832 |
| 16 | 496 | 992 |
| **32** | **512** | **1024** |
| **64** ★ | **512** | **1024** |
| 128 | 0 | 0 (OOM) |

**Best: batch=32/64 → 1024 nonces/min**

## Nonce collection

Collected 1056 nonces at 946 nonces/min (batch_size=32).

## Artifacts

- `artifacts/nonces_1000.json` — 1000 PoC nonce vectors (seq_len=1024, k_dim=12)

## Key observations

- INT4 (W4A16) works out of the box with vLLM 0.19 on B200 — no patches needed beyond standard MLNode setup
- MLA attention backend (FLASHINFER_MLA) auto-selected
- OOM at batch≥128 due to 14.44 GiB KV cache headroom
- Batch=32 and batch=64 yield identical throughput (1024/min)
