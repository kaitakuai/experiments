# PoC Benchmark: Kimi-K2.5-NVFP4 on 8×B200

**Date:** 2026-04-11
**Model:** `nvidia/Kimi-K2.5-NVFP4`
**Quantization:** NVFP4 (ModelOpt 0.41.0, W4A4, all Linear except self_attn, group_size=16)
**Hardware:** 8×NVIDIA B200 183GB
**vLLM:** 0.19.0

## Summary

First successful PoC v2 benchmark of an NVFP4-quantized model. Requires `--compilation-config '{"custom_ops": ["all", "-rms_norm"]}'` to prevent `rms_norm_kernel not implemented for 'Byte'` error during PoC forward pass. Model size: 551 GB.

## Hardware

| Parameter | Value |
|-----------|-------|
| GPU | 8× NVIDIA B200 (183,359 MiB each) |
| NVIDIA Driver | 595.58.03 |
| CPU | AMD EPYC 9575F 64-Core (256 vCPUs) |
| RAM | 3,096 GB |
| Disk | Samsung MZQL27T6HBLA-00A07, PCIe 5.0/16x |

## Software

| Component | Version |
|-----------|---------|
| vLLM | 0.19.0 |
| torch | 2.10.0+cu129 |
| Python | 3.12.13 |
| OS | Ubuntu 22.04.5 LTS |

## NVFP4 PoC compatibility fix

### Problem

NVFP4 default compilation enables `norm_quant` and `act_quant` fusions which pass `torch.uint8` (Byte) tensors through RMS norm during PoC forward. The CUDA kernel does not support this dtype:

```
NotImplementedError: "rms_norm_kernel" not implemented for 'Byte'
```

### Fix

Add to vLLM launch args:

```
--compilation-config '{"custom_ops": ["all", "-rms_norm"]}'
```

This keeps all custom ops enabled (maximum performance) but excludes `rms_norm` from compilation, so it falls back to the standard PyTorch implementation which handles dtype correctly.

**No code patches needed.** Only this launch flag.

### Alternative (also works but slower)

```
--compilation-config '{"custom_ops": ["none"]}'
```

Disables all custom ops. Simpler but ~3% slower.

## Reproduction

### 1. Download model

```bash
mkdir -p /dev/shm/hf
HF_HOME=/dev/shm/hf python3 -c \
  'from huggingface_hub import snapshot_download; snapshot_download("nvidia/Kimi-K2.5-NVFP4", max_workers=16)'
```

### 2. Install MLNode

Same as INT4 experiment — see `kimi-k25-int4-4xb200/README.md`.

### 3. Start vLLM

```bash
MODEL_PATH=/dev/shm/hf/hub/models--nvidia--Kimi-K2.5-NVFP4/snapshots/<hash>

curl -X POST http://127.0.0.1:8081/api/v1/inference/up/async \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "'"$MODEL_PATH"'",
    "dtype": "auto",
    "additional_args": [
      "--served-model-name", "nvidia/Kimi-K2.5-NVFP4",
      "--tensor-parallel-size", "8",
      "--gpu-memory-utilization", "0.92",
      "--max-num-seqs", "128",
      "--max-model-len", "131072",
      "--trust-remote-code",
      "--compilation-config", "{\"custom_ops\": [\"all\", \"-rms_norm\"]}"
    ]
  }'
```

### 4. Run benchmark & collect nonces

```bash
export HOST_IP=127.0.0.1
python3 -u run_pow_generation.py --phase 3 --skip-check

python3 -u collect_artifacts.py \
  --url http://127.0.0.1:5001 \
  --model 'nvidia/Kimi-K2.5-NVFP4' \
  --output-dir /tmp/artifacts \
  --nonces 1000 --batch-size 16 --logprobs-count 0
```

## Startup profile

| Phase | Duration |
|-------|----------|
| Model loading (from `/dev/shm`) | ~47 s |
| Model memory/GPU | 70.97 GiB |
| Available KV cache | 83.63 GiB (2,555,776 tokens) |
| CUDA graph memory | 1.12 GiB estimated |
| **Total cold start** | **~5 min 5 s** |

## Results

| Batch Size | Nonces (30s) | Nonces/min |
|-----------:|-------------:|-----------:|
| 8 | 1,496 | 2,991 |
| 16 | 1,792 | 3,583 |
| **32** ★ | **1,920** | **3,839** |
| 64 | 0 | 0 (OOM) |
| 128 | 0 | 0 (OOM) |

**Best: batch=32 → 3,839 nonces/min**

## Nonce collection

Collected 1,024 nonces at 2,792 nonces/min (batch_size=16).

## Artifacts

- `artifacts/nonces_1000.json` — 1000 PoC nonce vectors (seq_len=1024, k_dim=12)

## Key observations

- NVFP4 on 8×B200 yields **3.75× higher throughput** than INT4 on 4×B200 (3,839 vs 1,024 nonces/min)
- FLASHINFER_MLA attention backend, FP8 KV cache (block_size=32)
- Cross-validation between INT4 and NVFP4 is **not possible** — L2 distance ~1.5 (uncorrelated vectors). Different quantization = different model from PoC perspective.
- OOM at batch≥64 despite 83 GiB KV cache — PoC scratch tensor allocation exceeds available memory
