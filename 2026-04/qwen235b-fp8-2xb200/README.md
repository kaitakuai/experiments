# PoC Benchmark: Qwen3-235B-A22B-Instruct-2507-FP8 on 2×B200 (Vast.ai)

**Date:** 2026-04-10
**Purpose:** Standard PoC v2 nonce generation benchmark for `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` on 2×B200 using image `ghcr.io/product-science/mlnode:3.0.13-alpha3`.

## Infrastructure

| Parameter | Value |
|-----------|-------|
| Provider | Vast.ai |
| Instance ID | 34537523 |
| Host ID | 57669 |
| Location | Alabama, US |
| SSH | `ssh -p 17522 root@ssh8.vast.ai` |
| Cost | $7.68/hr |
| Network | 7723 Mbps down / 6916 Mbps up |
| Reliability | 98.6% (verified) |

## Hardware

| Parameter | Value |
|-----------|-------|
| GPU | 2× NVIDIA B200 |
| VRAM | 183,359 MiB (~179 GiB) per GPU — 358 GiB total |
| NVIDIA Driver | 590.48.01 |
| CPU | 64 vCPUs |
| RAM | 2,321 GB (~2.3 TB) |
| Disk (overlay) | 878 GB (553 GB pre-used by host cache, 281 GB free) |

## Software (from image `mlnode:3.0.13-alpha3`)

| Component | Version |
|-----------|---------|
| vLLM | 0.15.1 (product-science alpha3 build) |
| torch | 2.9.1+cu129 |
| Python | 3.12 |

## Model download

Model was **partially cached** on the host from a prior instance (~152 GB / 221 GB). Download resumed and completed remaining ~70 GB in ~60s at ~1.1 GB/s.

```bash
nohup python3 -c \
  'from huggingface_hub import snapshot_download; \
   snapshot_download("Qwen/Qwen3-235B-A22B-Instruct-2507-FP8", max_workers=8)' \
  > /tmp/download.log 2>&1 &
```

## Patches applied

```bash
sed -i 's/MAX_UNHEALTHY_COUNT = 3/MAX_UNHEALTHY_COUNT = 9999/' \
  /app/packages/api/src/api/watcher.py
sed -i '/await start_vllm_proxy()/a\    setup_vllm_proxy([5001])' \
  /app/packages/api/src/api/app.py
```

## vLLM startup command (via MLNode API)

```bash
MODEL_PATH=/root/.cache/huggingface/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/e156cb4efae43fbee1a1ab073f946a1377e6b969

curl -X POST http://127.0.0.1:8080/api/v1/inference/up/async \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "'"$MODEL_PATH"'",
    "dtype": "auto",
    "additional_args": [
      "--served-model-name", "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
      "--tensor-parallel-size", "2",
      "--gpu-memory-utilization", "0.92",
      "--max-num-seqs", "128",
      "--trust-remote-code"
    ]
  }'
```

### Key parameters

- `--tensor-parallel-size 2` — both GPUs
- `--gpu-memory-utilization 0.92`
- `--max-num-seqs 128`
- Default `max_model_len=262144`

### Startup profile

- Loading weights: fast (local disk, host cache hot)
- Available KV cache: **48.87 GiB** (545,136 tokens)
- DeepGEMM warmup: **24 s** (1379 kernels at 56 it/s)
- Total startup: **~151 s** (~2.5 min)

## PoC benchmark

### Parameters

- `seq_len=1024`, `k_dim=12`
- Batch sizes: `[8, 16, 32, 64, 128]`
- 5s warmup + 30s measurement per batch

### Run command

```bash
scp -P 17522 run_pow_generation.py root@ssh8.vast.ai:/tmp/
ssh -p 17522 root@ssh8.vast.ai "
  sed -i 's/BATCH_SIZES_TO_TEST = \[2, 8, 16, 32, 64\]/BATCH_SIZES_TO_TEST = [8, 16, 32, 64, 128]/' /tmp/run_pow_generation.py
  sed -i 's/if not start_vllm_if_needed():/if False:/' /tmp/run_pow_generation.py
  nohup python3 -u /tmp/run_pow_generation.py --phase 3 --skip-check \
    > /tmp/poc_tp2.log 2>&1 &
"
```

## Results

| Batch Size | Nonces (30s) | Nonces/min |
|-----------:|-------------:|-----------:|
| 8 | 744 | 1488 |
| 16 | 768 | 1536 |
| 32 | 768 | 1536 |
| **64** ★ | **768** | **1536** |
| 128 | 0 | 0 (OOM) |

**Best:** batch=16/32/64 → **1536 nonces/min**

## Full comparison across all tested GPUs (same image, same model)

| GPU | TP | Best batch | Nonces/min | KV cache |
|-----|---:|-----------:|-----------:|---------:|
| **2×B200** | 2 | 64 | **1536** | 48.87 GiB |
| **4×H100 SXM 80GB** | 4 | 16 | **928** | 14.03 GiB |
| **4×A100 SXM4 80GB** | 4 | 16 | **480** | ~10 GiB |

**B200 delivers 1.65× H100 and 3.2× A100 PoC throughput** while using only 2 GPUs (vs 4). The massive VRAM (179 GiB/GPU) allows much larger batch sizes (64 works vs 16 on H100) and nearly 50 GiB of KV cache.

## Artifacts

- `compressa-perf-results/vast_2xb200_qwen235b_poc_tp2.log`

## After completion

```bash
vastai destroy instance 34537523
```
