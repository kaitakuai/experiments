# PoC Benchmark: Qwen3-235B-A22B-Instruct-2507-FP8 on 4×A100 SXM4 80GB (Vast.ai)

**Date:** 2026-04-10
**Purpose:** Standard PoC v2 nonce generation benchmark for `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` on 4×A100 SXM4 80GB using image `ghcr.io/product-science/mlnode:3.0.13-alpha3`.

## Infrastructure

| Parameter | Value |
|-----------|-------|
| Provider | Vast.ai |
| Instance ID | 34505654 |
| Host ID | 42313 |
| Machine ID | 42313 |
| Location | Georgia, US |
| SSH | `ssh -p 22398 root@108.231.141.46` |
| Cost | $5.04/hr |
| Network | 1588 Mbps down / 1576 Mbps up |
| Reliability | 99.6% (verified) |

## Hardware

| Parameter | Value |
|-----------|-------|
| GPU | 4× NVIDIA A100-SXM4-80GB |
| VRAM | 81,920 MiB (~80 GiB) per GPU — 320 GiB total |
| Compute capability | sm_80 (Ampere) |
| NVIDIA Driver | 570.211.01 |
| CPU | 64 vCPUs |
| RAM | 1,031 GB (~1 TB) |
| Disk (overlay) | 300 GB |
| `/dev/shm` (tmpfs) | 251 GB |

## Software (from image `mlnode:3.0.13-alpha3`)

| Component | Version |
|-----------|---------|
| vLLM | 0.15.1 (product-science alpha3 build) |
| torch | 2.9.1+cu129 |
| Python | 3.12 |

## Model download

Downloaded to local disk (300 GB overlay):

```bash
nohup python3 -c \
  'from huggingface_hub import snapshot_download; \
   snapshot_download("Qwen/Qwen3-235B-A22B-Instruct-2507-FP8", max_workers=8)' \
  > /tmp/download.log 2>&1 &
```

Completed in ~20 min at ~200 MB/s (stable verified host). Model stored at: `/root/.cache/huggingface/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/e156cb4efae43fbee1a1ab073f946a1377e6b969/` (24 safetensors shards, ~221 GB).

## Patches applied

```bash
# 1. Disable watcher auto-kill
sed -i 's/MAX_UNHEALTHY_COUNT = 3/MAX_UNHEALTHY_COUNT = 9999/' \
  /app/packages/api/src/api/watcher.py

# 2. Register vLLM port 5001 with MLNode proxy
sed -i '/await start_vllm_proxy()/a\    setup_vllm_proxy([5001])' \
  /app/packages/api/src/api/app.py

# 3. Restart MLNode
kill <old-uvicorn-pid>
cd /app/packages/api && nohup .venv/bin/python -m uvicorn api.app:app \
  --host 0.0.0.0 --port 8080 --app-dir src > /tmp/mlnode.log 2>&1 &
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
      "--tensor-parallel-size", "4",
      "--gpu-memory-utilization", "0.92",
      "--max-num-seqs", "128",
      "--trust-remote-code"
    ]
  }'
```

### Key parameters

- `--tensor-parallel-size 4` — all 4 GPUs
- `--gpu-memory-utilization 0.92` — higher than default 0.9, needed for 262144 max_model_len to fit KV cache
- `--max-num-seqs 128` — lower than default to avoid sampler warmup OOM
- Default `max_model_len=262144` (model default, not overridden)

### Startup profile

- Loading weights: **71.6 s** (from local disk)
- Per-GPU model memory: ~55 GiB (TP=4 split of 221 GB FP8)
- torch.compile: **28.2 s**
- DeepGEMM warmup: ~1 min (cold cache)
- Total startup: **~228 s** (~3.8 min)

### MLNode runner notes

- `VLLM_USE_V1=0` is set by runner (V0 engine mode)
- Runner in alpha3 has no hardcoded PP/TP overrides (clean)

## PoC benchmark

### Parameters

- `seq_len=1024`, `k_dim=12`
- Batch sizes: `[8, 16, 32, 64, 128]`
- 5s warmup + 30s measurement per batch

### Run command

```bash
scp -P 22398 run_pow_generation.py root@108.231.141.46:/tmp/
ssh -p 22398 root@108.231.141.46 "
  sed -i 's/BATCH_SIZES_TO_TEST = \[2, 8, 16, 32, 64\]/BATCH_SIZES_TO_TEST = [8, 16, 32, 64, 128]/' /tmp/run_pow_generation.py
  sed -i 's/if not start_vllm_if_needed():/if False:/' /tmp/run_pow_generation.py
  nohup python3 -u /tmp/run_pow_generation.py --phase 3 --skip-check \
    > /tmp/poc_tp4.log 2>&1 &
"
```

## Results

| Batch Size | Nonces (30s) | Nonces/min |
|-----------:|-------------:|-----------:|
| 8 | 240 | 480 |
| **16** ★ | **240** | **480** |
| 32 | 224 | 448 |
| 64 | 0 | 0 (OOM) |
| 128 | 0 | 0 (OOM) |

**Best:** batch=8/16 → **480 nonces/min**

## Comparison with H100 SXM (same image, same model, same settings)

| GPU | VRAM/card | Best batch | Nonces/min |
|-----|----------:|-----------:|-----------:|
| 4×H100 SXM 80GB | 80 GB | 16 | **928** |
| **4×A100 SXM4 80GB** | 80 GB | 16 | **480** |
| Ratio | — | — | **A100 = 52% of H100** |

A100 delivers roughly half the PoC throughput of H100 at identical memory and TP configuration. This aligns with expected compute performance difference (A100 FP8: ~624 TFLOPS vs H100 FP8: ~1979 TFLOPS theoretical, but FP8 on A100 is emulated via FP16+cast, not native).

## Artifacts

- `compressa-perf-results/vast_4xa100sxm4_qwen235b_poc_tp4.log`

## After completion

```bash
vastai destroy instance 34505654
```
