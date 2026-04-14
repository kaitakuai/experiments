# PoC Benchmark: Qwen3-235B-A22B-Instruct-2507-FP8 on 4×H100 SXM (Vast.ai) — vLLM 0.19.0

**Date:** 2026-04-11
**Purpose:** Standard PoC v2 nonce generation benchmark for `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` on 4×H100 SXM using **vLLM 0.19.0** (upstream release image `vllm/vllm-openai:v0.19.0`) with MLNode API layer from `kaitakuai/rtx-pro-6000` gonka-source submodule. This is a follow-up to the alpha4 run to evaluate the upstream vLLM 0.19 release on the same hardware class.

## Request context

Tamaz asked to test vLLM 0.19.0 on H100 hardware. This run covers 4×H100 SXM on Vast.ai with the upstream vLLM image (not product-science fork).

## Infrastructure

| Parameter | Value |
|-----------|-------|
| Provider | Vast.ai |
| Instance ID | 34594413 |
| Host ID | 94202 |
| Machine ID | 58296 |
| Location | Massachusetts, US |
| SSH | `ssh -p 34413 root@ssh1.vast.ai` |
| Cost | ~$8.40/hr |
| Network | 6810 Mbps down / 6826 Mbps up |

## Hardware

| Parameter | Value |
|-----------|-------|
| GPU | 4× NVIDIA H100 80GB HBM3 |
| VRAM | 81,559 MiB (~80 GiB) per GPU — 320 GiB total |
| NVIDIA Driver | 580.126.09 |
| CPU | Intel Xeon Platinum 8462Y+ (64 vCPUs) |
| RAM | 2,064 GB (~2 TB) |
| Motherboard | HPE Cray XD670, PCIe 5.0/16x |
| Disk (overlay) | 258 GB |
| `/dev/shm` (tmpfs) | 503 GB |

## Software

| Component | Version |
|-----------|---------|
| vLLM | **0.19.0** (upstream `vllm/vllm-openai`) |
| torch | 2.10.0+cu129 |
| Python | 3.12.13 |
| OS | Ubuntu 22.04.5 LTS |
| MLNode | gonka-source (manual install from `kaitakuai/rtx-pro-6000` submodule) |

## Model storage: `/dev/shm` (RAM)

```bash
mkdir -p /dev/shm/hf
HF_HOME=/dev/shm/hf nohup python3 -c \
  'from huggingface_hub import snapshot_download; \
   snapshot_download("Qwen/Qwen3-235B-A22B-Instruct-2507-FP8", max_workers=16)' \
  > /tmp/download.log 2>&1 &
```

Download completed in ~4 min 29 sec at ~6.8 Gbps. Result: `/dev/shm/hf/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/e156cb4efae43fbee1a1ab073f946a1377e6b969/` (24 safetensors shards, 221 GB total).

## MLNode setup (manual)

The vLLM 0.19 image (`vllm/vllm-openai`) does not include MLNode. It was installed manually:

```bash
# 1. Copy MLNode packages from gonka-source submodule
tar czf mlnode_packages.tar.gz -C vendor/gonka-source/mlnode/packages api common pow train
scp mlnode_packages.tar.gz root@server:/tmp/
ssh root@server "mkdir -p /app/packages && cd /app/packages && tar xzf /tmp/mlnode_packages.tar.gz"

# 2. Install Python dependencies
pip install toml accelerate fire fastrlock h2 termcolor typer-slim setuptools-scm tenacity

# 3. Patch watcher (disable auto-kill) and runner (python path, V1 engine)
sed -i 's/MAX_UNHEALTHY_COUNT = 3/MAX_UNHEALTHY_COUNT = 9999/' /app/packages/api/src/api/watcher.py
sed -i 's|VLLM_PYTHON_PATH = "/usr/bin/python3.12"|VLLM_PYTHON_PATH = "/usr/bin/python3"|' /app/packages/api/src/api/inference/vllm/runner.py
sed -i 's/env\["VLLM_USE_V1"\] = "0"/env["VLLM_USE_V1"] = "1"/' /app/packages/api/src/api/inference/vllm/runner.py

# 4. Kill Jupyter (occupies port 8080), start MLNode
kill $(pgrep -f jupyter)
PYTHONPATH="/app:/app/packages/api/src:/app/packages/pow/src:/app/packages/train/src:/app/packages/common/src" \
  nohup python3 -m uvicorn api.app:app --host 0.0.0.0 --port 8080 --app-dir /app/packages/api/src \
  > /tmp/mlnode.log 2>&1 &
```

**Note:** Provider-specific patches (01-runner-compat, 02-pow-batch-size, 03-poc-v2-interrupt-inference) were NOT applied — they target miniconda paths and networking that don't exist on Vast.ai.

## vLLM startup: via MLNode API

### Working configuration

```bash
MODEL_PATH=/dev/shm/hf/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/e156cb4efae43fbee1a1ab073f946a1377e6b969

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
      "--max-model-len", "240000",
      "--trust-remote-code"
    ]
  }'
```

**Note:** `--max-model-len 240000` is required. Without it, vLLM 0.19 defaults to `max_seq_len=262144` which requires 11.75 GiB KV cache — exceeding the 10.89 GiB available. This was not needed in alpha4 (vLLM 0.15.1).

### Startup profile (cold cache)

| Phase | Duration |
|-------|----------|
| Model loading (from `/dev/shm`) | 26.79 s |
| Total model init (incl. framework) | 28.70 s |
| Dynamo bytecode transform | 8.72 s |
| torch.compile total | 30.67 s |
| **DeepGEMM warmup (5678 kernels, cold)** | **2 min 25 s** |
| CUDA graph capture (prefill-decode PIECEWISE, 35 graphs) | ~15 s |
| CUDA graph capture (decode FULL, 19 graphs) | ~3 s |
| **Total cold start** | **~5 min 49 s** |

- Per-GPU model memory: **55.22 GiB** (TP=4 split of 221 GB FP8)
- Available KV cache: **10.89 GiB** (242,912 tokens)
- Maximum concurrency for 240,000 tokens/request: **1.01x**

### Key differences from alpha4 (vLLM 0.15.1)

1. **vLLM 0.19.0 vs 0.15.1** — major version bump. V1 engine is now default (no need for `VLLM_USE_V1` override).

2. **torch 2.10.0 vs 2.9.1** — updated PyTorch version.

3. **DeepGEMM warmup: 5678 kernels** — significantly more than alpha4 (3630), but completed faster (2m25s vs 5m36s) due to better caching/parallelism.

4. **FlashAttention v3** — selected as attention backend (vs FlashAttention v2 in earlier versions).

5. **FLASHINFER_CUTLASS MoE backend** — same as alpha4, from expanded backend list including new options (FLASHINFER_TRTLLM, BATCHED_DEEPGEMM, BATCHED_VLLM_CUTLASS, BATCHED_TRITON).

6. **flashinfer allreduce: trtllm** — auto-selected `trtllm` allreduce backend (new in 0.19).

7. **`--max-model-len` required** — vLLM 0.19 fails without explicit max-model-len when default (262144) exceeds available KV cache. Alpha4 did not have this issue.

8. **KV cache reduced** — 10.89 GiB / 242,912 tokens (vs 14.03 GiB / 312,896 tokens in alpha4). The increased CUDA graph memory (1.76 GiB estimated) and changed memory profiling behavior consume more overhead.

### Critical environment/default notes

- **`--max-model-len 240000` required** — without it, startup fails with `ValueError: 11.75 GiB KV cache needed, 10.89 GiB available`
- **Default `gpu_memory_utilization=0.9`** too low — need 0.92+
- **Default `max_num_seqs`** too high — need 128 explicitly
- **DeepGEMM cold warmup now faster** (~2.5 min vs ~5.5 min) despite more kernels (5678 vs 3630)
- **Jupyter notebook** occupies port 8080 on vanilla vLLM image — must be killed before starting MLNode

## PoC benchmark parameters

Same as prior runs:
- `seq_len=1024`, `k_dim=12`
- Batch sizes: `[8, 16, 32, 64, 128]`
- 5s warmup + 30s measurement per batch

## Run command

```bash
scp -P 34413 run_pow_generation.py root@ssh1.vast.ai:/tmp/
ssh -p 34413 root@ssh1.vast.ai "
  sed -i 's/BATCH_SIZES_TO_TEST = \[2, 8, 16, 32, 64\]/BATCH_SIZES_TO_TEST = [8, 16, 32, 64, 128]/' /tmp/run_pow_generation.py
  sed -i 's/if not start_vllm_if_needed():/if False:/' /tmp/run_pow_generation.py
"

# Run
ssh -p 34413 root@ssh1.vast.ai "
  export HOST_IP=127.0.0.1
  nohup python3 -u /tmp/run_pow_generation.py --phase 3 --skip-check \
    > /tmp/poc_benchmark.log 2>&1 &
"
```

## Results

| Batch Size | Nonces (30s) | Nonces/min |
|-----------:|-------------:|-----------:|
| 8 | 456 | 912 |
| **16** ★ | **480** | **960** |
| 32 | 0 | 0 (OOM) |
| 64 | 0 | 0 (OOM) |
| 128 | 0 | 0 (OOM) |

**Best:** batch=16 → **960 nonces/min**

### Comparison with alpha4 (vLLM 0.15.1) and alpha3

| Metric | alpha3 (v0.15.1) | alpha4 (v0.15.1) | **vLLM 0.19.0** |
|--------|--------|--------|--------|
| Best nonces/min | 928 | 928 | **960** |
| Best batch size | 16 | 16 | 16 |
| batch=8 nonces (30s) | 448 | 448 | 456 |
| batch=16 nonces (30s) | 464 | 464 | 480 |
| OOM threshold | batch≥32 | batch≥32 | batch≥32 |
| vLLM version | 0.15.1 | 0.15.1 | **0.19.0** |
| vLLM engine | V0 (forced) | V1 | V1 |
| MoE backend | DeepGEMM | FLASHINFER_CUTLASS | FLASHINFER_CUTLASS |
| Attention backend | FlashAttention v2 | FlashAttention v2 | **FlashAttention v3** |
| Cold start | ~10 min | ~7 min 49 s | **~5 min 49 s** |
| DeepGEMM kernels | — | 3,630 | 5,678 |
| DeepGEMM warmup time | ~5 min | ~5 min 36 s | **~2 min 25 s** |
| Model memory/GPU | 55.22 GiB | 55.22 GiB | 55.22 GiB |
| KV cache | 14.03 GiB | 14.03 GiB | **10.89 GiB** |
| KV tokens | 312,896 | 312,896 | **242,912** |
| torch version | 2.9.1 | 2.9.1 | **2.10.0** |

**PoC throughput improved ~3.4%.** vLLM 0.19.0 achieves 960 nonces/min vs 928 for alpha3/alpha4 at batch=16.

## Raw artifact

- `compressa-perf-results/vast_4xh100sxm_qwen235b_poc_vllm019.log`

## Key observations

1. **~3.4% PoC throughput improvement** — 960 nonces/min (vLLM 0.19) vs 928 (alpha4 vLLM 0.15.1). Both batch=8 and batch=16 show consistent ~1.7-3.4% improvement (456 vs 448, 480 vs 464 nonces in 30s).

2. **Fastest cold start yet** — ~5 min 49 s total, down from ~7:49 (alpha4) and ~10:00 (alpha3). DeepGEMM warmup is 2× faster despite 56% more kernels (5678 vs 3630), likely due to improved kernel caching in vLLM 0.19.

3. **Reduced KV cache** — 10.89 GiB / 242,912 tokens vs 14.03 GiB / 312,896 tokens in alpha4. This is due to changed CUDA graph memory profiling behavior in vLLM 0.19 (1.76 GiB estimated graph memory) and requires explicit `--max-model-len 240000` to avoid startup failure.

4. **OOM at batch≥32 persists** — same memory pressure issue as alpha3/alpha4. The reduced KV cache makes this slightly worse, though it doesn't affect the optimal batch=16 operating point.

5. **FlashAttention v3 + trtllm allreduce** — vLLM 0.19 uses FlashAttention v3 and flashinfer's trtllm allreduce backend. These may contribute to the throughput improvement.

6. **MLNode requires manual installation** — the vanilla vLLM image has no MLNode. Setup requires copying gonka-source packages, installing pip dependencies, and patching runner.py for correct Python path and V1 engine. Provider-specific patches from other environments are incompatible.

7. **`--max-model-len` is a new required parameter** — vLLM 0.19 no longer silently caps model length to available KV cache. This is a breaking change for existing startup configurations.

## After completion

```bash
vastai destroy instance 34594413
```
