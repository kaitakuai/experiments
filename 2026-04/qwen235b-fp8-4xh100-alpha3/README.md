# PoC Benchmark: Qwen3-235B-A22B-Instruct-2507-FP8 on 4×H100 SXM (Vast.ai)

**Date:** 2026-04-10
**Purpose:** Standard PoC v2 nonce generation benchmark for `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` on 4×H100 SXM using the new release candidate images `ghcr.io/product-science/mlnode:3.0.13-alpha3` and `ghcr.io/product-science/vllm:v0.15.1-alpha3`. This test was requested by Tamaz as part of the "final image verification" — check that PoC performance holds on H100/A100/B200 with default image settings.

## Request context

From Slack (2026-04-09):
> [Tamaz] vLLM: ghcr.io/product-science/vllm:v0.15.1-alpha3
> MLnode: ghcr.io/product-science/mlnode:3.0.13-alpha3
> ready for testing
>
> [Tamaz] Запустить на Н100, А100, B200, посмотреть PoC performance

This is the first of the three runs.

## Infrastructure

| Parameter | Value |
|-----------|-------|
| Provider | Vast.ai |
| Instance ID | 34501231 |
| Host ID | 94202 |
| Machine ID | 58297 |
| Location | Massachusetts, US |
| SSH | `ssh -p 21230 root@ssh7.vast.ai` |
| Cost | $7.96/hr |
| Network | 6198 Mbps down / 7331 Mbps up |
| Reliability | 99% |

**Search criteria used:**
```bash
vastai search offers 'gpu_name="H100 SXM" num_gpus=4 disk_space>=100' -o 'inet_down-'
```

**Create command:**
```bash
vastai create instance 33715891 \
  --image ghcr.io/product-science/mlnode:3.0.13-alpha3 \
  --disk 400 --ssh --direct
```

## Hardware

| Parameter | Value |
|-----------|-------|
| GPU | 4× NVIDIA H100 80GB HBM3 |
| VRAM | 81,559 MiB (~80 GiB) per GPU — 320 GiB total |
| NVIDIA Driver | 580.126.09 |
| CPU | 64 vCPUs |
| RAM | 2,015 GB (2 TB) |
| Disk (overlay) | 154 GB |
| `/dev/shm` (tmpfs) | 503 GB |

## Software (from image `mlnode:3.0.13-alpha3`)

| Component | Version |
|-----------|---------|
| vLLM | 0.15.1 (product-science alpha3 build) |
| torch | 2.9.1+cu129 |
| Python | 3.12 |
| OS | Ubuntu 22.04.5 LTS |

## Model storage: `/dev/shm` (RAM)

Local overlay disk is only 154 GB, too small for Qwen 235B FP8 (~221 GB). Stored the model in `/dev/shm` (tmpfs, 503 GB free) via:

```bash
mkdir -p /dev/shm/hf
HF_HOME=/dev/shm/hf nohup python3 -c \
  'from huggingface_hub import snapshot_download; \
   snapshot_download("Qwen/Qwen3-235B-A22B-Instruct-2507-FP8", max_workers=16)' \
  > /tmp/download.log 2>&1 &
```

Download completed in ~5 min at ~6 Gbps. Result: `/dev/shm/hf/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/e156cb4efae43fbee1a1ab073f946a1377e6b969/` (24 safetensors shards, 221 GB total).

**Important:** keep `/dev/shm` > 250 GB or use a larger disk. tmpfs is volatile — lost on reboot.

## Patches applied

The `alpha3` image does NOT have the following patches by default — must be applied manually:

```bash
# 1. Disable watcher auto-kill of unhealthy managers
sed -i 's/MAX_UNHEALTHY_COUNT = 3/MAX_UNHEALTHY_COUNT = 9999/' \
  /app/packages/api/src/api/watcher.py

# 2. Register vLLM port 5001 with MLNode proxy
sed -i '/await start_vllm_proxy()/a\    setup_vllm_proxy([5001])' \
  /app/packages/api/src/api/app.py

# 3. Restart MLNode to pick up patches (it was already running from image entrypoint)
kill <old-uvicorn-pid>
cd /app/packages/api && nohup .venv/bin/python -m uvicorn api.app:app \
  --host 0.0.0.0 --port 8080 --app-dir src > /tmp/mlnode.log 2>&1 &
```

## vLLM startup: via MLNode API

The new `alpha3` image has **no hardcodes** in `runner.py` — it respects caller-provided `--tensor-parallel-size` and `--max-model-len` (unlike `3.0.13-alpha1` which had a forced PP=4 override). So we can launch vLLM via the standard MLNode API.

### Final working configuration

After iterating through several OOM issues, the working configuration is:

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
      "--trust-remote-code"
    ]
  }'
```

### Iteration history (what we tried and why each failed)

1. **First attempt** — `--tensor-parallel-size 4 --trust-remote-code` (pure defaults):
   - Failed at engine init: `max_model_len=262144` requires 11.75 GiB KV cache but only 9.27 GiB available at default `gpu_memory_utilization=0.9`. vLLM error suggested max 206864 tokens.

2. **Second attempt** — added `--max-model-len 200000`:
   - Succeeded past KV cache check but failed later during `_dummy_sampler_run` warmup with OOM: "CUDA out of memory when warming up sampler with 1024 dummy requests". Default `max_num_seqs` in vLLM 0.15.1 is too high for H100 80 GB.

3. **Third attempt** — `--gpu-memory-utilization 0.95 --trust-remote-code` (user override, default max_model_len):
   - Same OOM in `_dummy_sampler_run`. 0.95 utilization leaves no room for the 1024-seq warmup tensors.

4. **Fourth attempt** — `--gpu-memory-utilization 0.95 --max-num-seqs 128 --trust-remote-code`:
   - Succeeded through DeepGEMM warmup (5+ min on cold cache, ~2s on hot), but PoC forward crashed with OOM in `fp8_utils._run_deepgemm` — util 0.95 left <1 GB on each GPU for the PoC scratchpad tensors.

5. **Fifth attempt (working)** — `--gpu-memory-utilization 0.92 --max-num-seqs 128`:
   - All startup phases passed. PoC forward ran successfully for batch_size ≤ 16. batch_size ≥ 32 still OOMs (see Results below).

### Startup profile (working config)

- Model loading: **24 s** (from `/dev/shm`, very fast)
- Per-GPU model memory: **55.22 GiB** (TP=4 split of 221 GB FP8)
- Dynamo bytecode transform: ~10 s
- Inductor compile (range 1, 32768): ~12 s (cached on warm runs)
- **DeepGEMM warmup: ~5 minutes on cold cache, <2 s on warm cache**
- FlashInfer autotuning: ~3 s
- Graph capturing: auto (CUDA graphs enabled)
- Available KV cache: **14.03 GiB** (312,944 tokens)
- Total cold start: ~10 minutes. Warm start (with DeepGEMM cache in `~/.cache/vllm/deep_gemm/`): ~2 minutes.

### Critical environment/default notes

- **DeepGEMM warmup is mandatory for Qwen FP8 MoE.** The image does NOT set `VLLM_DEEP_GEMM_WARMUP=skip`. First launch takes 5-10 minutes of nvcc compilation of kernels. Subsequent launches are fast because kernels are cached in `/root/.cache/vllm/deep_gemm/cache/`.
- **Default `gpu_memory_utilization=0.9`** in vLLM 0.15.1 is too low for Qwen 235B at `max_model_len=262144` on H100 80 GB — need to bump to at least 0.92 or lower max_model_len.
- **Default `max_num_seqs`** is too high (>= 1024). Need to set explicitly to 128 to avoid sampler warmup OOM.
- MLNode runner env: `VLLM_USE_V1=0` is forced (compatibility).

## PoC benchmark parameters

Same as prior runs:
- `seq_len=1024`, `k_dim=12`
- Batch sizes: `[8, 16, 32, 64, 128]`
- 5s warmup + 30s measurement per batch

## Run command

```bash
# Upload and patch benchmark script
scp -P 21230 run_pow_generation.py root@ssh7.vast.ai:/tmp/
ssh -p 21230 root@ssh7.vast.ai "
  sed -i 's/BATCH_SIZES_TO_TEST = \[2, 8, 16, 32, 64\]/BATCH_SIZES_TO_TEST = [8, 16, 32, 64, 128]/' /tmp/run_pow_generation.py
  sed -i 's/if not start_vllm_if_needed():/if False:/' /tmp/run_pow_generation.py
  # MODEL_NAME already = Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
"

# Run
ssh -p 21230 root@ssh7.vast.ai "
  nohup python3 -u /tmp/run_pow_generation.py --phase 3 --skip-check \
    > /tmp/poc_tp4_v2.log 2>&1 &
"
```

## Results

| Batch Size | Nonces (30s) | Nonces/min |
|-----------:|-------------:|-----------:|
| 8 | 448 | 896 |
| **16** ★ | **464** | **928** |
| 32 | 0 | 0 (OOM) |
| 64 | 0 | 0 (OOM) |
| 128 | 0 | 0 (OOM) |

**Best:** batch=16 → **928 nonces/min**

### PoC OOM at batch ≥ 32

Starting at batch_size=32, the PoC forward crashes with `CUDA out of memory` during the DeepGEMM kernel execution (FP8 grouped GEMM for MoE experts). The scratchpad tensor allocated by the new PoC runner (PR#24) takes ~2 GB at batch=32, and the 0.92 util leaves ~5 GB free per GPU — not enough for DeepGEMM workspace + scratchpad + activations simultaneously.

Lowering `gpu_memory_utilization` to 0.88 or 0.85 would likely unblock larger batches, but reduces KV cache budget significantly (below `max_model_len=262144`).

## Raw artifact

- `compressa-perf-results/vast_4xh100sxm_qwen235b_poc_tp4.log`

## Key observations

1. **Default image settings don't "just work" for Qwen 235B on H100 80 GB.** You must set `--gpu-memory-utilization 0.92` (not default 0.9) AND `--max-num-seqs 128` (not default ≥1024) to avoid OOM at vLLM startup. These were chosen iteratively after 5 failed attempts.

2. **DeepGEMM cold warmup is slow (~5 min).** The image does not pre-warm or ship a pre-built cache. First launch is 10+ minutes total. Subsequent launches (with `~/.cache/vllm/deep_gemm/` hot) are ~2 minutes.

3. **Peak PoC throughput: 928 nonces/min at batch=16.** Matches typical H100 H100 performance for this model size. Higher batches fail because PoC scratchpad + DeepGEMM workspace together exceed the memory left by `gpu_memory_utilization=0.92`.

4. **MLNode runner in alpha3 is cleaner than alpha1** — no hardcoded PP=4 override, `max_model_len` is respected, proxy setup works via `setup_vllm_proxy([5001])`.

## After completion

```bash
vastai destroy instance 34501231
```
