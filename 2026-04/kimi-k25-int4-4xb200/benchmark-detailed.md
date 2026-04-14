# Benchmark: Kimi K2.5 INT4 on 4xB200

**Date:** 2026-04-08
**Purpose:** Full PoC v2 + inference benchmark for `moonshotai/Kimi-K2.5` on 4×B200 in two vLLM modes (enforce-eager vs compiled with `custom_ops=["all"]`), using Tamaz's production image `product-science/mlnode:3.0.13-alpha1` and the official Gonka `compressa-perf` tool.

## Context

Tamaz requested a benchmark of Kimi K2.5 INT4 with:
- Image `ghcr.io/product-science/mlnode:3.0.13-alpha1` (libcuda fix baked in, PR#5 applied, scratchpad KV cache reuse)
- Compiled mode with `--compilation-config '{"custom_ops": ["all"]}'`
- Both PoC v2 nonce generation and compressa-perf inference throughput

4×H200 was unavailable on Vast.ai (only 2×H200 or 8×H200 Japan with slow network), so 4×B200 Alabama was chosen instead.

## Infrastructure

| Parameter | Value |
|-----------|-------|
| Vast.ai Instance ID | 34401918 |
| Host | 436398 |
| Location | Alabama, US |
| SSH | `ssh -p 26078 root@162.120.84.105` (ports: 11918, 26078) |
| Public IP | 162.120.84.105 |
| Cost | ~$20/hr |

## Hardware

| Parameter | Value |
|-----------|-------|
| GPU | 4× NVIDIA B200 |
| VRAM | 183,359 MiB (179 GiB) per GPU — 720 GiB total |
| CPU | 256 vCPUs |
| RAM | 2,267 GB (2.2 TiB) |
| Disk | 700 GB allocated (overlay 878 GB) |
| NVIDIA Driver | 590.48.01 |

## Software

| Parameter | Value |
|-----------|-------|
| Docker Image | `ghcr.io/product-science/mlnode:3.0.13-alpha1` |
| vLLM | 0.15.1 |
| torch | 2.9.1+cu129 |
| triton | 3.5.1 |
| flashinfer-python | 0.6.1 |
| CUDA | 12.9 (V12.9.86) |
| Python | 3.12.12 |
| OS | Ubuntu 22.04.5 LTS |
| Kernel | 5.15.0-173-generic |
| compressa-perf | 0.2.5 |

## Model

| Parameter | Value |
|-----------|-------|
| Name | `moonshotai/Kimi-K2.5` |
| Architecture | `KimiK25ForConditionalGeneration` (text: `DeepseekV3ForCausalLM` + MoonViT vision 400M) |
| Parameters | 1.1T total, 32B activated (MoE) |
| Quantization | compressed-tensors INT4 (group_size=32, num_bits=4, strategy=group, type=int) |
| Size on disk | ~547 GB (64 safetensors shards) |
| Default max_model_len | 262,144 (256K) |
| Attention | MLA (Multi-head Latent Attention) |
| Required attention backend | **FLASHINFER_MLA** (auto-selection on Blackwell works for `qk_nope_head_dim==128`, but must be forced explicitly via `--attention-backend FLASHINFER_MLA`) |
| MoE method | CompressedTensorsWNA16MarlinMoEMethod |

## Instance setup (reproducible steps)

### 1. Create Vast.ai instance

Requirements:
- GPU: 4×B200 (or 4×H200 if available)
- Disk: ≥ 700 GB
- Image: `ghcr.io/product-science/mlnode:3.0.13-alpha1`
- Network: low-latency US host (Japan hosts are slower for HF downloads)

### 2. Apply patches

The `3.0.13-alpha1` image already includes:
- libcuda symlink fix (runtime in `/app/entrypoint.sh`)
- PoC PR#5 (scratchpad KV cache reuse, `input_ids=None` optimization)
- Proxy setup with `setup_vllm_proxy([5001])` in `app.py` (already present at line 47)

Only one patch to apply manually — watcher disable (prevents MLNode from killing vLLM during long benchmarks):

```bash
sed -i 's/MAX_UNHEALTHY_COUNT = 3/MAX_UNHEALTHY_COUNT = 9999/' \
  /app/packages/api/src/api/watcher.py
```

### 3. Record environment

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python3 -c 'import vllm, torch, triton, flashinfer; print(vllm.__version__, torch.__version__, triton.__version__, flashinfer.__version__)'
nvcc --version
```

### 4. Download model

~547 GB, ~10 min on fast network. With `HF_HUB_OFFLINE=0` it downloads on first vLLM start, but pre-downloading avoids startup race conditions:

```bash
python3 -c 'from huggingface_hub import snapshot_download; snapshot_download("moonshotai/Kimi-K2.5")'
```

## PoC benchmark parameters

| Parameter | Value |
|-----------|-------|
| seq_len | 1024 |
| k_dim | 12 |
| block_hash | `TEST_BLOCK` |
| block_height | 100 |
| Warmup | 5s |
| Measurement | 30s per batch size |
| Batch sizes tested | 8, 16, 32, 64, 128 |

## Experiment A: Enforce-Eager (PoC)

### vLLM startup command

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 -m vllm.entrypoints.openai.api_server \
  --model moonshotai/Kimi-K2.5 \
  --dtype auto \
  --port 5001 --host 0.0.0.0 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 128 \
  --max-model-len 262144 \
  --enforce-eager \
  --trust-remote-code \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --mm-encoder-tp-mode data \
  --attention-backend FLASHINFER_MLA
```

**Critical flags explained:**
- `--attention-backend FLASHINFER_MLA` — **required**. Default auto-selection on Blackwell may fall back to plain `FLASHINFER` which does not support MLA (error: `'head_size not supported', 'MLA not supported'`).
- `--gpu-memory-utilization 0.95` — at 0.9, KV cache is 11.66 GiB, not enough for `max_model_len=262144` (requires 17.16 GiB). At 0.95 → 20.58 GiB available.
- `--mm-encoder-tp-mode data` — 400M vision encoder is too small for TP=4, use DP instead.
- `HF_HUB_OFFLINE=1` — guards against HF API outages (observed 503 during testing). Requires model pre-downloaded.
- `--tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --trust-remote-code` — required for Kimi K2.5 compatibility.

### Startup profile

- Loading weights: **148 s**, 140.95 GiB/GPU model memory
- Available KV cache: **20.58 GiB**, 314,432 tokens (enforce-eager)
- Application startup: ~160 s total

### PoC Benchmark Results — Experiment A (eager)

| Batch Size | Nonces (30s) | Nonces/min |
|------------|-------------|------------|
| 8 | 456 | 912 |
| 16 | 416 | 832 |
| 32 | **512** | **1024** ★ |
| 64 | 0 | 0 (broken) |
| 128 | 0 | 0 (broken) |

**Best:** batch=32 → **1024 nonces/min**

### MLNode startup

```bash
cd /app/packages/api && \
  .venv/bin/python -m uvicorn api.app:app \
    --host 0.0.0.0 --port 8080 --app-dir src &
```

Wait for `127.0.0.1:5001 is UP` in MLNode logs.

## Experiment B: Compiled + custom_ops=["all"]

### Prerequisite: GPU memory cleanup

`kill -9` of vLLM does not free GPU memory on Vast.ai due to zombie worker processes. Either kill by exact PID from `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` or reboot the instance.

```bash
# Find and kill vllm worker PIDs explicitly
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
```

### vLLM startup command

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 -m vllm.entrypoints.openai.api_server \
  --model moonshotai/Kimi-K2.5 \
  --dtype auto \
  --port 5001 --host 0.0.0.0 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 128 \
  --max-model-len 262144 \
  --trust-remote-code \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --mm-encoder-tp-mode data \
  --attention-backend FLASHINFER_MLA \
  --compilation-config '{"custom_ops": ["all"]}'
```

### Effective compilation config (from startup log)

- `mode: VLLM_COMPILE (3)`
- `backend: inductor`
- `custom_ops: ['all']`
- `cudagraph_mode: FULL_AND_PIECEWISE`
- `cudagraph_capture_sizes: [1, 2, 4, 8, 16, ..., 248, 256]` (35 piecewise + 19 full shapes)
- `compile_ranges_split_points: [32768]` (default)
- `pass_config: fuse_norm_quant=True, fuse_act_quant=True, eliminate_noops=True`

### Startup profile

- Loading weights: **148 s**
- Dynamo bytecode transform: 2.95 s
- Inductor compile (range 1-32768): 8.69 s
- FlashInfer autotuning: ~1 s
- CUDA graph capturing (35 + 19 shapes): **6 s total**
- Graph capture freed 5.90 GiB (eliminate_noops)
- Available KV cache: **20.89 GiB** (+0.31 GiB vs eager due to fusions)
- Application startup: ~170 s total

### PoC Benchmark Results — Experiment B (compiled)

| Batch Size | Nonces (30s) | Nonces/min |
|------------|-------------|------------|
| 8 | 456 | 912 |
| 16 | 496 | 992 |
| 32 | **512** | **1024** ★ |
| 64 | 0 | 0 (broken) |
| 128 | 0 | 0 (broken) |

**Best:** batch=32 → **1024 nonces/min**

## PoC Comparison

| Batch | Eager | Compiled | Δ |
|-------|-------|----------|---|
| 8 | 912 | 912 | 0% |
| 16 | 832 | 992 | **+19%** |
| **32** ★ | **1024** | **1024** | **0%** |
| 64 | 0 | 0 | broken |
| 128 | 0 | 0 | broken |

**Key finding:** Unlike Qwen 235B FP8 (where compiled gave +25% on best batch), for Kimi K2.5 INT4 compiled mode provides **zero speedup** on the optimal batch size. Only `batch=16` benefits (+19%), which is below the optimal operating point anyway.

**batch=64/128 broken in both modes:** Unlike Qwen 235B where only enforce-eager failed at high batches, Kimi K2.5 fails in both compiled and eager. Root cause TBD (likely KV cache exhaustion: 314k tokens / 64 × 1024 = 4.8 sessions, may conflict with scratchpad allocation).

**Per-GPU PoC efficiency:** 1024 / 4 GPUs = **256 nonces/min/GPU**
For context, Qwen 235B compiled on 2×B200: 1920 / 2 = **960 nonces/min/GPU**.
Kimi K2.5 is ~3.75× less efficient per-GPU than Qwen 235B, reflecting the 1.1T total param count and MLA overhead.

## compressa-perf inference benchmark

### Install

```bash
pip install git+https://github.com/product-science/compressa-perf.git
```

### Config (`/tmp/compressa_kimi.yml`)

Adapted from the official Gonka config (6 scenarios) with:
- `model_name: moonshotai/Kimi-K2.5`
- `node_url: http://127.0.0.1:8080` (local MLNode)

Scenarios:
1. **small**: 10 prompts × 20k length, 5 tasks, 1 runner, 300 max_tokens
2. **concurrent**: 100 prompts × 2k, 200 tasks, 30 runners, 300 max_tokens
3. **longctx_single**: 10 prompts × 45k, 5 tasks, 1 runner, 1000 max_tokens
4. **longctx_5run**: 10 prompts × 45k, 10 tasks, 5 runners, 1000 max_tokens
5. **longctx_20run**: 20 prompts × 45k, 40 tasks, 20 runners, 1000 max_tokens
6. **longctx_10run**: 40 prompts × 45k, 60 tasks, 10 runners, 1000 max_tokens

### Run command

```bash
compressa-perf measure-from-yaml \
  --no-sign \
  --node_url http://127.0.0.1:8080 \
  --model_name moonshotai/Kimi-K2.5 \
  /tmp/compressa_kimi.yml
```

### Retrieve results

```bash
compressa-perf list --show-metrics --show-parameters
```

### Results — Experiment A (eager)

All 6 scenarios completed with **0 failed requests**.

| # | Scenario | Avg input tok | TTFT (s) | TPOT (s) | Latency (s) | In tok/s | Out tok/s | Total tok/s | RPS |
|---|----------|--------------:|---------:|---------:|------------:|---------:|----------:|------------:|----:|
| 1 | small (20k chars, 1 runner, 5 tasks, 300 out) | ~10k | 0.59 | 0.032 | 9.55 | 1043 | 31 | **1074** | 0.10 |
| 2 | concurrent (2k chars, 30 runners, 200 tasks, 300 out) | ~1k | 0.76 | 0.035 | 10.34 | 2784 | 830 | **3614** | 2.77 |
| 3 | longctx_single (45k chars, 1 runner, 5 tasks, 1000 out) | ~23k | 1.16 | 0.032 | 27.81 | 806 | 31 | **837** | 0.04 |
| 4 | longctx_5run (45k chars, 5 runners, 10 tasks, 1000 out) | ~23k | 2.69 | 0.036 | 34.19 | 3117 | 131 | **3248** | 0.14 |
| 5 | longctx_20run (45k chars, 20 runners, 40 tasks, 1000 out) | ~23k | **8.18** | 0.046 | 43.46 | **8414** | **355** | **8769** | 0.38 |
| 6 | longctx_10run (45k chars, 10 runners, 60 tasks, 1000 out) | ~23k | 2.33 | 0.041 | 37.31 | 5718 | 231 | **5950** | 0.25 |

**Peak throughput:** scenario 5 — **8769 total tok/s** with 20 concurrent runners at 23k input tokens, limited by prefill throughput (355 out vs 8414 in tok/s).

**TPOT stability:** 32-46 ms across all scenarios, no runaway latencies. Even at 20-way concurrency, no LATENCY > 120s.

### Results — Experiment B (compiled) — **SUCCESS after retry**

**First attempt failed.** The compiled vLLM engine crashed on scenario 1 (10k-token prefill) with `EngineDeadError` → `TimeoutError` in `shm_broadcast.acquire_read`. After the first timeout, all workers exited and the remaining 5 scenarios returned `503 No vLLM backend available`. First-attempt result: 60 FAILED_REQUESTS per scenario, 0 successful measurements.

**Root cause:** First-request cold start in compiled mode combines several heavy one-time costs that exceed the default `shm_broadcast` worker timeout of 300s:

1. **Marlin MoE kernel autotune.** `CompressedTensorsWNA16MarlinMoEMethod` autotunes Marlin kernels per unique shape on first use. For a 10k-token prefill across 60+ MoE layers, this can accumulate to tens of seconds (and per-shape autotunes stack up across the layers in a single forward pass).
2. **Inductor graph first-hit for range bucket [1, 32768].** First actual execution triggers kernel caching.
3. **FlashInfer MLA and DeepGEMM warmup** for specific shapes.
4. **Piecewise CUDA graph capture** for shapes not in pre-captured decode sizes `[1..256]`.

**Retry fix (which worked):**
1. Set `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900` before vLLM startup (default is 300s, set in `vllm/envs.py:199`, consumed in `multiproc_executor.py:277`).
2. Before compressa-perf, send two warmup chat requests directly to MLNode to trigger Marlin autotune outside the benchmark timing:
   - Request 1: ~5700 prompt tokens, 50 output tokens → took 1.0s
   - Request 2: ~11450 prompt tokens, 50 output tokens → took 1.3s
3. Run `compressa-perf` normally.

After the fix, all 6 scenarios completed with **0 failed requests**.

| # | Scenario | TTFT (s) | TPOT (s) | Latency (s) | In tok/s | Out tok/s | Total tok/s | RPS |
|---|----------|---------:|---------:|------------:|---------:|----------:|------------:|----:|
| 1 | small (20k chars, 1 runner) | 0.54 | 0.0124 | 3.72 | 2678 | 81 | **2759** | 0.269 |
| 2 | concurrent (2k chars, 30 runners) | 0.83 | 0.0275 | 8.25 | 3507 | 1045 | **4552** | 3.483 |
| 3 | longctx_single (45k chars, 1 runner) | 1.16 | 0.0126 | 10.09 | 2222 | 79 | **2301** | 0.099 |
| 4 | longctx_5run (45k chars, 5 runners) | 2.99 | 0.0217 | 21.35 | 5193 | 227 | **5420** | 0.231 |
| 5 | longctx_20run (45k chars, 20 runners) | 6.88 | 0.0414 | 37.90 | 11081 | 453 | **11534** | 0.494 |
| 6 | longctx_10run (45k chars, 10 runners) | 2.42 | 0.0328 | 30.01 | 7266 | 296 | **7562** | 0.324 |

**Peak throughput:** scenario 5 — **11,534 total tok/s** with 20 concurrent runners, 23k input tokens, 1000 max_tokens.

### Compressa-perf comparison: eager vs compiled

| # | Scenario | Eager tok/s | Compiled tok/s | Δ | Eager TPOT | Compiled TPOT | Δ |
|---|----------|------------:|---------------:|:--:|-----------:|--------------:|:--:|
| 1 | small (10k in, 1 runner) | 1074 | **2759** | **+157%** | 0.032 | 0.012 | **2.6× faster** |
| 2 | concurrent (1k in, 30 runners) | 3614 | **4552** | **+26%** | 0.035 | 0.028 | 1.3× faster |
| 3 | longctx_single (23k in, 1 runner) | 837 | **2301** | **+175%** | 0.032 | 0.013 | **2.5× faster** |
| 4 | longctx_5run (23k in, 5 runners) | 3248 | **5420** | **+67%** | 0.036 | 0.022 | 1.7× faster |
| 5 | longctx_20run (23k in, 20 runners) | 8769 | **11534** | **+32%** | 0.046 | 0.041 | 1.1× faster |
| 6 | longctx_10run (23k in, 10 runners) | 5950 | **7562** | **+27%** | 0.041 | 0.033 | 1.2× faster |

**Key findings:**

1. **Compiled beats eager in ALL scenarios**, peak speedup **+175%** on long-context single-runner. Average speedup ~+80%.
2. **Biggest wins on single-runner / low-concurrency scenarios** (1, 3) — these are prefill-bound and compile optimizations (FusedMoE, fused norm+quant, rotary embedding) pay off massively on the prefill path.
3. **Smaller wins on high-concurrency** (scenarios 2, 5) — saturated by KV cache bandwidth and scheduler overhead, compile gain shrinks to ~30%.
4. **TPOT consistently 1.1-2.6× faster in compiled mode** — Marlin INT4 MoE kernels + CUDA graphs pay off for decode too.

**Contrast with PoC benchmark (Test 1 vs 2):** PoC showed compiled == eager on best batch (both 1024 nonces/min). This is because PoC is a decode-only benchmark with fixed `seq_len=1024` and `k_dim=12` — the prefill path (where compile wins the most) is essentially unused, and decode throughput is limited by KV cache bandwidth which compile doesn't help with much at already-optimal batch sizes.

**Production recommendation:** for inference workloads (compressa-perf profile), **use compiled mode with custom_ops=["all"]** — it gives 27-175% throughput improvement with proper warmup. For PoC nonce generation, either mode works equally well, but eager is simpler and avoids cold-start complications.

## Artifacts

Saved under `gonka-deploy/artifacts/experiments/compressa-perf-results/` (relative to repo root):

- PoC benchmark logs: `b200_4x_kimi_poc_eager.log`, `b200_4x_kimi_poc_compiled.log`
- compressa-perf logs: `b200_4x_kimi_compressa_eager.log`, `b200_4x_kimi_compressa_compiled.log`
- compressa-perf SQLite (compiled): `b200_4x_kimi_compiled.sqlite`
- compressa-perf SQLite (eager): **lost** — overwritten before download (only metrics in log file remain)
- vLLM and MLNode logs were on the Vast.ai instance (`34401918`, now destroyed) and not preserved

## After completion

1. Fill in compressa-perf result tables above
2. Destroy instance: `vastai destroy instance 34401918`
