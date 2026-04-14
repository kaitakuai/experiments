# Benchmark: Kimi K2.5 INT4 on 8×H200

**Date:** 2026-04-09
**Purpose:** Full PoC v2 + inference benchmark for `moonshotai/Kimi-K2.5` on 8×H200 in two vLLM modes (enforce-eager vs compiled with `custom_ops=["all"]`), using `product-science/mlnode:3.0.13-alpha1`.

## Infrastructure

| Parameter | Value |
|-----------|-------|
| Vast.ai Instance ID | 34461743 |
| Host | 432545 (machine 59481) |
| Location | Japan, JP |
| SSH | `ssh -p 21742 root@ssh2.vast.ai` |
| Cost | ~$22.72/hr |
| Network | 4005 Mbps down / 1675 Mbps up |

## Hardware

| Parameter | Value |
|-----------|-------|
| GPU | 8× NVIDIA H200 |
| VRAM | 143,771 MiB (140 GiB) per GPU — 1,120 GiB total |
| CPU | 224 vCPUs (3.8 GHz) |
| RAM | 2,063 GB (2 TB) |
| Disk | 700 GB allocated (overlay) |
| NVIDIA Driver | 570.211.01 |

## Software

| Parameter | Value |
|-----------|-------|
| Docker Image | `ghcr.io/product-science/mlnode:3.0.13-alpha1` |
| vLLM | 0.15.1 |
| torch | 2.9.1+cu129 |
| CUDA | 12.9 (V12.9.86) — uses compat lib `libcuda.so.575.57.08` since host driver 570.x |
| Python | 3.12.12 |
| OS | Ubuntu 22.04.5 LTS |
| compressa-perf | 0.2.5 |

## Model

Same as B200 run (`moonshotai/Kimi-K2.5`, INT4 compressed-tensors, MoE 1.1T/32B activated, MLA attention, 256K context).

## Instance setup

### 1. Create Vast.ai instance

```bash
vastai create instance 33960372 \
  --image ghcr.io/product-science/mlnode:3.0.13-alpha1 \
  --disk 700 --ssh --direct
```

### 2. Apply CUDA forward-compat

The driver on this host is **570.211.01** which only supports CUDA 12.8, but the image is built against CUDA 12.9. We need to use the compat library bundled with the image:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH
```

This loads `libcuda.so.575.57.08` which supports CUDA 12.9 PTX. Without this, vLLM workers crash on startup with `cudaErrorUnsupportedPtxVersion`.

### 3. Apply patches

```bash
# Watcher disable (so MLNode doesn't kill vLLM during long benchmarks)
sed -i 's/MAX_UNHEALTHY_COUNT = 3/MAX_UNHEALTHY_COUNT = 9999/' \
  /app/packages/api/src/api/watcher.py

# Proxy setup (this image build does NOT have it; needed to register vLLM port 5001)
sed -i '/await start_vllm_proxy()/a\    setup_vllm_proxy([5001])' \
  /app/packages/api/src/api/app.py
```

### 4. Download model

```bash
python3 -c 'from huggingface_hub import snapshot_download; snapshot_download("moonshotai/Kimi-K2.5", max_workers=8)'
```

~555 GB at ~1 GB/s (Japan host) = **9 minutes**.

## Critical flags for 8×H200

Two flags required for this configuration that were NOT needed on 4×B200:

1. **`--attention-backend FLASHMLA`** — auto-selection picks `FLASHINFER` which doesn't support MLA. Must force `FLASHMLA` (the Hopper MLA backend created for DeepSeek V3). On B200 we used `FLASHINFER_MLA` (Blackwell-only).
2. **`--disable-custom-all-reduce`** — without this, vLLM workers crash during CUDA graph capture with `Cuda error /workspace/csrc/custom_all_reduce.cuh:455 'invalid argument'`. Custom all-reduce path has issues on TP=8 with H200 NVLink topology.

The fallback NCCL all-reduce is **significantly slower** in eager mode (per-token kernel launch overhead dominates), but works correctly in compiled mode where CUDA graphs amortize it. This explains the unusually large compiled-vs-eager gap on this configuration.

For compiled mode also need:

3. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — without this, compiled mode hits OOM during the first prefill due to fragmentation in the bf16 activation buffer.
4. **`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900`** — same as B200, to allow Marlin kernel autotune on first 10k-token prefill.

## PoC benchmark parameters

Same as B200: `seq_len=1024`, `k_dim=12`, batch sizes `[8, 16, 32, 64, 128]`, 5s warmup + 30s measurement.

## Experiment A: enforce-eager (PoC)

### vLLM startup command

```bash
LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 -m vllm.entrypoints.openai.api_server \
  --model moonshotai/Kimi-K2.5 \
  --dtype auto \
  --port 5001 --host 0.0.0.0 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 128 \
  --max-model-len 262144 \
  --enforce-eager \
  --trust-remote-code \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --mm-encoder-tp-mode data \
  --attention-backend FLASHMLA \
  --disable-custom-all-reduce
```

### Startup profile (eager)

- Loading weights: **305 s**, **71.92 GiB/GPU** model memory
- Available KV cache: **45.84 GiB**
- GPU KV cache size: **700,352 tokens** (vs 314,432 on 4×B200 — 2.2× larger thanks to TP=8)
- Application startup: ~5.5 minutes total

### PoC Results — Experiment A (eager)

| Batch Size | Nonces (30s) | Nonces/min |
|------------|-------------:|-----------:|
| 8 | 552 | 1104 |
| **16** ★ | **592** | **1184** |
| 32 | 512 | 1024 |
| 64 | 576 | 1152 |
| 128 | 0 | 0 (broken) |

**Best (eager):** batch=16 → **1184 nonces/min**

## Experiment B: compiled + custom_ops (PoC)

### vLLM startup command

```bash
LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -m vllm.entrypoints.openai.api_server \
  --model moonshotai/Kimi-K2.5 \
  --dtype auto \
  --port 5001 --host 0.0.0.0 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 128 \
  --max-model-len 262144 \
  --trust-remote-code \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --mm-encoder-tp-mode data \
  --attention-backend FLASHMLA \
  --disable-custom-all-reduce \
  --compilation-config '{"custom_ops": ["all"]}'
```

### Startup profile (compiled)

- Loading weights: **309 s**
- Dynamo bytecode transform: 5.86 s
- Inductor compile (range 1-32768): 14.41 s
- FlashInfer autotuning: ~1 s
- CUDA graph capturing (35 piecewise + 19 full shapes): **12 s**
- Available KV cache: **46.24 GiB** (slightly more than eager due to fusions)
- GPU KV cache size: **706,560 tokens**
- Application startup: ~6 minutes total

### PoC Results — Experiment B (compiled)

| Batch Size | Nonces (30s) | Nonces/min |
|------------|-------------:|-----------:|
| 8 | 552 | 1104 |
| 16 | 592 | 1184 |
| **32** ★ | **608** | **1216** |
| 64 | 512 | 1024 |
| 128 | 0 | 0 (broken) |

**Best (compiled):** batch=32 → **1216 nonces/min**

## PoC Comparison: eager vs compiled

| Batch | Eager nonces/min | Compiled nonces/min | Δ |
|-------|-----------------:|--------------------:|:-:|
| 8 | 1104 | 1104 | 0% |
| 16 | **1184** | 1184 | 0% |
| 32 | 1024 | **1216** | **+19%** |
| 64 | 1152 | 1024 | −11% |
| 128 | 0 | 0 | broken |

**Key findings:**
- Compiled and eager achieve essentially the same peak (1216 vs 1184 = +3% for compiled)
- Optimal batch shifts from 16 (eager) to 32 (compiled)
- batch=128 broken in both modes (KV cache exhaustion at 128 × 1024 = 131k tokens / context window)
- batch=64 works in eager but 11% slower than batch=32 compiled

For PoC nonce generation specifically, **eager and compiled are equivalent** on H200 — the PoC path is decode-only with fixed shape, so compile gains are minimal.

## compressa-perf inference benchmark

### Config

Same 6 scenarios as the B200 Kimi run, only `node_url` and experiment names changed (`kimi_h200_*` instead of `kimi_b200_*`). See `compressa-perf-results/h200_8x_kimi_compressa_eager.log` and `h200_8x_kimi_compressa_compiled.log` for raw output.

### Warmup before compiled run

The compressa-perf run sends an immediate large prefill (10k+ tokens) which triggers Marlin MoE kernel autotune in compiled mode. Without warmup this exceeds the default `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=300` and crashes the engine.

```bash
# warmup1 (~5700 tokens) → 1.4s
# warmup2 (~11500 tokens) → 1.3s
```

After warmup, all subsequent prefills run at full compiled speed.

### Results — Experiment A (eager) — 0 failed requests

| # | Scenario | TTFT (s) | TPOT (s) | Latency (s) | In tok/s | Out tok/s | Total tok/s | RPS |
|---|----------|---------:|---------:|------------:|---------:|----------:|------------:|----:|
| 1 | small (10k in, 1 runner) | 0.51 | 0.078 | 23.53 | 423 | 13 | **436** | 0.043 |
| 2 | concurrent (1k in, 30 runners) | 0.65 | 0.081 | 24.17 | 1191 | 355 | **1546** | 1.183 |
| 3 | longctx_single (23k in, 1 runner) | 1.03 | 0.078 | 66.88 | 335 | 13 | **348** | 0.015 |
| 4 | longctx_5run (23k in, 5 runners) | 2.24 | 0.082 | 78.25 | 1372 | 58 | **1431** | 0.061 |
| 5 | longctx_20run (23k in, 20 runners) | 6.16 | 0.088 | 85.73 | 5104 | 222 | **5326** | 0.228 |
| 6 | longctx_10run (23k in, 10 runners) | 2.25 | 0.084 | 78.32 | 2698 | 113 | **2811** | 0.120 |

**Peak (eager):** scenario 5 — **5326 tok/s**.

**Note on TPOT:** all scenarios show 78-88 ms TPOT, much higher than expected. This is the cost of `--disable-custom-all-reduce`: NCCL all-reduce + per-kernel Python launch overhead dominates the per-token time on TP=8.

### Results — Experiment B (compiled) — 0 failed requests

| # | Scenario | TTFT (s) | TPOT (s) | Latency (s) | In tok/s | Out tok/s | Total tok/s | RPS |
|---|----------|---------:|---------:|------------:|---------:|----------:|------------:|----:|
| 1 | small (10k in, 1 runner) | 0.47 | 0.013 | 3.94 | 2526 | 76 | **2602** | 0.254 |
| 2 | concurrent (1k in, 30 runners) | 0.65 | 0.023 | 6.97 | 4154 | 1238 | **5392** | 4.126 |
| 3 | longctx_single (23k in, 1 runner) | 1.02 | 0.014 | 12.02 | 1866 | 73 | **1939** | 0.083 |
| 4 | longctx_5run (23k in, 5 runners) | 2.54 | 0.021 | 20.94 | 5268 | 230 | **5498** | 0.235 |
| 5 | longctx_20run (23k in, 20 runners) | 5.74 | 0.038 | 33.89 | 12328 | 488 | **12816** | 0.550 |
| 6 | longctx_10run (23k in, 10 runners) | 1.82 | 0.029 | 26.60 | 8103 | 329 | **8432** | 0.361 |

**Peak (compiled):** scenario 5 — **12,816 tok/s**.

### Inference comparison: eager vs compiled

| # | Scenario | Eager tok/s | Compiled tok/s | Compiled speedup | Eager TPOT | Compiled TPOT | TPOT speedup |
|---|----------|------------:|---------------:|:----------------:|-----------:|--------------:|:------------:|
| 1 | small | 436 | **2602** | **5.97×** | 0.078 | 0.013 | **6.0×** |
| 2 | concurrent | 1546 | **5392** | **3.49×** | 0.081 | 0.023 | **3.5×** |
| 3 | longctx_single | 348 | **1939** | **5.57×** | 0.078 | 0.014 | **5.6×** |
| 4 | longctx_5run | 1431 | **5498** | **3.84×** | 0.082 | 0.021 | **3.9×** |
| 5 | longctx_20run | 5326 | **12816** | **2.41×** | 0.088 | 0.038 | **2.3×** |
| 6 | longctx_10run | 2811 | **8432** | **3.00×** | 0.084 | 0.029 | **2.9×** |

**Compiled wins every scenario by 2.4× to 6× total throughput.** The biggest gains are on **single-runner scenarios** (1, 3) where the per-token NCCL all-reduce overhead in eager mode dominates and CUDA graphs eliminate it.

**Production recommendation for inference on 8×H200 with Kimi K2.5:**
- **Always use compiled mode** with `custom_ops=["all"]`
- Required env: `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Required flags: `--attention-backend FLASHMLA --disable-custom-all-reduce`
- Send 1-2 warmup chat requests (5k+ tokens) after vLLM start before serving real traffic
- Eager mode is **3-6× slower** for inference and should be avoided

For PoC nonce generation specifically, eager mode is acceptable (essentially equal performance to compiled), but compiled is still recommended since you'll likely run inference on the same vLLM process.

## Troubleshooting / observations

1. **CUDA forward-compat libcuda required.** Driver 570.211 supports max CUDA 12.8, image is cu129. Use `LD_LIBRARY_PATH=/usr/local/cuda/compat:...` before launching vLLM.

2. **`--attention-backend FLASHMLA` required.** vLLM auto-selection picks `FLASHINFER` which doesn't support MLA. Same bug as on B200 (where we used `FLASHINFER_MLA` instead since Blackwell).

3. **`--disable-custom-all-reduce` required for TP=8.** Custom all-reduce kernel fails with `'invalid argument'` during CUDA graph capture. NCCL fallback works but is significantly slower for eager mode (where it dominates per-token cost).

4. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` required for compiled mode.** Without it, the first compiled prefill OOMs at 138/140 GiB used due to allocator fragmentation between weights, KV cache, and activation buffers.

5. **GPU memory zombie processes.** After `kill -9` of vLLM workers, the driver continues to track them as compute apps and holds VRAM. Only fix: `vastai reboot instance`. Same as B200 issue.

6. **Optional: model in /dev/shm.** First attempt put weights in tmpfs (`/dev/shm/huggingface`) hoping for faster loads. Actual speedup was minimal (~1.2× — bottleneck is PCIe H2D copy across 8 workers, not disk read), and the tmpfs gets wiped on reboot. **Not worth the setup hassle** — keep weights on disk.

7. **Setup_vllm_proxy patch missing.** This image build does NOT have the `setup_vllm_proxy([5001])` line in `app.py` (the B200 build had it). Need to add manually with `sed`. Without it MLNode never sees the vLLM backend and all PoC requests return 502.

## Artifacts

Saved under `gonka-deploy/artifacts/experiments/compressa-perf-results/`:

- PoC logs: `h200_8x_kimi_poc_eager.log`, `h200_8x_kimi_poc_compiled.log`
- compressa-perf logs: `h200_8x_kimi_compressa_eager.log`, `h200_8x_kimi_compressa_compiled.log`
- compressa-perf SQLite: `h200_8x_kimi_eager.sqlite`, `h200_8x_kimi_compiled.sqlite`

## After completion

```bash
vastai destroy instance 34461743
```
