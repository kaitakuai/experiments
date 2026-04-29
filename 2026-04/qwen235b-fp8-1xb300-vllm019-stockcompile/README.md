# Qwen3-235B-A22B FP8 — 1×B300 — vLLM 0.19.0 — FlashInfer TRTLLM + STOCK_TORCH_COMPILE

**Date:** 2026-04-29
**Model:** `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`
**Quantization:** FP8 (block-wise w8a8, block_shape=[128,128])
**Hardware:** 1× NVIDIA B300 SXM6 AC (Blackwell Ultra, sm_103a, 275 GB HBM3e)
**vLLM:** 0.19.0 (Kaitaku PoC v2 build, base `kaitakuai/vllm:v0.19.0-pocv2-alpha3`)
**MLNode:** 0.2.12 (image `ghcr.io/kaitakuai/mlnode-full:0.2.12-vllm0.19.0-b300-k3` / digest TBD)

## Summary

Backports the b300-k3 STOCK_TORCH_COMPILE optimization to the vLLM 0.19.0 line.
Best Phase-3 throughput: **1152 nonces/min @ batch=64**, stable across cold and
warm compile-cache runs. That is **+9 % over b300-k2** (vLLM 0.19 + FlashInfer
TRTLLM + `--enforce-eager`, 1056 nonces/min) and equal to the intermediate
**vLLM 0.20.0 + `--enforce-eager`** result measured on the same silicon (1152
nonces/min, see [qwen235b-fp8-1xb300-vllm020-flashinfer-trtllm/](../qwen235b-fp8-1xb300-vllm020-flashinfer-trtllm/)).

The fully-stacked `b300-k3` (vLLM 0.20 + STOCK_TORCH_COMPILE, 1280 nonces/min,
[qwen235b-fp8-1xb300-vllm020-stockcompile/](../qwen235b-fp8-1xb300-vllm020-stockcompile/))
remains the throughput leader. v0.19+mode=1 documented here is the appropriate
fall-back for operators on the more-validated vLLM 0.19.0 base.

Cumulative gain vs the original 0.19 + TRITON baseline (832 nonces/min):
**+38 %**.

## Configuration

### vLLM CLI (passed through the b300-k3 image's `runner.py` patcher)

```
--served-model-name Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
--trust-remote-code
--tensor-parallel-size 1
--max-num-seqs 128
--gpu-memory-utilization 0.95
--max-model-len 120000
--max-num-batched-tokens 65536
--logprobs-mode processed_logprobs
--compilation-config '{"mode": 1}'
```

This is bit-identical to the b300-k3 (vLLM 0.20) CLI. The `b300.py` runner
patcher is shared between the two image variants — only the underlying base
image (and therefore vLLM/torch/FlashInfer versions) differs.

### Image-level ENV (`_HW_ENV_BLOCKS["b300"]`)

```
VLLM_USE_FLASHINFER_MOE_FP8=1
VLLM_FLASHINFER_MOE_BACKEND=latency       # → FLASHINFER_TRTLLM (sm_100+)
VLLM_USE_V1=1
VLLM_MOE_USE_DEEP_GEMM=0
VLLM_USE_DEEP_GEMM_E8M0=1
VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES=1
VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=1073741824   # 1 GiB
```

Note: DeepGEMM E8M0/TMA env vars are likely no-ops on FlashInfer 0.6.6 (the
features were added in 0.6.8.post1). Kept for parity with the v0.20-k3 image
and harmless on this base.

### sm_103a fixes (image build time, see `mlnode/full/hw/b300/Dockerfile`)

1. Replace Triton's bundled `ptxas` with system CUDA `/usr/local/cuda/bin/ptxas`
   (resolves to CUDA 12.9 ptxas on this base; knows sm_103a)
2. `pip uninstall flashinfer-jit-cache -y` followed by `rm -rf` of the empty
   stub dir (the rm is harmless on FlashInfer 0.6.6 but mandatory on
   0.6.8.post1 — see b300-k3 source-fix history)
3. Symlink every `nvidia/*/include/*.h` into `/usr/local/cuda/include/`
4. Replace `/usr/local/cuda-*/compat/libcuda.so` stub with a symlink to
   `/usr/lib/x86_64-linux-gnu/libcuda.so.1` (host driver 580.126)

## Hardware

| Component | Value |
|---|---|
| GPU | 1× NVIDIA B300 SXM6 AC (sm_103a, 275 GB HBM3e) |
| Driver | 580.126.09 |
| Host CUDA | 13.0 |
| Image CUDA | 12.9.1 |
| Host RAM | 270 GiB |

## Software

| Component | Version |
|---|---|
| Image | `ghcr.io/kaitakuai/mlnode-full:0.2.12-vllm0.19.0-b300-k3` |
| Image base | `ghcr.io/kaitakuai/vllm:v0.19.0-pocv2-alpha3` |
| OS (image) | Ubuntu 22.04 (Python 3.12.13) |
| torch | 2.10.0+cu129 |
| vllm | 0.19.0 (Kaitaku PoC v2) |
| flashinfer-python | 0.6.6 (with our `flashinfer-jit-cache` rm + JIT-on-demand) |
| Triton | 3.6 (bundled ptxas swapped for system CUDA 12.9 ptxas) |

## Startup profile (cold cache)

| Phase | Time |
|---|---|
| Weight load (220.35 GiB into VRAM) | ~22 s |
| FlashInfer TRTLLM JIT compile + warmup | the bulk of cold-start |
| `compilation_mode=1` per-shape compile (decode shapes) | added on top |
| **Cold start total (vLLM up → MLNode "running")** | **~670 s first run** |

Subsequent restarts warm from `/root/.cache/flashinfer/<ver>/103a/` and
`/root/.cache/vllm/torch_compile_cache/` (~70-100 s).

## Results — Phase 3 (5×35 s sweep)

### Cold compile cache (first run after fresh container)

| Batch | Nonces (30 s) | Nonces/min | Note |
|---|---|---|---|
| 2 | 2 | 4 | per-shape compile dominates |
| 8 | 0 | 0 | per-shape compile dominates |
| 16 | 384 | 768 | per-shape compile dominates |
| 32 | 544 | 1088 | per-shape compile dominates |
| **64** | **576** | **1152** ★ | warmup absorbs compile |

### Warm compile cache (re-run, all shapes already compiled)

| Batch | Nonces (30 s) | Nonces/min |
|---|---|---|
| 2 | 448 | 896 |
| 8 | 536 | 1072 |
| 16 | 560 | 1120 |
| 32 | 480 | 960 |
| **64** | **576** | **1152** ★ |

batch=64 is rock-stable at 1152 nonces/min across cold and warm cache. Smaller
batches show noise (var ±10 %) typical of the 30 s measurement window.

## Results — comparison across the 1×B300 sweep

| Configuration | Best Phase-3 | Steady-state | Δ vs baseline |
|---|---:|---:|---|
| 0.19 TRITON, default MoE config | 832 | 798 | (baseline) |
| 0.19 FlashInfer TRTLLM + `--enforce-eager` (b300-k2) | 1056 | 1053 | +27 % / +32 % |
| **0.19 FlashInfer TRTLLM + `compilation_mode=1` (this run)** | **1152** | n/a | **+38 % / —** |
| 0.20 FlashInfer TRTLLM + `--enforce-eager` | 1152 | 1145 | +38 % / +43 % |
| 0.20 FlashInfer TRTLLM + `compilation_mode=1` (b300-k3) | **1280** | **1255** | **+54 % / +57 %** |

Layer-by-layer attribution of the gap to b300-k3 (1280):
- **Compile mode:** v0.19 + mode=1 (1152) − v0.19 + enforce-eager (1056) = **+96 nonces/min** (+9 %)
- **vLLM stack version:** v0.20 + mode=1 (1280) − v0.19 + mode=1 (1152) = **+128 nonces/min** (+11 %)
  - Composed of newer torch (2.11+cu130 vs 2.10+cu129), FlashInfer (0.6.8.post1 vs 0.6.6),
    and vLLM 0.19→0.20 source delta (notably DeepGEMM E8M0/TMA paths)

## Failed attempts to backport the v0.20 stack onto v0.19 base

We tried two cherry-pick paths to close the 1152→1280 gap without changing
the vLLM version. Both failed loud, documenting them here as negative
results:

### 1. Upgrade FlashInfer alone (0.6.6 → 0.6.8.post1)

```bash
pip install --upgrade flashinfer-python==0.6.8.post1 flashinfer-cubin==0.6.8.post1
```

vLLM started cleanly, FLASHINFER_TRTLLM Fp8 MoE backend selected, no obvious
errors. But measured throughput **regressed**:

| Batch | fi 0.6.6 (baseline) | fi 0.6.8.post1 | Δ |
|---|---:|---:|---:|
| 2 | 896 | 732 | −18 % |
| 8 | 1072 | 736 | −31 % |
| 16 | 1120 | 1088 ★ | −3 % |
| 32 | 960 | 1088 | +13 % |
| **64** | **1152 ★** | **1024** | **−11 %** |

Likely cause: FlashInfer 0.6.8.post1 ships pre-built CUBIN kernels expecting
CUDA 13.0 ABI / newer ptxas features; on the v0.19 base (CUDA 12.9 system ptxas
+ torch 2.10+cu129), the relevant kernels fall back to a slow path or hit a
suboptimal autotune branch. The CUBIN/JIT version-match check itself works
(we kept `flashinfer-cubin==0.6.8.post1` aligned), but runtime selection picks
worse kernels.

Artifact: [`artifacts/bench-flashinfer-068-regression.log`](artifacts/bench-flashinfer-068-regression.log).

### 2. Full torch 2.11 + CUDA 13.0 runtime upgrade

```bash
pip install torch==2.11.0    # pulls nvidia-{cublas,cudnn,nccl,nvshmem,...}-cu13
```

torch + CUDA 13 runtime libs installed cleanly:

```
torch: 2.11.0+cu130, cuda 13.0
flashinfer: 0.6.8.post1
cuda available: True, device count: 1
```

But on `import vllm`:

```
ImportError: /usr/local/lib/python3.12/dist-packages/vllm/_C.abi3.so:
  undefined symbol: _ZN3c1013MessageLoggerC1EPKciib
```

vLLM 0.19's compiled C++ extension `vllm/_C.abi3.so` is built against
torch 2.10's `libtorch` ABI. torch 2.11 changed the `MessageLogger`
constructor signature, so the extension's symbol resolution fails at
load. **vLLM 0.19 binary cannot use torch 2.11 — full source rebuild
against the new stack is required.**

Rebuilding vLLM 0.19 from source (`pip install --no-binary vllm
vllm==0.19.0` on a CUDA 13.0 + torch 2.11 environment) was estimated at
~45-60 min compile time and would still cap at the v0.20-k3 ceiling
(1280) in best case. We did not pursue it.

## Verdict

- **Ship `b300-k3` with vLLM 0.19** at 1152 nonces/min as the
  more-validated fall-back. The +9 % gain over `b300-k2` is real and
  comes purely from the `--compilation-config '{"mode": 1}'` switch on
  the same vLLM/CUDA/FlashInfer stack — no upstream churn.
- **Ship `vllm0.20.0-b300-k3`** at 1280 nonces/min for operators
  willing to track the newer vLLM line.
- The 1280 ceiling is **stack-coupled**: torch 2.11 + CUDA 13.0 +
  FlashInfer 0.6.8.post1 + vLLM 0.20 source come together. Cherry-picks
  are not viable.

## Files

- [`artifacts/bench-cold-cache.log`](artifacts/bench-cold-cache.log) —
  poller + Phase 3 sweep on first run after fresh container (cold
  compile cache; cold-cache batch≤32 numbers reflect compile time
  eating the 30 s window)
- [`artifacts/bench-warm-cache.log`](artifacts/bench-warm-cache.log) —
  same sweep re-run with warm cache (steady-state numbers)
- [`artifacts/bench-flashinfer-068-regression.log`](artifacts/bench-flashinfer-068-regression.log) —
  FlashInfer 0.6.8.post1 cherry-pick experiment showing the −11 % regression

## Reproduction

```bash
# 1. Pull image
docker pull ghcr.io/kaitakuai/mlnode-full:0.2.12-vllm0.19.0-b300-k3

# 2. Run container
docker run -d --name b300-k3 --gpus all --ipc=host \
  -p 5001:5001 -p 8081:8081 \
  -v <huggingface-cache-dir>:/data/hf:ro \
  ghcr.io/kaitakuai/mlnode-full:0.2.12-vllm0.19.0-b300-k3 \
  python -m uvicorn api.app:app --host 0.0.0.0 --port 8081 --app-dir /app/packages/api/src

# 3. Trigger model load (b300.py runner-patcher injects the b300 hardcodes)
curl -X POST http://127.0.0.1:8081/api/v1/inference/up/async \
  -H 'Content-Type: application/json' -d '{
    "model": "/data/hf/Qwen3-235B-A22B-Instruct-2507-FP8",
    "dtype": "auto",
    "additional_args": [
      "--served-model-name", "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
      "--trust-remote-code"
    ]
  }'

# 4. Wait ~11 min for cold start (FlashInfer JIT for sm_103a + per-shape compile)

# 5. Phase 3 sweep
docker exec b300-k3 python3 /tmp/run_pow_generation.py --phase 3 --skip-check
```
