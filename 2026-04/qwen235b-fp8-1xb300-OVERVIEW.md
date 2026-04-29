# Qwen3-235B-A22B FP8 on 1×B300 — optimization sweep overview

**Date:** 2026-04-26 → 2026-04-29
**Model:** `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`
**Hardware:** 1× NVIDIA B300 SXM6 AC (Blackwell Ultra, sm_103a, 275 GB HBM3e)

This is the index for a five-run sweep that took 1×B300 PoC v2 throughput
from **832 nonces/min** (the regression observed when porting from the
v0.15-based image to v0.19) up to **1280 nonces/min** (best Phase-3) /
**1255 nonces/min** (steady-state). Cumulative gain **+54 % / +57 %** on
the same silicon, all production-deployable under the chain's
`DistThreshold=0.2` validator threshold.

## Runs in this sweep

| # | Folder | Best Phase-3 | Steady (1000+ nonces) | One-line summary |
|---|---|---:|---:|---|
| 1 | [qwen235b-fp8-1xb300-vllm019-baseline](qwen235b-fp8-1xb300-vllm019-baseline/) | 832 | 798 | vLLM 0.19 + TRITON, default MoE config (regression baseline) |
| 2 | [qwen235b-fp8-1xb300-vllm019-tuned-M32768](qwen235b-fp8-1xb300-vllm019-tuned-M32768/) | 832 | 823 | TRITON + benchmark_moe.py-tuned `E=128,N=1536` config (silenced default-MoE warning, +3 %) |
| 3 | [qwen235b-fp8-1xb300-vllm019-flashinfer-trtllm](qwen235b-fp8-1xb300-vllm019-flashinfer-trtllm/) | 1056 | 1053 | TRITON → FLASHINFER_TRTLLM via `VLLM_FLASHINFER_MOE_BACKEND=latency`. **+27 %**. Frozen as image `b300-k2`. |
| 4 | [qwen235b-fp8-1xb300-vllm020-flashinfer-trtllm](qwen235b-fp8-1xb300-vllm020-flashinfer-trtllm/) | 1152 | 1145 | vLLM 0.19 → 0.20 (+9 %), `--enforce-eager` to dodge compile overhead |
| 5 | [qwen235b-fp8-1xb300-vllm020-stockcompile](qwen235b-fp8-1xb300-vllm020-stockcompile/) | **1280** | **1255** | `--compilation-config '{"mode": 1}'` (STOCK_TORCH_COMPILE) — keeps `torch.compile` active without vLLM piecewise/passes overhead. **+11 %** over run 4. **★ Best config.** |

Plus one **operability follow-up** (no perf change vs. run 5):

| Folder | Outcome | What changed |
|---|---|---|
| [qwen235b-fp8-8xb300-watcher-cold-start-fix](qwen235b-fp8-8xb300-watcher-cold-start-fix/) | k3 → k4 (1280/min preserved) | Env-driven `WATCHER_GRACE_FIRST_HEALTHY=1` + `VLLM_RUNNER_TIMEOUT=3600` so mlnode survives the 22-min cold start of 4× parallel vLLM instances on 8×B300; TP=1 moved from defaults to forced (override-resistant against the network node's TP=2 injection). |

## Best configuration (from run 5)

### vLLM CLI (passed via MLNode `additional_args`)

```
--served-model-name Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
--trust-remote-code
--max-num-batched-tokens 65536
--compilation-config '{"mode": 1}'
```

### `runner.py` B300 hardcodes

```python
_b300_forced = {
    "--tensor-parallel-size":  "1",         # k4: moved from defaults to forced
    "--gpu-memory-utilization": "0.95",
    "--max-model-len":          "120000",   # under KV pool ceiling 125408
    "--logprobs-mode":          "processed_logprobs",
    "--compilation-config":     '{"mode": 1}',
}
_b300_defaults = {                          # only set if absent
    "--max-num-seqs": "128",
}
```

> **k4 note:** `--tensor-parallel-size` is now in `_b300_forced` (was in
> `_b300_defaults` through k3) so the Gonka network node's
> `--tensor-parallel-size 2` injection in `additional_args` is overwritten
> with the throughput-optimal TP=1 layout. See
> [`qwen235b-fp8-8xb300-watcher-cold-start-fix/`](qwen235b-fp8-8xb300-watcher-cold-start-fix/)
> for the production case that motivated this.

### Subprocess env (set by `runner.py` before vLLM spawn)

```
VLLM_USE_V1=1
VLLM_USE_FLASHINFER_MOE_FP8=1
VLLM_FLASHINFER_MOE_BACKEND=latency       # → FLASHINFER_TRTLLM (sm_100+)
VLLM_MOE_USE_DEEP_GEMM=0
VLLM_USE_DEEP_GEMM_E8M0=1
VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES=1
VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=1073741824  # 1 GiB
```

### Image-level ENV (Dockerfile)

```
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.9 9.0 10.0 12.0"
ENV TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas
ENV VLLM_ALLOW_INSECURE_SERIALIZATION=1
ENV VLLM_ENABLE_CUDA_COMPATIBILITY=0
```

### Auto-derived under `compilation_mode=1`

- `cudagraph_mode=NONE` (no piecewise/full graph capture)
- `splitting_ops=[]` (single-graph compile, not split at attention ops)
- `inductor_passes={}` (no vLLM custom passes)
- `compile_sizes=[]` (recompile per encountered shape — see warm-up note)

### sm_103a fixes (apply at image build time, see `mlnode/full/hw/b300/Dockerfile`)

1. Replace Triton's bundled `ptxas` with CUDA-12.9 `ptxas` (knows sm_103a)
2. `pip uninstall -y flashinfer-jit-cache` (precompiled for sm_120, not sm_103a — JIT-compile on first run cached at `/root/.cache/flashinfer/<ver>/103a/`)
3. Symlink every `nvidia/*/include/*.h` into `/usr/local/cuda/include/` (FlashInfer JIT-compile finds CUDA dev headers)
4. Replace `/usr/local/cuda-12.9/compat/libcuda.so` stub with a symlink to `/usr/lib/x86_64-linux-gnu/libcuda.so.1` (host driver 580.126)

### MLNode

- **k3 (deprecated):** `MAX_UNHEALTHY_COUNT = 9999` in `watcher.py` (blunt
  hammer to disable the kill-on-unhealthy threshold during cold-start
  FlashInfer JIT)
- **k4 (current):** env-driven session-aware first-healthy grace —
  `WATCHER_GRACE_FIRST_HEALTHY=1` + `VLLM_RUNNER_TIMEOUT=3600` (60 min).
  Operator-tunable without rebuild; preserves the fast-restart
  `MAX_UNHEALTHY_COUNT=3` semantics for post-startup crashes. See
  [`qwen235b-fp8-8xb300-watcher-cold-start-fix/`](qwen235b-fp8-8xb300-watcher-cold-start-fix/)
  for the design and the 8×B300 production crash that motivated it.

## Operational notes

1. **Warm-up is mandatory.** With `compilation_mode=1`, every new
   `(batch, seq_len)` shape triggers a `torch.compile` recompile that
   typically eats the entire 30 s measurement window. On a cold cache,
   the first PoC run returned 0 nonces for batch=8 and batch=16. Operators
   must run a one-shot warm-up over expected shapes before serving live
   PoC traffic. After that, the compile cache lands on disk
   (`/root/.cache/vllm/torch_compile_cache/`) and second-and-later starts
   warm up in ~70-100 s.

2. **vLLM 0.20.0 needs Gonka PoC v2 patches.** `pip install vllm==0.20.0`
   from upstream wipes out the PoC routes; restore them from
   [`gonka-deploy/image/rtx_pro_6000/vendor/vllm`](../../gonka-deploy/image/rtx_pro_6000/vendor/vllm)
   either by `Dockerfile.quick`'s find-and-tar overlay (recommended) or
   manually via the same `find . -name "*.py" | tar | tar` recipe.

3. **`batch=128` still doesn't run.** 128 × seq_len=1024 = 131 072
   tokens > current `max-num-batched-tokens=65536` → the PoC engine
   returns 0 nonces. To unblock batch=128, raise
   `max-num-batched-tokens` to 131 072, but the larger scheduler buffers
   will drop available KV down to ~108 K tokens — `max-model-len` would
   need to come down with it. We did not measure whether batch=128 would
   then beat batch=64 throughput.

4. **Bit-compat across runs.**
   - **TRITON path is bit-identical** between runs (1000/1000 exact match,
     L2 = 0.0). Suitable for protocols requiring exact match.
   - **FLASHINFER_TRTLLM path drifts** L2 ≈ 0.085 between runs (likely
     FlashInfer Autotuner picking different kernels per session) — but
     this is **5× under** the chain's `DistThreshold = 0.2` and **50× under**
     `PMismatch = 0.1`, so it passes validation comfortably (p-value = 1.0,
     no fraud flag). All five runs above pairwise-validate against each
     other under the production thresholds.
   - Library defaults (`DistThreshold = 0.02`, `PMismatch = 0.001`) are
     stricter and would reject every FLASHINFER_TRTLLM run as fraud —
     do not use library defaults for cross-validator checks.

## Production deployment status

| Image | Per-card | 8-GPU host | Notes |
|---|---:|---:|---|
| `mlnode-full:0.2.12-vllm0.19.0-b300-k2` (FlashInfer TRTLLM) | 1056 | 8448 | Superseded by k3/k4. |
| `mlnode-full:0.2.12-vllm0.19.0-b300-k3` (mode=1 compile) | 1152 | 9216 | Superseded by k4. |
| `mlnode-full:0.2.12-vllm0.20.0-b300-k3` (mode=1 compile) | **1280** | **10240** | Superseded by k4. |
| `mlnode-full:0.2.12-vllm0.19.0-b300-k4` (cold-start fix, fall-back) | 1152 | 9216 | Same per-card perf as k3. Adds: env-driven cold-start tolerance, session-aware watcher grace, TP=1 forced (override-resistant). |
| `mlnode-full:0.2.12-vllm0.20.0-b300-k4` ★ **current production** | **1280** | **10240** | Same per-card perf as k3. Adds: env-driven cold-start tolerance, session-aware watcher grace, TP=1 forced (override-resistant). See [`qwen235b-fp8-8xb300-watcher-cold-start-fix/`](qwen235b-fp8-8xb300-watcher-cold-start-fix/). |

> The k3 GHCR images are NOT deleted (immutable -kN tag policy keeps
> existing pinned deployments working). The k3 dashboard entries are
> removed from <https://registry.kaitaku.ai/> so new operators do not
> pick the deprecated tag.

## Throughput table (canonical)

| Run | Best Phase-3 | Steady | Δ vs. baseline |
|---|---:|---:|---|
| 0.19 TRITON, default MoE | 832 | 798 | (baseline) |
| 0.19 TRITON, MoE M=32768 tuned | 832 | 823 | +0 % / +3 % |
| 0.19 FLASHINFER_TRTLLM | 1056 | 1053 | +27 % / +32 % |
| 0.20 FLASHINFER_TRTLLM, `--enforce-eager` | 1152 | 1145 | +38 % / +43 % |
| **0.20 FLASHINFER_TRTLLM, `compilation_mode=1`** | **1280** | **1255** | **+54 % / +57 %** |

Reference numbers from outside this sweep:
- `kaitakuai/segovchik/gonka-b300-image:3.0.13-b300-tp1` (vLLM 0.15.1, FlashInfer TRTLLM): 1024 nonces/min on the same B300. Beaten by all FLASHINFER_TRTLLM runs in this sweep (1056 / 1152 / **1280**).
