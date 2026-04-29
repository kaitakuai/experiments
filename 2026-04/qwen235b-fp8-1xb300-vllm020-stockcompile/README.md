# Qwen3-235B-A22B FP8 — 1×B300 — vLLM 0.20.0 — FLASHINFER TRTLLM + STOCK_TORCH_COMPILE

**Date:** 2026-04-29
**Model:** `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`
**Quantization:** FP8 (block-wise w8a8, block_shape=[128,128])
**Hardware:** 1× NVIDIA B300 SXM6 AC (Blackwell Ultra, sm_103a, 275 GB HBM3e)
**vLLM:** 0.20.0 (upstream `vllm/vllm-openai:v0.20.0` + Gonka PoC v2 patches)

## Summary

New best result on 1×B300: **1280 nonces/min @ batch=64** (Phase-3 peak),
**1255 nonces/min steady-state** on a 1088-nonce continuous collection.
That is **+11 % over the previous best** (0.20 + `--enforce-eager`,
1152 / 1145 nonces/min) and **+54 %** over the original vLLM 0.19 + TRITON
baseline (832 / 798).

The unlock here is the `compilation_mode` knob: switching from
**`VLLM_COMPILE` (mode 3, default)** to **`STOCK_TORCH_COMPILE` (mode 1)**
removes the vLLM-specific compile passes and graph splitting that were
adding a hidden ~12 % overhead to the PoC pipeline (visible as the 1024 cap
on every batch ≥ 16 in earlier runs). Stock `torch.compile` keeps inference
compiled (so non-PoC `/v1/*` requests still benefit) **without** dragging
the PoC eager forward into vLLM's compile machinery.

## Configuration changes vs. previous best (`--enforce-eager`)

```diff
- --enforce-eager
+ --compilation-config '{"mode": 1}'
```

Effect on auto-derived compile config (taken from the engine startup log):

| Field | enforce-eager (mode=NONE) | mode=1 (STOCK_TORCH_COMPILE) |
|---|---|---|
| `compilation_mode` | `CompilationMode.NONE` | `CompilationMode.STOCK_TORCH_COMPILE` |
| `cudagraph_mode` | `NONE` | `NONE` (auto, since stock compile does not capture) |
| `splitting_ops` | empty | empty (stock compile does not split) |
| `compile_sizes` | empty | empty |
| `pass_config` | all flags `False`/no-op | `fuse_norm_quant=True, fuse_act_quant=True` (defaults, but stock compile does not run vLLM's custom passes) |
| `inductor_passes` | none | none |

Net effect: torch.compile fires once per (batch, seq) shape and the
compiled graph is reused on warm calls. PoC `skip_compiled=True` continues
to run the model eager, but the rest of the pipeline (sampler, scheduler,
output processor) gets the benefit of the compiled main path without
paying for vLLM's piecewise/CUDA-graph machinery — which is what was
costing PoC the 1024 → 1152 → 1280 climb.

Full additional_args in the API call:

```json
{
  "additional_args": [
    "--served-model-name", "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
    "--trust-remote-code",
    "--max-num-batched-tokens", "65536",
    "--compilation-config", "{\"mode\": 1}"
  ]
}
```

The runner.py B300 hardcodes (still active) add:
```
--max-model-len 120000
--gpu-memory-utilization 0.95
--max-num-seqs 128
--tensor-parallel-size 1
--logprobs-mode processed_logprobs
```

Subprocess env (unchanged from previous best):
```
VLLM_USE_V1=1
VLLM_USE_FLASHINFER_MOE_FP8=1
VLLM_FLASHINFER_MOE_BACKEND=latency       # → FLASHINFER_TRTLLM
VLLM_MOE_USE_DEEP_GEMM=0
VLLM_USE_DEEP_GEMM_E8M0=1
VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES=1
VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=1073741824
```

## Hardware

| Component | Value |
|---|---|
| GPU | 1× NVIDIA B300 SXM6 AC (sm_103a, 275 GB HBM3e) |
| Driver | 580.126.09 |
| Host CUDA | 13.0 |
| Host RAM | 270 GiB |

## Software

| Component | Version |
|---|---|
| Image base | `ghcr.io/kaitakuai/mlnode-full:0.2.12-vllm0.19.0-h100-k1` (in-place upgrade) |
| OS (image) | Ubuntu 22.04 (Python 3.12.13) |
| torch | 2.11.0 |
| vllm | 0.20.0 (upstream) |
| flashinfer-python | 0.6.8.post1 |
| compressed-tensors | 0.15.0.1 |
| Triton ptxas | replaced with CUDA 12.9 ptxas (knows sm_103a) |

PoC v2 source overlay applied from
[`gonka-deploy/image/rtx_pro_6000/vendor/vllm`](../../../gonka-deploy/image/rtx_pro_6000/vendor/vllm)
(commit `0be8726de`).

## Startup behavior

Cold start has a **two-phase** characteristic with stock torch.compile:

1. **First run on a clean compile cache.** Each (batch_size, seq_len)
   shape that PoC v2 sweeps triggers a torch.compile recompile during
   the 5 s warmup window — long enough to swallow the entire
   measurement window for small batch sizes:

| Batch | first run (cold cache) | second run (warm cache) |
|---|---|---|
| 8 | **16** nonces/min (compile dominated) | 992 |
| 16 | **0** | 1248 |
| 32 | 704 | 1216 |
| 64 | 1280 | 1280 |

2. **Second run.** Compile cache hits → all batch sizes warm up
   normally and produce the numbers in the right column.

Production deployment must therefore **pre-warm** the compile cache
once per host (e.g., a single-pass warmup script that issues one prefill
at each expected `(batch, seq_len)` combo before serving live traffic).
Without warmup, the first ~30–60 s after vLLM start are unusable for
small-batch PoC requests.

## Results — Phase 3 (5×35 s sweep, **warm cache**)

| Batch | Nonces (30 s) | Nonces/min | Δ vs `--enforce-eager` |
|---|---|---|---|
| 8 | 496 | 992 | +7 % |
| 16 | 624 | 1248 | +8 % |
| 32 | 608 | 1216 | +6 % |
| **64** | **640** | **1280** ★ | **+11 %** |
| 128 | 0 | 0 | unchanged (PoC engine stuck) |

## Results — continuous collection (1000 nonces, batch=64)

| Metric | Value |
|---|---|
| Nonces collected | 1088 |
| Wall time | 51.0 s |
| **Throughput (n/min)** | **1255** |

Steady-state lands very close to the Phase-3 peak (1255 vs 1280),
suggesting we now sit slightly under the next downstream-pipeline
ceiling rather than directly on it.

## Comparison across all 1×B300 runs

| Run | Best Phase-3 | Steady (n/min) | Δ vs. baseline |
|---|---|---|---|
| 0.19, TRITON, default MoE | 832 | 798 | (baseline) |
| 0.19, TRITON, MoE config tuned for M=32768 | 832 | 823 | +3 % |
| 0.19, FLASHINFER_TRTLLM | 1056 | 1053 | +27 % / +32 % |
| 0.20, FLASHINFER_TRTLLM, `--enforce-eager` | 1152 | 1145 | +38 % / +43 % |
| **0.20, FLASHINFER_TRTLLM, `compilation_mode=1`** | **1280** | **1255** | **+54 % / +57 %** |

## Key observations

1. **`VLLM_COMPILE` mode 3 has hidden cost on PoC.** The 12 % gap we saw
   between `--enforce-eager` (1152) and the default no-eager run (1024)
   was not from CUDA graph capture or the `fuse_norm_quant` /
   `fuse_act_quant` passes — disabling those individually did not move
   the number. The cost is structural to vLLM's piecewise compilation
   pipeline: `splitting_ops`, the per-shape graph table, and the
   model-runner branch that toggles between compiled and `skip_compiled`
   PoC paths. `mode: 1` removes all of that and runs a single
   `torch.compile` over the model.
2. **Stock compile + PoC eager forward coexist cleanly.** PoC's
   `skip_compiled=True` (in `vllm/poc/poc_model_runner.py:241`) keeps
   the PoC forward path eager regardless of the compile mode, so nonces
   stay bit-identical to other validators using the same FlashInfer
   TRTLLM MoE backend. The compile only kicks in for non-PoC inference
   paths and the surrounding pipeline.
3. **Warm-up is not optional.** Stock `torch.compile` recompiles on
   every new shape. PoC v2's batch sweep starts from batch=8 and the
   30 s measurement window is shorter than the cold compile time — so
   small batches return zero nonces on a fresh start. A pre-warm script
   that issues one request per expected shape solves this.
4. **The 1024 plateau is gone.** With `mode: 1` we no longer see three
   batch sizes pinned to the same number — 16/32/64 each settle at
   1248 / 1216 / 1280, with batch=64 winning. The downstream-pipeline
   ceiling we identified at 1024 (and then 1152) has receded again,
   though we are likely close to a new one near 1280–1300.
5. **Cumulative gain is +54 %** (peak) / **+57 %** (steady) over the
   original 0.19 + TRITON baseline, all on the same 1×B300 silicon.

## Files

- `artifacts/nonces_1000.json` — 1088 nonces (1255 n/min steady-state, batch=64)
- `artifacts/config.json` — collector configuration
- `artifacts/logprobs_100.json` — empty (`--logprobs-count 0`)
- `artifacts/mlnode.log` — full MLNode + vLLM startup and serving log
- `artifacts/bench.log` — Phase-3 (warm-cache) sweep stdout

## Next steps

1. **Bake `compilation_mode: 1` into the new image.** Update
   `mlnode/full/hw/b300/Dockerfile` and `mlnode/tools/fragments/hw-patches/runner-py-patches/b300.py`
   to default `--compilation-config '{"mode": 1}'`, replacing the
   `--enforce-eager` force-flag from `b300-k2`. Bump to `b300-k3`.
2. **Add an explicit warm-up procedure** in the MLNode startup path so
   operators don't see the cold-cache zero-nonces window.
3. **Profile the new ~1280 plateau** to find the next bottleneck. Likely
   in the host-side PoC callback / scheduler again, but the gap to GPU
   peak should be measurable now.
