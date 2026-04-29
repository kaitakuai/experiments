# Qwen3-235B-A22B FP8 — 1×B300 — vLLM 0.20.0 — FLASHINFER TRTLLM MoE

**Date:** 2026-04-29
**Model:** `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`
**Quantization:** FP8 (block-wise w8a8, block_shape=[128,128])
**Hardware:** 1× NVIDIA B300 SXM6 AC (Blackwell Ultra, sm_103a, 275 GB HBM3e)
**vLLM:** 0.20.0 (upstream `vllm/vllm-openai:v0.20.0` + Gonka PoC v2 patches from
[`gonka-deploy/image/rtx_pro_6000/vendor/vllm`](../../../gonka-deploy/image/rtx_pro_6000/vendor/vllm))
**MLNode:** 0.2.12

## Summary

First successful PoC v2 run on **vLLM 0.20.0** with the **FlashInfer TRTLLM**
Fp8 MoE backend. Best Phase-3 throughput: **1152 nonces/min @ batch=64**;
steady-state 1145 nonces/min on a 1088-nonce continuous collection. That is
**+9 % over the equivalent vLLM 0.19** run (1056 / 1053 nonces/min) and
**+38 % over the original vLLM 0.19 + TRITON baseline** (832 / 798 nonces/min).

Both `b300-k2` (vLLM 0.19 + FlashInfer TRTLLM) and this 0.20.0 path stay on
the FlashInfer TRTLLM MoE backend, so the same cross-validator caveat
applies — bit-compat is only preserved across a homogeneous Blackwell pool.

## How the build was assembled

vLLM 0.20.0 was pip-installed in the existing `mlnode-full:0.2.12-vllm0.19.0-h100-k1`
container on top of the already-applied B300 setup (sm_103a fixes, runner.py
hardcodes, FlashInfer TRTLLM env vars). pip pulled the **upstream** wheel
which lacks the Gonka PoC v2 routes, so `/api/v1/inference/pow/*` returned
502/404 until the PoC source overlay from
[`gonka-deploy/image/rtx_pro_6000/vendor/vllm`](../../../gonka-deploy/image/rtx_pro_6000/vendor/vllm)
was applied — only the `.py` files (`vllm/poc/`, patched
`vllm/entrypoints/openai/api_server.py` and friends) merged into the
upstream installation.

The fork's `Dockerfile.quick` is the canonical recipe for this overlay
(it does the same find-and-tar of `*.py` files into the upstream image).
For a one-off pip-based test the steps were:

```bash
# system pip (vLLM lives in /usr/local/lib/python3.12/dist-packages/vllm)
/usr/bin/python3 -m pip install --break-system-packages --no-cache-dir vllm==0.20.0
# (also pulls torch 2.11.0, flashinfer-python 0.6.8.post1, cuda-bindings
#  13.x, compressed-tensors 0.15.0.1)

# overlay PoC patches (.py only)
tar -czf vllm-pocv2-py.tar.gz $(find vllm -name "*.py" -type f)   # in fork checkout
# scp + docker cp into container, then:
cd /usr/local/lib/python3.12/dist-packages/vllm
tar -xzf /tmp/vllm-pocv2-py.tar.gz --strip-components=1
```

Routes registered after overlay (under `/api/v1/pow/*` directly on vLLM,
proxied as `/api/v1/inference/pow/*` by MLNode):

```
INFO ... [fp8.py:266] Using FLASHINFER_TRTLLM Fp8 MoE backend
        out of potential backends:
        ['AITER', 'FLASHINFER_TRTLLM', 'FLASHINFER_CUTLASS',
         'DEEPGEMM', 'VLLM_CUTLASS', 'TRITON', 'MARLIN', ...]
```

## Configuration (B300 hardcodes inherited from `runner.py`)

```
--enforce-eager
--max-num-batched-tokens 65536
--max-model-len 65536
--gpu-memory-utilization 0.95
--max-num-seqs 128
--tensor-parallel-size 1
--logprobs-mode processed_logprobs
```

Subprocess env (also inherited):
```
VLLM_USE_V1=1
VLLM_USE_FLASHINFER_MOE_FP8=1
VLLM_FLASHINFER_MOE_BACKEND=latency
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
| Image base | `ghcr.io/kaitakuai/mlnode-full:0.2.12-vllm0.19.0-h100-k1` (then in-place upgrade) |
| OS (image) | Ubuntu 22.04 (Python 3.12.13) |
| torch | **2.11.0** (was 2.10.0+cu129) |
| vllm | **0.20.0** (was 0.19.0) |
| flashinfer-python | **0.6.8.post1** (was 0.6.6) |
| compressed-tensors | 0.15.0.1 |
| nvidia-cuda-runtime / cublas / etc. | 13.x (vendored by torch wheel) |
| Image system CUDA | 12.9.1 (unchanged; unused by torch 2.11 wheel) |
| Triton ptxas | replaced with CUDA 12.9 ptxas (knows sm_103a) |

## Startup profile

| Phase | Time |
|---|---|
| Weight load (220.35 GiB into VRAM) | 44.96 s |
| `MoEPrepareAndFinalizeNoDPEPMonolithic` selected | logged at +59 s |
| DeepGEMM warmup (2428 kernels) | ~1 s |
| FlashInfer TRTLLM JIT autotune | 13.2 s |
| FlashInfer attention warmup + TRTLLM prefill auto-detected | a few seconds |
| `init engine (profile, create kv cache, warmup model)` | **461.35 s** |
| **Cold start total (vLLM up → MLNode "running")** | ~510 s first run |

(The 461 s is dominated by FlashInfer JIT-compile-on-first-run for sm_103a,
just like the 0.19 TRTLLM run; second start should be much faster from the
filesystem cache at `/root/.cache/flashinfer/<ver>/103a/`.)

## Results — Phase 3 (5×35 s sweep)

| Batch | Nonces (30 s) | Nonces/min | Δ vs 0.19 TRTLLM |
|---|---|---|---|
| 8 | 464 | 928 | −6 % |
| 16 | 576 | 1152 | +9 % |
| 32 | 576 | 1152 | +12 % |
| **64** | **576** | **1152** ★ | **+12 %** |
| 128 | 0 | 0 | unchanged (PoC engine stuck) |

For comparison, 0.19 TRTLLM ([qwen235b-fp8-1xb300-vllm019-flashinfer-trtllm](../qwen235b-fp8-1xb300-vllm019-flashinfer-trtllm/)):

| Batch | 0.19 TRTLLM | 0.20 TRTLLM | Δ |
|---|---|---|---|
| 8 | 992 | 928 | −6 % |
| 16 | 1056 | 1152 | +9 % |
| 32 | 1024 | 1152 | +12 % |
| 64 | 1024 | 1152 | +12 % |
| 128 | 0 | 0 | unchanged |

Three batch sizes (16, 32, 64) settle on **exactly 1152 nonces/min** — the
same downstream-pipeline cap pattern we saw on 0.19 TRTLLM (which capped at
1024), just at a higher level. The compute path is faster on 0.20.0 but
something downstream of the GPU is still rate-limiting.

## Results — continuous collection (1000 nonces, batch=64, logprobs_count=0)

| Metric | Value |
|---|---|
| Nonces collected | 1088 |
| Wall time | 56.0 s |
| **Throughput (n/min)** | **1145** |

Steady-state matches the Phase-3 peak well (1145 ≈ 1152), suggesting we're
sitting right at the pipeline cap throughout the 56-second window.

## Comparison across all runs on 1×B300

| Run | Best Phase-3 | Steady (n/min) |
|---|---|---|
| 0.19, TRITON, default MoE config | 832 | 798 |
| 0.19, TRITON, MoE config tuned for M=32768 | 832 | 823 |
| 0.19, **FLASHINFER_TRTLLM** | 1056 | 1053 |
| **0.20, FLASHINFER_TRTLLM** | **1152** | **1145** |

vLLM 0.20.0 unlocks the next +9 % on top of the FlashInfer TRTLLM switch.
Cumulative gain over the original baseline: **+38 %** Phase-3 / **+43 %**
steady-state.

## Key observations

1. **Upstream `pip install vllm==0.20.0` is not enough.** vLLM 0.20.0 ships
   without Gonka's PoC v2 routes. They live in the
   [`gonka-deploy/image/rtx_pro_6000/vendor/vllm`](../../../gonka-deploy/image/rtx_pro_6000/vendor/vllm)
   fork (commit `0be8726de` at the time of writing) and must be overlaid
   onto the upstream installation. PoC route prefix in this fork is
   `/api/v1/pow/*` (was `/api/v1/inference/pow/*` on the 0.15.x fork);
   MLNode keeps a stable `/api/v1/inference/pow/*` proxy interface so
   benchmark / collector scripts do not need updating.
2. **+9 % over 0.19 TRTLLM is consistent across 16 / 32 / 64.** All three
   land on 1152 — same plateau pattern as 0.19 hitting 1024. The GPU is
   not the limit at this level; PoC-engine / callback-receiver / scheduler
   on the host side is.
3. **Cold start ~510 s** for the first run on a clean cache (FlashInfer
   JIT for sm_103a). Subsequent restarts should be much faster — this is a
   one-time cost per host, not a per-restart cost.
4. **batch=8 regressed slightly** (992 → 928, −6 %). Likely scheduler
   overhead in the new torch 2.11 / vLLM 0.20 path showing more cost at
   small batches; not a concern for our hot path (batch=32–64 wins).

## Files

- `artifacts/nonces_1000.json` — 1088 nonces (1145 nonces/min steady-state, batch=64)
- `artifacts/config.json` — collector configuration
- `artifacts/logprobs_100.json` — empty (`--logprobs-count 0`)
- `artifacts/mlnode.log` — full MLNode + vLLM startup and serving log
- `artifacts/bench.log` — Phase-3 sweep stdout

## Next steps

1. **Build a proper `kaitakuai/vllm:v0.20.0-pocv2-<rev>` base image** from
   the fork at
   [`gonka-deploy/image/rtx_pro_6000/vendor/vllm`](../../../gonka-deploy/image/rtx_pro_6000/vendor/vllm)
   (the included `Dockerfile.quick` does this in one step from
   `vllm/vllm-openai:v0.20.0`). Add it to
   `mlnode/.github/trusted-sources.yaml` under a new `vllm_base.0.20.0` key.
2. **Bump `mlnode-full` build matrix** to support `VLLM_VERSION=0.20.0` —
   add a new `b300-k3` (or similar) registry entry whose `vllm_base_version`
   is 0.20.0 and rebuild via the standard
   `KAITAKU_REV=3 VLLM_VERSION=0.20.0 HW_VARIANT=b300 scripts/build-full.sh`.
3. **Profile the 1152 cap** the same way we profiled the 1024 cap — trace
   PoC v2 callback handler and MLNode proxy with `nsys` or `py-spy` to
   confirm where the host-side bottleneck is.
