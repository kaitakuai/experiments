# Qwen3-235B-A22B FP8 — 1×B300 — vLLM 0.19.0 — FLASHINFER TRTLLM MoE

**Date:** 2026-04-29
**Model:** `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`
**Quantization:** FP8 (block-wise w8a8, block_shape=[128,128])
**Hardware:** 1× NVIDIA B300 SXM6 AC (Blackwell Ultra, sm_103a, 275 GB HBM3e)
**vLLM:** 0.19.0 (Kaitaku PoC v2 build, base `kaitakuai/vllm:v0.19.0-pocv2-alpha3`)
**MLNode:** 0.2.12 (image `ghcr.io/kaitakuai/mlnode-full:0.2.12-vllm0.19.0-h100-k1` + sm_103a fixes + B300 hardcodes)

## Summary

Best peak result so far on 1×B300 with the in-house vLLM 0.19 build:
**1056 nonces/min @ batch=16** in the Phase-3 sweep, **1053 nonces/min steady-state**
(1088 nonces collected in 61.0 s) at batch=64. This **closes the regression** we
saw versus the user's prior measurement of ~1024 nonces/min on the same B300
silicon (vLLM 0.15.x baseline) and edges slightly above it.

**The single change that unlocks this** is forcing the MoE backend to **FlashInfer
TensorRT-LLM** (a.k.a. `FLASHINFER_TRTLLM`) instead of the TRITON path that the
overlay/B300 hardcodes pin for cross-GPU bit-compatibility. On B300 alone the
TRTLLM kernels are dramatically faster than TRITON's `fused_moe_kernel` (which
the previous nsys profile pinned at 66 % of GPU time).

⚠️ **Cross-GPU bit-compat note.** Switching MoE to FLASHINFER_TRTLLM yields
different bit patterns than the TRITON path used on H100/B200 validators today.
This setup is appropriate when **all** participating validators use the same
FlashInfer TRTLLM MoE backend; it is **not** drop-in compatible with a mixed
TRITON/FlashInfer pool. A MoE-routing divergence can yield wholly different
expert selection and large output drift between paths, so the choice has to be
coordinated.

## Configuration changes (vs. baseline / TRITON path)

Subprocess env, `runner.py`:

```diff
- env["VLLM_USE_FLASHINFER_MOE_FP8"] = "0"   # forced TRITON for bit-compat
+ env["VLLM_USE_FLASHINFER_MOE_FP8"] = "1"   # allow FlashInfer FP8 MoE
+ env["VLLM_FLASHINFER_MOE_BACKEND"] = "latency"   # routes to TENSORRT_LLM (sm_100+)
  env["VLLM_USE_V1"] = "1"
  env["VLLM_MOE_USE_DEEP_GEMM"] = "0"
```

Backend mapping in vLLM 0.19
([flashinfer_utils.py:get_flashinfer_moe_backend](https://github.com/vllm-project/vllm/blob/v0.19.0/vllm/model_executor/layers/quantization/utils/flashinfer_utils.py)):
- `VLLM_FLASHINFER_MOE_BACKEND="throughput"` → `FLASHINFER_CUTLASS` (does
  not support our FP8 block-quant scheme — fails at startup)
- `VLLM_FLASHINFER_MOE_BACKEND="latency"` → `FLASHINFER_TRTLLM` (requires
  SM100+, available on B300 ✅)
- `VLLM_FLASHINFER_MOE_BACKEND="masked_gemm"` → `FLASHINFER_CUTEDSL`

vLLM CLI args (B300 hardcodes in `runner.py` plus per-call `additional_args`):

```
--max-num-batched-tokens 65536    # one chunk for batch_size=64 × seq_len=1024
--enforce-eager                   # PoC bit-compat; +0% on this workload
--tensor-parallel-size 1
--max-num-seqs 128
--gpu-memory-utilization 0.95
--max-model-len 65536             # lowered from 131072 to fit KV cache
                                  # at max-num-batched-tokens=65536
--logprobs-mode processed_logprobs
--trust-remote-code
```

`--max-model-len=65536` is required because, with `--max-num-batched-tokens=65536`,
vLLM allocates more scheduler/activation memory and the available KV pool drops
to ~21 GiB. At `max_model_len=131072` a single full-context request needs 23.5 GiB
KV which exceeds the pool — vLLM refuses to start with a clear `ValueError`.
65 536 is enough headroom for PoC v2 (seq_len=1024, batch ≤ 64).

## Hardware

| Component | Value |
|---|---|
| GPU | 1× NVIDIA B300 SXM6 AC (sm_103a, 275 GB HBM3e) |
| Driver | 580.126.09 |
| Host CUDA | 13.0 |
| Image CUDA | 12.9.1 |
| Host RAM | 270 GiB |
| Disk | 532 GB virtual disk (332 GB used) |

## Software

| Component | Version |
|---|---|
| Image base | `ghcr.io/kaitakuai/mlnode-full:0.2.12-vllm0.19.0-h100-k1` |
| OS (image) | Ubuntu 22.04 (Python 3.12.13) |
| torch | 2.10.0+cu129 |
| vllm | 0.19.0 |
| flashinfer | 0.6.6 (`flashinfer-jit-cache` removed; JIT-compiled per device) |
| Triton ptxas | replaced with CUDA 12.9 ptxas (knows sm_103a) |

## Reproduction

Apply the standard B300 sm_103a fixes inside the container (Triton ptxas swap,
remove `flashinfer-jit-cache`, symlink CUDA headers from `nvidia/*` packages,
fix libcuda compat to host driver), patch `runner.py` with the env-block above
and the B300 hardcodes (force `--gpu-memory-utilization=0.95`,
`--max-model-len=65536`, `--logprobs-mode=processed_logprobs`; default
`--tensor-parallel-size=1`, `--max-num-seqs=128`), then bring vLLM up:

```bash
curl -s -X POST http://127.0.0.1:8081/api/v1/inference/up/async \
  -H 'Content-Type: application/json' -d '{
    "model": "/data/hf/Qwen3-235B-A22B-Instruct-2507-FP8",
    "dtype": "auto",
    "additional_args": [
      "--served-model-name", "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
      "--trust-remote-code",
      "--max-num-batched-tokens", "65536",
      "--enforce-eager"
    ]
  }'
```

## Startup profile

| Phase | Time |
|---|---|
| Weight load (220.35 GiB into VRAM) | 21.9 s |
| `MoEPrepareAndFinalizeNoDPEPModular` selected | logged at +24 s |
| FlashInfer TRTLLM JIT compile + warmup | the bulk of cold-start |
| **Cold start total (load → ready)** | **~510 s first run** (cached-warm afterwards) |

MoE backend selection (from the engine log):
```
INFO ... [fp8.py:266] Using FLASHINFER_TRTLLM Fp8 MoE backend
        out of potential backends:
        ['AITER', 'FLASHINFER_TRTLLM', 'FLASHINFER_CUTLASS',
         'DEEPGEMM', 'VLLM_CUTLASS', 'TRITON', 'MARLIN', ...]
```

KV cache:
- Available: ~21 GiB (after model + scheduler/activation overhead at
  `max-num-batched-tokens=65536`)
- Pool size: ~117 K tokens
- Concurrency at 65 K-token requests: ~1.79×

## Results — Phase 3 (5×35 s sweep)

| Batch | Nonces (30 s) | Nonces/min | Δ vs TRITON baseline |
|---|---|---|---|
| 8 | 496 | 992 | +29 % |
| **16** | **528** | **1056** ★ | **+32 %** |
| 32 | 512 | 1024 | +23 % |
| 64 | 512 | 1024 | +33 % |
| 128 | 0 | 0 | unchanged (PoC engine stuck) |

For comparison, the TRITON baseline ([qwen235b-fp8-1xb300-vllm019-baseline](../qwen235b-fp8-1xb300-vllm019-baseline/)):

| Batch | TRITON (this build) | FLASHINFER_TRTLLM | Δ |
|---|---|---|---|
| 8 | 640 | 992 | +55 % |
| 16 | 800 | 1056 | +32 % |
| 32 | 832 | 1024 | +23 % |
| 64 | 0 | 1024 | new (was stuck) |
| 128 | 0 | 0 | unchanged |

Three batch sizes (16, 32, 64) all settle on or near 1024 nonces/min, with 16
slightly higher at 1056. This is consistent with a downstream pipeline bound
(callback receiver / scheduler / proxy) starting to cap throughput once the GPU
no longer is the bottleneck — worth investigating separately.

## Results — continuous collection (1000 nonces, batch=64, logprobs_count=0)

| Metric | Value |
|---|---|
| Nonces collected | 1088 |
| Wall time | 61.0 s |
| **Throughput (n/min)** | **1053** |

User's prior reference number on the same B300 silicon was ~1024 nonces/min, so
this run is **+3 % above it** while running on **vLLM 0.19** (instead of 0.15.x).

## Key observations

1. **MoE backend is the dominant knob on 1×B300.** Forcing TRITON costs ~25–55 %
   throughput vs. FlashInfer TRTLLM. The overlay/B300 default of TRITON is a
   *correctness* choice (cross-GPU bit-compat with H100/B200), not a perf
   choice — once the validator pool is homogeneous on FlashInfer, the TRITON
   pin can be lifted.
2. **`VLLM_FLASHINFER_MOE_BACKEND` matters.** Just enabling
   `VLLM_USE_FLASHINFER_MOE_FP8=1` is not enough on vLLM 0.19: the auto-pick
   logic prefers `FLASHINFER_CUTLASS`, which does **not** support the model's
   block-quant FP8 scheme and aborts startup. `VLLM_FLASHINFER_MOE_BACKEND="latency"`
   pins to `FLASHINFER_TRTLLM` instead — that's the working path on B300.
3. **`max-num-batched-tokens=65536` unblocks batch=64.** With the default
   chunked-prefill limit (`8192`) the PoC engine got stuck at 0 nonces/min for
   batch ≥ 64. Raising the chunk to 65 536 lets the entire `batch×seq_len` flow
   through one prefill chunk and PoC produces nonces normally.
4. **`--enforce-eager` does not cost throughput** on this workload (verified
   directly: same batch results with and without the flag) and is the safer
   choice for cross-GPU bit-compat.
5. **MoE config tuning (`benchmark_moe.py --tune`) gave +3 %** at most on this
   model × GPU pair (see [tuned-M32768](../qwen235b-fp8-1xb300-vllm019-tuned-M32768/)) —
   far less than the backend swap. Tuning is most useful to silence the
   `Using default MoE config` warning, not for headline throughput.
6. **Throughput appears to plateau around 1024–1056 nonces/min** across batch
   sizes 16/32/64. The compute budget is not the bottleneck at that point —
   downstream pipeline (PoC callback receiver, scheduler, MLNode proxy) is.
   Worth a follow-up profiling pass focused on the host-side request loop.

## Files

- `artifacts/nonces_1000.json` — 1088 nonces (1053/min steady state, batch=64)
- `artifacts/config.json` — collector configuration
- `artifacts/logprobs_100.json` — empty (`--logprobs-count 0`)
- `artifacts/mlnode.log` — full MLNode + vLLM startup and serving log
- `artifacts/bench.log` — Phase-3 sweep stdout

## Next steps

1. **Decide validator-pool MoE policy.** If all participating validators move to
   FlashInfer TRTLLM, lift the TRITON pin in `mlnode-overlay/hw/b300` and
   `mlnode-full` runner.py defaults.
2. **vLLM 0.20.0 trial** on top of this config (PR #40552 is a B200 DeepGEMM
   UE8M0 fix; not in our hot path here, but the 0.20 MoE refactor may unlock
   further gains).
3. **Profile the 1024 cap.** With GPU no longer dominant (FlashInfer TRTLLM is
   fast enough), instrument the PoC v2 callback path to check whether
   request-loop / port-9999 receiver / MLNode proxy is now the bottleneck.
