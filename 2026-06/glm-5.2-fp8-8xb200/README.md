# GLM-5.2 FP8 — 8×B200 — PoC v2 + inference (CUDA graphs +6.8× decode vs eager)

**Date:** 2026-06-22
**Model:** `zai-org/GLM-5.2-FP8` @ `31cba24fb749908a485082bdeed6eb1ac6cffc2f`
  FP8 block-wise e4m3 [128,128], arch `GlmMoeDsaForCausalLM` (DeepSeek-style MLA + DSA sparse
  attention, `index_topk=2048`), 753B total / ~40B active, 256 experts × top-8, 78 layers.
**Hardware:** 8× NVIDIA B200 SXM6 (1000 W, 183 GiB HBM3e, driver **595.71.05**, CUDA 13.2, sm_100).
  Vast.ai Virginia US (offer `41958960`).
**Image:** `ghcr.io/kaitakuai/mlnode-b300-kimi-k2-6:0.2.13-vllm0.23.0-k1`
**Digest:** `sha256:8975054a6a997fedcc688201a07f368d01ffa76a86aa76b8f3ef60a2cb38bd57`

> The image ships baked Kimi-K2.6 config. The GLM-5.2 model is activated via
> [`../glm-5.2-poc-backend-sweep/scripts/runner_patch_glm.py`](../glm-5.2-poc-backend-sweep/scripts/runner_patch_glm.py)
> which splices the GLM block after the Kimi block in `runner.py`.

## Summary

GLM-5.2 FP8 runs on B200 via the Kimi-B300 image once DeepGEMM is disabled and the runner
config is overridden. The only working backend is **triton/auto** (DeepGEMM crashes, FlashInfer-CUTLASS
hangs — see [Backend sweep](#backend-sweep)). KV cache dtype `fp8_e4m3` is accepted natively on sm_100.

| Metric | Value |
|---|---|
| **PoC throughput, eager (best)** | **1016 nonces/min** (batch 64) |
| **PoC throughput, CUDA graphs mode 3** | 768 nonces/min (batch 64) — **−24%** |
| **Inference output tok/s, eager** | 157 tok/s, TPOT 170 ms |
| **Inference output tok/s, CUDA graphs mode 3** | **817 tok/s, TPOT 25 ms** — **+6.8× decode** |
| **L2 gate vs AWQ-INT4 fraud** | **FRAUD** (mean 0.349, 27.4% mismatch, p≈7e-217) |

**Policy:** use eager for PoC mining, CUDA graphs (mode 3) for inference serving.

## Environment

| Parameter | Value |
|---|---|
| GPU | 8× B200 SXM6, 1000 W, 183 GiB HBM3e, sm_100 |
| NVIDIA driver | 595.71.05 |
| CUDA (vLLM build) | 13.2 |
| vLLM | 0.23.0 (in-image system python) |
| Python | 3.12 |
| OS / base image | mlnode-b300-kimi-k2-6 (Stage-3 baked) |

## Config

```bash
# Effective vLLM flags for FP8 PoC (eager):
--tensor-parallel-size 8
--gpu-memory-utilization 0.90
--max-model-len 1048576            # model maximum context (max_position_embeddings); B200 (sm_100) fits the full window
--max-num-seqs 128
--kv-cache-dtype fp8_e4m3
--logprobs-mode processed_logprobs
--tool-call-parser glm45
--reasoning-parser glm45
--enable-expert-parallel
--enable-auto-tool-choice
--trust-remote-code
--compilation-config '{"mode": 0, "cudagraph_mode": "NONE"}'

# For inference serving, swap to:
--compilation-config '{"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"}'

# Backend env (must be set before starting vLLM):
VLLM_USE_DEEP_GEMM=0
VLLM_MOE_USE_DEEP_GEMM=0
VLLM_USE_FLASHINFER_MOE_FP8=0
```

### What changed vs the image default (Kimi-K2.6)

| Parameter | Image default (Kimi) | This run (GLM-5.2 FP8) |
|---|---|---|
| Model | Kimi-K2.6 INT4 | GLM-5.2 FP8 |
| TP | 4 | **8** |
| Parsers | `kimi_k2` | `glm45` |
| Attention flag | `--attention-backend CUTLASS_MLA` (forced) | removed (GLM is DSA, not MLA) |
| MoE kernel | `VLLM_USE_FLASHINFER_MOE_INT4=1` | DeepGEMM **off** → triton MoE + cutlass linear |
| KV dtype | auto | `fp8_e4m3` |
| Compilation | mode 0 (eager) | mode 0 (PoC) / mode 3 (inference) |

## Results

### PoC v2 throughput sweep

Per-batch, via [`../glm-5.2-poc-backend-sweep/scripts/03_poc_sweep.sh`](../glm-5.2-poc-backend-sweep/scripts/03_poc_sweep.sh):

| batch | eager (mode 0) | CUDA graphs (mode 3) | mode 2 |
|---:|---:|---:|---:|
| 8  | 160  | 720 | 688 |
| 16 | 960  | 768 | **768** |
| 32 | 960  | 768 | 768 |
| 64 | **1016** | **768** | 732 |

CUDA graphs cost **−24%** on PoC (1016→768). Mode 2 (DYNAMO_TRACE_ONCE) gives no benefit over mode 3
at PoC (same 768), but cold-starts ~160 s slower than eager (287 vs 130 s). **Use eager for PoC.**

### Backend sweep (B200)

| MoE backend | env | Result |
|---|---|---|
| **triton / cutlass (auto)** | `VLLM_USE_DEEP_GEMM=0` | ✅ 1016 nonces/min |
| DeepGEMM | `VLLM_USE_DEEP_GEMM=1` | ❌ crash — `cudaErrorInvalidValue` in `BlockScaledMMLinearKernel` (`fused_qkv_a_proj`) during memory profiling |
| FlashInfer-CUTLASS MoE | `VLLM_USE_FLASHINFER_MOE_FP8=1` | ❌ hang — GPU util 0%, `shm_broadcast` 60 s timeout, never reaches healthy |

This differs from MiniMax/Kimi where DeepGEMM was the performance winner. GLM-5.2's DSA architecture
trips the DeepGEMM **linear** kernel (not the MoE sparse kernel) on B200 sm_100.

### Inference performance

`vllm bench serve`, 200 requests, 1024→256 tokens, concurrency 32, `/v1/chat/completions`:

| Config | Output tok/s | Total tok/s | TPOT mean | TTFT mean | Bench dur | Cold start |
|---|---:|---:|---:|---:|---:|---:|
| eager (mode 0) | 157 | 793 | 170.4 ms | 3251 ms | 326 s | ~130 s |
| mode 2 + FAP | 153 | 771 | 174.9 ms | 3464 ms | 335 s | ~287 s |
| **mode 3 + FAP** | **817** | **4124** | **25.1 ms** | 2999 ms | 63 s | ~450 s |

CUDA graphs give **6.8× faster decode** (TPOT 170→25 ms) and **5.2× output throughput** (157→817 tok/s).
Mode 2 is as slow as eager — piecewise decode graphs only engage under `VLLM_COMPILE` (mode 3).
TTFT (prefill) is graph-insensitive.

### L2 chain validation

Canonical Gonka L2 via [`../../tools/gonka-l2-validate/`](../../tools/gonka-l2-validate/), 988 common
nonces, k_dim 12, gate: dist_threshold=0.4, p_mismatch=0.02.

| Pair | mean L2 | mismatch | p_value | verdict |
|---|---:|---:|---:|---|
| FP8 B200 ↔ AWQ-INT4 fraud B200 | 0.349 | 271/988 (27.4%) | 6.98e-217 | **FRAUD** |
| FP8 B200 ↔ FP8 H200 | 0.188 | 7/1000 (0.7%) | 0.9998 | **PASS** |

See also: [`../glm-5.2-fp8-8xh200/README.md`](../glm-5.2-fp8-8xh200/README.md) (3-way L2).

## Files

- [`artifacts/nonces_fp8_auto_auto/nonces_1000.json`](artifacts/nonces_fp8_auto_auto/nonces_1000.json) — 1000 FP8 PoC nonces (eager, the reference vectors)
- [`artifacts/fp8_reference.json`](artifacts/fp8_reference.json) — same nonces, renamed for L2 (validator keys by basename)
- [`artifacts/fp8_b200.json`](artifacts/fp8_b200.json) — same nonces copy (3-way L2 label)
- [`artifacts/sweep_fp8_auto_auto.log`](artifacts/sweep_fp8_auto_auto.log) — eager PoC sweep log
- [`artifacts/sweep_fp8_cudagraph.log`](artifacts/sweep_fp8_cudagraph.log) — CUDA graph PoC sweep log
- [`artifacts/sweep_fp8_mode2.log`](artifacts/sweep_fp8_mode2.log) — mode 2 PoC sweep log
- [`artifacts/vllm_bench_fp8_c32.json`](artifacts/vllm_bench_fp8_c32.json) — inference bench, eager
- [`artifacts/vllm_bench_fp8_cudagraph_c32.json`](artifacts/vllm_bench_fp8_cudagraph_c32.json) — inference bench, mode 3
- [`artifacts/vllm_bench_fp8_mode2_c32.json`](artifacts/vllm_bench_fp8_mode2_c32.json) — inference bench, mode 2
- [`artifacts/l2_awq_fraud__vs__fp8_reference.json`](artifacts/l2_awq_fraud__vs__fp8_reference.json) — L2 B200 fraud verdict
- [`artifacts/l2_h200_b200_fraud_3way.json`](artifacts/l2_h200_b200_fraud_3way.json) — 3-way L2 (H200 vs B200 vs AWQ)
- [`scripts/config.env`](scripts/config.env) — B200-specific env overrides
- [`../glm-5.2-poc-backend-sweep/scripts/`](../glm-5.2-poc-backend-sweep/scripts/) — full reproduction scripts

## Findings / recommendation

1. **PoC → eager (mode 0).** CUDA graphs cost −24% (1016→768 nonces/min). Mode 2 is equally slow as mode 3 on PoC.
2. **Inference → CUDA graphs (mode 3).** 6.8× decode speedup, 5.2× throughput. Accept ~450 s cold start.
3. **Backend: triton/auto only.** `VLLM_USE_DEEP_GEMM=0` is mandatory. DeepGEMM crashes on GLM DSA linear kernel; FlashInfer-CUTLASS hangs.
4. **KV dtype: fp8_e4m3** works natively on B200 (sm_100). No FLASHMLA_SPARSE compatibility issue.
5. **L2 gate holds.** AWQ-INT4 fraud is caught (27.4% mismatch, p≈7e-217).

## How to reproduce

```bash
# 0. Set env
export VAST_API_KEY=<your-key>
export HF_TOKEN=<your-hf-token>

# 1. Rent 8×B200, Virginia, >=1.2 TB disk
bash ../glm-5.2-poc-backend-sweep/scripts/provision_vast.sh search b200 8
DISK=1500 bash ../glm-5.2-poc-backend-sweep/scripts/provision_vast.sh create <offer-id>
bash ../glm-5.2-poc-backend-sweep/scripts/provision_vast.sh list  # get ssh host:port

# 2. SSH in; clone experiments repo; cd to this folder
# source the B200-specific config, then run shared scripts:
source scripts/config.env

# 3. Hardware check + model download
bash ../glm-5.2-poc-backend-sweep/scripts/00_check_hw.sh
MODEL_ROLE=fp8 bash ../glm-5.2-poc-backend-sweep/scripts/01_download_model.sh

# 4. PoC (eager — best throughput):
COMPILATION_CONFIG='{"mode": 0, "cudagraph_mode": "NONE"}' \
  MODEL_ROLE=fp8 bash ../glm-5.2-poc-backend-sweep/scripts/02_start_vllm.sh
MODEL_ROLE=fp8 bash ../glm-5.2-poc-backend-sweep/scripts/03_poc_sweep.sh
MODEL_ROLE=fp8 bash ../glm-5.2-poc-backend-sweep/scripts/05_collect_nonces.sh

# 5. Inference (CUDA graphs):
COMPILATION_CONFIG='{"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"}' \
  MODEL_ROLE=fp8 bash ../glm-5.2-poc-backend-sweep/scripts/02_start_vllm.sh
bash ../glm-5.2-poc-backend-sweep/scripts/06_inference_bench.sh

# 6. L2 validation vs AWQ fraud:
bash ../glm-5.2-poc-backend-sweep/scripts/07_l2_validate.sh \
  artifacts/fp8_reference.json \
  ../glm-5.2-awq4bit-8xb200/artifacts/awq_fraud.json

# 7. Destroy when done:
bash ../glm-5.2-poc-backend-sweep/scripts/provision_vast.sh destroy <instance-id>
```

## Known gotchas

- **DeepGEMM crash**: `cudaErrorInvalidValue` during memory profiling, not at forward pass. Keep `VLLM_USE_DEEP_GEMM=0` and `VLLM_MOE_USE_DEEP_GEMM=0`.
- **CUTLASS hang**: GPU util drops to 0%, `shm_broadcast` 60 s timeout. Keep `VLLM_USE_FLASHINFER_MOE_FP8=0`.
- **Runner patch ordering**: the GLM block must be inserted **after** the baked Kimi block in `runner.py` (anchored to `# --- end Kaitaku B300-Kimi...`). Inserting before causes the Kimi block to overwrite GLM settings.
- **API startup**: Vast replaces Docker CMD with SSH. Start the API manually via venv: `source /app/packages/api/.venv/bin/activate && PYTHONPATH=/app/packages/api/src uvicorn api.app:app --port 8081`.
- **collect_artifacts.py**: prefix is `/api/v1/inference/pow/`, not `/inference/pow/`. Add a 12×5s retry loop on init (502/503 transient).
- **HF_HUB_OFFLINE**: the image sets `HF_HUB_OFFLINE=1`. Override: `export HF_HUB_OFFLINE=0`.
- **L2 self-comparison trap**: if both nonce files have the same basename, the validator detects L2=0 (PASS). Copy to distinct filenames (`fp8_reference.json`, `awq_fraud.json`) before running L2.

## Related

- AWQ fraud variant: [`../glm-5.2-awq4bit-8xb200/README.md`](../glm-5.2-awq4bit-8xb200/README.md)
- H200 run (cross-hardware L2 pair): [`../glm-5.2-fp8-8xh200/README.md`](../glm-5.2-fp8-8xh200/README.md)

## Reproducibility checklist

- [x] A reader with only this folder (and `../glm-5.2-poc-backend-sweep/scripts/`) can reach the headline result by following this README.
- [x] Hardware stated exactly: 8× B200 SXM6, 1000 W, driver 595.71.05, sm_100, NVLink.
- [x] Image pinned by tag + digest; model pinned by repo + commit SHA.
- [x] Every command is copy-pasteable; placeholders (`<offer-id>`, `<instance-id>`) are obvious.
- [x] All scripts referenced are committed under `../glm-5.2-poc-backend-sweep/scripts/`.
- [x] No links to `.claude/...` and no sibling-repo paths.
- [x] All artifacts referenced exist in `artifacts/` (as symlinks to umbrella).
- [x] Expected outputs stated: 1016 nonces/min eager, 817 tok/s cudagraph, FRAUD on L2.
- [x] Known gotchas and fixes listed.
