# GLM-5.2 AWQ-INT4 — 8×B200 — PoC v2 fraud (1536 nonces/min per TP=4 engine, caught by L2)

**Date:** 2026-06-22
**Model:** `cyankiwi/GLM-5.2-AWQ-INT4` @ `431c1cd297c7a2f38d17c7b9520b10c15101df25`
  Packaged as **compressed-tensors W4A16** (NOT classic AWQ — auto-detected, do not pass
  `--quantization awq_marlin`). ~390 GB on disk. Marlin MoE sparse kernel.
**Hardware:** 8× NVIDIA B200 SXM6 (1000 W, 183 GiB HBM3e, driver **595.71.05**, CUDA 13.2, sm_100).
  Vast.ai Virginia US (offer `41958960`, same box as [`../glm-5.2-fp8-8xb200/`](../glm-5.2-fp8-8xb200/)).
**Image:** `ghcr.io/kaitakuai/mlnode-b300-kimi-k2-6:0.2.13-vllm0.23.0-k1`
**Digest:** `sha256:8975054a6a997fedcc688201a07f368d01ffa76a86aa76b8f3ef60a2cb38bd57`

> This is the **fraud** experiment. The model is designed to mimic the reference (FP8) PoC
> output at lower cost. The L2 chain gate catches it.

## Summary

The AWQ-INT4 model runs TP=4, allowing **2 engines per 8-GPU box** (~3072 combined nonces/min vs 1016
for the FP8 reference on TP=8). The gate catches the fraud:

| Metric | Value |
|---|---|
| **PoC throughput per TP=4 engine (eager)** | **1536 nonces/min** (batch 64) |
| **Extrapolated 8-GPU throughput (2 engines)** | ~3072 nonces/min |
| **FP8 reference throughput** | 1016 nonces/min (1 engine, TP=8) |
| **L2 vs FP8 reference** | **FRAUD** (mean 0.349, 27.4% mismatch, p≈6.98e-217) |
| **L2 vs FP8 H200** | **FRAUD** (mean 0.342, 26.8% mismatch, p≈2.59e-209) |

Despite the **~3× throughput advantage**, the L2 chain gate rejects the AWQ-INT4 model on all
reference pairs. The 4-bit weight compression shifts the PoC activation vectors far enough that
27% exceed the L2 threshold (0.4), which binomial test flags with p≈7e-217.

## Environment

| Parameter | Value |
|---|---|
| GPU | 8× B200 SXM6, 1000 W, 183 GiB HBM3e, sm_100 |
| NVIDIA driver | 595.71.05 |
| CUDA (vLLM build) | 13.2 |
| vLLM | 0.23.0 |
| Python | 3.12 |
| OS / base image | mlnode-b300-kimi-k2-6 (Stage-3 baked) |

## Config

```bash
# Effective vLLM flags for AWQ-INT4 PoC (eager, TP=4):
--tensor-parallel-size 4
--gpu-memory-utilization 0.90
--max-model-len 131072
--max-num-seqs 128
--kv-cache-dtype auto                # bf16; AWQ doesn't need fp8 KV
--logprobs-mode processed_logprobs
--tool-call-parser glm45
--reasoning-parser glm45
--enable-expert-parallel
--enable-auto-tool-choice
--trust-remote-code
--compilation-config '{"mode": 0, "cudagraph_mode": "NONE"}'
# NOTE: do NOT pass --quantization awq_marlin — compressed-tensors auto-detects; forcing it causes ValidationError

VLLM_USE_DEEP_GEMM=0
VLLM_MOE_USE_DEEP_GEMM=0
```

### What changed vs the image default (Kimi-K2.6)

| Parameter | Image default (Kimi) | This run (GLM-5.2 AWQ-INT4) |
|---|---|---|
| Model | Kimi-K2.6 INT4 | GLM-5.2 AWQ-INT4 (compressed-tensors W4A16) |
| TP | 4 | 4 (fits the box; 2 engines possible) |
| Parsers | `kimi_k2` | `glm45` |
| Attention flag | `--attention-backend CUTLASS_MLA` | removed |
| `--quantization` | — | none (auto-detect) |
| Compilation | mode 0 | mode 0 (PoC) |

## Results

### PoC v2 throughput sweep (TP=4, 1 engine)

| batch | AWQ-INT4 eager |
|---:|---:|
| 8  | 1152 |
| 16 | **1216** |
| 32 | 1152 |
| 64 | **1536** |

Best: **1536 nonces/min at batch 64**.

With 2×TP=4 engines running concurrently on the same 8×B200 box: **~3072 nonces/min**.
This was **extrapolated** (not measured concurrently — concurrent 2-engine PoC measurement is
listed as a follow-up in the umbrella TODO).

### L2 chain validation — fraud detection

Canonical Gonka L2 via [`../../tools/gonka-l2-validate/`](../../tools/gonka-l2-validate/),
k_dim 12, gate: dist_threshold=0.4, p_mismatch=0.02.

| Pair (n common nonces) | mean L2 | mismatch | p_value | verdict |
|---|---:|---:|---:|---|
| AWQ-INT4 ↔ FP8 B200 reference | 0.349 | 271/988 (27.4%) | 6.98e-217 | **FRAUD** |
| AWQ-INT4 ↔ FP8 H200 reference | 0.342 | 265/988 (26.8%) | 2.59e-209 | **FRAUD** |

The gate holds on **both** reference hardware configs. The 4-bit compression shifts activations
far enough that the fraud is detectable regardless of which honest node you compare against.

## Files

- [`artifacts/nonces_awq_marlin_auto/nonces_1000.json`](artifacts/nonces_awq_marlin_auto/nonces_1000.json) — 1000 AWQ-INT4 PoC nonces
- [`artifacts/awq_fraud.json`](artifacts/awq_fraud.json) — same nonces, renamed for L2 (validator keys by basename)
- [`artifacts/awq_b200_fraud.json`](artifacts/awq_b200_fraud.json) — same nonces copy (3-way L2 label)
- [`artifacts/sweep_awq_marlin_auto.log`](artifacts/sweep_awq_marlin_auto.log) — PoC sweep log
- [`artifacts/l2_awq_fraud__vs__fp8_reference.json`](artifacts/l2_awq_fraud__vs__fp8_reference.json) — L2 verdict (AWQ vs B200 FP8)
- [`artifacts/l2_h200_b200_fraud_3way.json`](artifacts/l2_h200_b200_fraud_3way.json) — 3-way L2 (H200 vs B200 vs AWQ)
- [`scripts/config.env`](scripts/config.env) — AWQ-specific env overrides
- [`../glm-5.2-poc-backend-sweep/scripts/`](../glm-5.2-poc-backend-sweep/scripts/) — full reproduction scripts

## Findings / recommendation

1. **AWQ-INT4 is caught.** L2 fraud gate rejects it on all reference pairs (>26% mismatch, p<10e-200). The 4-bit weight compression leaves a detectable fingerprint in PoC activation vectors.
2. **TP=4 runs cleanly.** Marlin MoE sparse kernel auto-selected by compressed-tensors. No `--quantization` flag needed or wanted.
3. **Economics don't matter.** Even at ~3× throughput advantage (3072 vs 1016 nonces/min), the model is fraudulent. The gate was not calibrated for GLM-5.2 specifically (generic `dist_threshold=0.4`), yet it catches with overwhelming statistical confidence.
4. **Hopper (H200) is unstable for this AWQ model.** Marlin MoE crashes / idles on sm_90 (sparse_decode_fwd). B200 (sm_100) is the only validated platform.

## How to reproduce

```bash
export VAST_API_KEY=<your-key>
export HF_TOKEN=<your-hf-token>

# Same box as the FP8 reference (offer 41958960); or rent a new 8×B200:
bash ../glm-5.2-poc-backend-sweep/scripts/provision_vast.sh search b200 8
DISK=1500 bash ../glm-5.2-poc-backend-sweep/scripts/provision_vast.sh create <offer-id>

# SSH in; clone repo; cd to this folder:
source scripts/config.env

MODEL_ROLE=awq bash ../glm-5.2-poc-backend-sweep/scripts/01_download_model.sh

# PoC sweep (TP=4, eager):
COMPILATION_CONFIG='{"mode": 0, "cudagraph_mode": "NONE"}' \
  MODEL_ROLE=awq bash ../glm-5.2-poc-backend-sweep/scripts/02_start_vllm.sh
MODEL_ROLE=awq bash ../glm-5.2-poc-backend-sweep/scripts/03_poc_sweep.sh
MODEL_ROLE=awq bash ../glm-5.2-poc-backend-sweep/scripts/05_collect_nonces.sh
# Nonces saved to artifacts/nonces_awq_marlin_auto/nonces_1000.json

# L2 validation vs FP8 reference:
cp artifacts/nonces_awq_marlin_auto/nonces_1000.json artifacts/awq_fraud.json
bash ../glm-5.2-poc-backend-sweep/scripts/07_l2_validate.sh \
  artifacts/awq_fraud.json \
  ../glm-5.2-fp8-8xb200/artifacts/fp8_reference.json

bash ../glm-5.2-poc-backend-sweep/scripts/provision_vast.sh destroy <instance-id>
```

## Known gotchas

- **Do NOT pass `--quantization awq_marlin`**: the model uses compressed-tensors format; forcing the flag causes a `ValidationError`. Remove it and let vLLM auto-detect.
- **Hopper unstable**: Marlin MoE PoC crashes or idles on sm_90 (H200). Only validated on B200 (sm_100).
- **L2 baseline naming**: both nonce files must have distinct basenames before running L2 — the validator keys by filename. Copy to `awq_fraud.json` and `fp8_reference.json`.
- **Concurrent 2-engine measurement**: running 2×TP=4 engines simultaneously on the same box requires separate port/PID management. The `~3072/min` figure is extrapolated, not directly measured.
- Same image-startup and HF_HUB_OFFLINE gotchas as [`../glm-5.2-fp8-8xb200/README.md`](../glm-5.2-fp8-8xb200/README.md#known-gotchas).

## Related

- FP8 reference (this fraud is compared against): [`../glm-5.2-fp8-8xb200/README.md`](../glm-5.2-fp8-8xb200/README.md)
- H200 FP8 (second cross-hardware L2 pair): [`../glm-5.2-fp8-8xh200/README.md`](../glm-5.2-fp8-8xh200/README.md)

## Reproducibility checklist

- [x] A reader with only this folder (and `../glm-5.2-poc-backend-sweep/scripts/`) can reach the headline result by following this README.
- [x] Hardware stated exactly: 8× B200 SXM6, 1000 W, driver 595.71.05, sm_100.
- [x] Image pinned by tag + digest; model pinned by repo + commit SHA.
- [x] Every command is copy-pasteable.
- [x] All scripts referenced are committed under `../glm-5.2-poc-backend-sweep/scripts/`.
- [x] No links to `.claude/...` and no sibling-repo paths.
- [x] All artifacts exist in `artifacts/` (symlinks to umbrella).
- [x] Expected output stated: 1536 nonces/min per engine, FRAUD on L2.
- [x] Known gotchas listed (compressed-tensors auto-detect, Hopper instability, basename collision).
