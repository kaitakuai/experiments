# GLM-5.2 FP8 — 8×H200 — PoC v2 + inference + context (fp8_ds_mla required on Hopper)

**Date:** 2026-06-22
**Model:** `zai-org/GLM-5.2-FP8` @ `31cba24fb749908a485082bdeed6eb1ac6cffc2f`
  FP8 block-wise e4m3 [128,128], arch `GlmMoeDsaForCausalLM`, 753B / ~40B active, 256 experts × top-8.
**Hardware:** 8× NVIDIA H200 SXM5 (700 W, 141 GiB HBM3e/GPU, driver **570.xx**, CUDA 12.8, sm_90).
  Vast.ai Virginia US.
**Image:** `ghcr.io/kaitakuai/mlnode-b300-kimi-k2-6:0.2.13-vllm0.23.0-k1`
**Digest:** `sha256:8975054a6a997fedcc688201a07f368d01ffa76a86aa76b8f3ef60a2cb38bd57`

> Same image and scripts as the B200 run. Key Hopper-specific difference: the FLASHMLA_SPARSE
> backend (sm_90) rejects `fp8_e4m3` KV — use `auto` (bf16) for PoC and `fp8_ds_mla` for
> long-context inference.

## Summary

GLM-5.2 FP8 runs on H200 at **576 nonces/min** (eager PoC) — **−43% vs B200** (1016). The gap is
attributed to B200's higher memory bandwidth and sm_100 FLASHMLA support. Cross-hardware L2 confirms
honesty: H200 and B200 FP8 nonces are statistically indistinguishable (0.7% mismatch, p≈1.0).

| Metric | H200 | B200 (reference) | Δ |
|---|---|---|---|
| **PoC throughput, eager** | **576 nonces/min** | 1016 | −43% |
| **PoC throughput, CUDA graphs** | 512 nonces/min | 768 | −33% |
| **Inference output tok/s (450K ctx, cudagraph)** | **561 tok/s** | 817 (131K ctx) | — |
| **Max context (gmu 0.90, fp8_ds_mla)** | ~553K tokens | ~131K+ | — |
| **L2 vs B200 FP8** | mean 0.188, 0.7% mismatch | — | **PASS** |
| **L2 vs AWQ-INT4 fraud** | mean 0.342, 26.8% mismatch | — | **FRAUD** |

**Hopper-specific rules:**
1. For PoC: use `KV_CACHE_DTYPE=auto` (bf16) — FLASHMLA_SPARSE rejects `fp8_e4m3`.
2. For max-context inference: use `KV_CACHE_DTYPE=fp8_ds_mla` (the only fp8 KV format FLASHMLA_SPARSE accepts on sm_90).
3. Keep `GPU_MEM_UTIL=0.90` — `sparse_decode_fwd` needs ~2 GB/card scratch; at gmu=0.97 decode OOMs with 500 errors.

## Environment

| Parameter | Value |
|---|---|
| GPU | 8× H200 SXM5, 700 W, 141 GiB HBM3e, sm_90 |
| NVIDIA driver | 570.xx |
| CUDA (vLLM build) | 12.8 |
| vLLM | 0.23.0 (in-image system python) |
| Python | 3.12 |
| OS / base image | mlnode-b300-kimi-k2-6 (Stage-3 baked) |
| Attention backend | FLASHMLA_SPARSE (auto on H200 + GLM DSA) |

## Config

```bash
# Effective vLLM flags for H200 FP8 PoC (eager, bf16 KV):
--tensor-parallel-size 8
--gpu-memory-utilization 0.90
--max-model-len 450000
--max-num-seqs 128
--max-num-batched-tokens 65536     # max tokens per prefill batch (NOT context length); lowered because H200 OOMs during KV profiling at 131072+bf16 KV
--kv-cache-dtype auto              # bf16 — FLASHMLA_SPARSE rejects fp8_e4m3 on sm_90
--logprobs-mode processed_logprobs
--tool-call-parser glm45
--reasoning-parser glm45
--enable-expert-parallel
--enable-auto-tool-choice
--trust-remote-code
--compilation-config '{"mode": 0, "cudagraph_mode": "NONE"}'

# For max-context inference (450K+ tokens):
--kv-cache-dtype fp8_ds_mla        # only fp8 KV format accepted by FLASHMLA_SPARSE on Hopper
--max-model-len 450000
--gpu-memory-utilization 0.90      # ~12.5 GB free/card for sparse_decode_fwd scratch

VLLM_USE_DEEP_GEMM=0
VLLM_MOE_USE_DEEP_GEMM=0
VLLM_USE_FLASHINFER_MOE_FP8=0
```

### What changed vs the image default (Kimi-K2.6)

| Parameter | Image default (Kimi) | This run (GLM-5.2 H200) |
|---|---|---|
| Model | Kimi-K2.6 INT4 | GLM-5.2 FP8 |
| TP | 4 | **8** |
| Parsers | `kimi_k2` | `glm45` |
| Attention flag | `--attention-backend CUTLASS_MLA` | removed (DSA; H200 uses FLASHMLA_SPARSE) |
| MoE kernel | `VLLM_USE_FLASHINFER_MOE_INT4=1` | DeepGEMM **off** → triton MoE |
| KV dtype | auto | `auto` (bf16 for PoC) / `fp8_ds_mla` (long-context inference) |
| `--max-num-batched-tokens` | 131072 | **65536** (H200 KV profiling OOM fix) |

## Results

### PoC v2 throughput sweep (eager vs CUDA graphs)

| batch | eager | CUDA graphs (mode 3) |
|---:|---:|---:|
| 8  | — | 464 |
| 16 | — | 480 |
| 32 | — | 448 |
| 64 | **576** | **512** |

Full eager per-batch table not captured before box destruction (best=576 at batch 64 was recorded).
CUDA graphs: **−11%** vs eager (512 vs 576 nonces/min). Policy same as B200: **use eager for PoC**.

### Context investigation

| Config | KV cache tokens | Max tokens @ 1.0x concurrency | Notes |
|---|---:|---:|---|
| `auto` (bf16), gmu=0.90, maxnbt=65536 | 553,668 | 553K | PoC config; bf16 KV |
| `fp8_ds_mla`, gmu=0.90, maxnbt=65536 | 711,167 | 711K → cap at 450K | fp8 KV; OOM at gmu=0.97 |
| `fp8_ds_mla`, gmu=0.97 | OOM | — | `sparse_decode_fwd` needs ~2 GB scratch |

The official GLM-5.2 recipe claims 1M context requires 8×B200 — our H200 run confirms ~553K max
with bf16 KV (or ~711K token cache with fp8_ds_mla, but serving at 450K is the safe ceiling).

### Inference performance (CUDA graphs, 450K context, fp8_ds_mla)

`vllm bench serve`, 200 requests, 1024→256 tokens, concurrency 32:

| Metric | Value |
|---|---|
| Output tok/s | **561** |
| Peak output tok/s | 1056 |
| Total tok/s | 2832 |
| TTFT mean | 3117 ms |
| TPOT mean | **42 ms** |
| P99 TPOT | 58 ms |
| KV cache | 553,668 tokens |

For comparison, B200 inference at 131K context (cudagraph): 817 tok/s, TPOT 25 ms. H200 is slower
per-GPU, but fits ~553K tokens per box vs B200's ~711K (fp8_ds_mla). Note: 600K context bench
produced degenerate results (0 output tokens, all requests prefill-only) — use ≤450K for real serving.

### L2 chain validation

Canonical Gonka L2 via [`../../tools/gonka-l2-validate/`](../../tools/gonka-l2-validate/),
k_dim 12, gate: dist_threshold=0.4, p_mismatch=0.02.

| Pair | n nonces | mean L2 | mismatch | p_value | verdict |
|---|---:|---:|---:|---:|---|
| FP8 H200 ↔ FP8 B200 | 1000 | **0.188** | 7 (0.7%) | 0.9998 | **PASS** |
| FP8 H200 ↔ AWQ-INT4 fraud | 988 | 0.342 | 265 (26.8%) | 2.59e-209 | **FRAUD** |

Cross-hardware honesty holds: H200 and B200 FP8 produce equivalent PoC vectors (0.7% beyond
threshold, which is within random noise). The AWQ-INT4 fraud is caught on H200 just as firmly as on B200.

## Files

- [`artifacts/nonces_fp8_h200_auto/nonces_1000.json`](artifacts/nonces_fp8_h200_auto/nonces_1000.json) — 1000 H200 FP8 PoC nonces
- [`artifacts/fp8_h200.json`](artifacts/fp8_h200.json) — same nonces (3-way L2 label)
- [`artifacts/vllm_bench_fp8_h200_cudagraph_c32.json`](artifacts/vllm_bench_fp8_h200_cudagraph_c32.json) — H200 inference bench (450K ctx, cudagraph)
- [`artifacts/l2_h200_b200_fraud_3way.json`](artifacts/l2_h200_b200_fraud_3way.json) — 3-way L2 result (H200 vs B200 vs AWQ fraud)
- [`artifacts/model_revisions_h200.txt`](artifacts/model_revisions_h200.txt) — pinned model revisions as observed on H200 box
- [`scripts/config.env`](scripts/config.env) — H200-specific env overrides
- [`../glm-5.2-poc-backend-sweep/scripts/`](../glm-5.2-poc-backend-sweep/scripts/) — full reproduction scripts

**Note:** H200 backend sweep logs (`sweep_fp8_h200_*.log`) were not pulled before box destruction.
Numbers are captured in this README. To regenerate: re-rent an 8×H200 and re-run the sweep.

## Findings / recommendation

1. **PoC → eager (mode 0), `KV_CACHE_DTYPE=auto`.** CUDA graphs −11% on H200 (512 vs 576). bf16 KV is mandatory for PoC (FLASHMLA_SPARSE rejects fp8_e4m3).
2. **Long-context inference → `fp8_ds_mla`, gmu=0.90, ≤450K tokens.** fp8_ds_mla is the only fp8 KV format Hopper's FLASHMLA_SPARSE accepts. Keep gmu≤0.90 to leave ~12.5 GB/card free for `sparse_decode_fwd` scratch.
3. **1M context requires B200.** H200 ceiling is ~553K tokens (bf16) or ~711K (fp8_ds_mla), not 1M.
4. **Cross-hardware L2 passes.** H200 and B200 FP8 are interchangeable for chain validation purposes.
5. **`MAX_NUM_BATCHED_TOKENS=65536` is mandatory on H200.** Default 131072 OOMs during KV cache profiling with bf16 KV.
6. **AWQ-INT4 is caught cross-hardware.** Fraud is detectable from H200 reference nonces just as well as from B200.

## How to reproduce

```bash
export VAST_API_KEY=<your-key>
export HF_TOKEN=<your-hf-token>

# Rent 8×H200, Virginia:
bash ../glm-5.2-poc-backend-sweep/scripts/provision_vast.sh search h200 8
DISK=1500 bash ../glm-5.2-poc-backend-sweep/scripts/provision_vast.sh create <offer-id>
bash ../glm-5.2-poc-backend-sweep/scripts/provision_vast.sh list

# SSH in; clone repo; cd to this folder:
source scripts/config.env   # sets H200-specific overrides

bash ../glm-5.2-poc-backend-sweep/scripts/00_check_hw.sh
MODEL_ROLE=fp8 bash ../glm-5.2-poc-backend-sweep/scripts/01_download_model.sh

# PoC (eager, bf16 KV):
COMPILATION_CONFIG='{"mode": 0, "cudagraph_mode": "NONE"}' \
  KV_CACHE_DTYPE=auto \
  MAX_NUM_BATCHED_TOKENS=65536 \
  MODEL_ROLE=fp8 bash ../glm-5.2-poc-backend-sweep/scripts/02_start_vllm.sh
MODEL_ROLE=fp8 bash ../glm-5.2-poc-backend-sweep/scripts/03_poc_sweep.sh
MODEL_ROLE=fp8 bash ../glm-5.2-poc-backend-sweep/scripts/05_collect_nonces.sh

# Inference (CUDA graphs, fp8_ds_mla, 450K context):
COMPILATION_CONFIG='{"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"}' \
  KV_CACHE_DTYPE=fp8_ds_mla \
  MAX_MODEL_LEN=450000 \
  GPU_MEM_UTIL=0.90 \
  MODEL_ROLE=fp8 bash ../glm-5.2-poc-backend-sweep/scripts/02_start_vllm.sh
bash ../glm-5.2-poc-backend-sweep/scripts/06_inference_bench.sh

# L2 vs B200 reference:
bash ../glm-5.2-poc-backend-sweep/scripts/07_l2_validate.sh \
  artifacts/fp8_h200.json \
  ../glm-5.2-fp8-8xb200/artifacts/fp8_reference.json

bash ../glm-5.2-poc-backend-sweep/scripts/provision_vast.sh destroy <instance-id>
```

## Known gotchas

- **`fp8_e4m3` rejected on H200**: `No valid attention backend` when combining FLASHMLA_SPARSE + fp8_e4m3. Use `auto` (bf16) for PoC, `fp8_ds_mla` for long-context inference.
- **`MAX_NUM_BATCHED_TOKENS=65536` mandatory**: default 131072 + bf16 KV → `Available KV cache: -3.95 GiB` (negative). Lower maxnbt first.
- **`sparse_decode_fwd` OOM at gmu=0.97**: the fp8_ds_mla sparse decode kernel needs ~2 GB scratch per card. At gmu=0.97 only 0.7 GB free → 500 errors during inference. Set `GPU_MEM_UTIL=0.90`.
- **600K context bench degenerate**: 200-request bench at 600K context returned 0 output tokens (all prefill-only, bench computed TPOT=0). Safe serving ceiling is ≤450K tokens.
- **AWQ-INT4 unstable on H200**: Marlin MoE sparse_decode_fwd crashes / idles on sm_90. For fraud comparison, the B200 AWQ nonces are used in the 3-way L2 (not H200 AWQ).
- **H200 sweep logs lost**: backend sweep logs were not pulled before box destruction. Numbers in this README are from session output during the run.
- Same image-startup and HF_HUB_OFFLINE gotchas as [`../glm-5.2-fp8-8xb200/README.md`](../glm-5.2-fp8-8xb200/README.md#known-gotchas).

## Related

- B200 FP8 reference (cross-hardware L2 pair): [`../glm-5.2-fp8-8xb200/README.md`](../glm-5.2-fp8-8xb200/README.md)
- AWQ-INT4 fraud (cross-hardware fraud detected): [`../glm-5.2-awq4bit-8xb200/README.md`](../glm-5.2-awq4bit-8xb200/README.md)

## Reproducibility checklist

- [x] A reader with only this folder (and `../glm-5.2-poc-backend-sweep/scripts/`) can reach the headline result.
- [x] Hardware stated exactly: 8× H200 SXM5, 700 W, driver 570.xx, sm_90, NVLink.
- [x] Image pinned by tag + digest; model pinned by repo + commit SHA.
- [x] Every command is copy-pasteable; placeholders obvious.
- [x] All scripts referenced are in `../glm-5.2-poc-backend-sweep/scripts/`.
- [x] No links to `.claude/...` and no sibling-repo paths.
- [x] All artifacts referenced exist in `artifacts/` (symlinks to umbrella).
- [x] Expected outputs stated: 576 nonces/min eager, 561 tok/s inference, PASS on cross-hardware L2.
- [x] Known gotchas listed (fp8_e4m3 rejection, maxnbt OOM, gmu scratch OOM, 600K degenerate bench).
