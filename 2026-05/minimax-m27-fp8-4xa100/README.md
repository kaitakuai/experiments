# MiniMax-M2.7 FP8 — 4×A100 SXM4 80GB

## Hardware
- **GPU**: 4× NVIDIA A100 SXM4 80GB (sm_80, NVLink)
- **Host**: Vast.ai (Massachusetts, US)
- **Disk**: 280 GB
- **OS**: Ubuntu 22.04.5 LTS

## Software
- **Image**: `ghcr.io/kaitakuai/vllm:0.20.0-pocv2`
- **vLLM**: 0.20.0
- **MoE backend**: `MARLIN` (forced via `--moe-backend marlin` — A100 has no native FP8 hardware, MARLIN is the only fallback that works)
- **Attention backend**: auto-selected

## Patches applied
1. `MAX_UNHEALTHY_COUNT=9999` in `api/watcher.py`
2. `VLLM_USE_V1=1` in `api/inference/vllm/runner.py`
3. **kv_scratch dtype check** in `vllm/poc/poc_model_runner.py` — required for FP8 KV cache models
4. **PR#36** (`@torch.compile` on `apply_householder` in `vllm/poc/gpu_random.py`)

## vLLM startup command
```
vllm serve <MODEL_PATH> \
  --served-model-name MiniMaxAI/MiniMax-M2.7 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 128 \
  --max-model-len 131072 \
  --kv-cache-dtype fp8 \
  --moe-backend marlin \
  --trust-remote-code \
  --logprobs-mode processed_logprobs
```

Without `--moe-backend marlin` vLLM tries FLASHINFER FP8 path which fails on Ampere (`NotImplementedError: VLLM_USE_FLASHINFER_MOE_FP8=1, but no FlashInfer FP8 MoE backend supports the configuration`).

## Phase 3 batch sweep (5s warmup + 30s measurement)

| Batch | Nonces | nonces/min |
|---|---|---|
| 2 | 302 | 604 |
| 8 | 416 | 832 |
| **16** | **448** | **896** ★ |
| 32 | 448 | 896 |
| 64 | 0 | hung (PoC engine OOM-stuck) |

**Best: 896 nonces/min @ batch_size=16** (single instance, 4 GPUs)

batch=32 saturates same as 16 (MARLIN compute-bound on A100).

## 8×GPU normalized
2 × 4-GPU instances: **1 792 nonces/min**

## Comparison with published 4×A100 result
| Source | n/min @ best batch |
|---|---|
| **This run** | **896** ★ |
| [Published 2026-04](https://github.com/kaitakuai/experiments/tree/main/2026-04/minimax-m27-fp8-4xa100) | 864 |
| Δ | **+4%** |

## Artifacts
- `artifacts/config.json` — benchmark configuration
- `artifacts/nonces_1000.json` — 1000 PoC v2 nonces (batch_size=16)
- `artifacts/inference_5langs.json` — 5-language inference probe (sp/en/ch/ar/hi)

## Notes
- A100 is the **slowest** GPU tested for this model (~3× slower than B200, ~2× slower than H200)
- MARLIN is software-emulated FP4/FP8 path — no native FP8 tensor cores on Ampere
- batch_size=16 vs 32 saturates because MARLIN is compute-bound, not memory-bound at these sizes
