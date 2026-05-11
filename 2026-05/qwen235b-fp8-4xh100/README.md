# Qwen3-235B-A22B-Instruct-2507-FP8 — 4×H100 SXM5

## Hardware
- **GPU**: 4× NVIDIA H100 SXM5 80GB HBM3 (sm_90, NVLink full-mesh)
- **Host**: Vast.ai (US, machine 36493648)
- **Disk**: 280 GB
- **OS**: Ubuntu 22.04.5 LTS
- **Network**: 5.2 Gbps download

## Software
- **Image**: `ghcr.io/kaitakuai/vllm:0.20.0-pocv2`
- **vLLM**: 0.20.0
- **MoE backend**: auto (`TRITON` selected)
- **Attention backend**: `FLASHINFER` (forced via env var)

## Patches applied
1. `MAX_UNHEALTHY_COUNT=9999` in `api/watcher.py`
2. `VLLM_USE_V1=1` in `api/inference/vllm/runner.py`
3. **kv_scratch dtype check** in `vllm/poc/poc_model_runner.py` — required for FP8 KV cache
4. **PR#36** (`@torch.compile` on `apply_householder`) — applied during best run; collected artifacts here are without it (see notes)

## Env vars (from [PR#24 comment](https://github.com/gonka-ai/vllm/pull/24#issuecomment-4276298420))
```
VLLM_ATTENTION_BACKEND=FLASHINFER
VLLM_ALLOW_INSECURE_SERIALIZATION=1
POC_RPC_TIMEOUT_MS=300000
POC_BATCH_SIZE_DEFAULT=16
```

## vLLM startup command
```
vllm serve <MODEL_PATH> \
  --served-model-name Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 128 \
  --max-model-len 32768 \
  --max-num-batched-tokens 65536 \
  --kv-cache-dtype fp8 \
  --enable-expert-parallel \
  --disable-custom-all-reduce \
  --trust-remote-code \
  --logprobs-mode processed_logprobs
```

PR#24 рекомендовал `max-model-len 240000` + `--num-gpu-blocks-override 15000`, но на 4×H100 80GB не влезает (max KV cache available = 1.32 GiB, нужно 5.38 GiB). Уменьшил до 32K.

## Phase 3 batch sweep (5s warmup + 30s measurement)

### С PR#36 (`@torch.compile` on apply_householder)
| Batch | nonces/min |
|---|---|
| **16** | **1248** ★ |
| 32, 64, 96 | 0 (PoC engine stuck after first batch — known issue) |

### Без PR#36
| Batch | nonces/min |
|---|---|
| **16** | **1120** ★ |
| 32, 64 | 0 (PoC engine stuck) |

**PR#36 даёт +11.4% boost** (1248 vs 1120).

## 8×GPU normalized
2 × 4-GPU instances: **2 496 nonces/min** (with PR#36)

## Comparison
| Source | n/min @ best batch |
|---|---|
| **This run with PR#36** | **1248** ★ |
| This run without PR#36 | 1120 |
| [Published in PR#24 comment](https://github.com/gonka-ai/vllm/pull/24#issuecomment-4276298420) (4×H100, mlnode 3.0.13-alpha5) | 1295 |
| Δ vs published | -3.6% |

## Artifacts
- `artifacts/config.json` — benchmark configuration
- `artifacts/nonces_1000.json` — 1000 PoC v2 nonces (batch_size=16, **without PR#36** — captured during second collection)
- `artifacts/inference_5langs.json` — 5-language inference probe

## Notes
- batch ≥32 caused PoC engine to permanently stuck (0 nonces) requiring vLLM restart between attempts
- На H100 best batch = 16 (in contrast to B200/H200 batch=32-64)
- `--num-gpu-blocks-override 15000` from PR#24 не использован (несовместимо с max-model-len 32K)
