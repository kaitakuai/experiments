# Qwen3-235B-A22B-Instruct-2507-FP8 — 4×H200

## Hardware
- **GPU**: 4× NVIDIA H200 141GB HBM3e (sm_90, NVLink)
- **Host**: Vast.ai (Texas, US, machine 36493649)
- **Disk**: 280 GB
- **OS**: Ubuntu 22.04.5 LTS
- **Network**: 3.4 Gbps download

## Software
- **Image**: `ghcr.io/kaitakuai/vllm:0.20.0-pocv2`
- **vLLM**: 0.20.0
- **MoE backend**: auto (`TRITON` selected)
- **Attention backend**: `FLASHINFER` (forced via env var)

## Patches applied
1. `MAX_UNHEALTHY_COUNT=9999` in `api/watcher.py`
2. `VLLM_USE_V1=1` in `api/inference/vllm/runner.py`
3. **kv_scratch dtype check** in `vllm/poc/poc_model_runner.py`
4. **PR#36** (`@torch.compile` on `apply_householder`) — applied during best run

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
  --max-model-len 240000 \
  --max-num-batched-tokens 65536 \
  --kv-cache-dtype fp8 \
  --enable-expert-parallel \
  --disable-custom-all-reduce \
  --trust-remote-code \
  --logprobs-mode processed_logprobs
```

H200 имеет 141GB/GPU vs H100 80GB — позволяет PR#24 рекомендованный `max-model-len 240000` (на H100 пришлось урезать до 32K).

## Phase 3 batch sweep (5s warmup + 30s measurement)

### С PR#36
| Batch | nonces/min |
|---|---|
| 16 | 1344 |
| 32 | 1408 |
| **64** | **1408** ★ (saturation) |
| 96 | 0 (stuck) |

**Best: 1408 nonces/min @ batch=32 or 64** (saturation kicks in at 32)

## 8×GPU normalized
2 × 4-GPU instances: **2 816 nonces/min**

## Artifacts
- `artifacts/config.json` — benchmark configuration
- `artifacts/nonces_1000.json` — 1000 PoC v2 nonces (batch_size=64, без PR#36 — захвачены при второй сборке)
- `artifacts/inference_5langs.json` — 5-language inference probe

## Notes
- H200 показывает **+12.8%** vs H100 4×H100 (1408 vs 1248) при тех же настройках
- batch=96 застрял (PoC engine OOM-stuck — known issue)
- Per-GPU H200 = 352, vs H100 = 312 — ожидаемая разница HBM3e vs HBM3 + больше памяти
