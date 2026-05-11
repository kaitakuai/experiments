# Qwen3-235B-A22B-Instruct-2507-FP8 — 2×B200

## Hardware
- **GPU**: 2× NVIDIA B200 (192 GB each, sm_100, NVLink)
- **Host**: Vast.ai (Ohio, US, machine_id 58629)
- **Disk**: 280 GB
- **OS**: Ubuntu 22.04.5 LTS
- **Network**: 59 Gbps download

## Software
- **Image**: `ghcr.io/kaitakuai/vllm:0.20.0-pocv2`
- **vLLM**: 0.20.0
- **MoE backend**: `FLASHINFER_TRTLLM` (auto-selected on Blackwell)
- **Attention backend**: `FLASHINFER`

## Patches applied
1. `MAX_UNHEALTHY_COUNT=9999` in `api/watcher.py`
2. `VLLM_USE_V1=1` in `api/inference/vllm/runner.py`
3. **kv_scratch dtype check** in `vllm/poc/poc_model_runner.py` — required for FP8 KV cache models
4. **PR#36** (`@torch.compile` on `apply_householder` in `vllm/poc/gpu_random.py`)

## vLLM startup command
```
vllm serve <MODEL_PATH> \
  --served-model-name Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 128 \
  --max-model-len 120000 \
  --max-num-batched-tokens 65536 \
  --kv-cache-dtype fp8 \
  --compilation-config '{"mode": 1}' \
  --trust-remote-code \
  --logprobs-mode processed_logprobs
```

Settings adapted from published [B300 8×GPU report](https://github.com/kaitakuai/experiments/tree/main/2026-04/qwen235b-fp8-8xb300-watcher-cold-start-fix) but with TP=2 instead of TP=1.

## Phase 3 batch sweep (5s warmup + 30s measurement)

| Batch | nonces/min |
|---|---|
| 32 | 1920 |
| **64** | **1984** ★ |
| 96 | 1856 |
| 128 | 192 (stuck) |

**Best: 1984 nonces/min @ batch_size=64** (single instance, 2 GPUs)

## 8×GPU normalized
4 × 2-GPU instances: **7 936 nonces/min**

## Comparison with published B300 8×GPU
| Source | n/min/GPU | 8×GPU n/min |
|---|---|---|
| **This run (2×B200)** | 992 | 7 936 |
| [Published 8×B300](https://github.com/kaitakuai/experiments/tree/main/2026-04/qwen235b-fp8-8xb300-watcher-cold-start-fix) | 1280 | 10 240 |
| Δ | -23% | -22.5% |

The gap is consistent with B200 vs B300 hardware spec difference (B300 = Blackwell Ultra, sm_103a).

## Artifacts
- `artifacts/config.json` — benchmark configuration
- `artifacts/nonces_1000.json` — 1024 PoC v2 nonces (batch_size=64)

(inference_5langs.json not generated — collect timeout hit during 5-language probe; nonces produced successfully)

## Notes
- batch_size=64 is optimal — same as published B300 config
- batch_size=128 hangs (PoC engine OOM-stuck — known issue across all GPU/model combos tested)
- DeepGEMM warmup ~1 min, FlashInfer attention warmup adds ~30s, total cold start ~4 min
