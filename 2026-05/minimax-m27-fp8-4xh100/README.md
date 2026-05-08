# MiniMax-M2.7 FP8 — 4×H100 SXM5

## Hardware
- **GPU**: 4× NVIDIA H100 80GB HBM3 SXM5 (NV18 NVLink full-mesh, sm_90)
- **Host**: shadecloud (orion@192.222.54.186)
- **Disk**: 11 TB / 443 GB /dev/shm
- **OS**: Ubuntu 22.04.5 LTS, kernel 6.8.0-60-generic
- **Network**: 1.27 Gbps download

## Software
- **Image**: `ghcr.io/kaitakuai/vllm:0.20.0-pocv2`
- **Container runtime**: Docker 28.3.1 with `--gpus all --network host --shm-size=400g`
- **vLLM**: 0.20.0
- **MoE backend**: `TRITON` (forced via `--moe-backend triton`)
- **Attention backend**: `FLASHINFER` (forced via `--attention-backend FLASHINFER`)

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
  --moe-backend triton \
  --attention-backend FLASHINFER \
  --trust-remote-code \
  --logprobs-mode processed_logprobs
```

Without explicit backends, H100 auto-selects FLASHINFER_CUTLASS Fp8 MoE which is significantly slower for this model.

## Phase 3 batch sweep (5s warmup + 30s measurement)

| Batch | Nonces | nonces/min |
|---|---|---|
| 2 | 444 | 888 |
| 8 | 1024 | 2048 |
| 16 | 1120 | 2240 |
| **32** | **1184** | **2368** ★ |
| 64 | 0 | hung (PoC engine OOM-stuck) |

**Best: 2368 nonces/min @ batch_size=32** (single instance, 4 GPUs)

## 8×GPU normalized
2 × 4-GPU instances: **4 736 nonces/min**

## Comparison with published 4×H100 result
| Source | n/min @ best batch |
|---|---|
| **This run** | **2368** ★ |
| [Published 2026-04](https://github.com/kaitakuai/experiments/tree/main/2026-04/minimax-m27-fp8-4xh100) | 1664 |
| Δ | **+42%** ↑ |

The +42% gap likely from combination of:
- PR#36 (`@torch.compile` on apply_householder) — +7-12%
- TRITON MoE explicitly forced (vs auto-default FLASHINFER_CUTLASS)
- FLASHINFER attention forced (vs FLASH_ATTN auto-default)
- shadecloud H100 hardware (NV18 NVLink full-mesh, fresh driver)

## Artifacts
- `artifacts/config.json` — benchmark configuration
- `artifacts/nonces_1000.json` — 1000 PoC v2 nonces (batch_size=32)
- `artifacts/inference_5langs.json` — 5-language inference probe (sp/en/ch/ar/hi)

## Notes
- 4×H100 SXM5 with TRITON+FLASHINFER attn now matches B200 throughput (2368 vs 2624) — Hopper SXM5 is genuinely competitive on this model
- Container approach: `docker run --gpus all --network host --shm-size=400g ghcr.io/kaitakuai/vllm:0.20.0-pocv2`
- batch_size=64 hangs (PoC engine OOM-stuck — known issue)
