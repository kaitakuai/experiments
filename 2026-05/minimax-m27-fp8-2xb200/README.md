# MiniMax-M2.7 FP8 — 2×B200

## Hardware
- **GPU**: 2× NVIDIA B200 (192 GB each, sm_100, NVLink)
- **Host**: Vast.ai (Alabama, US, machine_id 41067)
- **Disk**: 280 GB
- **OS**: Ubuntu 22.04.5 LTS

## Software
- **Image**: `ghcr.io/kaitakuai/vllm:0.20.0-pocv2`
- **vLLM**: 0.20.0
- **MoE backend**: `FLASHINFER_TRTLLM` (auto-selected on Blackwell)
- **Attention backend**: `FLASHINFER`

## Patches applied
1. `MAX_UNHEALTHY_COUNT=9999` in `api/watcher.py`
2. `VLLM_USE_V1=1` in `api/inference/vllm/runner.py`
3. **kv_scratch dtype check** in `vllm/poc/poc_model_runner.py` — required for FP8 KV cache models, otherwise PoC reuses uint8-storage KV cache as inputs_embeds and feeds Byte tensor to per_token_group_quant kernel
4. **PR#36** (`@torch.compile` on `apply_householder` in `vllm/poc/gpu_random.py`)

## vLLM startup command
```
vllm serve <MODEL_PATH> \
  --served-model-name MiniMaxAI/MiniMax-M2.7 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 128 \
  --max-model-len 131072 \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --logprobs-mode processed_logprobs
```
No `--moe-backend` flag — vLLM auto-selects `FLASHINFER_TRTLLM Fp8 MoE` on B200.

## Phase 3 batch sweep (5s warmup + 30s measurement)

| Batch | Nonces | nonces/min |
|---|---|---|
| 2 | 682 | 1364 |
| 8 | 1128 | 2256 |
| 16 | 1280 | 2560 |
| **32** | **1312** | **2624** ★ |
| 64 | 0 | hung (PoC engine OOM-stuck) |

**Best: 2624 nonces/min @ batch_size=32** (single instance, 2 GPUs)

## 8×GPU normalized
4 × 2-GPU instances: **10 496 nonces/min**

## Comparison with published 2×B200 result
| Source | n/min @ best batch |
|---|---|
| **This run** | **2624** ★ |
| [Published 2026-04](https://github.com/kaitakuai/experiments/tree/main/2026-04/minimax-m27-fp8-2xb200) | 2367 |
| Δ | **+11%** |

## Artifacts
- `artifacts/config.json` — benchmark configuration
- `artifacts/nonces_1000.json` — 1000 PoC v2 nonces (batch_size=32)
- `artifacts/inference_5langs.json` — 5-language inference probe (sp/en/ch/ar/hi)

## Reproduction
1. Rent 2×B200 on Vast.ai with image `ghcr.io/kaitakuai/vllm:0.20.0-pocv2`
2. Apply 4 patches above
3. Download `MiniMaxAI/MiniMax-M2.7` to `/dev/shm/hf` via `huggingface_hub.snapshot_download`
4. Start vLLM via MLNode `/api/v1/inference/up/async` with the additional_args above
5. After DeepGEMM warmup completes (~2-3 min), run:
```
HOST_IP=127.0.0.1 python3 run_pow_generation.py --phase 3 --skip-check
```

## Known issues
- batch_size=64 hangs (PoC engine OOM-stuck) — same as `flashinfer_moe_int4_blackwell.md` memory entry
- After phase 3 sweep PyTorch caching allocator accumulates ~170 GB; subsequent collect_artifacts.py 1000 nonces requires vLLM restart to clear cache
