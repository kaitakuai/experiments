# PoC Benchmark: MiniMax-M2.7 (FP8) on 2×B200

**Date:** 2026-04-15
**Model:** `MiniMaxAI/MiniMax-M2.7`
**Quantization:** FP8 block-wise 128×128 (E4M3, dynamic activations)
**Hardware:** 2×NVIDIA B200 183GB
**vLLM:** 0.19.0

## Summary

PoC v2 benchmark for MiniMax-M2.7 FP8 on 2×B200. Tested with both default backends (FLASHINFER_TRTLLM MoE + FLASHINFER attention) and forced TRITON MoE + FlashInfer attention for cross-GPU comparison.

Default is **19% faster** (2,367 vs 1,983 nonces/min). Fraud detection identical (~0.74 L2) regardless of backend.

## Results: default vs TRITON

| Config | MoE Backend | Attention | Nonces/min |
|--------|-------------|-----------|-----------|
| **Default** | FLASHINFER_TRTLLM | FLASHINFER | **2,367** |
| TRITON forced | TRITON | FLASHINFER | 1,983 |

### Benchmark (default, batch sweep)

| Batch Size | Nonces (30s) | Nonces/min |
|-----------:|-------------:|-----------:|
| 8 | 1,088 | 2,175 |
| 16 | 1,152 | 2,303 |
| **32** ★ | **1,184** | **2,367** |
| 64 | 0 | 0 (OOM) |
| 128 | 0 | 0 (OOM) |

## Backend comparison

| Metric | Default | TRITON |
|--------|---------|--------|
| FP8 nonces/min | **2,367** | 1,983 |
| AWQ nonces/min | 1,181 | 1,145 |
| Fraud L2 (FP8 vs AWQ) | 0.740 | 0.738 |
| Same-model cross-backend L2 | — | 0.28 (FP8), 0.15 (AWQ) |

Fraud detection works identically on both backends. Cross-backend L2 on same model is small (0.15-0.28).

## Cross-GPU honest (FP8)

| Pair | Backend | L2 mean |
|------|---------|---------|
| B200 vs H200 (TRITON both) | TRITON | 0.46 |
| B200 default vs H200 TRITON | mixed | 0.48 |

Cross-GPU L2 ~0.46-0.48 regardless of backend — dominated by GPU architecture difference (Blackwell vs Hopper), not backend.

## Reproduction

### Default (recommended for B200)

```bash
curl -X POST http://127.0.0.1:8081/api/v1/inference/up/async \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "'"$MODEL_PATH"'",
    "dtype": "auto",
    "additional_args": [
      "--served-model-name", "MiniMaxAI/MiniMax-M2.7",
      "--tensor-parallel-size", "2",
      "--gpu-memory-utilization", "0.92",
      "--max-num-seqs", "128",
      "--max-model-len", "131072",
      "--trust-remote-code"
    ]
  }'
```

No special flags — vLLM auto-selects FLASHINFER_TRTLLM MoE + FLASHINFER attention on B200.

### TRITON forced (for cross-GPU consistency)

Add to runner.py env:
```python
env["VLLM_USE_FLASHINFER_MOE_FP8"] = "0"
env["VLLM_MOE_USE_DEEP_GEMM"] = "0"
env["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
```
Plus move TRITON to top of `_AVAILABLE_BACKENDS` in `fp8.py`.

## Artifacts

- `artifacts/nonces_1000.json` — 1000 nonces (default: FLASHINFER_TRTLLM + FLASHINFER)
- `artifacts/nonces_triton.json` — 1000 nonces (forced TRITON + FlashInfer)
- `artifacts/config.json` — benchmark configuration

## Key observations

- **Default is fastest** — 2,367/min with FLASHINFER_TRTLLM, no reason to force TRITON on B200
- **Fraud detection unaffected by backend** — L2 ~0.74 on both
- **Cross-GPU L2 = 0.46** between B200 and H200 — GPU architecture is the main factor, not backend
- **B200 is fastest GPU tested** — 2,367/min vs H200 1,728/min vs A100 864/min
