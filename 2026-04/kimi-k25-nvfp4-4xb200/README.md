# PoC Benchmark: Kimi-K2.5-NVFP4 on 4×B200

**Date:** 2026-04-13
**Model:** `nvidia/Kimi-K2.5-NVFP4`
**Quantization:** NVFP4 (ModelOpt 0.41.0, W4A4, all Linear except self_attn, group_size=16)
**Hardware:** 4×NVIDIA B200 183GB
**vLLM:** 0.19.0

## Summary

NVFP4 PoC benchmark on 4×B200 (TP=4). Initial attempts failed with OOM and `rms_norm Byte` errors. After systematic investigation, the **only required fix** is a vLLM launch flag:

```
--compilation-config '{"custom_ops": ["all", "-rms_norm"]}'
```

No code patches needed. Model works on 4×B200 at batch=32 with 1,816 nonces/min.

## Problem investigation

### Error 1: `rms_norm_kernel not implemented for 'Byte'`

Default NVFP4 compilation enables `norm_quant` and `act_quant` fusions which pass uint8 tensors to rms_norm CUDA kernel during PoC forward.

**Fix:** `--compilation-config '{"custom_ops": ["all", "-rms_norm"]}'` excludes rms_norm from custom ops.

### Error 2: OOM on 4×B200 (earlier attempts without fix)

Without the compilation fix, vLLM allocated 172 GiB/GPU (vs 165 with fix). The extra memory came from fused norm_quant/act_quant kernels and their workspace buffers.


### Fixes that were NOT needed

| Fix | Needed? |
|-----|---------|
| `--compilation-config '{"custom_ops": ["all", "-rms_norm"]}'` | **YES** |
| `--dtype bfloat16` (instead of auto) | No |
| poc_model_runner.py dtype fix | No |
| modelopt.py scale fix (torch.max) | No |
| layernorm.py rms_norm guard | No |
| `--enforce-eager` | No |
| `--pass_config fuse_norm_quant=false` | No (doesn't work) |

## Hardware

| Parameter | Value |
|-----------|-------|
| GPU | 4× NVIDIA B200 (183,359 MiB each) |
| NVIDIA Driver | 595.58.03 |
| CPU | AMD EPYC 9575F 64-Core (256 vCPUs) |
| RAM | ~1.5 TB |

## Reproduction

### 1. Download model

```bash
mkdir -p /dev/shm/hf
HF_HOME=/dev/shm/hf python3 -c \
  'from huggingface_hub import snapshot_download; snapshot_download("nvidia/Kimi-K2.5-NVFP4", max_workers=16)'
```

### 2. Install MLNode

Same as INT4 experiment.

### 3. Start vLLM (key: compilation-config flag)

```bash
MODEL_PATH=/dev/shm/hf/hub/models--nvidia--Kimi-K2.5-NVFP4/snapshots/<hash>

curl -X POST http://127.0.0.1:8081/api/v1/inference/up/async \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "'"$MODEL_PATH"'",
    "dtype": "auto",
    "additional_args": [
      "--served-model-name", "nvidia/Kimi-K2.5-NVFP4",
      "--tensor-parallel-size", "4",
      "--gpu-memory-utilization", "0.92",
      "--max-num-seqs", "128",
      "--max-model-len", "131072",
      "--trust-remote-code",
      "--compilation-config", "{\"custom_ops\": [\"all\", \"-rms_norm\"]}"
    ]
  }'
```

### 4. Run benchmark & collect nonces

```bash
export HOST_IP=127.0.0.1
python3 -u run_pow_generation.py --phase 3 --skip-check

python3 -u collect_artifacts.py \
  --url http://127.0.0.1:5001 \
  --model 'nvidia/Kimi-K2.5-NVFP4' \
  --output-dir /tmp/artifacts \
  --nonces 1000 --batch-size 32 --logprobs-count 0
```

## Startup profile

| Phase | Duration |
|-------|----------|
| Model loading (from `/dev/shm`) | ~77 s |
| Model memory/GPU | 139.49 GiB |
| Available KV cache | ~15 GiB (~470,000 tokens) |
| **Total cold start** | **~5 min** |

## Results

Collected with `--compilation-config '{"custom_ops": ["all", "-rms_norm"]}'`, no other patches:

| Batch Size | Nonces (30s) | Nonces/min |
|-----------:|-------------:|-----------:|
| **32** ★ | **~900** | **~1,816** |

**Best: batch=32 → ~1,816 nonces/min**

## Comparison: 4×B200 vs 8×B200

| Metric | NVFP4 4×B200 (TP=4) | NVFP4 8×B200 (TP=8) |
|--------|---------------------|---------------------|
| Best nonces/min | 1,816 | 3,839 |
| Model memory/GPU | 139.49 GiB | 70.97 GiB |
| KV cache/GPU | ~15 GiB | 83.63 GiB |
| OOM threshold | batch≥64 (est.) | batch≥64 |

## Cross-validation: INT4 vs NVFP4

**Not compatible.** L2 distance between INT4 and NVFP4 nonce vectors is ~1.5 (equivalent to random/uncorrelated vectors). Cross-validation will fail — different quantization methods produce fundamentally different numerical outputs.

See `../NVFP4_INT4_COMPATIBILITY_ANALYSIS.md` for full analysis.

## Artifacts

- `artifacts/nonces_1000.json` — 1000 PoC nonce vectors (compilation-config fix only, no code patches)
- `artifacts/nvfp4_tp4_allfixes_nonces.json` — nonces with all fixes (for comparison)
- `artifacts/nvfp4_tp4_noscalefix_nonces.json` — nonces without scale fix
- `artifacts/nvfp4_tp4_dtypeauto_nonces.json` — nonces with dtype=auto
- `artifacts/nvfp4_tp4_nopocfix_nonces.json` — nonces without poc dtype fix

## Key findings

1. **Only one launch flag needed** — `--compilation-config '{"custom_ops": ["all", "-rms_norm"]}'`
2. **No code patches required** — not in poc_model_runner.py, not in layernorm.py, not in modelopt.py
3. **`--dtype auto` works fine** — no need for `--dtype bfloat16`
4. **NVFP4 on 4×B200 gives 1,816/min** vs 1,024/min for INT4 on same hardware (+77%)
5. **INT4 and NVFP4 nonces are NOT cross-validatable** — L2=1.5, different quantization = different model
