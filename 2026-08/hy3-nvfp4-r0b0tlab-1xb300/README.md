# Hy3 NVFP4 (ModelOpt) — 1×B300 — the attack economics: +48 % by dropping tensor parallelism

**Date:** 2026-08-19
**Model:** `r0b0tlab/Hy3-295B-NVFP4` — 186 GB, 100 shards, built with **NVIDIA ModelOpt**.
vLLM reports `quantization=modelopt_mixed` and detects a *mix* of `FP8`, `NVFP4`,
`W4A16_NVFP4` and `MXFP8` quant algos, so the repo's "W4A4" label is a simplification.
MoE kernel: **`FLASHINFER_TRTLLM` NvFp4** — real FP4 tensor cores, not MARLIN emulation.
Blackwell-only (sm_100).
**Hardware:** 1× NVIDIA **B300 SXM6** (1100 W, 275 GB, driver **610.57.04**, sm_100) — one
card of Vast.ai instance 48124506, the second left idle (`CUDA_VISIBLE_DEVICES=0`).
**Image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash-0731:3.0.16-overlay-k5`
**Digest:** `sha256:8ce8830b4182b3dfd642c5e00f548f50a830611e4bc64ec4fbd84fe91070e3aa`

> The DeepSeek-V4-Flash foundry image, reused only as a vLLM 0.25.1 + PoC-plugin runtime.
> Its `runner.py` hardcodes V4-specific flags that must be replaced — `scripts/patch_hy3.py`.

## Summary

This is the run that quantifies why quantising is worth it for an attacker, and it is not
the reason one would assume. At matched topology the fraud arm is exactly as fast as honest
FP8 ([1599 = 1599 at TP=2](../hy3-nvfp4-r0b0tlab-2xb300/)). The gain comes from **fitting
into one card**, which the honest model cannot do, and therefore not paying for tensor
parallelism at all.

| Scenario on 2× B300 | topology | nonces/min | per card |
|---|---|---:|---:|
| honest FP8 | TP=2 | 1599 | 800 |
| this model | TP=2 | 1599 | 800 |
| **this model, two independent instances** | **2 × TP=1** | **2366** | **1183** |

**+48 % on identical hardware.** TP=2 scales only **1.35×** over TP=1, so the wider the
fleet, the further the honest and fraudulent economics diverge.

Measured, not inferred: weights occupy **166.48 GiB on a single card**, leaving 415 104 KV
tokens. Honest FP8 needs **276 GiB** and a B300 offers ~242 GiB at `gmu 0.90`, so it does
not fit — verified by the loader, not by arithmetic alone.

## Environment

| Parameter | Value |
|---|---|
| CUDA | 13.0.2 (image), driver 610.57.04 |
| vLLM | 0.25.1, build `752a3a504485790a2e8491cacbb35c137339ad34` |
| Python | 3.12.13 |
| mlnode | 3.0.16, `gonka_poc.entrypoint.api_router` |

## Config

```bash
TP=1 python3 scripts/patch_hy3.py     # the patch script is parameterised by env
# restart the mlnode API afterwards — uvicorn caches runner.py and would otherwise
# silently keep the previous topology
CUDA_VISIBLE_DEVICES=0 python -m uvicorn api.app:app --host 0.0.0.0 --port 8081
curl -X POST http://127.0.0.1:8081/api/v1/inference/up/async \
  -H 'Content-Type: application/json' \
  -d '{"model":"r0b0tlab/Hy3-295B-NVFP4","dtype":"auto","additional_args":[]}'
```

```
--tensor-parallel-size 1   --gpu-memory-utilization 0.90
--max-model-len 262144        --max-num-batched-tokens 65536
--kv-cache-dtype fp8          --logprobs-mode processed_logprobs
--worker-extension-cls gonka_poc.worker.PoCWorkerExtension
--tool-call-parser hy_v3      --reasoning-parser hy_v3
--trust-remote-code --enable-auto-tool-choice --enable-expert-parallel
```

Measurement window **120 s**, batches 16/32/64.

## Validation

### Throughput (120 s window)

| batch | 16 | 32 | 64 |
|---:|---:|---:|---:|
| this arm, TP=1 (one card) | 1120 | 1168 | **1183** |
| same model, TP=2 (`ref_sweep_same_model_tp2_120s.log`) | 1304 | 1552 | **1599** |
| honest FP8, TP=2 (`ref_sweep_honest_fp8_tp2_120s.log`) | 1439 | 1535 | **1599** |

Both reference sweeps are committed here so the +48 % claim can be checked without leaving
this folder.

### Resources

| | value |
|---|---:|
| weights on one card | **166.48 GiB** |
| KV cache | 415 104 tokens |
| usable at `gmu 0.90` | ≈242 GiB |
| honest FP8 requirement | 276 GiB — **does not fit** |

A batch-64 sweep needs 65 536 KV tokens, so the single-card configuration still has a ~6×
margin.

## Findings

1. **The attack is a topology attack, not a quantisation attack.** Identical throughput at
   equal TP; +48 % only by running two single-card instances.
2. **The right economic metric is nonces per card at the minimum viable topology**, not
   nonces per box. Measured per box, this fraud looks free of benefit.
3. **TP=2 scales 1.35×, not 2×** — the inefficiency the attacker monetises.
4. **`runner.py` edits are invisible to a running uvicorn.** The API must be restarted after
   changing TP, otherwise the "new" configuration silently measures the old one.

## Files

```
artifacts/
  sweep_tp1_120s.log                       this run — one card
  ref_sweep_same_model_tp2_120s.log        same model at TP=2, for the comparison
  ref_sweep_honest_fp8_tp2_120s.log        honest FP8 at TP=2, for the comparison
scripts/
  patch_hy3.py  run_pow_generation.py  collect_artifacts.py  l2_matrix.py  poc_seeds.json
```

No nonce sets were collected in this configuration: the fingerprint of this checkpoint is
established on [2×B300](../hy3-nvfp4-r0b0tlab-2xb300/) and [4×B200](../hy3-nvfp4-r0b0tlab-4xb200/),
and topology does not change it (both of those agree to three decimals).

Related: [same model on 2×B300](../hy3-nvfp4-r0b0tlab-2xb300/) ·
[honest FP8 on 2×B300](../hy3-fp8-2xb300/)

## Reproducibility checklist

- [x] Image pinned by digest
- [x] Both reference sweeps committed alongside — the +48 % claim is checkable in-folder
- [x] Weights/KV figures quoted from the engine log, not computed
- [x] 120 s window for every number
- [x] Scope stated: no nonce collection in this configuration, and why that is sufficient
- [x] No internal-tooling links, absolute paths, or sibling-repo references
