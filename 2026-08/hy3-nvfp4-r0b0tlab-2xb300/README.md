# Hy3 NVFP4 (ModelOpt) — 2×B300 — fraud arm, bit-exact and loud (L2 0.49)

**Date:** 2026-08-19
**Model:** `r0b0tlab/Hy3-295B-NVFP4` — 186 GB, 100 shards, built with **NVIDIA ModelOpt**.
vLLM reports `quantization=modelopt_mixed` and detects a *mix* of `FP8`, `NVFP4`,
`W4A16_NVFP4` and `MXFP8` quant algos, so the repo's "W4A4" label is a simplification.
MoE kernel: **`FLASHINFER_TRTLLM` NvFp4** — real FP4 tensor cores, not MARLIN emulation.
Blackwell-only (sm_100).
**Honest reference:** [`../hy3-fp8-2xb300/`](../hy3-fp8-2xb300/) — same host, same session;
its nonce sets are duplicated here as `ref_nonces_*`.
**Hardware:** 2× NVIDIA **B300 SXM6** (1100 W, 275 GB, NV18, driver **610.57.04**, sm_100).
Vast.ai instance 48124506.
**Image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash-0731:3.0.16-overlay-k5`
**Digest:** `sha256:8ce8830b4182b3dfd642c5e00f548f50a830611e4bc64ec4fbd84fe91070e3aa`

> The DeepSeek-V4-Flash foundry image, reused only as a vLLM 0.25.1 + PoC-plugin runtime.
> Its `runner.py` hardcodes V4-specific flags that must be replaced — `scripts/patch_hy3.py`.

## Summary

At matched topology this fraud buys **nothing**: 1599 nonces/min against the honest arm's
1599 — identical to the nonce. Four-bit weights do not make the prefill proof faster on this
hardware.

What it does buy is the ability to leave TP=2 behind; that measurement lives in
[1×B300](../hy3-nvfp4-r0b0tlab-1xb300/) and is worth **+48 %** on the same two cards.

For detection this arm is the **loudest** measured: median L2 0.49 against honest, versus
0.37 for the [llm-compressor build](../hy3-nvfp4-redhatai-4xb200/) of the same scheme.
Both arms here are bit-exact on repeat, so on matched hardware one differing nonce already
decides.

| Comparison | bit-identical | L2 median | >0.40 |
|---|---:|---:|---:|
| this arm ↔ itself (repeat) | **100.0 %** | 0.0000 | 0.0 % |
| honest ↔ this arm, s1 | 0.0 % | **0.4926** | 73.3 % |

## Environment

| Parameter | Value |
|---|---|
| CUDA | 13.0.2 (image), driver 610.57.04 |
| vLLM | 0.25.1, build `752a3a504485790a2e8491cacbb35c137339ad34` |
| Python | 3.12.13 |
| mlnode | 3.0.16, `gonka_poc.entrypoint.api_router` |

## Config

```bash
TP=2 python3 scripts/patch_hy3.py
curl -X POST http://127.0.0.1:8081/api/v1/inference/up/async \
  -H 'Content-Type: application/json' \
  -d '{"model":"r0b0tlab/Hy3-295B-NVFP4","dtype":"auto","additional_args":[]}'
```

```
--tensor-parallel-size 2   --gpu-memory-utilization 0.90
--max-model-len 262144        --max-num-batched-tokens 65536
--kv-cache-dtype fp8          --logprobs-mode processed_logprobs
--worker-extension-cls gonka_poc.worker.PoCWorkerExtension
--tool-call-parser hy_v3      --reasoning-parser hy_v3
--trust-remote-code --enable-auto-tool-choice --enable-expert-parallel
```

Measurement window **120 s**, batches 16/32/64.

### What changed vs the honest arm

| Parameter | Honest | This arm |
|---|---|---|
| model | `tencent/Hy3-FP8` (276 GiB) | `r0b0tlab/Hy3-295B-NVFP4` |
| MoE kernel | FP8 path | `FLASHINFER_TRTLLM` NvFp4 |
| KV cache | 1 172 144 tokens | 1 914 384 tokens (+63 %) |
| everything else | — | unchanged |

## Validation

### Throughput (120 s window)

| batch | 16 | 32 | 64 |
|---:|---:|---:|---:|
| this arm | 1304 | 1552 | **1599** |
| honest FP8 | 1439 | 1535 | **1599** |

### Fingerprint

```bash
python3 scripts/l2_matrix.py artifacts ref_nonces_fp8_s1.json nonces_nvfp4_s1.json
```

| Pair | bit-identical | L2 median | p95 | >0.40 |
|---|---:|---:|---:|---:|
| this arm ↔ itself (repeat) | **100.0 %** | 0.0000 | 0.0000 | 0.0 % |
| honest ↔ this arm, s1 | 0.0 % | 0.4926 | 0.8580 | 73.3 % |
| honest ↔ this arm, s2 | 0.0 % | 0.4897 | 0.8543 | 70.6 % |
| honest ↔ this arm, s3 | 0.0 % | 0.5038 | 0.8783 | 75.0 % |

The same checkpoint measured on a different host
([4×B200](../hy3-nvfp4-r0b0tlab-4xb200/)) reproduces these to three decimals — the distance
belongs to the build, not the hardware.

### Inference

Warm-engine numbers.

| Scenario | honest out tok/s | this arm out tok/s | Δ |
|---|---:|---:|---:|
| s1 long, sequential | 60.3 | 114.3 | **+90 %** |
| s2 short, 30 runners | 840.8 | 1294.5 | **+54 %** |
| s3 very long, sequential | 87.1 | 93.4 | +7 % |
| s4 very long, 20 runners | 269.2 | 356.7 | +33 % |

### Resources

| | value |
|---|---:|
| KV cache | 1 914 384 tokens |
| bring-up | 363 s |

## Findings

1. **No PoC gain at matched topology** — 1599 = 1599.
2. **Bit-exact on repeat**, like the honest arm: an architecture property, not a
   numeric-format one.
3. **The loudest fraud measured for Hy3** (0.49). A gate calibrated on this build would let
   the quieter llm-compressor build through.
4. **Not a pure W4A4 checkpoint** — vLLM reports a mixed FP8/NVFP4/MXFP8 configuration.

## Files

```
artifacts/
  nonces_nvfp4_{s1,s1_r2,s2,s3}.json   fraud arm (s1_r2 = repeat of s1)
  ref_nonces_fp8_{s1,s1_r2,s2,s3}.json honest reference, same host
  sweep_120s.log                        valid sweep
  serving.sqlite                        compressa-perf database
scripts/
  patch_hy3.py  run_pow_generation.py  collect_artifacts.py  l2_matrix.py  poc_seeds.json
```

Related: [honest FP8 on this host](../hy3-fp8-2xb300/) ·
[same model on 1×B300 — the economics](../hy3-nvfp4-r0b0tlab-1xb300/) ·
[same model on 4×B200](../hy3-nvfp4-r0b0tlab-4xb200/)

## Reproducibility checklist

- [x] Image pinned by digest; quantisation described from `config.json` / loader output
- [x] Every script referenced above committed under `scripts/`
- [x] L2 tables reproducible from committed artifacts via `scripts/l2_matrix.py`
- [x] 3 seeds behind every fingerprint claim
- [x] Invalid or superseded measurements labelled in place, not dropped
- [x] No internal-tooling links, absolute paths, or sibling-repo references
