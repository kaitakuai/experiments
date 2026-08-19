# Hy3 INT4 W4A16 by `cyankiwi` — 8×H100 at TP=8 — 30 % slower than honest at matched topology

**Date:** 2026-08-19
**Model:** `cyankiwi/Hy3-AWQ-INT4` — 182 GB, 34 shards. **Despite the repo name this is not
AWQ**: `quant_method: compressed-tensors`, format `pack-quantized`, INT4 **W4A16**
asymmetric, `group_size 32`, observer `mse`, `input_activations: null` (activations stay
BF16), `kv_cache_scheme: null`. Of the 915 `ignore` entries, **588 are the whole MTP layer
80**, plus layer 0, router gates, `lm_head` and every dense MLP. Kernels:
`MarlinLinearKernel` + `CompressedTensorsWNA16MarlinMoEMethod`. Weights: **166 GiB**.
**Honest reference:** [`../hy3-fp8-8xh100/`](../hy3-fp8-8xh100/) — same host, same session,
same topology; its nonce sets and sweep are duplicated here as `ref_*`.
**Hardware:** 8× NVIDIA **H100 80GB HBM3 SXM** (700 W, NV18, driver **580.126.09**, sm_90).
**Image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash-0731:3.0.16-overlay-k5`
**Digest:** `sha256:8ce8830b4182b3dfd642c5e00f548f50a830611e4bc64ec4fbd84fe91070e3aa`

> The DeepSeek-V4-Flash foundry image, reused only as a vLLM 0.25.1 + PoC-plugin runtime.
> Its `runner.py` hardcodes V4-specific flags that must be replaced — `scripts/patch_hy3.py`.

## Summary

Measured against the honest arm at the **same** topology, so the only variable is the
checkpoint. The fraud is worse on every axis except memory:

| | honest FP8 | this arm |
|---|---:|---:|
| nonces/min | 1344 | **944 (−30 %)** |
| weights | 276 GiB | **166 GiB** |
| KV tokens | 1 058 752 | 1 799 216 |
| s4 output tok/s | 333.5 | 267.1 (−20 %) |
| s4 TTFT | 3.394 | 5.669 |

Its fingerprint distance from honest is **0.3755**, against a 0.2013 honest floor — and that
number matches the 0.3741 measured for the *same checkpoint* on
[4×H200 at TP=4](../hy3-int4-cyankiwi-4xh200/) to the third decimal. Neither the chip nor the
tensor-parallel width moves it.

The attack only becomes profitable at a narrower topology — see
[the same model on 4 cards](../hy3-int4-cyankiwi-4xh100/).

## Environment

| Parameter | Value |
|---|---|
| CUDA | 13.0.2 (image), driver 580.126.09 |
| vLLM | 0.25.1, build `752a3a504485790a2e8491cacbb35c137339ad34` |
| Python | 3.12.13 |
| mlnode | 3.0.16, `gonka_poc.entrypoint.api_router` |

Bare-metal host: everything runs inside the container (`docker exec`; use `-i` for heredocs).

## Config

Identical to the honest arm; only the model id changes.

```bash
docker exec -e TP=8 hy3 python3 /root/patch_hy3.py
docker exec hy3 curl -s -X POST http://127.0.0.1:8081/api/v1/inference/up/async \
  -H 'Content-Type: application/json' \
  -d '{"model":"cyankiwi/Hy3-AWQ-INT4","dtype":"auto","additional_args":[]}'
```

```
--tensor-parallel-size 8   --gpu-memory-utilization 0.90
--max-model-len 262144        --max-num-batched-tokens 65536
--kv-cache-dtype fp8          --logprobs-mode processed_logprobs
--worker-extension-cls gonka_poc.worker.PoCWorkerExtension
--tool-call-parser hy_v3      --reasoning-parser hy_v3
--trust-remote-code --enable-auto-tool-choice --enable-expert-parallel
```

### What changed vs the honest arm

| Parameter | Honest | This arm |
|---|---|---|
| model | `tencent/Hy3-FP8` (276 GiB) | `cyankiwi/Hy3-AWQ-INT4` (166 GiB) |
| MoE kernel | FP8 path | MARLIN WNA16 |
| KV scales | static, baked in | `kv_cache_scheme: null` → runtime under forced `--kv-cache-dtype fp8` |
| everything else | — | unchanged |

## Validation

### Throughput (120 s window)

| batch | 16 | 32 | 64 |
|---:|---:|---:|---:|
| this arm | 896 | **944** | 928 |
| honest FP8 | 1232 | 1312 | **1344** |

Note the shape: the fraud peaks at batch 32 and **declines** at 64, while the honest arm
grows monotonically. MARLIN saturates earlier — an early hint that this checkpoint prefers
fewer cards.

On 4×H200 the same checkpoint was 36 % behind honest; here 30 %. Marlin dequantisation is
expensive in prefill regardless of tensor-parallel width.

### Fingerprint

```bash
python3 scripts/l2_matrix.py artifacts ref_nonces_fp8_s1.json nonces_int4_s1.json
```

| Pair | bit-identical | L2 median | p95 | >0.40 |
|---|---:|---:|---:|---:|
| honest ↔ honest (floor) | 0.0 % | 0.2013 | 0.3744 | 3.5 % |
| this arm ↔ itself (repeat) | **0.0 %** | — | — | — |
| **honest ↔ this arm** | 0.0 % | **0.3755** | 0.6370 | **43.0 %** |

The fraud arm is non-deterministic on repeat as well — on Hopper that holds for quantised
weights too, exactly as bit-exactness holds for quantised weights on Blackwell.

Same checkpoint, different hardware:

| host | topology | L2 median vs honest |
|---|---|---:|
| 4×H200 | TP=4 | 0.3741 |
| **8×H100** | **TP=8** | **0.3755** |

### Inference

| Scenario | honest out tok/s | this arm out tok/s | Δ |
|---|---:|---:|---:|
| s1 long, sequential | 90.8 | 85.0 | −6 % |
| s2 short, 30 runners | 1253.3 | **1458.6** | **+16 %** |
| s3 very long, sequential | 78.7 | 66.8 | −15 % |
| s4 very long, 20 runners | 333.5 | 267.1 | −20 % |

The only win is short prompts at high concurrency, where decode is memory-bound and 4-bit
weights help. Everywhere else it loses, and TTFT degrades badly.

### Resources

| | value |
|---|---:|
| weights / rank | 20.7 GiB (**166 GiB** total) |
| KV cache | 1 799 216 tokens |
| bring-up | 197 s (compilation 118 s) |

KV is capped by `--max-model-len 262144` rather than by free memory here — the same 1.79 M
tokens as on 4×H200, despite twice the cards.

## Findings

1. **A bad fraud at matched topology** — 30 % slower in PoC, worse serving, louder
   fingerprint (0.3755 vs a 0.2013 floor).
2. **The fingerprint is invariant to chip and topology**: 0.3741 on H200 TP=4, 0.3755 here.
3. **MARLIN saturates at batch 32**, unlike the honest arm — the reason the same checkpoint
   does better on fewer cards.
4. **Hopper non-determinism is quantisation-independent** — 0/1000 for this arm too.

## Files

```
artifacts/
  nonces_int4_{s1,s1_r2,s2,s3}.json      this arm (s1_r2 = repeat of s1)
  ref_nonces_fp8_{s1,s1_r2,s2,s3}.json   honest reference, same host and topology
  sweep_120s.log, ref_sweep_honest_tp8_120s.log
  serving.sqlite
scripts/
  patch_hy3.py  run_pow_generation.py  collect_artifacts.py  l2_matrix.py  poc_seeds.json
```

Related: [honest FP8 on this host](../hy3-fp8-8xh100/) ·
[same model on 4 cards — the economics](../hy3-int4-cyankiwi-4xh100/) ·
[same model on 4×H200](../hy3-int4-cyankiwi-4xh200/)

## Reproducibility checklist

- [x] Image pinned by digest; quantisation described from `config.json` / loader output
- [x] Every script referenced above committed under `scripts/`
- [x] L2 tables reproducible from committed artifacts via `scripts/l2_matrix.py`
- [x] 3 seeds behind every fingerprint claim
- [x] Invalid or superseded measurements labelled in place, not dropped
- [x] No internal-tooling links, absolute paths, or sibling-repo references
