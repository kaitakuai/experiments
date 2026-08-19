# Hy3 FP8 — 4×B200 — honest baseline (bit-exact at TP=4, which rules out topology)

**Date:** 2026-08-19
**Model:** `tencent/Hy3-FP8` — 295B total / 21B active MoE, 192 experts × top-8, 80 layers
+ 1 MTP layer (`num_nextn_predict_layers: 1`), GQA 64 heads / **8 KV heads** × 128, 256K
context. `quant_method: fp8`, `activation_scheme: static`, **`kv_cache_scheme: static`**;
only `lm_head` and `embed_tokens` excluded. 300 GB, 101 shards, weights **276 GiB** in VRAM.
**Hardware:** 4× NVIDIA **B200 SXM** (1000 W, 183 GB, NV18, driver **580.126.20**, sm_100).
Vast.ai instance 48135501, Virginia US.
**Image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash-0731:3.0.16-overlay-k5`
**Digest:** `sha256:8ce8830b4182b3dfd642c5e00f548f50a830611e4bc64ec4fbd84fe91070e3aa`

> The DeepSeek-V4-Flash foundry image, reused only as a vLLM 0.25.1 + PoC-plugin runtime.
> Its `runner.py` hardcodes V4-specific flags that must be replaced — `scripts/patch_hy3.py`.

## Summary

This run isolates the variable behind the determinism result. [4×H200](../hy3-fp8-4xh200/)
reproduces 0/1000 nonces at TP=4; [2×B300](../hy3-fp8-2xb300/) reproduces 1000/1000 at TP=2.
Here Blackwell runs at **TP=4** and still reproduces **1000/1000** — so the divide is the
architecture, not the tensor-parallel width.

It also supplies the second half of the cross-machine result: honest FP8 here versus honest
FP8 on B300 gives 0 % identical and a **0.2028** median — two Blackwell machines are no
closer to each other than Blackwell is to Hopper.

Note also that the driver here is **580.126.20**, i.e. CUDA 13.0, while the image ships CUDA
13.0.2. It works: minor-version compatibility within CUDA 13 is sufficient, and the offer's
`cuda_max_good: 13.0` was not a blocker.

## Environment

| Parameter | Value |
|---|---|
| CUDA | 13.0.2 (image), driver 580.126.20 (CUDA 13.0) |
| vLLM | 0.25.1, build `752a3a504485790a2e8491cacbb35c137339ad34` |
| Python | 3.12.13 |
| mlnode | 3.0.16, `gonka_poc.entrypoint.api_router` |

## Config

```bash
TP=4 python3 scripts/patch_hy3.py
curl -X POST http://127.0.0.1:8081/api/v1/inference/up/async \
  -H 'Content-Type: application/json' \
  -d '{"model":"tencent/Hy3-FP8","dtype":"auto","additional_args":[]}'
```

```
--tensor-parallel-size 4   --gpu-memory-utilization 0.90
--max-model-len 262144        --max-num-batched-tokens 65536
--kv-cache-dtype fp8          --logprobs-mode processed_logprobs
--worker-extension-cls gonka_poc.worker.PoCWorkerExtension
--tool-call-parser hy_v3      --reasoning-parser hy_v3
--trust-remote-code --enable-auto-tool-choice --enable-expert-parallel
```

Measurement window **120 s**, batches 16/32/64.

### What changed vs the default

| Parameter | Image as-shipped (V4 leaf) | This run |
|---|---|---|
| `--tensor-parallel-size` | 1 | 4 |
| `--max-model-len` | 400000 | 262144 |
| `--max-num-batched-tokens` | 32768 | 65536 (batch 64 × seq_len 1024) |
| `--tokenizer-mode` | `deepseek_v4` | removed (auto) |
| `--tool-call-parser` / `--reasoning-parser` | `deepseek_v4` | `hy_v3` |
| `--speculative-config` | DSpark, 7 tokens | removed |
| `--enable-expert-parallel` | absent | added |

## Validation

### Throughput (120 s window, corrected accounting)

| batch | 16 | 32 | 64 |
|---:|---:|---:|---:|
| nonces/min | 1736 | 1840 | **1888** |

472 nonces per card, against 800 per card on [2×B300](../hy3-fp8-2xb300/): doubling the card
count buys **+18 %**. This is the honest-side reason the fraud arms profit from dropping
tensor parallelism rather than from quantisation itself.

### Fingerprint

```bash
python3 scripts/l2_matrix.py artifacts nonces_fp8_s1.json nonces_fp8_s1_r2.json
```

| Pair | bit-identical | L2 median | >0.40 |
|---|---:|---:|---:|
| **FP8 ↔ FP8 repeat (same machine, TP=4)** | **100.0 %** | 0.0000 | 0.0 % |
| FP8 here ↔ FP8 on 2×B300 | 0.0 % | 0.2028 | 3.3 % |
| FP8 here ↔ FP8 on 4×H200 | 0.0 % | 0.2043 | 3.8 % |

### Inference

| Scenario | TTFT | out tok/s |
|---|---:|---:|
| s1 long, sequential | 0.794 | 84.2 |
| s2 short, 30 runners | 0.304 | 1183.2 |
| s3 very long, sequential | 0.798 | 84.8 |
| s4 very long, 20 runners | 2.204 | **369.3** |

s1 TTFT here is 0.794 s against 1.467 s for the same checkpoint on 2×B300 — evidence that
the B300 TTFT anomaly noted in that folder is topology-related rather than a property of the
FP8 checkpoint.

### Resources

| | value |
|---|---:|
| weights / rank | 69.28 GiB (**276 GiB** total) |
| KV cache | 2 046 704 tokens |
| bring-up | 293 s (compilation 85 s) |

## Findings

1. **Bit-exact at TP=4 on Blackwell** — topology is not the cause of Hopper's 0/1000.
2. **Two Blackwell machines agree no better than Blackwell and Hopper** (0.2028 vs 0.2043).
3. **A CUDA-13.0 driver runs the CUDA-13.0.2 image** without the forward-compat layer.
4. **+18 % for double the cards** — the honest side of the TP economics.

## Files

```
artifacts/
  nonces_fp8_{s1,s1_r2,s2,s3}.json   honest FP8 (s1_r2 = repeat of s1)
  sweep_120s.log                      valid sweep
  serving.sqlite                      compressa-perf database
scripts/
  patch_hy3.py  run_pow_generation.py  collect_artifacts.py  l2_matrix.py  poc_seeds.json
```

Related: [FP8 on 4×H200](../hy3-fp8-4xh200/) · [FP8 on 2×B300](../hy3-fp8-2xb300/) ·
[NVFP4 ModelOpt](../hy3-nvfp4-modelopt-4xb200/) · [NVFP4 llm-compressor](../hy3-nvfp4-llmcompressor-4xb200/)

## Reproducibility checklist

- [x] Image pinned by digest; quantisation described from `config.json` / loader output
- [x] Every script referenced above committed under `scripts/`
- [x] L2 tables reproducible from committed artifacts via `scripts/l2_matrix.py`
- [x] 3 seeds behind every fingerprint claim
- [x] Invalid or superseded measurements labelled in place, not dropped
- [x] No internal-tooling links, absolute paths, or sibling-repo references
