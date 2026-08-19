# Hy3 FP8 — 2×B300 — honest baseline (PoC is **bit-exact** on Blackwell, but only per machine)

**Date:** 2026-08-19
**Model:** `tencent/Hy3-FP8` — 295B total / 21B active MoE, 192 experts × top-8, 80 layers
+ 1 MTP layer (`num_nextn_predict_layers: 1`), GQA 64 heads / **8 KV heads** × 128, 256K
context. `quant_method: fp8`, `activation_scheme: static`, **`kv_cache_scheme: static`**;
only `lm_head` and `embed_tokens` excluded. 300 GB, 101 shards, weights **276 GiB** in VRAM.
**Hardware:** 2× NVIDIA **B300 SXM6** (1100 W, 275 GB, NV18, driver **610.57.04**, sm_100).
Vast.ai instance 48124506, Utah US.
**Image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash-0731:3.0.16-overlay-k5`
**Digest:** `sha256:8ce8830b4182b3dfd642c5e00f548f50a830611e4bc64ec4fbd84fe91070e3aa`

> The DeepSeek-V4-Flash foundry image, reused only as a vLLM 0.25.1 + PoC-plugin runtime.
> Its `runner.py` hardcodes V4-specific flags that must be replaced — `scripts/patch_hy3.py`.

## Summary

Repeating the identical run on this machine reproduces **1000 of 1000 nonces bit-exactly**
(median L2 = 0.0000). The same procedure on [4×H200](../hy3-fp8-4xh200/) reproduces 0 of
1000. Against a zero floor, any deviation is a verdict — a fraud arm can be caught with a
single nonce.

**That property does not survive a change of machine.** Honest FP8 here versus honest FP8 on
[4×B200](../hy3-fp8-4xb200/) — both Blackwell — gives 0 % identical and a **0.2028** median,
the same floor as Blackwell↔Hopper. Exact-match validation is therefore usable only inside a
homogeneous pool or for a node checking itself, not across a heterogeneous network.

This host also anchors the fraud economics: FP8 needs **276 GiB** of weights and a single
B300 offers ~242 GiB at `gmu 0.90`, so the honest model **cannot** run on one card — while a
4-bit fraud arm can. See [NVFP4 on 1×B300](../hy3-nvfp4-r0b0tlab-1xb300/).

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
  -d '{"model":"tencent/Hy3-FP8","dtype":"auto","additional_args":[]}'
```

```
--tensor-parallel-size 2   --gpu-memory-utilization 0.90
--max-model-len 262144        --max-num-batched-tokens 65536
--kv-cache-dtype fp8          --logprobs-mode processed_logprobs
--worker-extension-cls gonka_poc.worker.PoCWorkerExtension
--tool-call-parser hy_v3      --reasoning-parser hy_v3
--trust-remote-code --enable-auto-tool-choice --enable-expert-parallel
```

Measurement window **120 s** (`GENERATION_DURATION_S=120`), batches 16/32/64.

### What changed vs the default

| Parameter | Image as-shipped (V4 leaf) | This run |
|---|---|---|
| `--tensor-parallel-size` | 1 | 2 |
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
| nonces/min | 1439 | 1535 | **1599** |

800 nonces per card. For comparison [4×B200](../hy3-fp8-4xb200/) reaches 1888 on twice the
cards — 472 per card — so tensor parallelism barely pays for itself.

**Why 120 s and not 30 s.** `sweep_30s_noise_demo.log` is the same arm measured on a 30 s
window *after* the accounting fix: 1439 / 1535 / **1279**. The top point is pure noise —
callbacks arrive in ~5 s bulks, so a 30 s window holds ~6 deliveries and one delivery is
±17 %. Before the fix (`kaitakuai/experiments` PR #7) the same points read 1695 / 1791 /
1791, i.e. +17.8 % / +16.7 % / **+40 %** inflation.

### Fingerprint

```bash
python3 scripts/l2_matrix.py artifacts nonces_fp8_s1.json nonces_fp8_s1_r2.json
```

| Pair | bit-identical | L2 median | >0.40 |
|---|---:|---:|---:|
| **FP8 ↔ FP8 repeat (same machine)** | **100.0 %** | 0.0000 | 0.0 % |
| FP8 here ↔ FP8 on 4×B200 | 0.0 % | 0.2028 | 3.3 % |
| FP8 here ↔ FP8 on 4×H200 | 0.0 % | 0.1977 | 4.3 % |

The cross-host rows are reproduced from the sibling folders' artifacts; the sets are
committed there.

### Inference

Warm-engine numbers — a cold-boot run reported 56 tok/s on s3 where the warm run gives 87,
so serving must be measured **after** PoC load, not straight after bring-up.

| Scenario | TTFT | out tok/s |
|---|---:|---:|
| s1 long, sequential | 1.467 | 60.3 |
| s2 short, 30 runners | 0.657 | 840.8 |
| s3 very long, sequential | 0.498 | 87.1 |
| s4 very long, 20 runners | 4.137 | 269.2 |

**Unexplained:** s1 TTFT is 1.467 s here but 0.794 s on 4×B200 with the identical
checkpoint, and 0.246 s for the NVFP4 arm on this very host — even though PoC prefill
throughput is identical between the two arms (1599 = 1599). Not a warm-up effect (cold
1.468, warm 1.467). Looks topology-related; needs profiling rather than another benchmark.

### Resources

| | value |
|---|---:|
| weights / rank | 138.16 GiB (**276 GiB** total) |
| KV cache | 1 172 144 tokens |
| bring-up | 508 s (compilation 129 s) |

## Findings

1. **Bit-exact PoC on repeat** (1000/1000) — and the same holds for the quantised arm on
   this host, so it is an architecture property, not a numeric-format one.
2. **Bit-exactness is scoped to one machine.** B300↔B200 honest-to-honest sits at 0.2028.
3. **FP8 does not fit one B300** (276 GiB vs ~242 GiB usable) — the root of the fraud
   economics measured in the sibling folders.
4. **30 s windows are not measurable**; 120 s is the minimum for comparing arms.
5. **Serving must be measured warm.**

## Files

```
artifacts/
  nonces_fp8_{s1,s1_r2,s2,s3}.json   honest FP8 (s1_r2 = repeat of s1)
  sweep_120s.log                      valid sweep
  sweep_30s_noise_demo.log            same arm, 30 s window — why it is not enough
  serving.sqlite                      compressa-perf database (warm run)
scripts/
  patch_hy3.py  run_pow_generation.py  collect_artifacts.py  l2_matrix.py  poc_seeds.json
```

Related: [FP8 on 4×H200](../hy3-fp8-4xh200/) · [FP8 on 4×B200](../hy3-fp8-4xb200/) ·
[NVFP4 fraud on this host](../hy3-nvfp4-r0b0tlab-2xb300/)

## Reproducibility checklist

- [x] Image pinned by digest; quantisation described from `config.json` / loader output
- [x] Every script referenced above committed under `scripts/`
- [x] L2 tables reproducible from committed artifacts via `scripts/l2_matrix.py`
- [x] 3 seeds behind every fingerprint claim
- [x] Invalid or superseded measurements labelled in place, not dropped
- [x] No internal-tooling links, absolute paths, or sibling-repo references
