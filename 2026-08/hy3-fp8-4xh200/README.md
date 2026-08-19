# Hy3 FP8 — 4×H200 — honest baseline (PoC fingerprint is **not** bit-reproducible on Hopper)

**Date:** 2026-08-19
**Model:** `tencent/Hy3-FP8` — 295B total / 21B active MoE, 192 experts × top-8, 80 layers
+ 1 MTP layer (`num_nextn_predict_layers: 1`), GQA 64 heads / **8 KV heads** × 128, 256K
context. `quant_method: fp8`, `activation_scheme: static`, **`kv_cache_scheme: static`**;
only `lm_head` and `embed_tokens` excluded. 300 GB, 101 shards, weights **276 GiB** in VRAM.
**Hardware:** 4× NVIDIA **H200 SXM** (700 W, 143 GB HBM3e, NV18 full mesh, driver
**610.57.04**, sm_90). Vast.ai instance 48115352, Virginia US.
**Image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash-0731:3.0.16-overlay-k5`
**Digest:** `sha256:8ce8830b4182b3dfd642c5e00f548f50a830611e4bc64ec4fbd84fe91070e3aa`

> The DeepSeek-V4-Flash foundry image, reused only as a vLLM 0.25.1 + PoC-plugin runtime.
> Its `runner.py` hardcodes V4-specific flags that must be replaced — `scripts/patch_hy3.py`.

## Summary

The honest reference for every Hopper fraud comparison, and the run that establishes the
main constraint on validating Hy3 at all: **on Hopper the proof is not reproducible.**
Re-running the identical configuration on the identical machine reproduces **0 of 1000**
nonces bit-exactly, with a 0.2025 median L2 between two honest runs.

The same experiment on Blackwell reproduces 1000/1000
([2×B300](../hy3-fp8-2xb300/), [4×B200](../hy3-fp8-4xb200/)), and both hosts there ran the
same or a wider topology — so this is an architecture property, not a TP=4 artefact.

**Consequence:** the 0.40 gate inherited from earlier campaigns is unusable here. An honest
prover exceeds it on **4.1 %** of nonces. Detection on Hopper must be aggregate; see
[the INT4 arm](../hy3-int4-cyankiwi-4xh200/) for the separability analysis.

Hy3 also needs **no vLLM port**: `HYV3ForCausalLM` and `HYV3MTPModel` are already registered
in 0.25.1. The recipe's "≥0.26" requirement concerns optimizations (PR #47433 + HPC-Ops
kernels), not support.

## Environment

| Parameter | Value |
|---|---|
| CUDA | 13.0.2 (image), driver 610.57.04 |
| vLLM | 0.25.1, build `752a3a504485790a2e8491cacbb35c137339ad34` |
| Python | 3.12.13 |
| mlnode | 3.0.16, `gonka_poc.entrypoint.api_router` |

## Config

```bash
cd /app && source /app/packages/api/.venv/bin/activate   # system python cannot import `common`
WATCHER_MAX_UNHEALTHY_COUNT=9999 VLLM_RUNNER_TIMEOUT=3600 \
  python -m uvicorn api.app:app --host 0.0.0.0 --port 8081

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

### What changed vs the default

| Parameter | Image as-shipped (V4 leaf) | This run |
|---|---|---|
| `--tensor-parallel-size` | 1 | 4 |
| `--max-model-len` | 400000 | 262144 |
| `--max-num-batched-tokens` | 32768 | 65536 (batch 64 × seq_len 1024) |
| `--tokenizer-mode` | `deepseek_v4` | removed (auto) |
| `--tool-call-parser` / `--reasoning-parser` | `deepseek_v4` | `hy_v3` |
| `--speculative-config` | DSpark, 7 tokens | removed (baseline) / MTP-2 (variant below) |
| `--enable-expert-parallel` | absent | added |

## Validation

### Throughput

| batch | 8 | 16 | 32 | 64 |
|---:|---:|---:|---:|---:|
| FP8 | 1248 | 1376 | **1408** | 1408 |
| FP8 + MTP-2 | 1248 | 1376 | **1408** | 1408 |

> ⚠️ **These throughput numbers are invalid.** They were taken with a 30 s measurement
> window *and* the pre-fix boundary-accounting bug (`kaitakuai/experiments` PR #7), which
> inflates by an unknown 0–40 %. On top of that, callbacks arrive in ~5 s bulks, so a 30 s
> window carries ±17 % noise on its own. They are published rather than deleted because the
> relative comparisons on this host were taken identically. For valid numbers see the
> Blackwell runs, which use the corrected script and a 120 s window.

The two rows being identical to the nonce is still informative: MTP cannot influence a
prefill-only proof.

### MTP speculative decoding (same model, same host)

`--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`. No draft checkpoint
is needed — `hf_config_override` maps `hy_v3` → `hy_v3_mtp` (`HYV3MTPModel`) and reuses the
target weights (log: *"Detected MTP model. Sharing target model embedding / lm_head weights
with the draft model"*).

| | fingerprint vs honest | KV tokens | s1 out tok/s | s3 out tok/s |
|---|---:|---:|---:|---:|
| FP8 | — | 1 049 888 | 93.5 | 79.1 |
| FP8 + MTP-2 | **0.1993** (floor: 0.2025) | 951 680 (−9.4 %) | 112.6 (**+20 %**) | 102.1 (**+29 %**) |

MTP is statistically indistinguishable from a plain repeat, so it is undetectable in the
proof and free to enable — a policy question, not a detection problem. If allowed, it should
be baked into the image, otherwise the gain accrues only to operators who discover it.
Cost: 9.4 % of KV and a doubled s2 TTFT (0.188 → 0.432).

### Fingerprint

```bash
python3 scripts/l2_matrix.py artifacts nonces_fp8_s1.json nonces_fp8_s1_r2.json
```

| Pair | bit-identical | L2 median | p95 | max | >0.40 |
|---|---:|---:|---:|---:|---:|
| FP8 ↔ FP8 repeat (**honest floor**) | 0.0 % | 0.2025 | 0.3808 | 0.9540 | **4.1 %** |
| FP8 ↔ FP8 + MTP-2 | 0.0 % | 0.1993 | 0.3765 | 0.9367 | 3.6 % |
| FP8 + MTP ↔ its own repeat | 0.0 % | 0.2004 | 0.3744 | 0.7692 | 4.0 % |
| control: seed s1 ↔ seed s2 | 0.0 % | 1.4041 | 1.7055 | 1.8872 | 100 % |

The cross-seed control lands on the expected ~1.4 asymptote. Divergence is spread over
**all** nonces rather than concentrating on `n % batch_size == 0`, so the "first nonce in
batch" artefact seen with other models does not dominate here.

### Inference

| Scenario | TTFT | out tok/s |
|---|---:|---:|
| s1 long prompt, sequential | 0.278 | 93.5 |
| s2 short prompt, 30 runners | 0.188 | 1224.2 |
| s3 very long, sequential | 0.625 | 79.1 |
| s4 very long, 20 runners | 4.405 | 275.8 |

Throughput is recomputed from the `measurements` table: compressa-perf 0.2.7 inserts metrics
in a loop with `conn.commit()` commented out and then crashes generating a PDF
(`OSError: Cannot open resource "logo.png"`), so `THROUGHPUT_*` and `RPS` are usually lost.

### Resources

| | value |
|---|---:|
| weights / rank | 69.28 GiB (**276 GiB** total) |
| KV cache | 1 049 888 tokens |
| bring-up | 225 s (compilation 75 s) |

KV costs ≈ **164 KiB/token** at fp8 (2 × 80 layers × 8 KV heads × 128) — heavy, though
irrelevant for a prefill proof (batch 64 × 1024 = 65 536 tokens).

## Findings

1. **No vLLM port needed** — 0.25.1 already registers Hy3 and its MTP model.
2. **PoC is not bit-reproducible on Hopper** (0/1000 on repeat), unlike Blackwell.
3. **The 0.40 gate is invalid here** — 4.1 % honest false positives.
4. **MTP is free for the proof, worth +20…29 % for serving.**
5. **compressa-perf 0.2.7 loses metrics**; recompute from `measurements`:
   `sum(n_output) / (max(end_time) - min(start_time))`.
6. **The mlnode API must run from `/app/packages/api/.venv`** — the system python cannot
   import `common`.

## Files

```
artifacts/
  nonces_fp8_{s1,s1_r2,s2,s3}.json      honest FP8 (s1_r2 = repeat of s1)
  nonces_fp8_mtp2_{s1,s1_r2}.json       MTP-2 variant
  sweep_30s_BUGGY.log, sweep_mtp2_30s_BUGGY.log
  serving.sqlite, serving_mtp2.sqlite    compressa-perf databases
scripts/
  patch_hy3.py  run_pow_generation.py  collect_artifacts.py  l2_matrix.py  poc_seeds.json
```

Related: [FP8 on 2×B300](../hy3-fp8-2xb300/) · [FP8 on 4×B200](../hy3-fp8-4xb200/) ·
[INT4 fraud on this host](../hy3-int4-cyankiwi-4xh200/)

## Reproducibility checklist

- [x] Image pinned by digest; quantisation described from `config.json` / loader output
- [x] Every script referenced above committed under `scripts/`
- [x] L2 tables reproducible from committed artifacts via `scripts/l2_matrix.py`
- [x] 3 seeds behind every fingerprint claim
- [x] Invalid or superseded measurements labelled in place, not dropped
- [x] No internal-tooling links, absolute paths, or sibling-repo references
