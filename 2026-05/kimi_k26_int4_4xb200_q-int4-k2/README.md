# Kimi-K2.6 — 4×B200 — q-int4-k2 image validation

**Date:** 2026-05-21
**Model:** `moonshotai/Kimi-K2.6` (compressed-tensors, MLA, 384 routed × top_k=8)
**Hardware:** 4× NVIDIA B200 SXM (host 86.38.182.57, node `mlnode-050`, 179.06 GiB HBM/GPU, NVLink 53.125 GB/s/link, driver 580.126.09, sm_100)
**Image:** `ghcr.io/kaitakuai/mlnode-b200-kimi-k2-6:0.2.13-vllm0.20.0-q.int4-k2`
**Digest:** `sha256:43943c0b478808383cdbb1f95ee405f63bd8b52b3eab5e1ab1037bda12cdd914` (built 2026-05-19)

## Summary

Validation of the **dedicated B200 image** `mlnode-b200-kimi-k2-6:0.2.13-vllm0.20.0-q.int4-k2`.
Unlike the generic `b200-k5-kimi-1` image (eager mode), this image runs with
**CUDA graphs** (`cudagraph_mode=FULL_AND_PIECEWISE`), expert-parallel on, at
`gpu-memory-utilization=0.93`.

The first build of this image (digest `f8f3eca…`) crashed PoC generation with
an `AssertionError` in MLA attention; kaitakuai shipped a fixed build (digest
`43943c0b…`, validated here). With the fixed image, vLLM cold-starts in ~34 s
(engine init), sustains **~2240 nonces/min @ batch=32** on a single 4×B200
instance, and produces nonces that **PASS** the canonical Gonka chain
validation against every B200/B300 baseline on the network (mean L2 in the
0.187–0.189 cross-anything floor).

## Image identity (verified)

```
Config.Image  : ghcr.io/kaitakuai/mlnode-b200-kimi-k2-6:0.2.13-vllm0.20.0-q.int4-k2
Image ID      : sha256:43943c0b478808383cdbb1f95ee405f63bd8b52b3eab5e1ab1037bda12cdd914
RepoDigest    : ghcr.io/kaitakuai/mlnode-b200-kimi-k2-6@sha256:43943c0b4788...12cdd914
Quant         : compressed-tensors  ("quant_method":"compressed-tensors")
Created       : 2026-05-19T19:35:21Z
```

## Symptom: PoC crash on the first build (digest `f8f3eca…`)

The initial build of this image crashed during PoC v2 generation with:

```
File ".../vllm/v1/attention/backends/mla/mla_attention.py", line 1843, in build
    assert seq_lens_cpu is not None
AssertionError
```

Inference (`/v1/chat/completions`) worked, but any PoC `init/generate`
immediately killed the engine. The MLA attention metadata builder requires
`seq_lens_cpu` to be populated on the PoC forward path.

## Root cause

The PoC model runner (`poc_model_runner.py`) lost the line that forwards the
CPU-side sequence-length upper bound into the attention metadata:

```python
seq_lens_cpu_upper_bound=seq_lens_cpu,
```

Without it, the MLA backend's `build()` asserts on `seq_lens_cpu is not None`.
This is a porting regression — the inference forward path sets it, the PoC
forward path did not. Confirmed by adding the single line back and observing
PoC generate succeed.

## Fix

kaitakuai shipped a rebuilt image with the line restored, digest
`43943c0b…`. This is the build validated in this report. PoC generation runs
clean on it; no live patches required.

## Config (baked into the image / launched via `inference/up`)

```
--tensor-parallel-size 4
--gpu-memory-utilization 0.93
--max-model-len 120000
--max-num-batched-tokens 32768          # caps PoC effective batch at 32 (32×1024=32768)
--max-num-seqs 32
--compilation-config '{"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"}'  # CUDA graphs
--attention-backend CUTLASS_MLA
--enable-expert-parallel
--enable-auto-tool-choice
--tool-call-parser kimi_k2
--reasoning-parser kimi_k2
--logprobs-mode processed_logprobs
--mm-encoder-tp-mode data
--trust-remote-code
```

Differs from the `b200-k5-kimi-1` config in three ways: **CUDA graphs**
(vs eager), **expert-parallel enabled** (vs removed), **gmu 0.93** (vs 0.95).

## Validation

### Cold start (fixed image, tuned knobs)

| Step | Result |
|---|---:|
| Model load to VRAM (TP=4) | 141.37 GiB / GPU, 30.3 s |
| Available KV cache (gmu=0.93) | 14.3 GiB pool |
| CUDA graph capture (PIECEWISE 11 + FULL decode 7) | 4 s, 0.45 GiB |
| init engine (profile + KV + warmup) | 33.9 s (compilation 3.0 s) |
| First `inference_healthy=true` | ~34 s after model resident |

### Phase-3 PoC throughput sweep (`run_pow_generation.py`, callback aggregator)

30 s measurement window per batch, 5 s warmup, single 4×B200 instance. Two
independent measurements agree: the sweep callback aggregator (nonces received
in the 30 s window) and the PoC engine's own steady-state counter
(`routes.py:346 "Generated: N nonces (X/min)"`).

| batch | engine counter (steady-state) | sweep aggregator (30 s) |
|---:|---:|---:|
|  8 | ~1 918/min | 1 904/min |
| 16 | ~2 116/min | window-noisy |
| 32 | **~2 235/min ★** | **2 240/min ★** |
| 64 | OOM | OOM — exceeds 32 768 token budget |
| 128 | OOM | OOM |

**Best: ~2240 nonces/min @ batch=32** (sweep aggregator 1120/30 s = 2240; engine
counter converges to ~2235; reproduced across 4 sweeps). Batch is capped at 32
because `--max-num-batched-tokens 32768 = 32 × seq_len 1024`; batch 64/128
exceed the token budget and OOM. Intermediate batches are noisy in the sweep's
30 s callback window (PoC engine stop/start between batches eats into the
window), but the engine's own counter gives a clean monotonic curve
1918 → 2116 → 2235; the peak at batch=32 is rock-solid.

Per-GPU: 2240 / 4 = **560 nonces/min/GPU**, matching the `b200-k5-kimi-1`
result (also 2240 @ batch=32) — i.e. CUDA graphs do not change PoC throughput
vs the k5 eager build (PoC runs a separate forward-only path, not the
inference decode/CUDA-graph path).

### Canonical Gonka chain validation (mean L2 + binomtest)

Run via [`gonka-l2-validate`](../../../.claude/skills/gonka-l2-validate/SKILL.md)
skill (formulas lifted byte-for-byte from `vllm/poc/data.py` +
`vllm/poc/validation.py`). 200 common nonces per pair, k_dim=12.

| Pair | mean L2 | n_mismatch @ thr=0.4 | verdict (chain proto + calibrated p_mis=0.02) |
|---|---:|---:|---|
| **050 (this image) ↔ 051 (b200 k5 eager)** | **0.1874** | 5 (2.5%) | **PASS** p=0.37 |
| **050 (this image) ↔ 001 (b300 eager)** | **0.1888** | 3 (1.5%) | **PASS** p=0.77 |
| **050 (this image) ↔ 002 (b300 CG)** | **0.1879** | 3 (1.5%) | **PASS** p=0.77 |

All mean L2 in the **0.187–0.189** corridor — the fundamental Kimi-K2.6
cross-anything floor (384 routed experts × top_k=8 + discrete MoE routing: a
single-bit numeric perturbation flips the expert set → activations diverge to
the end of the network). **At chain-default `dist_threshold=0.4` + calibrated
`p_mismatch=0.02` every pair PASSES the binomial fraud test.** Strict
`0.02/0.001` self-validation defaults FAIL across all cross-anything pairs
(expected).

This confirms the image produces honest, cross-validatable nonces
indistinguishable on-chain from the k5 image and from the B300 nodes.

### Inference sanity (FP-side regression probe)

Same instance, `/v1/chat/completions`, streaming, concurrency=10, 30 req,
max_tokens=300 (CUDA-graphs warm):

| Metric | Value |
|---|---:|
| TTFT mean | 0.365 s |
| Latency mean | 5.14 s |
| TPOT mean | 16.0 ms/tok |
| Throughput output | 583 tok/s (aggregate) |
| requests ok/err | 30/0 |

Coherent output, no degradation. CUDA graphs give ≈2.2× decode throughput vs
eager on the same hardware (see related B300 report).

## Files

- [`artifacts/nonces.json`](artifacts/nonces.json) — Kimi-K2.6 PoC v2 nonces
  collected on this image (200 artifacts, seq_len 1024, k_dim 12), input to the
  canonical L2 validation
- [`artifacts/l2_4way.json`](artifacts/l2_4way.json) — canonical pairwise L2 +
  binomtest results across all 4 live network nodes (050 this image, 051 k5,
  001/002 b300)
- [`artifacts/up_config.json`](artifacts/up_config.json) — exact
  `inference/up/async` payload (full vLLM arg set) used on `mlnode-050`

## Related

- Generic B200 k5 image (eager):
  [`../kimi_k26_4xb200_b200-k5-kimi-1/README.md`](../kimi_k26_4xb200_b200-k5-kimi-1/README.md)
- B300 perf attribution (eager vs CUDA-graphs):
  [`../kimi_k26_b300_eager_flashinfer/README.md`](../kimi_k26_b300_eager_flashinfer/README.md)
- Canonical L2 skill:
  [`../../../.claude/skills/gonka-l2-validate/SKILL.md`](../../../.claude/skills/gonka-l2-validate/SKILL.md)
