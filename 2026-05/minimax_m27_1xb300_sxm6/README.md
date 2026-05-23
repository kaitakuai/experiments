# MiniMax-M2.7 — 1×B300 SXM6 — b300 image validation + inference

**Date:** 2026-05-23
**Model:** `MiniMaxAI/MiniMax-M2.7` (FP8 weights, FP8 KV cache, MoE)
**Hardware:** 1× NVIDIA **B300 SXM6 AC** (bare server 31.22.104.128, host `host-u1ab7iu2vbb`, **275 GiB HBM**, 1100 W, driver 580.126.09, sm_100 Blackwell Ultra)
**Image:** `ghcr.io/kaitakuai/mlnode-b300-minimax-m2-7:0.2.13-vllm0.20.0-k1`
**Digest:** `sha256:45e14a9177764cb51501d95c00dbdf8674bd8bf2de03716598110403d06ff1b0` (PR#36 baked in)

## Summary

Validation of the `mlnode-b300-minimax-m2-7` image on a **single B300 SXM6** (275 GiB HBM), **TP=1**, FLASHINFER_TRTLLM Fp8 MoE + FLASHINFER attention. Run on a bare server (host docker, `docker run` — not Vast).

The headline: **MiniMax-M2.7 FP8 (230 GB) fits and runs on ONE B300** (275 GiB HBM), no tensor-parallelism needed.

1. **PoC throughput = 1792 nonces/min @ batch=32 on a single GPU** — the **highest per-GPU** of any hardware tested (B200 1312/GPU, H100 576/GPU, H200 864/GPU, A100 224/GPU).
2. **Inference (compressa-perf): TTFT 1.28 s (fastest of all), TPOT 26.6 ms/tok, 746 output tok/s, RPS 2.51, 0 failures** — a single B300 ≈ a 2×B200 instance (803 tok/s) for inference.
3. **Nonces cross-validate with the B200 baseline under the MiniMax chain gate** (mean L2 0.266, 0.9 % > thr=0.75 → **PASS** p=1.0).

The model fits with **1.23× concurrency** at 180 000-token max_model_len (221 616-token KV pool) — the GPU is ~91 % full (249/275 GiB), so it's tight on KV but runs single-GPU. The image starts out-of-the-box (no env override needed — unlike A100; Blackwell auto-selects FLASHINFER_TRTLLM correctly).

## Image identity (verified)

```
Config.Image  : ghcr.io/kaitakuai/mlnode-b300-minimax-m2-7:0.2.13-vllm0.20.0-k1
Docker digest : sha256:45e14a9177764cb51501d95c00dbdf8674bd8bf2de03716598110403d06ff1b0
PR#36         : PRESENT (torch.compile on apply_householder, count=1)
Quant         : fp8 weights + fp8 KV cache
vLLM          : 0.20.0 (system python)
```

## Config (launched via `inference/up`, TP=1)

```
--tensor-parallel-size 1          # single B300 — 230 GB model fits in 275 GiB HBM
--max-model-len 180000
--kv-cache-dtype fp8
--gpu-memory-utilization 0.92
--max-num-seqs 128
--logprobs-mode processed_logprobs
--enable-auto-tool-choice --tool-call-parser minimax_m2 --reasoning-parser minimax_m2_append_think
--trust-remote-code
# No MoE env override — B300 (sm_100) auto-selects FLASHINFER_TRTLLM (correct). Image starts out-of-the-box.
```

## Validation

### Cold start (TP=1, native CUDA 13 — driver 580)

| Step | Result |
|---|---:|
| Loading weights | 29 s (fast — B300 HBM) |
| MoE backend | FLASHINFER_TRTLLM Fp8 (auto, Blackwell) + FLASHINFER attention |
| Kernel JIT (`ptxas`, FlashInfer/DeepGEMM cold cache) | ~4 min on one kernel (CPU-bound; faster than B200's ~8 min; **wait it out, not a hang**) |
| **GPU KV cache size** | **221 616 tokens** (1.23× concurrency @ 180 000-token req — tightest of all; model nearly fills the 275 GiB) |
| GPU memory after warm-up | 249 / 275 GiB (91 %) |
| Cold start total | ~567 s (ptxas-dominated, one-time; warm restart faster) |

### Phase-3 PoC throughput sweep (`run_pow_generation.py`)

| batch | nonces (30 s) | nonces/min |
|---:|---:|---:|
|  8 | 696 | 1392 |
| 16 | 864 | 1728 |
| **32** | **896** | **1792 ★** |
| 64 | 0 | hung |

**Best: 1792 nonces/min @ batch=32 — on a single GPU.** `batch=64` hangs the PoC engine (documented MiniMax behavior).

Cross-hardware (best batch, MiniMax-M2.7 FP8):

| Config | GPUs | nonces/min | **nonces/min/GPU** |
|---|---:|---:|---:|
| **1×B300 SXM6 (this run)** | 1 | 1792 | **1792 ★** |
| 2×B200 SXM | 2 | 2624 | 1312 |
| 2×H200 | 2 | 1728 | 864 |
| 4×H100 SXM5 | 4 | 2304 | 576 |
| 4×A100 SXM4 | 4 | 896 | 224 |

**B300 has the highest per-GPU PoC throughput by a wide margin** — and uniquely runs the model on a single GPU.

### Inference performance — `compressa-perf` (gonka §3.2.3)

60 requests, 20 concurrent runners, ~542 input tok, 300 output tok.

| Metric | Value | vs others |
|---|---:|---|
| **TTFT (mean)** | **1.28 s** | fastest (B200 2.12, H200 2.07, A100 2.35) |
| LATENCY (mean) | 7.92 s | |
| TPOT (mean) | **26.6 ms/tok** | ≈ B200 (24.9), better than H200/A100 |
| THROUGHPUT (total) | 2104 tok/s | |
| **THROUGHPUT_OUTPUT_TOKENS** | **746 tok/s** | ≈ 2×B200 (803) on ONE GPU |
| RPS | 2.51 | |
| FAILED_REQUESTS | **0 / 60** | |

A single B300 delivers near-2×B200 inference throughput and the lowest TTFT of any config.

### Canonical Gonka chain validation (mean L2 + binomtest)

| Pair | mean L2 | n_mismatch @ thr=0.75 | strict (0.02) | MiniMax chain (0.75/0.10) |
|---|---:|---:|---|---|
| **1×B300 (TP=1, TRTLLM) ↔ 2×B200 published baseline** | **0.2655** | 9/1000 (0.9 %) | FRAUD | **PASS** p=1.0 |

Note: B300 (TP=1) ↔ B200 (TP=2) gives L2 0.27 — both Blackwell + TRTLLM MoE, but **different TP topology changes the reduction order** → nonces diverge (vs the bit-identical 0.0 we saw for B200-TP=2 vs B200-TP=2 baseline). Still well within the MiniMax chain gate; B300 nodes cross-validate with the B200 fleet.

## Files

- [`artifacts/nonces_1000.json`](artifacts/nonces_1000.json) — 1024 B300 PoC nonces (TP=1, TRTLLM, PR#36), batch=32
- [`artifacts/inference_5langs.json`](artifacts/inference_5langs.json) — 5-language probe
- [`artifacts/bench.log`](artifacts/bench.log) — Phase-3 PoC sweep
- [`artifacts/cp.log`](artifacts/cp.log) / [`artifacts/compressa_perf_metrics.txt`](artifacts/compressa_perf_metrics.txt) — compressa-perf
- [`artifacts/l2_vs_b200.json`](artifacts/l2_vs_b200.json) — canonical L2 vs B200 baseline

## Findings

1. **MiniMax-M2.7 runs on a single B300** (230 GB model in 275 GiB HBM, TP=1) — no multi-GPU needed. The chain `VRam=320 GB` nominal requirement is conservative; 1×B300 (275 GB) works at 1.23× concurrency.
2. **Best per-GPU PoC throughput of all hardware: 1792 nonces/min/GPU** (vs B200 1312, H100 576). PoC weight = 1792 × 0.3024 = 542/GPU — highest economic value per GPU for MiniMax.
3. **Best inference latency: TTFT 1.28 s, 746 output tok/s on one GPU** ≈ a 2×B200 instance.
4. **Image starts out-of-the-box** (unlike A100 — no `VLLM_USE_FLASHINFER_MOE_FP8` override needed; Blackwell auto-picks TRTLLM). PR#36 baked in.
5. **KV is the constraint at TP=1** (1.23× concurrency @ 180k) — for high-concurrency long-context inference, 2×B300 would give much more KV headroom.
6. ptxas cold-JIT (~4 min) on first start — wait it out.

## Related

- 2×B200 (Ohio, L2=0 vs ref): [`../minimax_m27_2xb200_ohio_pr36/README.md`](../minimax_m27_2xb200_ohio_pr36/README.md)
- 4×H100 SXM PR#36 A/B: [`../minimax_m27_4xh100_sxm_pr36/README.md`](../minimax_m27_4xh100_sxm_pr36/README.md)
- 2×H200: [`../minimax_m27_2xh200_h100img_pr36/README.md`](../minimax_m27_2xh200_h100img_pr36/README.md)
- 4×A100 SXM4 MARLIN: [`../minimax_m27_4xa100_sxm4_marlin/README.md`](../minimax_m27_4xa100_sxm4_marlin/README.md)
- compressa-perf: https://github.com/product-science/compressa-perf
