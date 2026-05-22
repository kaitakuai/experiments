# MiniMax-M2.7 — 4×A100 SXM4 — a100 image (MARLIN MoE) validation + inference

**Date:** 2026-05-22
**Model:** `MiniMaxAI/MiniMax-M2.7` (FP8 weights, FP8 KV cache, MoE)
**Hardware:** 4× NVIDIA A100-SXM4-80GB, NV12 full NVLink mesh, 500 W, sm_80. Tested on **two hosts**:
- **host1** = Vast inst 37359446, machine 49463, Georgia US, driver 580.105.08 — **throttled** (see below)
- **host2** = Vast inst 37373485, machine 44000, Massachusetts US, driver 570.195.03 (forward-compat) — **canonical**
**Image:** `ghcr.io/kaitakuai/mlnode-a100-minimax-m2-7:0.2.13-vllm0.20.0-k1`
**Digest:** `sha256:6f1958581cb108979d2d237147e6032c7255058bfc01666ed3596610e29caf06` (PR#36 baked in)

## Summary

Validation of the `mlnode-a100-minimax-m2-7` image on 4×A100 SXM4 80GB, TP=4, **MARLIN Fp8 MoE** (A100 sm_80 has no native FP8 — MARLIN dequantizes) + FLASHINFER attention.

Canonical results (host2, the un-throttled host):

1. **PoC throughput = 896 nonces/min @ batch=32 — exactly matches the profile reference.** (batch=16 = 864.) ~2.9× slower per-instance than 2×B200 (2624), as expected for Ampere.
2. **Inference (compressa-perf): TTFT 2.35 s, TPOT 37.4 ms/tok, 535 output tok/s, RPS 1.78, 0 failures** — slowest tier (B200 803, H200 700), but the un-throttled host.
3. **Nonces cross-validate with the B200 baseline under the MiniMax chain gate** (mean L2 0.329, 2.4 % > thr=0.75 → **PASS** p=1.0). Cross-arch divergence higher than H100/H200 (MARLIN-on-Ampere vs TRTLLM-on-Blackwell) but inside the deployed tolerance.

**Host-throttling finding (two-host test).** The first A100 host (Georgia, machine 49463) gave only **768 nonces/min @ batch=32** (and slower inference: TTFT 3.04 s, 478 tok/s) — ~14 % below profile. Suspecting a bad host, we re-ran on a **second** A100 SXM4 host (Massachusetts, machine 44000): **896 nonces/min** (= profile) and faster inference across the board. **The ~14 % shortfall was the host, not the image/config** — identical image, args, MARLIN backend, TP=4 on both. A100 PoC throughput is host-sensitive; pick hosts carefully (or benchmark before committing). Per-GPU L2 between the two hosts (same arch, same MARLIN) = 0.175 — cross-host numeric variance, still PASS the chain gate.

**Critical image bug found:** the A100 image ships with `VLLM_USE_FLASHINFER_MOE_FP8=1` baked in (a Hopper/Blackwell value). On A100 (sm_80) this **crashes the engine** — `NotImplementedError: Found VLLM_USE_FLASHINFER_MOE_FP8=1, but no FlashInfer FP8 MoE backend supports the configuration` — and the mlnode watcher then shuts down the API after 3 unhealthy counts. **The image does not start out-of-the-box on A100.** We had to override `VLLM_USE_FLASHINFER_MOE_FP8=0` to get the profile's MARLIN backend. This must be fixed in mlnode-foundry (the a100 profile env should set `VLLM_USE_FLASHINFER_MOE_FP8=0`).

## Image identity (verified)

```
Config.Image  : ghcr.io/kaitakuai/mlnode-a100-minimax-m2-7:0.2.13-vllm0.20.0-k1
Docker digest : sha256:6f1958581cb108979d2d237147e6032c7255058bfc01666ed3596610e29caf06
PR#36         : PRESENT (torch.compile on apply_householder, count=1)
Quant         : fp8 weights (MARLIN-dequantized on Ampere) + fp8 KV cache
vLLM          : 0.20.0 (system python)
```

## Config (launched via `inference/up`, a100 profile = TP=4, MARLIN MoE)

```
--tensor-parallel-size 4
--max-model-len 180000
--kv-cache-dtype fp8
--gpu-memory-utilization 0.92
--max-num-seqs 128
--logprobs-mode processed_logprobs
--enable-auto-tool-choice
--tool-call-parser minimax_m2
--reasoning-parser minimax_m2_append_think
--trust-remote-code
ENV VLLM_USE_FLASHINFER_MOE_FP8=0   # REQUIRED override — image bakes =1 which crashes on sm_80; =0 → MARLIN (profile intent)
ENV VLLM_USE_FLASHINFER_MOE_FP4=0
ENV VLLM_MOE_USE_DEEP_GEMM=0
```

All runtime args match the `a100-minimax-m2-7` profile (TP=4, MARLIN, FLASHINFER attention). MARLIN is the only viable MoE backend on A100 for this fp8-block-quantized model — vLLM's fp8 oracle tries `['AITER','TRITON','MARLIN',…]`, TRITON fails the support check for this config, MARLIN is selected. (No TRITON A/B possible.)

## Validation

### Cold start (TP=4, MARLIN, native CUDA 13 — driver 580)

| Step | Result |
|---|---:|
| Loading weights (TP=4) | ~38 s |
| Model load to VRAM | 54.58 GiB / GPU (×4 ≈ 218 GiB) |
| MoE backend | **MARLIN** Fp8 (Ampere) + FLASHINFER attention |
| Available KV cache / GPU | 12.72 GiB |
| **GPU KV cache size** | **430 112 tokens** (2.38× concurrency @ 180 000-token req — ~same as 4×H100) |
| CUDA graph capture | 8 s, 1.05 GiB |
| Cold start total | **~168 s** (no ptxas pathology — MARLIN, unlike B200's trtllm kernel) |

### Phase-3 PoC throughput sweep (`run_pow_generation.py`, MARLIN)

| batch | host2 (canonical) | host1 (throttled) |
|---:|---:|---:|
|  8 | 816 | 704 |
| 16 | 864 | 736 |
| **32** | **896 ★** | 768 |
| 64 | hung | hung |

**Best: 896 nonces/min @ batch=32 (host2) = profile reference exactly** (224/min/GPU). The first host (host1, Georgia) gave only 768 — a ~14 % throttle, not an image/config issue (identical setup on both). `batch=64` hangs the PoC engine on both (documented MiniMax behavior).

Cross-hardware (best batch, MiniMax-M2.7 FP8):

| Config | nonces/min | nonces/min/GPU |
|---|---:|---:|
| 2×B200 SXM | 2624 | 1312 |
| 4×H100 SXM5 | 2304 | 576 |
| 2×H200 | 1728 | 864 |
| **4×A100 SXM4 (host2, = profile)** | **896** | **224** |
| 4×A100 SXM4 (host1, throttled) | 768 | 192 |

### Inference performance — `compressa-perf` (gonka §3.2.3)

60 requests, 20 concurrent runners, ~542 input tok, 300 output tok.

| Metric | host2 (canonical) | host1 (throttled) |
|---|---:|---:|
| TTFT (mean) | **2.35 s** | 3.04 s |
| TPOT (mean) | **37.4 ms/tok** | 41.8 ms/tok |
| THROUGHPUT_OUTPUT_TOKENS | **534.9 tok/s** | 478.2 tok/s |
| RPS | 1.78 | 1.60 |
| FAILED_REQUESTS | 0 / 60 | 0 / 60 |

Slowest tier of all hardware (host2 TPOT 37.4 ms vs B200 24.9, H200 28.5; output 535 vs 803/700 tok/s). Expected — Ampere + MARLIN dequant overhead. host1 was ~12 % slower again (same throttle as PoC).

### Canonical Gonka chain validation (mean L2 + binomtest)

| Pair | mean L2 | max | n_mismatch @ thr=0.75 | strict (0.02) | MiniMax chain (0.75/0.10) |
|---|---:|---:|---:|---|---|
| **4×A100 (MARLIN, PR#36) ↔ 2×B200 published baseline** | **0.3295** | 1.109 | 23/1000 (2.3 %) | FRAUD | **PASS** p=1.0 |

A100 MARLIN nonces diverge from B200 TRTLLM more than H100/H200 do (cross-arch + cross-MoE-kernel: MARLIN dequant vs TRTLLM FP8), L2 0.33 vs ~0.29 for H100. Still **PASS** the deployed MiniMax chain gate (2.3 % ≪ 10 % allowance) → A100 nodes cross-validate with the B200/Hopper fleet.

## Files

- [`artifacts/nonces_1000.json`](artifacts/nonces_1000.json) — 1056 A100 PoC nonces (MARLIN, PR#36), batch=32
- [`artifacts/inference_5langs.json`](artifacts/inference_5langs.json) — 5-language probe
- [`artifacts/bench.log`](artifacts/bench.log) — Phase-3 PoC sweep
- [`artifacts/cp.log`](artifacts/cp.log) / [`artifacts/compressa_perf_metrics.txt`](artifacts/compressa_perf_metrics.txt) — compressa-perf run + metrics
- [`artifacts/l2_vs_b200.json`](artifacts/l2_vs_b200.json) — canonical L2 vs B200 baseline

## Findings

1. **A100 image does NOT start out-of-the-box** — baked `VLLM_USE_FLASHINFER_MOE_FP8=1` crashes on sm_80 (`NotImplementedError`), watcher then kills the API. Must override to `0`. **Fix the a100 profile env in mlnode-foundry.**
2. **PoC 896 nonces/min @ batch=32 on a good host = profile reference exactly.** A first host gave 768 (~14 % throttle) — the gap was the **host**, not the image/config (confirmed by re-test: identical image/args/MARLIN/TP=4, 768→896). A100 throughput is host-sensitive — benchmark before committing a host.
3. **Inference (host2): TTFT 2.35 s, 535 output tok/s, TPOT 37.4 ms** — slowest tier, as expected for Ampere.
4. **Nonces cross-validate with B200 under the MiniMax chain gate** (L2 0.33, 2.4 % mismatch, PASS) despite cross-arch + MARLIN. Cross-host same-arch L2 = 0.175 (also PASS).
5. Per-GPU PoC weight (×0.3024 scale): 224 nonces/min/GPU (good host) — still lowest of all MiniMax hardware → A100 least economical for MiniMax PoC.

## Related

- 2×B200 (Ohio, L2=0 vs ref): [`../minimax_m27_2xb200_ohio_pr36/README.md`](../minimax_m27_2xb200_ohio_pr36/README.md)
- 4×H100 SXM PR#36 A/B: [`../minimax_m27_4xh100_sxm_pr36/README.md`](../minimax_m27_4xh100_sxm_pr36/README.md)
- 2×H200 fit + inference: [`../minimax_m27_2xh200_h100img_pr36/README.md`](../minimax_m27_2xh200_h100img_pr36/README.md)
- compressa-perf: https://github.com/product-science/compressa-perf
