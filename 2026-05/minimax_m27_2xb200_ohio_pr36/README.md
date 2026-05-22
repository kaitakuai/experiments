# MiniMax-M2.7 — 2×B200 — b200 image (PR#36 baked) validation + inference

**Date:** 2026-05-21
**Model:** `MiniMaxAI/MiniMax-M2.7` (FP8 weights, FP8 KV cache, MoE)
**Hardware:** 2× NVIDIA B200 SXM (Vast.ai inst 37296947, Ohio US, 183.4 GiB HBM/GPU, NV18 NVLink 53.125 GB/s/link, 1000 W, driver 595.71.05, sm_100)
**Image:** `ghcr.io/kaitakuai/mlnode-b200-minimax-m2-7:0.2.13-vllm0.20.0-k1`
**Digest:** `sha256:9643206e1b421efe3f326352fec806c72ecd2dae9c083ff8d75bff13e0a830a2` (updated build, PR#36 baked in)

## Summary

Validation of the updated `mlnode-b200-minimax-m2-7` image (PR#36 now baked in) on 2×B200 SXM, TP=2, auto-selected FLASHINFER_TRTLLM Fp8 MoE + FLASHINFER attention + DeepGEMM E8M0 (Blackwell).

Three headline results:

1. **PoC throughput = 2624 nonces/min @ batch=32 — bit-for-bit identical to the published 2×B200 reference.**
2. **Nonces are bit-identical to the published baseline** (mean L2 = **0.0000**, 0/1000 mismatch, PASS even under strict vLLM self-validation 0.02). This proves **PR#36 is numerically neutral on B200** (contrast: on 4×H100 the same patch shifted nonces by L2≈0.23) **and** that `max_model_len` 131072 vs 180000 does not affect B200 PoC outputs.
3. **Inference (compressa-perf, gonka §3.2.3): TTFT 2.12 s, TPOT 24.9 ms/tok, 803 output tok/s, RPS 2.68, 0 failures.**

**Operational caveat (cold start):** first `/inference/up` took **~15 min**, of which **~8 min was a single `ptxas` process** compiling one FlashInfer TRTLLM allreduce kernel (`trtllm_comm`) on a cold kernel cache — CPU-bound, GPU idle, looks like a hang but is not. It completes; the kernel cache (`/root/.cache/flashinfer`) is then warm and restarts are ~150 s. **Do not kill it** — wait it out (no workaround flags needed).

**Host note:** the cheaper Alabama 2×B200 (machine 57668, $10.63/hr) was **broken** — `cudaErrorDevicesUnavailable` on a trivial torch matmul (GPU unusable despite nvidia-smi showing idle/Default). Switched to this Ohio box ($18.62/hr), which passed a pre-download GPU sanity test.

## Image identity (verified)

```
Config.Image  : ghcr.io/kaitakuai/mlnode-b200-minimax-m2-7:0.2.13-vllm0.20.0-k1
Docker digest : sha256:9643206e1b421efe3f326352fec806c72ecd2dae9c083ff8d75bff13e0a830a2
PR#36         : PRESENT (torch.compile on apply_householder, count=1)
Quant         : fp8 (weights + KV cache)
vLLM          : 0.20.0 (system python)
```

## Config (launched via `inference/up`, b200 profile = TP=2, auto MoE)

```
--tensor-parallel-size 2
--max-model-len 180000
--kv-cache-dtype fp8
--gpu-memory-utilization 0.92
--max-num-seqs 128
--logprobs-mode processed_logprobs
--enable-auto-tool-choice
--tool-call-parser minimax_m2
--reasoning-parser minimax_m2_append_think
--trust-remote-code
# NO MoE-backend env override — B200 auto-selects FLASHINFER_TRTLLM (correct for Blackwell).
# NO --disable-custom-all-reduce — kept stock trtllm allreduce for faithful comparison (waited out the cold compile).
```

## Validation

### Cold start (TP=2, native CUDA 13 — driver 595)

| Step | Result |
|---|---:|
| Loading weights (TP=2) | ~44 s |
| Model load to VRAM | 107.32 GiB / GPU (×2 ≈ 214 GiB) |
| MoE backend | **FLASHINFER_TRTLLM** (auto, Blackwell) + DeepGEMM E8M0 |
| **Cold kernel JIT (`ptxas` on `trtllm_comm`)** | **~8 min single kernel** (cold `/root/.cache/flashinfer`); CPU-bound, GPU idle |
| torch.compile (graph) | 136 s cold / 6.7 s warm |
| Available KV cache / GPU | 50.0 GiB |
| **GPU KV cache size** | **845 568 tokens** (4.70× concurrency @ 180 000-token req — same as the earlier foundry 2×B200 run) |
| **Cold start total** | **~918 s** (ptxas-dominated, one-time) |
| Warm restart (`down`+`up`, kernel cache hot) | ~150 s |

### Phase-3 PoC throughput sweep (`run_pow_generation.py`)

| batch | nonces (30 s) | nonces/min |
|---:|---:|---:|
|  8 | 1144 | 2288 |
| 16 | 1232 | 2464 |
| **32** | **1312** | **2624 ★** |
| 64 | 0 | hung |

**Best: 2624 nonces/min @ batch=32 — exactly matches the published 2×B200 reference.** `batch=64` hangs the PoC engine (documented MiniMax behavior); recovery = vLLM `down`/`up` (fast now, kernel cache warm).

Cross-hardware (best batch=32, MiniMax-M2.7 FP8):

| Config | nonces/min | nonces/min/GPU |
|---|---:|---:|
| **2×B200 SXM (this run, PR#36)** | **2624** | **1312** |
| 2×B200 published reference (vllm:pocv2) | 2624 | 1312 |
| 4×H100 SXM5 + PR#36 | 2304 | 576 |
| 2×H200 + PR#36 | 1728 | 864 |

### Inference performance — `compressa-perf` (gonka §3.2.3)

Workload: 60 requests, 20 concurrent runners, ~542 input tok, 300 output tok.

| Metric | Value |
|---|---:|
| TTFT (mean) | **2.12 s** (p95 6.17 s) |
| LATENCY (mean) | 7.46 s (p95 11.40 s) |
| TPOT (mean) | **24.9 ms/tok** |
| THROUGHPUT (total) | 2253.5 tok/s |
| THROUGHPUT_INPUT_TOKENS | 1450.3 tok/s |
| **THROUGHPUT_OUTPUT_TOKENS** | **803.2 tok/s** |
| RPS | 2.68 |
| FAILED_REQUESTS | **0 / 60** |

B200 vs H200 inference: TPOT 24.9 vs 28.5 ms, output 803 vs 700 tok/s — B200 ~15 % faster decode, as expected.

### Canonical Gonka chain validation (mean L2 + binomtest)

Run via [`gonka-l2-validate`](../../../.claude/skills/gonka-l2-validate/SKILL.md).

| Pair | mean L2 | max L2 | n_mismatch | strict (0.02) | MiniMax chain (0.75/0.10) |
|---|---:|---:|---:|---|---|
| **2×B200 Ohio (PR#36, 180k) ↔ published 2×B200 baseline (no patch, 131k)** | **0.0000** | 0.0000 | 0/1000 | **PASS** | **PASS** |

**Bit-identical.** This single result establishes three things at once:
- The foundry image reproduces the published reference exactly on B200.
- **PR#36 (`@torch.compile` on `apply_householder`) does not change B200 nonces** — it is numerically neutral on Blackwell (whereas on 4×H100 the identical patch produced L2≈0.23, FRAUD under strict self-validation). The compiled householder reduction matches eager on sm_100.
- **`max_model_len` 131072 vs 180000 has no effect on PoC outputs** on B200 (PoC seq_len is 1024 regardless).

### Inference sanity

5-language probe (sp/en/ch/ar/hi) coherent, processed-logprob sentinels present, reasoning trace via `minimax_m2_append_think`. 0 failed requests in the compressa-perf run.

## Files

- [`artifacts/nonces_1000.json`](artifacts/nonces_1000.json) — 1088 B200 PoC nonces (PR#36 build), batch=32
- [`artifacts/inference_5langs.json`](artifacts/inference_5langs.json) — 5-language probe
- [`artifacts/bench.log`](artifacts/bench.log) — Phase-3 PoC sweep
- [`artifacts/cp.log`](artifacts/cp.log) / [`artifacts/compressa_perf_metrics.txt`](artifacts/compressa_perf_metrics.txt) — compressa-perf run + metrics
- [`artifacts/l2_vs_baseline.json`](artifacts/l2_vs_baseline.json) — canonical L2 vs published baseline (0.0000)

## Findings

1. **B200 foundry image (PR#36) is production-faithful: 2624 nonces/min and bit-identical nonces vs the published reference.**
2. **PR#36 is numerically neutral on B200** (L2=0), but **changes nonces on H100** (L2≈0.23). The patch is safe on both under the MiniMax chain gate (0.75/0.10), but is arch-dependent in its numeric effect — worth noting for cross-arch fleets.
3. **Cold-start ptxas pathology**: one FlashInfer TRTLLM kernel takes ~8 min to compile cold on B200/CUDA 13. Pre-warm the kernel cache in the image, or operators should expect ~15 min first start (then ~150 s). Do not mistake it for a hang (check `ps` for cicc/ptxas at 100 % CPU).
4. **Vast host lottery**: a cheaper Alabama B200 host was broken (`cudaErrorDevicesUnavailable`). Always run a torch-matmul GPU sanity check before downloading the 230 GB model.
5. Image still lacks `openssh` (works on Vast hosts that mount `/usr/bin/ssh`; this Ohio host did).

## Related

- 2×B200 foundry image (earlier, pre-PR#36 digest): [`../minimax_m27_2xb200_b200-minimax-m2-7/README.md`](../minimax_m27_2xb200_b200-minimax-m2-7/README.md)
- 4×H100 SXM PR#36 A/B: [`../minimax_m27_4xh100_sxm_pr36/README.md`](../minimax_m27_4xh100_sxm_pr36/README.md)
- 2×H200 fit + inference: [`../minimax_m27_2xh200_h100img_pr36/README.md`](../minimax_m27_2xh200_h100img_pr36/README.md)
- Published 2×B200 baseline: [`../minimax-m27-fp8-2xb200/README.md`](../minimax-m27-fp8-2xb200/README.md)
- compressa-perf: https://github.com/product-science/compressa-perf
