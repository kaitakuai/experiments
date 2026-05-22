# MiniMax-M2.7 — 2×H200 — h100 image (PR#36 baked) fit + perf validation

**Date:** 2026-05-21
**Model:** `MiniMaxAI/MiniMax-M2.7` (FP8 weights, FP8 KV cache, MoE)
**Hardware:** 2× NVIDIA H200 (Vast.ai inst 37293975, Sweden, 140.4 GiB HBM/GPU, NV18 NVLink pair, 700 W, driver 580.126.09, sm_90)
**Image:** `ghcr.io/kaitakuai/mlnode-h100-minimax-m2-7:0.2.13-vllm0.20.0-k1`
**Digest:** `sha256:9643206e1b421efe3f326352fec806c72ecd2dae9c083ff8d75bff13e0a830a2` (updated build, PR#36 now baked in)

## Summary

Two questions: **(1) does MiniMax-M2.7 fit on 2×H200**, and **(2) what inference speed
does it sustain** (measured with gonka's `compressa-perf`). There is no dedicated
`mlnode-h200-minimax-m2-7` image published (ghcr returns 404), so we run the
**h100 image** on H200 hardware (both Hopper sm_90 — fully compatible), TP=2,
TRITON Fp8 MoE, FLASHINFER attention.

The model **fits** on 2×H200 (107.3 GiB weights/GPU, leaving a tight 12.2 GiB KV
pool → 241 760 tokens, **1.34× concurrency** at the 180 000-token max_model_len —
much tighter than B200's 4.69× or 4×H100's 2.29×, but it runs). PoC throughput is
**1728 nonces/min @ batch=32**, matching the `h200-minimax-m2-7` profile reference
exactly. Inference (compressa-perf, conc=20, ~542-tok prompts, 300-tok outputs):
**TTFT 2.07 s, TPOT 28.5 ms/tok, 700 output tok/s, RPS 2.33, 0 failures**.

This image is the **updated build** — `apply_householder` now carries the
`@torch.compile` decorator (**PR#36 is baked in**, confirmed `torch.compile`
count=1 in `vllm/poc/gpu_random.py`), unlike the digest `2fffb79e…` build we
tested earlier on 4×H100. H200 nonces cross-validate against both the 4×H100
(PR#36) and the 2×B200 (no-patch) baselines under the deployed MiniMax chain gate.

## Image identity (verified)

```
Config.Image  : ghcr.io/kaitakuai/mlnode-h100-minimax-m2-7:0.2.13-vllm0.20.0-k1
Docker digest : sha256:9643206e1b421efe3f326352fec806c72ecd2dae9c083ff8d75bff13e0a830a2
PR#36         : PRESENT (torch.compile on apply_householder, count=1) — baked into this build
Quant         : fp8 (weights + KV cache)
vLLM          : 0.20.0 (system python /usr/local/lib/python3.12/dist-packages/vllm)
```

There is **no** `mlnode-h200-minimax-m2-7` package on ghcr (anonymous → DENIED,
authenticated as `clanster` → **404**). The `h100` image is used on H200 because
H100 and H200 are the same Hopper sm_90 arch (H200 differs only by HBM size/BW).

## Config (launched via `inference/up`, H200 = TP=2 per `h200-minimax-m2-7` profile)

```
--tensor-parallel-size 2          # 2×H200 = 280 GB HBM; profile mandates TP=2
--max-model-len 180000
--kv-cache-dtype fp8
--gpu-memory-utilization 0.92
--max-num-seqs 128
--logprobs-mode processed_logprobs
--enable-auto-tool-choice
--tool-call-parser minimax_m2
--reasoning-parser minimax_m2_append_think
--trust-remote-code
ENV VLLM_USE_FLASHINFER_MOE_FP8=0   # force TRITON Fp8 MoE (Hopper sm_90)
ENV VLLM_MOE_USE_DEEP_GEMM=0
```

## Validation

### Fit / cold start (TP=2, native CUDA 13 — driver 580)

| Step | Result |
|---|---:|
| Loading weights (TP=2) | ~50 s |
| **Model load to VRAM** | **107.31 GiB / GPU** (×2 ≈ 214 GiB) — fits in 140 GiB H200 |
| torch.compile | 39.1 s |
| DeepGEMM warmup | ~3-4 min (Hopper JIT, cold-start bottleneck) |
| Available KV cache / GPU (gmu=0.92) | **12.2 GiB** (very tight — only ~22 GiB headroom after weights) |
| **GPU KV cache size** | **241 760 tokens** |
| **Maximum concurrency @ 180 000-token req** | **1.34×** (B200: 4.69×, 4×H100: 2.29×) |
| Cold start total (`/inference/up` → running) | ~269 s |

**Verdict on fit:** MiniMax-M2.7 FP8 runs on 2×H200, but the KV pool is the
binding constraint — 1.34× concurrency means barely more than one full-length
(180k) request in flight. For production with long contexts, 2×H200 is the floor;
more headroom (lower max_model_len, or 4×H200) would be needed for real
concurrency.

### Phase-3 PoC throughput sweep (`run_pow_generation.py`)

30 s measurement + 5 s warmup, single 2×H200 instance, PR#36 baked.

| batch | nonces (30 s) | nonces/min |
|---:|---:|---:|
|  8 | 776 | 1552 |
| 16 | 848 | 1696 |
| **32** | **864** | **1728 ★** |
| 64 | 0 | hung |

**Best: 1728 nonces/min @ batch=32 — matches the `h200-minimax-m2-7` profile
reference exactly** (864 nonces/min/GPU). `batch=64` hangs the PoC engine (KV too
tight + the documented MiniMax batch=64 hazard); recovery = vLLM `down`/`up`.

Cross-hardware context (best batch=32, MiniMax-M2.7 FP8):

| Config | nonces/min | nonces/min/GPU |
|---|---:|---:|
| 2×B200 reference (vllm:pocv2) | 2624 | 1312 |
| 2×B200 foundry image | 2496 | 1248 |
| 4×H100 SXM5 + PR#36 | 2304 | 576 |
| 4×H100 SXM5 no patch | 2176 | 544 |
| **2×H200 + PR#36 (this run)** | **1728** | **864** |

Note: 2×H200 reaches 864 nonces/min/GPU — **higher per-GPU than 4×H100** (576),
because TP=2 has less all-reduce overhead than TP=4 and the H200 pair is
NVLink-connected. Aggregate is lower simply because it's 2 GPUs vs 4.

### Inference performance — `compressa-perf` (gonka §3.2.3)

Tool: `compressa-perf` (`pip install git+https://github.com/product-science/compressa-perf.git`).
Command:

```
compressa-perf measure-from-yaml --no-sign \
  --account_address 0x0000000000000000000000000000000000000000 \
  --node_url http://127.0.0.1:8081 \
  --model_name /root/models/MiniMax-M2.7 \
  cp_config.yml
```

Workload: 60 requests, 20 concurrent runners, generated prompts (~542 input tok),
300 output tok each.

| Metric | Value |
|---|---:|
| TTFT (mean) | **2.07 s** (p95 6.07 s) |
| LATENCY (mean) | 8.56 s (p95 12.36 s) |
| TPOT (mean) | **28.5 ms/tok** |
| THROUGHPUT (total) | 1964.6 tok/s |
| THROUGHPUT_INPUT_TOKENS | 1264.4 tok/s |
| **THROUGHPUT_OUTPUT_TOKENS** | **700.2 tok/s** |
| RPS | 2.33 |
| FAILED_REQUESTS | **0 / 60** |

Coherent output, no errors. (Note: the PoC `--max-num-seqs 128` runtime ceiling
plus the tight KV pool mean inference concurrency is effectively KV-bound here, as
in the PoC sweep.)

### Canonical Gonka chain validation (mean L2 + binomtest)

Run via [`gonka-l2-validate`](../../../.claude/skills/gonka-l2-validate/SKILL.md).
Thresholds = the deployed v0.2.13 MiniMax PoC config (`DistThr=0.75 / PMis=0.10`).

| Pair | mean L2 | n_mismatch @ thr=0.75 | MiniMax chain verdict |
|---|---:|---:|---|
| **2×H200 (PR#36) ↔ 4×H100 SXM (PR#36)** | 0.2535 | 6/1000 (0.60 %) | **PASS** p=1.0 |
| **2×H200 (PR#36) ↔ 2×B200 baseline (no patch, 131k)** | 0.2855 | 11/1000 (1.10 %) | **PASS** p=1.0 |

Both pairs **PASS** the deployed MiniMax chain gate by a wide margin (mismatch
≪ 10 %). Under strict vLLM self-validation (0.02) they are FRAUD (expected for
cross-GPU / cross-arch / cross-patch). The H200 nonces are honest and
cross-validatable on-chain with the existing H100 and B200 nodes.

## Files

- [`artifacts/nonces_1000.json`](artifacts/nonces_1000.json) — 1024 H200 PoC nonces (PR#36 build), batch=32
- [`artifacts/inference_5langs.json`](artifacts/inference_5langs.json) — 5-language probe (processed_logprobs)
- [`artifacts/bench.log`](artifacts/bench.log) — Phase-3 PoC sweep
- [`artifacts/cp.log`](artifacts/cp.log) / [`artifacts/compressa_perf_metrics.txt`](artifacts/compressa_perf_metrics.txt) — compressa-perf run + full metrics table
- [`artifacts/l2_h200_vs_h100.json`](artifacts/l2_h200_vs_h100.json), [`artifacts/l2_h200_vs_b200.json`](artifacts/l2_h200_vs_b200.json) — canonical L2

## Findings

1. **MiniMax-M2.7 fits on 2×H200 but KV is the bottleneck** (1.34× concurrency @ 180k). Fine for PoC; for long-context inference at concurrency, needs 4×H200 or a lower max_model_len.
2. **PoC 1728 nonces/min @ batch=32** — matches the `h200-minimax-m2-7` profile reference exactly. Per-GPU (864) beats 4×H100 (576) due to lower TP overhead.
3. **Inference: TTFT 2.07 s, 700 output tok/s, TPOT 28.5 ms** at conc=20 — measured with gonka's `compressa-perf`.
4. **The updated h100 image (digest 9643206e) has PR#36 baked in** (confirmed). It still lacks `openssh` — works on Vast only on hosts that provide `/usr/bin/ssh` (this Sweden host did; the NL host 20897 did not).
5. **No dedicated `mlnode-h200-minimax-m2-7` image exists** on ghcr (404). The h100 image is the correct artifact for H200 (same Hopper arch).

## Related

- 4×H100 SXM PR#36 A/B: [`../minimax_m27_4xh100_sxm_pr36/README.md`](../minimax_m27_4xh100_sxm_pr36/README.md)
- 2×B200 foundry image: [`../minimax_m27_2xb200_b200-minimax-m2-7/README.md`](../minimax_m27_2xb200_b200-minimax-m2-7/README.md)
- compressa-perf: https://github.com/product-science/compressa-perf
- Canonical L2 skill: [`../../../.claude/skills/gonka-l2-validate/SKILL.md`](../../../.claude/skills/gonka-l2-validate/SKILL.md)
