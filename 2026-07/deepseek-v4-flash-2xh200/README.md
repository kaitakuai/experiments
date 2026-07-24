# PoC + Inference Benchmark: DeepSeek-V4-Flash on 2×H200 (eager vs CUDA-graph)

**Date:** 2026-07-24
**Model:** `deepseek-ai/DeepSeek-V4-Flash` (FP8, sparse-MLA, hash-routed MoE)
**Hardware:** 2× NVIDIA H200 NVL, TP=2
**Image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash:0.2.13-vllm0.25.1-overlay-k4`
**Digest:** `sha256:2af898fa516424ea2884b77e40ae480ce61b19a7d0112f8c9b8cc866c8bcb28a`
**PoC:** gonka-poc plugin (`--worker-extension-cls gonka_poc.worker.PoCWorkerExtension`)
**vLLM:** 0.25.1

## Summary

V4 fits on **two** H200 cards (149 GiB of weights against 2 × 140 GiB), so this is the
same benchmark as the 4×H100 run one topology step down: TP=2 instead of TP=4. Two runs
of the same image differing only in the compilation-related arguments, each measured for
PoC throughput, nonce collection and inference.

- **PoC: 1024 nonces/min** at batch 16 (eager), 992 at batch 16 (compiled) — the two runs
  differ by ~3 %, which is run-to-run noise; the PoC forward is `skip_compiled` and does
  not see either setting. Unlike the H100 run, the curve **peaks at batch 16 and drops at
  32**.
- **Inference: 4.4–6.7× faster with CUDA graphs** (i.e. without `--enforce-eager`), and
  ~1.9× at maximum concurrency.
- **Per-GPU efficiency beats the 4×H100 box**: 512 nonces/min per H200 at TP=2 versus 384
  per H100 at TP=4 — less sharding, less loss. A single B300 at TP=1 still delivers more
  than either (1472 nonces/min on one card).
- **KV headroom is the real advantage**: 1,225,837 tokens against 656,967 on 4×H100, i.e.
  3× concurrency at a 400k context versus 1.64×.

Nonce cross-validation is reported as **distances only** — DeepSeek-V4 does not have
calibrated validation thresholds yet, so no pass/fail is drawn (see below).

## Hardware

| Parameter | Value |
|-----------|-------|
| GPU | 2× NVIDIA H200 NVL, 143,771 MiB each |
| Form factor | NVL (PCIe) — verified by `power.max_limit = 600 W` |
| Interconnect | NV6 NVLink bridge (`nvidia-smi topo -m`) |
| NVIDIA Driver | 580.159.03 |
| Host | Vast.ai, Mississippi US |

Note for comparison with the 4×H100 report: that box was **SXM5 (700 W) with NV18 full
mesh**; this one is NVL with a 6-link bridge.

## Software

| Component | Version |
|-----------|---------|
| vLLM | 0.25.1 |
| gonka-poc plugin | 0.1.0a0 |
| torch | 2.11.0+cu130 |
| Python | 3.12.13 |
| OS | Ubuntu 22.04.5 LTS |

## Configuration

Only **`runner.py` parameters** were changed relative to the image defaults:

| Flag | Image default | This run | Why |
|------|---------------|----------|-----|
| `--tensor-parallel-size` | 1 | **2** | 149 GiB of weights do not fit on one 140 GB H200 |
| `--max-model-len` | 200000 | **400000** | benchmark at a realistic serving context |
| `--max-num-batched-tokens` | 16384 | **32768** | sweep batch 32 × seq_len 1024 |

Unchanged: `--gpu-memory-utilization 0.90`, `--kv-cache-dtype fp8` (mandatory — FlashMLA
asserts without it), `--logprobs-mode processed_logprobs`, `--trust-remote-code`,
`--worker-extension-cls gonka_poc.worker.PoCWorkerExtension`.

Per-mode arguments (passed through `additional_args`, not part of the forced block):

| Mode | Extra args | Observed at startup |
|------|-----------|---------------------|
| **eager** | `--enforce-eager` | no CUDA graphs |
| **compiled** | `--compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'` | **breakable CUDA graph enabled**, capture 61 s / 2.41 GiB |

### sm_90 prerequisite

H200 is Hopper, and the image is built for the B300 (sm_100) profile without an
unversioned `libnvrtc.so`. The V4 sparse-MLA warmup compiles a FlashInfer kernel that
links `-lnvrtc`; without the symlink every worker dies during startup. Applied before both
runs (see `scripts/setup_box.sh`):

```bash
ln -sf /usr/local/cuda/lib64/libnvrtc.so.13 /usr/local/cuda/lib64/libnvrtc.so
ln -sf /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so.13 \
       /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so
ldconfig
```

## Startup profile

| Phase | eager | compiled |
|-------|------:|---------:|
| Cold start (model load → serving) | 255 s | 225 s |
| CUDA graph capture | — | 61 s, 2.41 GiB |
| Available KV cache per GPU | 39.69 GiB | 36.88 GiB |
| GPU KV cache size | 1,225,837 tokens | 1,139,114 tokens |
| Max concurrency @ 400k context | 3.0× | 2.8× |

## Results — PoC throughput sweep

`run_pow_generation.py --phase 3`, 5 s warmup + 30 s steady state, seq_len 1024:

| Batch Size | eager: nonces (30 s) | eager: nonces/min | compiled: nonces (30 s) | compiled: nonces/min |
|-----------:|---------------------:|------------------:|------------------------:|---------------------:|
| 8 | 480 | 960 | 488 | 976 |
| **16** ★ | **512** | **1024** | **496** | **992** |
| 32 | 480 | 960 | 448 | 896 |

The two modes land within ~3 % of each other in both directions (batch 8 favours
compiled, 16 and 32 favour eager) — run-to-run noise, not a compilation effect. Note the
shape: throughput **peaks at batch 16**, whereas on 4×H100 it rose monotonically to
batch 32.

Nonce collection (batch 32): eager **1056 nonces @ 769/min**, compiled **1024 nonces @
793/min**.

### Topology comparison

| Configuration | total nonces/min | per GPU | KV cache tokens | concurrency @400k |
|---------------|-----------------:|--------:|----------------:|------------------:|
| 1×B300 TP=1 | 1472 | **1472** | 2,604,694 | 6.5× |
| 2×H200 TP=2 (this run) | 1024 | **512** | 1,225,837 | 3.0× |
| 4×H100 TP=4 | 1536 | **384** | 656,967 | 1.64× |

Per-GPU throughput falls as the model is sharded wider: 1472 → 512 → 384. Two H200s carry
2× the KV of four H100s, because at TP=4 the weights are replicated across more cards and
eat the headroom.

## Results — inference (compressa-perf)

Full tables in [`compressa-perf-comparison.md`](compressa-perf-comparison.md). Headline
numbers (eager → compiled, 0 failed requests in every scenario):

| Scenario | TTFT | Latency | TPOT | Output tok/s |
|----------|-----:|--------:|-----:|-------------:|
| s1 long prompt, sequential, short decode | 2.12 → **0.42** s | 6.97 → **1.29** s | 0.078 → **0.012** s | 12.7 → **81.9** |
| s2 short prompt, high concurrency | 5.49 → **0.82** s | 19.27 → **4.34** s | 0.127 → **0.029** s | 221 → **974** |
| s3 very long, sequential, long decode | 4.22 → **0.92** s | 12.53 → **2.09** s | 0.083 → **0.015** s | 12.1 → **67.5** |
| s4 very long, max concurrency | 6.96 → 6.23 s | 28.85 → **15.11** s | 0.227 → **0.125** s | 80.1 → **151.8** |

The CUDA-graph win is 4.4–6.7× here versus 8–15× on 4×H100 — H200 eager is already
faster (fewer shards, less cross-GPU traffic), so there is less overhead left to remove.

## Nonce cross-validation (distances)

`scripts/compare_nonces.py` — canonical `decode_vector` → per-nonce L2. 1000 common
nonces per pair.

**No pass/fail verdict is drawn.** The validation thresholds used elsewhere are
calibrated for other models; DeepSeek-V4's own limits are still to be set, so the numbers
below are reported as measurements only. The mismatch column counts nonces further apart
than 0.4 purely as a descriptive statistic.

| A | B | median L2 | mean L2 | max L2 | nonces >0.4 |
|---|---|----------:|--------:|-------:|------------:|
| eager 2×H200 | compiled 2×H200 (same box) | 0.1840 | 0.1967 | 0.663 | 3.50 % |
| eager 2×H200 | B300 TP1 plugin | 0.1812 | 0.1971 | 0.682 | 2.90 % |
| compiled 2×H200 | B300 TP1 plugin | 0.1898 | 0.1996 | 0.708 | 2.50 % |
| compiled 2×H200 | B300 TP1 in-tree fork | 0.1910 | 0.1996 | 0.712 | 2.30 % |
| eager 2×H200 | eager 4×H100 | 0.1824 | 0.1977 | 1.342 | 3.30 % |
| compiled 2×H200 | compiled 4×H100 | 0.1864 | 0.1989 | 1.336 | 3.30 % |

Every pair sits in the same narrow band — median ≈ 0.18–0.19, 2.3–3.5 % of nonces beyond
0.4 — **including the two runs on the same box with the same hardware** (3.50 %, the
widest of the set). Repeating a run on identical hardware moves the nonces at least as
much as changing the GPU model does. On a multi-GPU topology V4 PoC is not
bit-reproducible run to run, and that spread is what any future threshold has to absorb.

## Files

- `README.md` — this report
- `compressa-perf-comparison.md` — full inference tables, eager vs compiled
- `artifacts/config.json` — hardware, versions, runner.py overrides, per-mode startup
- `artifacts/sweep.json` — PoC sweep and collection numbers
- `artifacts/l2_matrix.json` — L2 distances and mismatch counts for all pairs
- `artifacts/nonces_eager.json`, `artifacts/nonces_compiled.json` — 1000+ nonce vectors each
- `artifacts/h200_2x_v4_poc_{eager,compiled}.log` — raw sweep output
- `artifacts/h200_2x_v4_compressa_{eager,compiled}.log` — raw compressa output
- `artifacts/compressa_all_metrics.txt`, `artifacts/compressa_parsed.json` — metric dumps
- `artifacts/env.txt` — versions, topology, KV sizing as read off the box
- `artifacts/runner_forced_args.txt` — the forced block as it ran
- `scripts/setup_box.sh` — full box preparation (sm_90 fix, deps, runner.py, API)
- `scripts/` — sweep, collection and L2 scripts

## Findings

- **Two H200s are the better shape for V4 than four H100s.** More throughput per GPU
  (512 vs 384 nonces/min), 2× the KV cache, 3× concurrency at 400k context, and the box
  rented for less than half the price.
- **A single B300 still beats both** at 1472 nonces/min on one card — where the weights
  fit on one GPU, that is the configuration to use.
- **Do not pass `--enforce-eager` when serving V4** — 4.4–6.7× on TTFT, latency and
  throughput for the price of 61 s of startup and 2.41 GiB of KV.
- **PoC throughput peaks at batch 16 on this topology** and falls at 32, unlike 4×H100
  where it kept rising.
- **PoC throughput is insensitive to the compilation arguments** — the ±3 % between the
  two runs is noise in both directions.
- **The B300-profile image needs a `libnvrtc.so` symlink on Hopper** (H200 as well as
  H100), otherwise the V4 sparse-MLA warmup kills the engine at startup.
- **V4 nonces are not run-to-run reproducible on multi-GPU topologies**: two runs on the
  same box differ as much as two different GPU models do (3.5 % of nonces beyond 0.4).

## Reproducibility checklist

- [x] Hardware verified, not assumed (600 W ⇒ NVL, NV6 topology)
- [x] Image pinned by digest; PoC plugin identified
- [x] Every changed `runner.py` parameter listed against its image default
- [x] Per-mode arguments and what was observed at startup
- [x] sm_90 prerequisite documented with the exact fix, and scripted
- [x] Sweep method, warmup and measurement windows stated
- [x] Nonce sets, raw logs and metric dumps committed
- [x] L2 computed with the canonical script, which is committed
- [x] No `.claude/` paths, no sibling-repo paths
