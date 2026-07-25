# PoC + Inference Benchmark: DeepSeek-V4-Flash on 2×H200 SXM (eager vs CUDA-graph)

**Date:** 2026-07-24
**Model:** `deepseek-ai/DeepSeek-V4-Flash` (FP8, sparse-MLA, hash-routed MoE)
**Hardware:** 2× NVIDIA H200 SXM, TP=2
**Image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash:0.2.13-vllm0.25.1-overlay-k4`
**Digest:** `sha256:2af898fa516424ea2884b77e40ae480ce61b19a7d0112f8c9b8cc866c8bcb28a`
**PoC:** gonka-poc plugin (`--worker-extension-cls gonka_poc.worker.PoCWorkerExtension`)
**vLLM:** 0.25.1

## Summary

V4 fits on **two** H200 cards (149 GiB of weights against 2 × 140 GiB), so this is the same
benchmark as the 4×H100 run one topology step down: TP=2 instead of TP=4. Two runs of the
same image differing only in the compilation-related arguments.

- **PoC: 1216 nonces/min** at batch 32 — **the same in both modes within measurement
  resolution (< 5 %)**, batch for batch. The
  PoC forward runs `skip_compiled`, and on Hopper neither CUDA graphs nor the compilation
  arguments change it.
- **Inference: 1.2–18.8× faster with CUDA graphs** (i.e. without `--enforce-eager`),
  with the biggest wins on sequential profiles and the smallest at saturation.
- **608 nonces/min per GPU** versus 384 per H100 at TP=4 — less sharding, less loss. A
  single B300 at TP=1 still delivers more than either (1472 nonces/min on one card).
- **KV headroom**: 1,214,683 tokens against 656,967 on 4×H100 — ~2× the concurrency at a
  400k context.
- **The two runs share no nonce**: 0 / 1000 bit-identical. On Hopper the V4 PoC forward is
  not reproducible run to run — see *Reproducibility* below, it is the most consequential
  result here.

Nonce cross-validation is reported as **distances only** — DeepSeek-V4 does not have
calibrated validation thresholds yet, so no pass/fail is drawn.

## Hardware

| Parameter | Value |
|-----------|-------|
| GPU | 2× NVIDIA H200 SXM, 143,771 MiB each |
| Form factor | SXM — verified by `power.max_limit = 700 W` |
| Interconnect | NV18 full mesh (`nvidia-smi topo -m`) |
| NVIDIA Driver | 580.159.03 (CUDA 13) |
| Host | Vast.ai, Saudi Arabia |

### Interconnect matters — an NVL run was discarded

An earlier attempt used an **H200 NVL** box (600 W, NV6 bridge). Its PoC numbers came out
**~19 % lower** and with a different curve shape:

| Batch | H200 **SXM** (NV18) | H200 NVL (NV6) |
|------:|--------------------:|---------------:|
| 8 | 1104 | 960 |
| 16 | 1184 | 1024 ★ |
| 32 | **1216** ★ | 960 |

SXM rises monotonically to batch 32; NVL peaks at 16 and falls. Only SXM numbers are
reported here — NVL is not representative of the fleet. Pick boxes by `gpu_max_power`
(700 W = SXM, 600 W = NVL) *before* renting.

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
| **compiled** | `--compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'` | **breakable CUDA graph enabled** |

> **What "compiled" actually toggles.** vLLM auto-enables `VLLM_USE_BREAKABLE_CUDAGRAPH`
> for DeepSeek-V4, which disables the torch.compile pipeline outright ("Equivalent to
> `-cc.mode=none`"). The `--compilation-config '{"mode":3,...}'` shown here therefore has
> **no effect** — the only difference between the two configurations is whether
> `--enforce-eager` is present, i.e. whether the **breakable CUDA graph** is active.
> Evidence: `../deepseek-v4-flash-2xb200/artifacts/control_startup_cudagraph_evidence.txt`.


### sm_90 prerequisite

H200 is Hopper, and the image is built for the B300 (sm_100) profile without an
unversioned `libnvrtc.so`. The V4 sparse-MLA warmup compiles a FlashInfer kernel that links
`-lnvrtc`; without the symlink every worker dies during startup. Applied before both runs
(see `scripts/setup_box.sh`):

```bash
ln -sf /usr/local/cuda/lib64/libnvrtc.so.13 /usr/local/cuda/lib64/libnvrtc.so
ln -sf /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so.13 \
       /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so
ldconfig
```

Also note: the host driver must be **≥ 580**. A CUDA-13 image on a 570 host dies with
`RuntimeError: The NVIDIA driver on your system is too old (found version 12080)`, and the
image's own `/usr/local/cuda/compat` does **not** rescue it (it ships libcuda 525, older
than the host). Filter offers by `cuda_max_good >= 13.0`.

## Startup profile

| Phase | eager | compiled |
|-------|------:|---------:|
| Cold start (model load → serving) | 315 s | ~330 s |
| Available KV cache per GPU | 39.33 GiB | 36.6 GiB |
| GPU KV cache size | 1,214,683 tokens | 1,127,930 tokens |

## Results — PoC throughput sweep

`run_pow_generation.py --phase 3`, 5 s warmup + 30 s steady state, seq_len 1024:

| Batch Size | eager: nonces (30 s) | eager: nonces/min | compiled: nonces (30 s) | compiled: nonces/min |
|-----------:|---------------------:|------------------:|------------------------:|---------------------:|
| 8 | 552 | 1104 | 552 | 1104 |
| 16 | 592 | 1184 | 592 | 1184 |
| **32** ★ | **608** | **1216** | **608** | **1216** |

**Identical in every cell.** Nonce collection (batch 32) agrees too: 1056 nonces @
1011/min in both modes. Contrast this with 2×B200, where the same comparison shows a large
CUDA-graph gain on PoC — see `2026-07/deepseek-v4-flash-2xb200`.

### Topology comparison

| Configuration | best nonces/min | per GPU | KV cache tokens |
|---------------|----------------:|--------:|----------------:|
| 1×B300 TP=1 | 1472 | **1472** | 2,604,694 |
| 2×B200 TP=2 | 2304 (compiled) | **1152** | 2,359,479 |
| 2×H200 SXM TP=2 (this run) | 1216 | **608** | 1,214,683 |
| 4×H100 SXM TP=4 | 1536 | **384** | 656,967 |

Per-GPU throughput falls as the model is sharded wider, and Blackwell is ahead of Hopper at
equal TP.

## Results — inference (compressa-perf)

Full tables in [`compressa-perf-comparison.md`](compressa-perf-comparison.md). Headline
numbers (eager → compiled, 0 failed requests in every scenario):

| Scenario | TTFT | Latency | TPOT | Output tok/s |
|----------|-----:|--------:|-----:|-------------:|
| s1 long prompt, sequential, short decode | 3.28 → **0.39** s | 9.52 → **1.04** s | 0.138 → **0.012** s | 7.2 → **83.4** |
| s2 short prompt, high concurrency | 5.51 → **4.27** s | 27.56 → **12.27** s | 0.184 → **0.079** s | 152 → **371** |
| s3 very long, sequential, long decode | 6.24 → **0.86** s | 35.61 → **1.89** s | 0.112 → **0.014** s | 8.9 → **72.3** |
| s4 very long, max concurrency | 6.45 → **5.40** s | 38.10 → **12.97** s | 0.295 → **0.103** s | 53.9 → **170.8** |

CUDA graphs are worth 7–19× on the sequential profiles and 2–3× under concurrency.

## Reproducibility — Hopper does not reproduce

Repeating the collection on the **same box** (only the compilation flag differed):

| GPU | architecture | bit-identical nonces |
|-----|--------------|---------------------:|
| 2×B200 | Blackwell | 968 / 1000 (96.8 %) |
| 1×B300 | Blackwell | 875 / 1000 (87.5 %) |
| **2×H200 SXM (this run)** | **Hopper** | **0 / 1000 (0 %)** |
| 4×H100 SXM | Hopper | 0 / 1000 (0 %) |

Not a single nonce repeats on H200, while on Blackwell the overwhelming majority do — with
the *same* configuration difference. The V4 PoC forward is deterministic on Blackwell
kernels and not on the Hopper path, which sets the noise floor any future validation
threshold has to absorb on Hopper hardware.

## Nonce cross-validation (distances)

`scripts/compare_nonces.py` — canonical `decode_vector` → per-nonce L2. 1000 common nonces
per pair. **No pass/fail verdict is drawn** — V4's limits are still to be calibrated; the
mismatch column counts nonces further apart than 0.4 as a descriptive statistic.

| A | B | median L2 | mean L2 | max L2 | nonces >0.4 |
|---|---|----------:|--------:|-------:|------------:|
| eager 2×H200 | compiled 2×H200 (same box) | 0.1830 | 0.1963 | 1.347 | 2.90 % |
| eager 2×H200 | B300 TP1 plugin | 0.1891 | 0.1991 | 1.370 | 2.80 % |
| compiled 2×H200 | B300 TP1 plugin | 0.1871 | 0.2000 | 1.292 | 3.20 % |
| eager 2×H200 | eager 4×H100 | 0.1884 | 0.1978 | 1.021 | 2.50 % |
| eager 2×H200 | eager 2×B200 | 0.1891 | 0.2030 | 1.327 | 3.70 % |

Every pair lands in the same narrow band — median ≈ 0.183–0.189, 2.5–3.7 % beyond 0.4 —
**including the two runs on this very box**. Repeating a run on identical hardware moves
the nonces as much as changing the GPU model does.

## Files

- `README.md` — this report
- `compressa-perf-comparison.md` — full inference tables, eager vs compiled
- `artifacts/config.json` — hardware, versions, runner.py overrides, per-mode startup
- `artifacts/sweep.json` — PoC sweep and collection numbers
- `artifacts/l2_matrix.json` — L2 distances and mismatch counts
- `artifacts/nonces_eager.json`, `artifacts/nonces_compiled.json` — 1000+ nonce vectors each
- `artifacts/h200_2x_v4_poc_{eager,compiled}.log` — raw sweep output
- `artifacts/h200_2x_v4_compressa_{eager,compiled}.log` — raw compressa output
- `artifacts/compressa_all_metrics.txt`, `artifacts/compressa_parsed.json` — metric dumps
- `artifacts/env.txt` — versions, topology, KV sizing as read off the box
- `artifacts/runner_forced_args.txt` — the forced block as it ran
- `scripts/setup_box.sh` — box preparation (sm_90 fix, deps, runner.py, API)
- `scripts/` — sweep, collection and L2 scripts

## Findings

- **Two H200s are a better shape for V4 than four H100s**: 608 vs 384 nonces/min per GPU,
  ~2× the KV cache, and the box rents for less.
- **A single B300 still beats both**, and 2×B200 beats all of them — where the weights fit
  on fewer cards, use fewer cards.
- **Do not pass `--enforce-eager` when serving V4** — 7–19× on sequential latency and
  2–3× under concurrency, at no cost to PoC throughput.
- **PoC throughput on Hopper is insensitive to the compilation arguments** — identical in
  every cell. This does *not* hold on Blackwell (B200 gains ~70 %).
- **Buy SXM, not NVL** — the NV6 bridge costs ~19 % of PoC throughput versus NV18.
- **The image needs a `libnvrtc.so` symlink on Hopper**, and a host driver ≥ 580.
- **V4 PoC does not reproduce on Hopper** (0/1000 bit-identical between two runs on one
  box), while Blackwell reproduces 87–97 %.

## Reproducibility checklist

- [x] Hardware verified, not assumed (700 W ⇒ SXM, NV18 topology)
- [x] Image pinned by digest; PoC plugin identified
- [x] Every changed `runner.py` parameter listed against its image default
- [x] Per-mode arguments and what was observed at startup
- [x] sm_90 and driver prerequisites documented with the exact fixes, and scripted
- [x] Sweep method, warmup and measurement windows stated
- [x] Nonce sets, raw logs and metric dumps committed
- [x] L2 computed with the canonical script, which is committed
- [x] Discarded NVL run disclosed with its numbers
- [x] No `.claude/` paths, no sibling-repo paths
