# PoC + Inference Benchmark: DeepSeek-V4-Flash on 4×H100 (eager vs CUDA-graph)

**Date:** 2026-07-24
**Model:** `deepseek-ai/DeepSeek-V4-Flash` (FP8, sparse-MLA, hash-routed MoE)
**Hardware:** 4× NVIDIA H100 80GB HBM3 SXM5, TP=4
**Image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash:0.2.13-vllm0.25.1-overlay-k4`
**Digest:** `sha256:2af898fa516424ea2884b77e40ae480ce61b19a7d0112f8c9b8cc866c8bcb28a`
**PoC:** gonka-poc plugin (`--worker-extension-cls gonka_poc.worker.PoCWorkerExtension`)
**vLLM:** 0.25.1

## Summary

Two runs of the same image on the same box, differing only in the compilation-related
vLLM arguments, each measured for **PoC throughput**, **nonce collection** and
**inference** (compressa-perf), followed by canonical L2 cross-validation against the
1×B300 baselines.

Two results:

1. **PoC throughput is identical between the two runs — bit for bit** (1408 / 1504 /
   1536 nonces/min at batch 8 / 16 / 32). The PoC forward runs eager on its own
   (`skip_compiled`), so it is unaffected by either setting.
2. **Inference is 8–15× faster in the compiled configuration.** `--enforce-eager`
   disables CUDA graphs; the compiled run captures the *breakable CUDA graph* path (the
   supported one for V4) — 85 s / 2.37 GiB — and pays for it many times over. This is
   the practically important knob for V4 serving.

Cross-validating the nonces: 4×H100 TP=4 agrees with the 1×B300 TP=1 baselines at
~0.19 median L2 and 2.6–3.1 % mismatch — **PASS under the chain threshold with the
calibrated `p_mismatch=0.02`, FRAUD under the stricter `p_mismatch=0.001`**.

## Hardware

| Parameter | Value |
|-----------|-------|
| GPU | 4× NVIDIA H100 80GB HBM3 |
| Form factor | SXM5 — verified by `power.max_limit = 700 W` |
| Interconnect | NV18 full mesh (`nvidia-smi topo -m`) |
| NVIDIA Driver | 580.95.05 (CUDA 13.0) |
| CPU | AMD EPYC 9575F 64-Core (240 vCPUs) |
| RAM | 1,511 GB |
| Host | Vast.ai, Taiwan |

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
| `--tensor-parallel-size` | 1 | **4** | 149 GiB of weights do not fit on one 80 GB H100 |
| `--max-model-len` | 200000 | **400000** | benchmark at a realistic serving context |
| `--max-num-batched-tokens` | 16384 | **32768** | sweep batch 32 × seq_len 1024 = 32768 tokens |

Unchanged: `--gpu-memory-utilization 0.90`, `--kv-cache-dtype fp8` (mandatory — FlashMLA
asserts without it), `--logprobs-mode processed_logprobs`, `--trust-remote-code`,
`--worker-extension-cls gonka_poc.worker.PoCWorkerExtension`.

Per-mode arguments passed through `additional_args` (not part of the forced block):

| Mode | Extra args | Observed at startup |
|------|-----------|---------------------|
| **eager** | `--enforce-eager` | no CUDA graphs |
| **compiled** | `--compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'` | **breakable CUDA graph enabled**, capture 85 s / 2.37 GiB |

### sm_90 prerequisite

The image is built for the B300 (sm_100) profile and lacks the unversioned
`libnvrtc.so` symlink. On H100 the V4 sparse-MLA warmup compiles a FlashInfer kernel via
ninja, which links `-lnvrtc` and fails:

```
/usr/bin/ld: cannot find -lnvrtc
  → deepseek_v4_sparse_mla_attention_warmup → all four workers die → engine exits
```

Fix applied before the runs:

```bash
ln -sf /usr/local/cuda/lib64/libnvrtc.so.13 /usr/local/cuda/lib64/libnvrtc.so
ln -sf /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so.13 \
       /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so
ldconfig
```

## Startup profile

| Phase | eager | compiled |
|-------|------:|---------:|
| Cold start (model load → serving) | **180 s** | **285 s** |
| CUDA graph capture | — | 85 s, 2.37 GiB |
| Available KV cache per GPU | 21.27 GiB | 18.61 GiB |
| GPU KV cache size | 656,967 tokens | 574,611 tokens |
| Max concurrency @ 400k context | 1.64× | 1.44× |

The 105 s difference in cold start is the graph capture plus the KV it reserves.

## Results — PoC throughput sweep

`run_pow_generation.py --phase 3`, 5 s warmup + 30 s steady state, seq_len 1024:

| Batch Size | eager: nonces (30 s) | eager: nonces/min | compiled: nonces (30 s) | compiled: nonces/min |
|-----------:|---------------------:|------------------:|------------------------:|---------------------:|
| 8 | 704 | 1408 | 704 | 1408 |
| 16 | 752 | 1504 | 752 | 1504 |
| **32** ★ | **768** | **1536** | **768** | **1536** |

**Identical in every cell** — the PoC forward runs `skip_compiled`, so neither CUDA
graphs nor the compilation arguments touch this path.

Nonce collection (batch 32): eager **1088 nonces @ 1229/min**, compiled **1088 nonces @
1227/min**.

### Per-GPU efficiency

4×H100 TP=4 delivers 1536 nonces/min, i.e. **384 nonces/min per GPU**. A single B300 at
TP=1 delivers 1472 nonces/min on one GPU (see `2026-07/deepseek-v4-flash-poc-1xb300`).
For V4 PoC, four H100s barely match one B300 — the model fits on a single Blackwell card,
and TP=4 sharding costs roughly 4× the per-GPU throughput.

## Results — inference (compressa-perf)

Full tables in [`compressa-perf-comparison.md`](compressa-perf-comparison.md). Headline
numbers (eager → compiled, all runs with 0 failed requests):

| Scenario | TTFT | Latency | TPOT | Output tok/s |
|----------|-----:|--------:|-----:|-------------:|
| s1 long prompt, sequential, short decode | 2.55 → **0.30** s | 15.83 → **1.07** s | 0.125 → **0.010** s | 8.0 → **102.7** |
| s2 short prompt, high concurrency | 3.87 → **0.26** s | 32.63 → **3.22** s | 0.210 → **0.021** s | 133.8 → **1293.8** |
| s3 very long, sequential, long decode | 5.10 → **0.64** s | 15.21 → **1.82** s | 0.161 → **0.011** s | 6.2 → **91.6** |
| s4 very long, max concurrency | 4.89 → 4.75 s | 25.65 → **15.72** s | 0.208 → **0.109** s | 80.7 → **156.2** |

Turning CUDA graphs on (i.e. *not* passing `--enforce-eager`) is worth **8–15×** on the
sequential and moderate-concurrency profiles and ~1.9× at maximum concurrency, where the
GPU is already saturated and kernel-launch overhead is amortised.

## Cross-validation (canonical L2)

`scripts/compare_nonces.py` — decode_vector → per-nonce L2 → `binomtest(alternative="greater")`,
`fraud_threshold=0.01`. 1000 common nonces per pair.

| A | B | median L2 | mean L2 | mismatch @0.4 | chain default (p_mis=0.001) | chain + calibrated (p_mis=0.02) |
|---|---|----------:|--------:|--------------:|:---------------------------:|:-------------------------------:|
| eager 4×H100 | compiled 4×H100 | 0.1857 | 0.1962 | 2.70 % | FRAUD | **PASS** |
| eager 4×H100 | B300 TP1 plugin | 0.1889 | 0.2033 | 3.00 % | FRAUD | **PASS** |
| compiled 4×H100 | B300 TP1 plugin | 0.1889 | 0.2005 | 2.60 % | FRAUD | **PASS** |
| compiled 4×H100 | B300 TP1 in-tree fork | 0.1870 | 0.1998 | 2.90 % | FRAUD | **PASS** |

The two runs on the **same box** differ by as much as the **cross-hardware** pairs
(2.70 % vs 2.6–3.1 %). The dominant term is therefore not hardware, topology or CUDA
graphs — it is TP=4 run-to-run nondeterminism in the V4 kernels. All pairs sit in the
same envelope and pass the production threshold with the calibrated `p_mismatch`.

## Files

- `README.md` — this report
- `compressa-perf-comparison.md` — full inference tables, eager vs compiled
- `artifacts/config.json` — hardware, versions, runner.py overrides, per-mode resolution
- `artifacts/sweep.json` — PoC sweep and collection numbers
- `artifacts/l2_matrix.json` — canonical L2 verdicts for all pairs
- `artifacts/nonces_eager.json`, `artifacts/nonces_compiled.json` — 1000+ nonce vectors each
- `artifacts/h100_4x_v4_poc_{eager,compiled}.log` — raw sweep output
- `artifacts/h100_4x_v4_compressa_{eager,compiled}.log` — raw compressa output
- `artifacts/compressa_all_metrics.txt`, `artifacts/compressa_parsed.json` — metric dumps
- `artifacts/compressa_config.yml` — the four inference scenarios
- `artifacts/runner_forced_args.txt` — the forced block as it ran
- `scripts/` — sweep, collection and L2 scripts

## Findings

- **Do not pass `--enforce-eager` when serving V4.** It is the only thing separating a
  usable serving profile from one 8–15× slower, and it costs nothing in PoC throughput.
- **PoC throughput is insensitive to both settings** — 1408/1504/1536 nonces/min in both
  runs, identical per batch.
- **V4 PoC scales poorly across GPUs.** 384 nonces/min per H100 at TP=4 versus 1472 on a
  single B300 at TP=1; prefer one card per engine where the weights fit.
- **The B300-profile image needs a `libnvrtc.so` symlink to run on sm_90**, otherwise the
  V4 sparse-MLA warmup kills the engine during startup.
- **TP=4 introduces ~2.7 % nonce mismatch run-to-run**, on par with the cross-hardware
  difference. V4 nonces from a TP>1 topology pass the chain gate only with the calibrated
  `p_mismatch=0.02`.

## Reproducibility checklist

- [x] Hardware verified, not assumed (700 W ⇒ SXM5, NV18 topology)
- [x] Image pinned by digest; PoC plugin identified
- [x] Every changed `runner.py` parameter listed against its image default
- [x] Per-mode arguments and what vLLM actually resolved them to
- [x] sm_90 prerequisite documented with the exact fix
- [x] Sweep method, warmup and measurement windows stated
- [x] Nonce sets, raw logs and metric dumps committed
- [x] L2 computed with the canonical script, which is committed
- [x] No `.claude/` paths, no sibling-repo paths
