# compressa-perf B200 × Kimi K2.5: enforce-eager vs compiled

**Date:** 2026-04-08
**Model:** `moonshotai/Kimi-K2.5` (INT4 compressed-tensors, MoE 1.1T/32B, MLA)
**Hardware:** 4× NVIDIA B200 (Vast.ai, Alabama)
**Image:** `ghcr.io/product-science/mlnode:3.0.13-alpha1`
**Baseline:** compiled (100%)

All percentages are relative to **compiled** mode. For TTFT/TPOT/Latency: lower is better (negative % = better). For throughput and RPS: higher is better (positive % = better).

## Configurations Compared

| Mode | vLLM args |
|------|-----------|
| **enforce-eager** | `--enforce-eager --attention-backend FLASHINFER_MLA` |
| **compiled** (baseline) | `--attention-backend FLASHINFER_MLA --compilation-config '{"custom_ops": ["all"]}'` |

Common args: `--tensor-parallel-size 4 --gpu-memory-utilization 0.95 --max-num-seqs 128 --max-model-len 262144 --trust-remote-code --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --mm-encoder-tp-mode data`

Environment: `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`

**Important:** compiled mode **requires warmup** before benchmarking to avoid `shm_broadcast` `TimeoutError` from Marlin MoE kernel autotune on the first ~10k-token prefill. See [b200-kimi-k25-benchmark.md](./b200-kimi-k25-benchmark.md) for the crash + fix story.

## Benchmark Scenarios

| # | Profile | Prompt chars | ~Prompt tokens | Tasks | Runners | Max output tokens |
|---|---------|-------------:|---------------:|------:|--------:|------------------:|
| 1 | Long prompt, sequential, short decode | 20,000 | ~10,200 | 5 | 1 | 300 |
| 2 | Short prompt, high concurrency | 2,000 | ~1,040 | 200 | 30 | 300 |
| 3 | Very long prompt, sequential, long decode | 45,000 | ~23,000 | 5 | 1 | 1000 |
| 4 | Very long, 5 runners | 45,000 | ~23,000 | 10 | 5 | 1000 |
| 5 | Very long, max concurrency | 45,000 | ~23,000 | 40 | 20 | 1000 |
| 6 | Very long, 10 runners, 60 tasks | 45,000 | ~23,000 | 60 | 10 | 1000 |

## TTFT — Time To First Token (s) — lower is better

| # | enforce-eager | compiled |
|---|---:|---:|
| 1 | 0.59 (+9%) | **0.54** (100%) |
| 2 | **0.76 (-8%)** | 0.83 (100%) |
| 3 | 1.16 (0%) | **1.16** (100%) |
| 4 | **2.69 (-10%)** | 2.99 (100%) |
| 5 | 8.18 (+19%) | **6.88** (100%) |
| 6 | **2.33 (-4%)** | 2.42 (100%) |

## TPOT — Time Per Output Token (s) — lower is better

| # | enforce-eager | compiled |
|---|---:|---:|
| 1 | 0.032 (+158%) | **0.012** (100%) |
| 2 | 0.035 (+27%) | **0.028** (100%) |
| 3 | 0.032 (+154%) | **0.013** (100%) |
| 4 | 0.036 (+66%) | **0.022** (100%) |
| 5 | 0.046 (+11%) | **0.041** (100%) |
| 6 | 0.041 (+25%) | **0.033** (100%) |

## Latency (s) — lower is better

| # | enforce-eager | compiled |
|---|---:|---:|
| 1 | 9.55 (+157%) | **3.72** (100%) |
| 2 | 10.34 (+25%) | **8.25** (100%) |
| 3 | 27.81 (+176%) | **10.09** (100%) |
| 4 | 34.19 (+60%) | **21.35** (100%) |
| 5 | 43.46 (+15%) | **37.90** (100%) |
| 6 | 37.31 (+24%) | **30.01** (100%) |

## Throughput Input Tokens (tok/s) — higher is better

| # | enforce-eager | compiled |
|---|---:|---:|
| 1 | 1043 (-61%) | **2678** (100%) |
| 2 | 2784 (-21%) | **3507** (100%) |
| 3 | 806 (-64%) | **2222** (100%) |
| 4 | 3117 (-40%) | **5193** (100%) |
| 5 | 8414 (-24%) | **11081** (100%) |
| 6 | 5718 (-21%) | **7266** (100%) |

## Throughput Output Tokens (tok/s) — higher is better

| # | enforce-eager | compiled |
|---|---:|---:|
| 1 | 31.4 (-61%) | **80.7** (100%) |
| 2 | 829.7 (-21%) | **1045.0** (100%) |
| 3 | 31.1 (-61%) | **79.0** (100%) |
| 4 | 130.9 (-42%) | **227.0** (100%) |
| 5 | 354.9 (-22%) | **453.0** (100%) |
| 6 | 231.3 (-22%) | **296.0** (100%) |

## Total Throughput (tok/s) — higher is better

| # | enforce-eager | compiled |
|---|---:|---:|
| 1 | 1074 (-61%) | **2759** (100%) |
| 2 | 3614 (-21%) | **4552** (100%) |
| 3 | 837 (-64%) | **2301** (100%) |
| 4 | 3248 (-40%) | **5420** (100%) |
| 5 | 8769 (-24%) | **11534** (100%) |
| 6 | 5950 (-21%) | **7562** (100%) |

## RPS — Requests Per Second — higher is better

| # | enforce-eager | compiled |
|---|---:|---:|
| 1 | 0.105 (-61%) | **0.269** (100%) |
| 2 | 2.77 (-21%) | **3.48** (100%) |
| 3 | 0.036 (-64%) | **0.099** (100%) |
| 4 | 0.139 (-40%) | **0.231** (100%) |
| 5 | 0.375 (-24%) | **0.494** (100%) |
| 6 | 0.255 (-21%) | **0.324** (100%) |

## Summary vs compiled

### enforce-eager

- **Total throughput:** −21% to −64% (always worse)
- **Latency:** +15% to +176% worse
- **TPOT:** +11% to +158% worse (largest gap on prefill-bound single-runner scenarios)
- **TTFT:** mixed (better on scenarios 2/4/6 due to no compile/capture overhead on first-runner bursts)

## Key Insight

**Compiled mode wins in every throughput-oriented metric for Kimi K2.5.** The gap is biggest on **prefill-heavy, low-concurrency scenarios** (1 and 3) where compile optimizations (FusedMoE + fused norm/quant + rotary) dominate the critical path.

- Biggest throughput gap: scenario 3 (23k in, 1 runner, long decode) — eager −64%
- Smallest throughput gap: scenario 2 (1k in, 30 runners) — eager −21%
- **Average throughput uplift from compiled: +80%**

### Contrast with PoC benchmark

The PoC v2 nonce benchmark (fixed `seq_len=1024`, `k_dim=12`, decode-only) showed **compile == eager** on Kimi K2.5 (both 1024 nonces/min on best batch=32). This is because:
1. PoC is decode-only — the prefill path (where compile wins the most) is not exercised
2. At the optimal batch, decode throughput is limited by KV cache bandwidth, not kernel latency
3. On Qwen 235B the PoC gap was +25% compile vs eager, but Qwen uses GQA (not MLA) and FP8 (not INT4 Marlin) — different bottlenecks

### Gotcha: compiled mode requires warmup

Our first attempt at compiled compressa-perf **crashed** on the first 10k-token prefill with `shm_broadcast TimeoutError` in the multiproc executor (default 300s timeout exceeded by Marlin MoE kernel autotune + first-hit Inductor compile). Fix:
1. `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900`
2. Warmup requests (5.7k tok, then 11.4k tok) before benchmark

## Recommendation

**For Kimi K2.5 inference on B200 (and similar architectures), use compiled mode:**

```bash
--attention-backend FLASHINFER_MLA \
--compilation-config '{"custom_ops": ["all"]}'
```

With the prerequisite:
- `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900` env var
- Warmup with a 5-10k token request after vLLM starts, before taking real traffic

**Do NOT use enforce-eager for Kimi K2.5 production inference** — you lose 21-64% total throughput and 11-158% on TPOT.

**For PoC-only workload:** either mode works equally (1024 nonces/min), eager is simpler and has no cold-start complications.

## Raw Data

- SQLite database (compiled): `compressa-perf-results/b200_4x_kimi_compiled.sqlite`
- compressa-perf logs: `compressa-perf-results/b200_4x_kimi_compressa_eager.log`, `compressa-perf-results/b200_4x_kimi_compressa_compiled.log`
- PoC logs: `compressa-perf-results/b200_4x_kimi_poc_eager.log`, `compressa-perf-results/b200_4x_kimi_poc_compiled.log`

Full reproduction details: [b200-kimi-k25-benchmark.md](./b200-kimi-k25-benchmark.md)
