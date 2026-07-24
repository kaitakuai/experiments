# compressa-perf 2×H200 × DeepSeek-V4-Flash: enforce-eager vs CUDA-graph

**Date:** 2026-07-24
**Model:** `deepseek-ai/DeepSeek-V4-Flash` (FP8, sparse-MLA, hash-routed MoE)
**Hardware:** 2× NVIDIA H200 SXM (Vast.ai, Saudi Arabia), TP=2
**Image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash:0.2.13-vllm0.25.1-overlay-k4`
**Baseline:** **compiled** (100 %)

All percentages are relative to **compiled**. For TTFT / TPOT / Latency lower is better
(negative % = better); for throughput and RPS higher is better.

## Configurations Compared

| Mode | vLLM args | Observed at startup |
|------|-----------|---------------------|
| **enforce-eager** | `--enforce-eager` | **CUDA graphs off** |
| **compiled** (baseline) | `--compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'` | **breakable CUDA graph on** |

Common args (from `runner.py`): `--tensor-parallel-size 2 --gpu-memory-utilization 0.90
--max-model-len 400000 --max-num-batched-tokens 32768 --kv-cache-dtype fp8
--logprobs-mode processed_logprobs --trust-remote-code
--worker-extension-cls gonka_poc.worker.PoCWorkerExtension`

CUDA graphs are what separates the two columns: `--enforce-eager` turns them off, the
compiled configuration captures them at startup.

## Benchmark Scenarios

| # | Profile | Prompt chars | Tasks | Runners | Max output tokens |
|---|---------|-------------:|------:|--------:|------------------:|
| 1 | Long prompt, sequential, short decode | 20,000 | 5 | 1 | 300 |
| 2 | Short prompt, high concurrency | 2,000 | 200 | 30 | 300 |
| 3 | Very long prompt, sequential, long decode | 45,000 | 5 | 1 | 1000 |
| 4 | Very long prompt, max concurrency | 45,000 | 40 | 20 | 1000 |

Failed requests: **0** in every scenario, both modes.

## TTFT — Time To First Token (s) — lower is better

| # | enforce-eager | compiled |
|---|---:|---:|
| 1 | 3.2849 (+738 %) | **0.3919** (100 %) |
| 2 | 5.5083 (+29 %) | **4.2710** (100 %) |
| 3 | 6.2380 (+624 %) | **0.8610** (100 %) |
| 4 | 6.4483 (+19 %) | **5.3985** (100 %) |

## Latency (s) — lower is better

| # | enforce-eager | compiled |
|---|---:|---:|
| 1 | 9.5211 (+813 %) | **1.0425** (100 %) |
| 2 | 27.5646 (+125 %) | **12.2669** (100 %) |
| 3 | 35.6095 (+1784 %) | **1.8898** (100 %) |
| 4 | 38.1041 (+194 %) | **12.9703** (100 %) |

## TPOT — Time Per Output Token (s) — lower is better

| # | enforce-eager | compiled |
|---|---:|---:|
| 1 | 0.1380 (+1050 %) | **0.0120** (100 %) |
| 2 | 0.1837 (+133 %) | **0.0787** (100 %) |
| 3 | 0.1122 (+713 %) | **0.0138** (100 %) |
| 4 | 0.2953 (+186 %) | **0.1033** (100 %) |

## Output token throughput (tok/s) — higher is better

| # | enforce-eager | compiled |
|---|---:|---:|
| 1 | 7.25 (−91 %) | **83.44** (100 %) |
| 2 | 151.50 (−59 %) | **371.12** (100 %) |
| 3 | 8.91 (−88 %) | **72.28** (100 %) |
| 4 | 53.85 (−68 %) | **170.85** (100 %) |

## RPS — higher is better

| # | enforce-eager | compiled |
|---|---:|---:|
| 1 | 0.1050 (−89 %) | **0.9591** (100 %) |
| 2 | 1.0095 (−58 %) | **2.3800** (100 %) |
| 3 | 0.0281 (−95 %) | **0.5291** (100 %) |
| 4 | 0.4174 (−69 %) | **1.3602** (100 %) |

## Cold start

| Phase | enforce-eager | compiled |
|-------|--------------:|---------:|
| Model load → serving | 315 s | ~330 s |
| CUDA graph capture | — | logged |
| KV cache per GPU | 39.33 GiB | 36.6 GiB |
| GPU KV cache size | 1,214,683 tok | 1,127,930 tok |

## Comparison with 4×H100

| Metric (scenario 2, high concurrency) | 2×H200 SXM TP=2 | 4×H100 TP=4 |
|---------------------------------------|------------:|------------:|
| TTFT, compiled | 4.27 s | 0.26 s |
| Latency, compiled | 12.27 s | 3.22 s |
| Output tok/s, compiled | 371 | 1294 |
| RPS, compiled | 2.38 | 8.59 |
| CUDA-graph win over eager | 2.2× | 10.1× |

Four H100s serve more concurrent throughput than two H200s, as expected from twice the
card count. The CUDA-graph advantage is smaller on H200 because its eager baseline is
already faster — fewer shards mean less cross-GPU traffic to hide.

## Conclusions

- **CUDA graphs dominate V4 serving performance here too**: 7–19× on the sequential
  profiles, 2–3× under concurrency.
- **Cost of graphs:** ~2.7 GiB less KV cache per GPU (1,214,683 → 1,127,930 tokens) plus
  the capture time at startup.
- **PoC throughput is unaffected on Hopper** — 1216 nonces/min at batch 32 in both modes,
  identical in every cell (see `README.md`). On Blackwell it is *not* unaffected: 2×B200
  gains ~70 % from CUDA graphs.
