# compressa-perf 2×H200 × DeepSeek-V4-Flash: enforce-eager vs CUDA-graph

**Date:** 2026-07-24
**Model:** `deepseek-ai/DeepSeek-V4-Flash` (FP8, sparse-MLA, hash-routed MoE)
**Hardware:** 2× NVIDIA H200 NVL (Vast.ai, Mississippi US), TP=2
**Image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash:0.2.13-vllm0.25.1-overlay-k4`
**Baseline:** **compiled** (100 %)

All percentages are relative to **compiled**. For TTFT / TPOT / Latency lower is better
(negative % = better); for throughput and RPS higher is better.

## Configurations Compared

| Mode | vLLM args | Observed at startup |
|------|-----------|---------------------|
| **enforce-eager** | `--enforce-eager` | **CUDA graphs off** |
| **compiled** (baseline) | `--compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'` | **breakable CUDA graph on**, capture 61 s / 2.41 GiB |

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
| 1 | 2.1196 (+403 %) | **0.4217** (100 %) |
| 2 | 5.4932 (+570 %) | **0.8194** (100 %) |
| 3 | 4.2189 (+359 %) | **0.9200** (100 %) |
| 4 | 6.9596 (+12 %) | **6.2318** (100 %) |

## Latency (s) — lower is better

| # | enforce-eager | compiled |
|---|---:|---:|
| 1 | 6.9658 (+439 %) | **1.2922** (100 %) |
| 2 | 19.2729 (+344 %) | **4.3401** (100 %) |
| 3 | 12.5345 (+500 %) | **2.0873** (100 %) |
| 4 | 28.8464 (+91 %) | **15.1141** (100 %) |

## TPOT — Time Per Output Token (s) — lower is better

| # | enforce-eager | compiled |
|---|---:|---:|
| 1 | 0.0784 (+543 %) | **0.0122** (100 %) |
| 2 | 0.1267 (+343 %) | **0.0286** (100 %) |
| 3 | 0.0825 (+458 %) | **0.0148** (100 %) |
| 4 | 0.2265 (+81 %) | **0.1254** (100 %) |

## Output token throughput (tok/s) — higher is better

| # | enforce-eager | compiled |
|---|---:|---:|
| 1 | 12.75 (−84 %) | **81.87** (100 %) |
| 2 | 221.14 (−77 %) | **973.55** (100 %) |
| 3 | 12.13 (−82 %) | **67.55** (100 %) |
| 4 | 80.09 (−47 %) | **151.77** (100 %) |

## RPS — higher is better

| # | enforce-eager | compiled |
|---|---:|---:|
| 1 | 0.1436 (−81 %) | **0.7738** (100 %) |
| 2 | 1.4543 (−77 %) | **6.4064** (100 %) |
| 3 | 0.0798 (−83 %) | **0.4791** (100 %) |
| 4 | 0.6288 (−50 %) | **1.2590** (100 %) |

## Cold start

| Phase | enforce-eager | compiled |
|-------|--------------:|---------:|
| Model load → serving | 255 s | 225 s |
| CUDA graph capture | — | 61 s (2.41 GiB) |
| KV cache per GPU | 39.69 GiB | 36.88 GiB |
| GPU KV cache size | 1,225,837 tok | 1,139,114 tok |

## Comparison with 4×H100

| Metric (scenario 2, high concurrency) | 2×H200 TP=2 | 4×H100 TP=4 |
|---------------------------------------|------------:|------------:|
| TTFT, compiled | 0.82 s | 0.26 s |
| Latency, compiled | 4.34 s | 3.22 s |
| Output tok/s, compiled | 974 | 1294 |
| RPS, compiled | 6.41 | 8.59 |
| CUDA-graph win over eager | 4.4× | 10.1× |

Four H100s serve more concurrent throughput than two H200s, as expected from twice the
card count. The CUDA-graph advantage is smaller on H200 because its eager baseline is
already faster — fewer shards mean less cross-GPU traffic to hide.

## Conclusions

- **CUDA graphs dominate V4 serving performance here too**: 4.4–6.7× on TTFT, latency,
  TPOT and throughput for sequential and moderately concurrent traffic, ~1.9× at
  saturation.
- **Cost of graphs:** 61 s of startup and 2.81 GiB less KV cache per GPU
  (1,225,837 → 1,139,114 tokens).
- **PoC throughput is unaffected** — 1024 vs 992 nonces/min at batch 16, a ~3 % spread in
  both directions (see `README.md`).
