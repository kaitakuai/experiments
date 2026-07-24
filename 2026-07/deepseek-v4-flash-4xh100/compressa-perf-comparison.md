# compressa-perf 4×H100 × DeepSeek-V4-Flash: enforce-eager vs CUDA-graph

**Date:** 2026-07-24
**Model:** `deepseek-ai/DeepSeek-V4-Flash` (FP8, sparse-MLA, hash-routed MoE)
**Hardware:** 4× NVIDIA H100 80GB HBM3 SXM5 (Vast.ai, Taiwan), TP=4
**Image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash:0.2.13-vllm0.25.1-overlay-k4`
**Baseline:** **compiled** (100 %)

All percentages are relative to **compiled**. For TTFT / TPOT / Latency lower is better
(negative % = better); for throughput and RPS higher is better.

## Configurations Compared

| Mode | vLLM args | What vLLM resolved |
|------|-----------|--------------------|
| **enforce-eager** | `--enforce-eager` | `CompilationMode.NONE`, **CUDA graphs off** |
| **compiled** (baseline) | `--compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'` | `CompilationMode.NONE` (forced for `DeepseekV4ForCausalLM`), **breakable CUDA graph on** |

Common args (from `runner.py`): `--tensor-parallel-size 4 --gpu-memory-utilization 0.90
--max-model-len 400000 --max-num-batched-tokens 32768 --kv-cache-dtype fp8
--logprobs-mode processed_logprobs --trust-remote-code
--worker-extension-cls gonka_poc.worker.PoCWorkerExtension`

**Important:** the requested `mode: 3` never takes effect — vLLM 0.25.1 auto-enables
`VLLM_USE_BREAKABLE_CUDAGRAPH=1` for this architecture and that forces
`CompilationMode.NONE`. The real difference between the two columns is therefore
**CUDA graphs on/off**, which `--enforce-eager` controls.

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
| 1 | 2.5500 (+757 %) | **0.2976** (100 %) |
| 2 | 3.8680 (+1361 %) | **0.2648** (100 %) |
| 3 | 5.1027 (+695 %) | **0.6415** (100 %) |
| 4 | 4.8939 (+3 %) | **4.7540** (100 %) |

## Latency (s) — lower is better

| # | enforce-eager | compiled |
|---|---:|---:|
| 1 | 15.8338 (+1381 %) | **1.0689** (100 %) |
| 2 | 32.6348 (+913 %) | **3.2213** (100 %) |
| 3 | 15.2081 (+735 %) | **1.8217** (100 %) |
| 4 | 25.6529 (+63 %) | **15.7182** (100 %) |

## TPOT — Time Per Output Token (s) — lower is better

| # | enforce-eager | compiled |
|---|---:|---:|
| 1 | 0.1247 (+1181 %) | **0.0097** (100 %) |
| 2 | 0.2095 (+879 %) | **0.0214** (100 %) |
| 3 | 0.1608 (+1375 %) | **0.0109** (100 %) |
| 4 | 0.2081 (+91 %) | **0.1090** (100 %) |

## Output token throughput (tok/s) — higher is better

| # | enforce-eager | compiled |
|---|---:|---:|
| 1 | 8.02 (−92 %) | **102.72** (100 %) |
| 2 | 133.76 (−90 %) | **1293.77** (100 %) |
| 3 | 6.22 (−93 %) | **91.56** (100 %) |
| 4 | 80.66 (−48 %) | **156.17** (100 %) |

## RPS — higher is better

| # | enforce-eager | compiled |
|---|---:|---:|
| 1 | 0.0632 (−93 %) | **0.9355** (100 %) |
| 2 | 0.8587 (−90 %) | **8.5876** (100 %) |
| 3 | 0.0658 (−88 %) | **0.5489** (100 %) |
| 4 | 0.6544 (−40 %) | **1.0832** (100 %) |

## Cold start

| Phase | enforce-eager | compiled |
|-------|--------------:|---------:|
| Model load → serving | 180 s | 285 s |
| CUDA graph capture | — | 85 s (2.37 GiB) |
| KV cache per GPU | 21.27 GiB | 18.61 GiB |
| GPU KV cache size | 656,967 tok | 574,611 tok |

## Conclusions

- **CUDA graphs dominate V4 serving performance.** Leaving `--enforce-eager` off is worth
  8–15× on TTFT, latency, TPOT and throughput for sequential and moderately concurrent
  traffic.
- **The advantage narrows at saturation.** At 20 concurrent runners on 45k-char prompts
  (scenario 4) the win drops to ~1.6–1.9×: with the GPU already busy, kernel-launch
  overhead matters less.
- **Cost of graphs:** +105 s cold start and 2.66 GiB less KV cache per GPU
  (656,967 → 574,611 tokens, i.e. max concurrency at 400k context 1.64× → 1.44×).
- **PoC throughput is unaffected** — 1536 nonces/min at batch 32 in both modes
  (see `README.md`).
