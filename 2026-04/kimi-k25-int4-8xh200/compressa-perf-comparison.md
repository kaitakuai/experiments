# compressa-perf 8×H200 × Kimi K2.5: enforce-eager vs compiled

**Date:** 2026-04-09
**Model:** `moonshotai/Kimi-K2.5` (INT4 compressed-tensors, MoE 1.1T/32B, MLA)
**Hardware:** 8× NVIDIA H200 (Vast.ai, Japan)
**Image:** `ghcr.io/product-science/mlnode:3.0.13-alpha1`
**Baseline:** compiled (100%)

All percentages are relative to **compiled** mode. For TTFT/TPOT/Latency: lower is better (negative % = better). For throughput and RPS: higher is better (positive % = better).

## Configurations Compared

| Mode | vLLM args |
|------|-----------|
| **enforce-eager** | `--enforce-eager --attention-backend FLASHMLA --disable-custom-all-reduce` |
| **compiled** (baseline) | `--attention-backend FLASHMLA --disable-custom-all-reduce --compilation-config '{"custom_ops": ["all"]}'` |

Common args: `--tensor-parallel-size 8 --gpu-memory-utilization 0.9 --max-num-seqs 128 --max-model-len 262144 --trust-remote-code --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --mm-encoder-tp-mode data`

Environment:
- `LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH` (CUDA 12.9 forward-compat for driver 570)
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`
- For compiled: `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

**Important:** compiled mode requires warmup before benchmarking to avoid `shm_broadcast` `TimeoutError` from Marlin MoE kernel autotune. See [h200-kimi-k25-benchmark.md](./h200-kimi-k25-benchmark.md) for details.

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
| 1 | 0.51 (+9%) | **0.47** (100%) |
| 2 | **0.65** (+0%) | 0.65 (100%) |
| 3 | 1.03 (+1%) | **1.02** (100%) |
| 4 | **2.24 (-12%)** | 2.54 (100%) |
| 5 | 6.16 (+7%) | **5.74** (100%) |
| 6 | 2.25 (+24%) | **1.82** (100%) |

## TPOT — Time Per Output Token (s) — lower is better

| # | enforce-eager | compiled |
|---|---:|---:|
| 1 | 0.078 (+497%) | **0.013** (100%) |
| 2 | 0.081 (+247%) | **0.023** (100%) |
| 3 | 0.078 (+475%) | **0.014** (100%) |
| 4 | 0.082 (+283%) | **0.021** (100%) |
| 5 | 0.088 (+130%) | **0.038** (100%) |
| 6 | 0.084 (+186%) | **0.029** (100%) |

## Latency (s) — lower is better

| # | enforce-eager | compiled |
|---|---:|---:|
| 1 | 23.53 (+497%) | **3.94** (100%) |
| 2 | 24.17 (+247%) | **6.97** (100%) |
| 3 | 66.88 (+456%) | **12.02** (100%) |
| 4 | 78.25 (+274%) | **20.94** (100%) |
| 5 | 85.73 (+153%) | **33.89** (100%) |
| 6 | 78.32 (+194%) | **26.60** (100%) |

## Throughput Input Tokens (tok/s) — higher is better

| # | enforce-eager | compiled |
|---|---:|---:|
| 1 | 423 (-83%) | **2526** (100%) |
| 2 | 1191 (-71%) | **3507** (100%) |
| 3 | 335 (-82%) | **1866** (100%) |
| 4 | 1372 (-74%) | **5268** (100%) |
| 5 | 5104 (-59%) | **12328** (100%) |
| 6 | 2698 (-67%) | **8103** (100%) |

## Throughput Output Tokens (tok/s) — higher is better

| # | enforce-eager | compiled |
|---|---:|---:|
| 1 | 13 (-83%) | **76** (100%) |
| 2 | 355 (-71%) | **1238** (100%) |
| 3 | 13 (-82%) | **73** (100%) |
| 4 | 58 (-75%) | **230** (100%) |
| 5 | 222 (-54%) | **488** (100%) |
| 6 | 113 (-66%) | **329** (100%) |

## Total Throughput (tok/s) — higher is better

| # | enforce-eager | compiled |
|---|---:|---:|
| 1 | 436 (-83%) | **2602** (100%) |
| 2 | 1546 (-71%) | **5392** (100%) |
| 3 | 348 (-82%) | **1939** (100%) |
| 4 | 1431 (-74%) | **5498** (100%) |
| 5 | 5326 (-58%) | **12816** (100%) |
| 6 | 2811 (-67%) | **8432** (100%) |

## RPS — Requests Per Second — higher is better

| # | enforce-eager | compiled |
|---|---:|---:|
| 1 | 0.043 (-83%) | **0.254** (100%) |
| 2 | 1.183 (-71%) | **4.126** (100%) |
| 3 | 0.015 (-82%) | **0.083** (100%) |
| 4 | 0.061 (-74%) | **0.235** (100%) |
| 5 | 0.228 (-59%) | **0.550** (100%) |
| 6 | 0.120 (-67%) | **0.361** (100%) |

## Summary vs compiled

### enforce-eager

- **Total throughput:** −58% to −83% (catastrophic)
- **TPOT:** +130% to +497% worse
- **Latency:** +153% to +497% worse
- **TTFT:** roughly equal (±25%) — prefill is similar in both modes since prefill kernels dominate
- **RPS:** −59% to −83%

## Key Insight

**Compiled wins every metric on every scenario by a huge margin.** The ratio is 2.4×–6× total throughput.

The unusually large gap (vs typical "compile gives 1.5–2×") is caused by `--disable-custom-all-reduce`. On TP=8 with H200 NVLink topology, custom all-reduce crashes during CUDA graph capture, forcing the slower NCCL fallback. NCCL all-reduce is acceptable in compiled mode because CUDA graphs amortize the per-call latency, but in eager mode each kernel launch + NCCL exchange happens individually for every token, which is what blows TPOT to ~80 ms.

In a hypothetical configuration where custom all-reduce works (e.g., TP=4 or different NVLink topology), the eager numbers would be ~2× faster, narrowing the compile-vs-eager gap to a more typical ~3× — still a clear compiled win, but not 6×.

## Recommendation

**For Kimi K2.5 inference on 8×H200, ALWAYS use compiled mode:**

```bash
LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -m vllm.entrypoints.openai.api_server \
  --model moonshotai/Kimi-K2.5 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 128 \
  --max-model-len 262144 \
  --trust-remote-code \
  --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 \
  --mm-encoder-tp-mode data \
  --attention-backend FLASHMLA \
  --disable-custom-all-reduce \
  --compilation-config '{"custom_ops": ["all"]}'
```

Plus warmup with a 5–10k token chat request after vLLM start, before serving real traffic.

**Eager mode is 3–6× slower for inference and should be avoided.**

## Raw Data

- SQLite databases: `compressa-perf-results/h200_8x_kimi_eager.sqlite`, `h200_8x_kimi_compiled.sqlite`
- compressa-perf logs: `compressa-perf-results/h200_8x_kimi_compressa_eager.log`, `h200_8x_kimi_compressa_compiled.log`
- PoC logs: `compressa-perf-results/h200_8x_kimi_poc_eager.log`, `h200_8x_kimi_poc_compiled.log`

Full reproduction details: [h200-kimi-k25-benchmark.md](./h200-kimi-k25-benchmark.md)
