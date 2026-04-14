# Inference Benchmark (compressa-perf): MiniMax-M2.7 (FP8) on 4×RTX PRO 6000 Blackwell

**Date:** 2026-04-14
**Purpose:** Inference performance characterisation across 6 workload profiles using the official Gonka benchmark tool [`compressa-perf`](https://github.com/product-science/compressa-perf), per the methodology in [§3.2.3 of the gonka host guide](https://gonka.ai/host/benchmark-to-choose-optimal-deployment-config-for-llms/#323-measure-performance).

Companion to [`README.md`](README.md) (PoC v2 nonce-generation benchmark on the same model+hardware).

## Hardware

| Parameter | Value |
|-----------|-------|
| GPU | 4× NVIDIA RTX PRO 6000 Blackwell Server Edition (97 887 MiB each, SM 12.0) |
| NVIDIA Driver | 580.95.05 (CUDA 13.0 capable) |
| CPU | Intel(R) Xeon(R) Platinum 8470Q (208 vCPUs) |
| RAM | 1.0 TiB total, 961 GiB available |
| Local disk (model) | `/dev/md0` XFS, 350 GiB |
| Provider | partner GPU provider (bare-metal node) |

## Software

| Component | Version |
|-----------|---------|
| vLLM | 0.19.0 |
| PoC v2 overlay (vLLM) | `kaitakuai/vllm` `mb/feat/port-pocv2-vllm-0.19` @ `f9477d1` |
| MLNode packages | `kaitakuai/rtx-pro-6000` `vendor/gonka-source/mlnode/packages` |
| torch | 2.10.0+cu128 |
| triton | 3.6.0 |
| flashinfer-python / -cubin | 0.6.7.post3 |
| compressa-perf | latest from `git+https://github.com/product-science/compressa-perf.git` |
| Python | 3.12.3 (Miniconda) |
| OS | Ubuntu 22.04.5 LTS |

## Model

- **Name:** `MiniMaxAI/MiniMax-M2.7`
- **Quantization:** FP8 block-wise 128×128 (E4M3, dynamic activations; `gate`, `e_score_correction_bias`, `lm_head` unquantized)
- **Architecture:** MoE, `MiniMaxM2ForCausalLM`, 62 layers, 48 attention heads, 8 KV heads, max position 196 608
- **On-disk size:** 215 GiB / 125 safetensors shards
- **Attention backend:** FLASH_ATTN
- **MoE backend:** **TRITON** — the only working FP8 MoE backend on SM120 with this layout (FlashInfer FP8 MoE rejects 128×128 block layout; DeepGEMM doesn't support SM120)
- **Dense FP8 backend:** CutlassFP8ScaledMMLinearKernel

## Setup

The model and stack are produced by the `vllm0.19.0-poc` golden image (see `gonka-deploy/image/rtx_pro_6000/stage{1,2}_*.sh`) and started via MLNode exactly as in the [`README.md`](README.md) reproduction steps 0–6.

### Critical override: `--max-num-batched-tokens 65536`

A first attempt with vLLM defaults (`--max-num-batched-tokens 16384` derived from `max-model-len`) **crashed the engine** on the first long-prompt request:

```
RuntimeError: Worker failed with error 'Workspace is locked but allocation from
'modular_kernel.py:1063:_allocate_buffers' requires 6144.00 MB,
current size is 3072.00 MB. Workspace growth is not allowed after locking.'
```

Origin trace: `vllm/poc/poc_model_runner.py:239 execute_poc_forward → minimax_m2.py:541 forward → MoE expert dispatch → modular_kernel.py:1063`. The PoC v2 model runner pre-allocates a 3 GiB MoE workspace at warmup and locks it; compressa-perf scenarios 3–6 use 22 600-token prompts that need ~6 GiB. Once the engine dies, MLNode's watchdog cannot bring it back without a full `inference/down` + `inference/up`.

Fix: explicitly raise the upper bound at startup so the warmup pass allocates a workspace large enough for the longest prompt the benchmark sends.

```diff
   "additional_args": [
     "--served-model-name", "MiniMaxAI/MiniMax-M2.7",
     "--tensor-parallel-size", "4",
     "--gpu-memory-utilization", "0.92",
     "--max-num-seqs", "128",
+    "--max-num-batched-tokens", "65536",
     "--max-model-len", "32768",
     "--trust-remote-code",
     ...
   ]
```

After this change all 6 scenarios completed with **0 failed requests**.

### Install compressa-perf and download config

```bash
uv pip install --python /root/miniconda3/bin/python3.12 \
    git+https://github.com/product-science/compressa-perf.git
curl -sL https://raw.githubusercontent.com/product-science/inference-ignite/main/mlnode/packages/benchmarks/resources/config.yml \
    -o /root/compressa_config.yml
```

### Run the sweep

vLLM is exposed by MLNode on port 5001. We hit it directly (compressa-perf only needs an OpenAI-compatible endpoint).

```bash
compressa-perf measure-from-yaml \
    --no-sign \
    --node_url http://127.0.0.1:5001 \
    --model_name 'MiniMaxAI/MiniMax-M2.7' \
    /root/compressa_config.yml
```

### View results

```bash
compressa-perf list --show-metrics --show-parameters
```

Results are persisted in `compressa-perf-db.sqlite` in the working directory.

## Benchmark scenarios

From the upstream `config.yml` (6 profiles, `prompt_length` is in characters):

| # | num_prompts | prompt_len (chars) | num_tasks | num_workers | max_tokens | avg input tok | seed |
|---|------------:|-------------------:|----------:|------------:|-----------:|--------------:|-----:|
| 1 | 10 | 20 000 | 5 | 1 | 300 | ~10 071 | 1 |
| 2 | 100 | 2 000 | 200 | 30 | 300 | ~1 042 | 1 |
| 3 | 10 | 45 000 | 5 | 1 | 1 000 | ~22 612 | 1 |
| 4 | 10 | 45 000 | 10 | 5 | 1 000 | ~22 623 | 2 |
| 5 | 20 | 45 000 | 40 | 20 | 1 000 | ~22 605 | 3 |
| 6 | 40 | 45 000 | 60 | 10 | 1 000 | ~22 629 | 2 |

## Results

All scenarios completed with **`FAILED_REQUESTS = 0`**.

| # | TTFT (s) | TTFT₉₅ (s) | LATENCY (s) | LATENCY₉₅ (s) | TPOT (s) | THROUGHPUT (tok/s) | THROUGHPUT_INPUT (tok/s) | THROUGHPUT_OUTPUT (tok/s) | RPS | over 60s |
|---|---------:|-----------:|------------:|--------------:|---------:|-------------------:|-------------------------:|--------------------------:|----:|---------:|
| 1 | 0.81 | 1.35 | 4.44 | 4.99 | 0.0148 | 2 334.5 | 2 267.0 | 67.5 | 0.225 | 0 |
| 2 | 1.50 | 6.74 | 11.31 | 16.54 | 0.0377 | 3 416.2 | 2 653.3 | 762.9 | 2.545 | 0 |
| 3 | 1.48 | 1.83 | 13.32 | 14.51 | 0.0143 | 1 768.1 | 1 698.0 | 70.1 | 0.075 | 0 |
| 4 | 4.28 | 7.08 | 23.82 | 33.29 | 0.0309 | 4 314.9 | 4 172.5 | 142.3 | 0.184 | 0 |
| 5 | 9.07 | 19.42 | 50.08 | 67.28 | 0.0586 | 8 506.5 | 8 196.9 | 309.7 | 0.363 | **12** |
| 6 | 3.64 | 13.72 | 40.33 | 51.91 | 0.0452 | 5 555.3 | 5 344.6 | 210.7 | 0.236 | 0 |

Notes:
- `TTFT` / `LATENCY` / `TPOT` are means; `_95` columns are 95th percentile.
- `THROUGHPUT` = combined input+output tokens/s aggregated across all in-flight requests.
- `over 60s` counts requests that took > 60 s end-to-end (`LONGER_THAN_60_LATENCY`); 12 of 40 in scenario 5 because 20-way concurrency on 22 K-token prompts saturates the prefill path.
- Average output token count varies because compressa-perf stops at EOS; some completions are shorter than `max_tokens`.

## Where each scenario stresses the system

| # | Bottleneck | Observation |
|---|---|---|
| 1 | Single-request prefill on 10 K input | Best-case TTFT (0.81 s); decode runs at 67.5 tok/s, the cold steady-state for one in-flight request |
| 2 | Concurrent short prompts (30 workers, 1 K input each) | Highest **output** throughput (763 tok/s) — decode bandwidth scales linearly with active sequences while prefill is cheap |
| 3 | Single-request prefill on 22 K input + 1 K decode | Prefill dominates (LATENCY 13.3 s for 1 K decoded tokens); useful baseline for long-context single-user UX |
| 4 | Moderate concurrency (5 workers) on 22 K input | Throughput 4.3 K combined tok/s; LATENCY₉₅ 33 s — the practical ceiling for "agentic" multi-turn workloads with long contexts |
| 5 | Heavy concurrency (20 workers) on 22 K input | Combined throughput peaks at **8.5 K tok/s**, but TTFT₉₅ jumps to 19.4 s and 30 % of requests breach 60 s — server is queue-bound |
| 6 | Medium concurrency (10 workers) on 22 K input × 60 tasks | Sustained mid-load: 5.6 K combined tok/s, TTFT₉₅ 13.7 s — comfortable saturation point |

## Key observations

- **`--max-num-batched-tokens 65536` is mandatory for any compressa-style workload on this stack.** PoC v2's locked MoE workspace assumes warmup-time prompt sizes; raising the batched-token cap forces vLLM to size the workspace for full-length prompts at warmup. Without the override the engine dies on the first long prompt and MLNode cannot recover without a manual `inference/down`+`up`.
- **Short prompts give the highest output throughput** (scenario 2: 763 tok/s output). 30-way concurrency on 1 K-token prompts is the workload class where MiniMax-M2.7 on this hardware looks competitive.
- **Long prompts saturate prefill quickly.** Scenarios 4–6 trade output throughput for input throughput (8 K tok/s on input vs 300 tok/s on output in scenario 5). The MoE expert dispatch in TRITON costs more per token than the MoE backends used on Hopper/datacenter Blackwell, so the prefill ceiling is correspondingly lower.
- **No 503 / priority-gating events** — the PoC v2 priority gating that returns 503 on `/v1/chat/completions` only fires when a PoC generation session is active. The compressa benchmark ran with no concurrent PoC traffic, so all 320 requests were served normally.
- **Per-GPU memory** held steady at 89.8 GiB / 96 GiB through all 6 scenarios (overhead = workspace + KV cache + cuda graphs); peak GPU temperature 34–37 °C, no throttling observed.

## Comparison with the B200 reference

[`gonka-deploy/artifacts/experiments/compressa-perf-b200-inference.md`](../../../gonka-deploy/artifacts/experiments/compressa-perf-b200-inference.md) ran the same 6-scenario sweep against `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` on **2×B200**. Across-the-board the B200 reference shows ~3-5× higher combined throughput than RTX PRO 6000 Blackwell on these scenarios, primarily because:

1. B200 SM100 has access to FlashInfer TRTLLM/CUTLASS FP8 MoE kernels — vendor-tuned, ~2-3× faster than the TRITON fallback we are forced to use here.
2. B200's 183 GiB VRAM allows much higher `--max-num-seqs` and bigger KV cache, which directly improves the high-concurrency scenarios (5–6).
3. Qwen3-235B's FP8 layout is supported by FlashInfer; MiniMax-M2.7's 128×128 block FP8 is not.

This run is therefore a fair characterisation of MiniMax-M2.7 on **workstation Blackwell with the TRITON-only path**, not an upper bound on the model itself.

## Artifacts

- [`artifacts/compressa-perf.sqlite`](artifacts/compressa-perf.sqlite) — full sqlite database produced by `compressa-perf measure-from-yaml` (re-queryable with `compressa-perf list --show-metrics --show-parameters` after copying it back to a host with the tool installed)
- [`artifacts/compressa-results.txt`](artifacts/compressa-results.txt) — text dump of the same `list --show-metrics --show-parameters` output captured immediately after the run
