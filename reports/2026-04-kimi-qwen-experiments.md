# PoC v2 on vLLM 0.19.0 — Experiment Results (2026-04-07 … 2026-04-14)

## Summary

The [kaitaku.ai](https://github.com/kaitakuai) team (Pavlo [@clanster](https://github.com/clanster), Mykola [@baychak](https://github.com/baychak)) evaluated PoC v2 on the upstream vLLM 0.19.0 release and prepared the port that ships in [gonka-ai/vllm#29](https://github.com/gonka-ai/vllm/pull/29). The work covers:

- Port of PoC v2 from vLLM 0.15.1 → 0.19.0 (PR #29; image `ghcr.io/kaitakuai/vllm:v0.19.0-pocv2-alpha1`)
- PoC + inference benchmarks for **Qwen3-235B-A22B-Instruct-2507-FP8** on A100 / H100 / B200 (vLLM 0.15.1 alpha3) and on H100 under the new vLLM 0.19.0
- PoC + inference benchmarks for **Kimi K2.5** on 4×B200 and 8×H200 (INT4, vLLM 0.15.1 alpha1)
- PoC on **Kimi K2.5 NVFP4** on 4×B200 under vLLM 0.19.0 — first working configuration on Blackwell FP4 path
- PoC on **MiniMax-M2.7 (FP8)** on 4×RTX PRO 6000 Blackwell under vLLM 0.19.0 — first working configuration on workstation Blackwell (SM120) via TRITON FP8 MoE backend

---

## PoC nonces/min — Qwen3-235B-A22B-Instruct-2507-FP8

All runs use `seq_len=1024`, `k_dim=12`, 5 s warmup + 30 s measurement per batch size.

| GPU (TP) | vLLM | Image | Best batch | Nonces/min | Notes |
|----------|------|-------|-----------:|-----------:|-------|
| [4×A100 SXM4 80GB](https://github.com/kaitakuai/experiments/tree/main/2026-04/qwen235b-fp8-4xa100) | 0.15.1 alpha3 | `product-science/mlnode:3.0.13-alpha3` | 8/16 | **480** | `--gpu-memory-utilization 0.92 --max-num-seqs 128` |
| [4×H100 SXM 80GB](https://github.com/kaitakuai/experiments/tree/main/2026-04/qwen235b-fp8-4xh100-alpha3) | 0.15.1 alpha3 | `product-science/mlnode:3.0.13-alpha3` | 16 | **928** | same flags; OOM ≥ batch 32 |
| [4×H100 SXM 80GB](https://github.com/kaitakuai/experiments/tree/main/2026-04/qwen235b-fp8-4xh100-vllm019) | **0.19.0** | `vllm/vllm-openai:v0.19.0` + gonka-source MLNode | 16 | **960** (+3.4 %) | `--max-model-len 240000`|
| [2×B200 180GB](https://github.com/kaitakuai/experiments/tree/main/2026-04/qwen235b-fp8-2xb200) | 0.15.1 alpha3 | `product-science/mlnode:3.0.13-alpha3` | 64 | **1536** | 48.87 GiB KV cache, 179 GiB / GPU |

**Conclusion.** KV cache shrinks to 10.89 GiB / 242 912 tokens due to new CUDA-graph memory profiling.

---

## PoC nonces/min — Kimi K2.5

| Model | GPU (TP) | vLLM | Mode | Best batch | Nonces/min |
|-------|----------|------|------|-----------:|-----------:|
| Kimi K2.5 **INT4** | [4×B200](https://github.com/kaitakuai/experiments/tree/main/2026-04/kimi-k25-int4-4xb200) | 0.15.1 alpha1 | eager | 32 | 1024 |
| Kimi K2.5 **INT4** | [4×B200](https://github.com/kaitakuai/experiments/tree/main/2026-04/kimi-k25-int4-4xb200) | 0.15.1 alpha1 | compiled, `custom_ops=["all"]` | 32 | 1024 (= eager) |
| Kimi K2.5 **INT4** | [8×H200](https://github.com/kaitakuai/experiments/tree/main/2026-04/kimi-k25-int4-8xh200) | 0.15.1 alpha1 | eager | 16 | 1184 |
| Kimi K2.5 **INT4** | [8×H200](https://github.com/kaitakuai/experiments/tree/main/2026-04/kimi-k25-int4-8xh200) | 0.15.1 alpha1 | compiled, `custom_ops=["all"]` | 32 | **1216** |
| **Kimi K2.5 NVFP4** | [**4×B200**](https://github.com/kaitakuai/experiments/tree/main/2026-04/kimi-k25-nvfp4-4xb200) | **0.19.0** | compiled, `custom_ops=["all","-rms_norm"]` | **32** | **1937** ★ |

**Key new result (Pavlo, 2026-04-13).** Kimi K2.5 NVFP4 on vLLM 0.19.0 / 4×B200 — **1937 nonces/min**, the first stable Blackwell FP4 PoC configuration. Minimal compile fix required:

```
--compilation-config '{"custom_ops": ["all", "-rms_norm"]}'
```

Without `-rms_norm` the kernel fails with `NotImplementedError: "rms_norm_kernel" not implemented for 'Byte'`.

**INT4 ↔ FP4 cross-validation.** Tested whether the two quantizations can share one PoC artifact — they cannot (L2 distance = 1.51 between nonce vectors). Recommendation: ship INT4 and FP4 as two separate models; FP4 unlocks Blackwell speedups without blocking INT4 rollout on Hopper/Ampere.

---

## PoC nonces/min — MiniMax-M2.7 (FP8)

| Model | GPU (TP) | vLLM | Backend | Best batch | Nonces/min |
|-------|----------|------|---------|-----------:|-----------:|
| **MiniMax-M2.7 FP8** | [**4×RTX PRO 6000 Blackwell**](https://github.com/kaitakuai/experiments/tree/main/2026-04/minimax-m27-fp8-4xrtxpro6000) (SM120) | **0.19.0** | TRITON FP8 MoE (fallback) | **8** | **848** |

**First working MiniMax-M2.7 PoC on workstation Blackwell.** Both FlashInfer FP8 MoE kernels (0.6.6, 0.6.7.post3) and DeepGEMM reject MiniMax's 128×128 block-wise FP8 layout on SM120 — only the universal TRITON fallback works in vLLM 0.19.0. Throughput is ~30–50 % below what vendor-tuned kernels would deliver on the same silicon, but **PoC v2 self-validation passes byte-for-byte** (`p_value = 1.0`, async↔sync determinism confirmed on 424/424 nonces).

Comparison: **~83 % of the 4×B200 Kimi INT4 baseline** (848 vs 1024 nonces/min) despite using a smaller workstation Blackwell — the PoC path is bound by MoE expert dispatch latency, not raw matmul. Artifact `artifacts/nonces_1000.json` (1 056 nonces collected at sustained 823 nonces/min) ships alongside the report.

---

## Inference throughput — Kimi K2.5 INT4 (compressa-perf, 6 scenarios)

Peak totals (scenario 5: 45k-char prompts, 20 runners, 1000 max_tokens):

| GPU (TP) | Mode | TTFT (s) | TPOT (ms) | **Peak total tok/s** |
|----------|------|---------:|----------:|---------------------:|
| [4×B200](https://github.com/kaitakuai/experiments/tree/main/2026-04/kimi-k25-int4-4xb200) | eager | 8.18 | 46 | 8 769 |
| [4×B200](https://github.com/kaitakuai/experiments/tree/main/2026-04/kimi-k25-int4-4xb200) | compiled | 6.88 | 41 | **11 534** (+32 %) |
| [8×H200](https://github.com/kaitakuai/experiments/tree/main/2026-04/kimi-k25-int4-8xh200) | eager | 6.16 | 88 | 5 326 |
| [8×H200](https://github.com/kaitakuai/experiments/tree/main/2026-04/kimi-k25-int4-8xh200) | compiled | 5.74 | 38 | **12 816** (+141 %) |

Compiled mode wins **every** inference scenario — 2.4×–6.0× on single-runner long-context workloads on 8×H200 (NCCL all-reduce overhead in eager mode vanishes under CUDA graphs). Production recommendation: always enable `--compilation-config '{"custom_ops": ["all"]}'` with `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900` and a 1–2-request warmup.

---

## vLLM 0.19.0 port

Artifacts:

| Resource | Link |
|----------|------|
| PR | [gonka-ai/vllm#29](https://github.com/gonka-ai/vllm/pull/29) |
| Branch | [`kaitakuai/vllm:mb/feat/port-pocv2-vllm-0.19`](https://github.com/kaitakuai/vllm/tree/mb/feat/port-pocv2-vllm-0.19) |
| Alpha image | `ghcr.io/kaitakuai/vllm:v0.19.0-pocv2-alpha1` |

Port validated end-to-end: standard Qwen PoC path works out of the box; first NVFP4 run confirmed on 4×B200. Outstanding items tracked in [kaitakuai/vllm#8](https://github.com/kaitakuai/vllm/pull/8).

---

## Participants

[kaitakuai](https://github.com/kaitakuai) ([@baychak](https://github.com/baychak), [@clanster](https://github.com/clanster)).
Gonka core team ([@gmorgachev](https://github.com/gmorgachev), [@tamazgadaev](https://github.com/tamazgadaev)).

| Participant | GitHub | Role | Contribution |
|-------------|--------|------|--------------|
| Pavlo | [@clanster](https://github.com/clanster) | kaitakuai | Kimi NVFP4 on 4×B200 + vLLM 0.19.0 (first working Blackwell FP4 PoC), compilation-config minimal fix, INT4↔FP4 cross-validation experiment |
| Mykola | [@baychak](https://github.com/baychak) | kaitakuai | PoC v2 → vLLM 0.19.0 port (PR #29), slim 6.28 GB image, Qwen/Kimi benchmarks across A100/H100/H200/B200, MiniMax-M2.7 FP8 benchmark on RTX PRO 6000 Blackwell (SM120 TRITON path), dual-logprobs migration design |
| Gleb | [@gmorgachev](https://github.com/gmorgachev) | Gonka core team | Review and acceptance of experiments and PR #29 |
| Tamaz | [@tamazgadaev](https://github.com/tamazgadaev) | Gonka core team | Upstream PoC v2 architecture (PR #24 baseline for the 0.19.0 port), review and acceptance |
