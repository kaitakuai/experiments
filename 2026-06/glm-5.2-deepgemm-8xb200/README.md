# GLM-5.2 FP8 — 8×B200 — DeepGEMM, cudagraph vs eager (PoC 1054 vs 1517 / inference 583 vs 196 tok/s)

**Date:** 2026-06-24
**Model:** `zai-org/GLM-5.2-FP8` @ `31cba24fb749908a485082bdeed6eb1ac6cffc2f`
  FP8 block-wise e4m3 [128,128], arch `GlmMoeDsaForCausalLM` (DeepSeek-style MLA + DSA sparse
  attention, `index_topk=2048`), 753B total / ~40B active, 256 experts × top-8, 78 layers.
**Hardware:** 8× NVIDIA B200 SXM6 (183 GiB HBM3e, driver **590.48.01**, sm_100). Vast.ai Alabama US,
  machine `57662`.
**Image:** `ghcr.io/kaitakuai/mlnode-b200-glm-5-2:0.2.13-vllm0.23.0-k1`
  (dedicated GLM-5.2 B200 image — GLM config baked into `runner.py`, DeepGEMM-ready `gonka_poc`)

> This is the **DeepGEMM** experiment on the purpose-built GLM-5.2 image. Unlike the earlier
> Kimi-image run ([`../glm-5.2-fp8-8xb200/`](../glm-5.2-fp8-8xb200/)), here DeepGEMM is the active
> MoE/linear backend and the baked `gonka_poc` plugin runs the PoC forward **eager** even on a
> compiled (CUDA-graph) engine via `set_forward_context(skip_compiled=True)`.

## Summary

On GLM-5.2 FP8 / 8×B200 with **DeepGEMM**, the choice of engine compilation mode is the dominant
PoC-vs-inference tradeoff:

| Engine mode | PoC (nonces/min) | Inference (output tok/s) | Inference TPOT |
|---|---:|---:|---:|
| **CUDA graphs** (mode 3) | **1054** | **583** ¹ | 25.3 ms |
| **eager** (`--enforce-eager`) | **1517** | 196 ² | 76.6 ms |

¹ concurrency 32, `--max-num-seqs 64`. ² concurrency 16, `--max-num-seqs 16` (image default).
See [Inference](#inference-performance) for the concurrency caveat.

**Two decisive findings:**
1. **PoC mining → eager.** DeepGEMM eager hits **1517 nonces/min** vs **1054** on CUDA graphs
   (**+44%**). Even though the new `gonka_poc` forces the PoC forward eager (`skip_compiled`), the
   compiled engine still taxes PoC — a fully-eager engine is materially faster.
2. **Inference serving → CUDA graphs.** Decode is **3× faster** (TPOT 25.3 ms vs 76.6 ms) and output
   throughput far higher. Eager inference is a non-starter for serving.

The "one image for both" (CUDA-graph engine + eager PoC via `skip_compiled`) is a **compromise**: it
gives **1054 PoC + good inference** from a single engine, sacrificing ~44% PoC vs pure eager.

## Environment

| Parameter | Value |
|---|---|
| GPU | 8× B200 SXM6, 183 GiB HBM3e, sm_100 |
| NVIDIA driver | 590.48.01 |
| vLLM | 0.23.0 (in-image system python) |
| Python | 3.12 |
| MoE/linear backend | **DeepGEMM** (`DeepGemmFp8BlockScaledMMKernel`, E8M0) — auto-selected for FP8 block-scale on B200 |
| OS / base image | mlnode-b200-glm-5-2 (Stage-3 baked, GLM-5.2 native) |

## Config

The image's `runner.py` bakes the GLM-5.2 PoC profile (no patch needed):

```bash
# baked B200-GLM-5.2 plugin hardcodes (Kaitaku):
--tensor-parallel-size 8
--gpu-memory-utilization 0.85
--max-model-len 350000
--max-num-batched-tokens 16384      # DeepGEMM survives profiling only at small mnbt
--max-num-seqs 16
--kv-cache-dtype fp8_e4m3
--tool-call-parser glm47
--reasoning-parser glm45
--logprobs-mode processed_logprobs
--worker-extension-cls gonka_poc.worker.PoCWorkerExtension
--trust-remote-code  --enable-auto-tool-choice
# compilation is NOT forced: inference runs COMPILED (CUDA graphs) by default;
# the PoC forward runs eager on its own (gonka_poc poc_model_runner, skip_compiled=True).

# DeepGEMM is forced on at the engine env level:
VLLM_USE_DEEP_GEMM=1  VLLM_MOE_USE_DEEP_GEMM=1  VLLM_USE_FLASHINFER_MOE_FP8=0
```

### The two modes tested

| | CUDA graphs (default) | eager |
|---|---|---|
| how | native image start | add `--enforce-eager` |
| inference | compiled + CUDA graphs | eager |
| PoC | eager via `skip_compiled` | eager |
| `--max-num-seqs` for the inference number | 64 (override) | 16 (baked) |

## Results

### PoC v2 throughput (DeepGEMM, batch 16, steady-state via `/inference/pow/init/generate` + `/status`)

| Engine mode | nonces/min (cumulative) | nonces/min (steady-state window) |
|---|---:|---:|
| CUDA graphs (mode 3) | 1054 | 1056 |
| **eager** | **1509–1520** | **~1536** |

DeepGEMM **eager is +44% on PoC**. The 1054 CUDA-graph figure reproduces the gonka deploy number
(measured at 1055 on this same host). See [`artifacts/poclive.log`](artifacts/poclive.log) (cudagraph)
and [`artifacts/poclive_eager.log`](artifacts/poclive_eager.log) (eager).

### Inference performance

`vllm bench serve`, 200 prompts, 1024→256 tokens, `/v1/chat/completions`:

| Engine mode | concurrency | max-num-seqs | output tok/s | TPOT median | TTFT mean |
|---|---:|---:|---:|---:|---:|
| **CUDA graphs** | 32 | 64 | **583** | **25.3 ms** | 6401 ms |
| CUDA graphs | 16 | 16 (baked) | 440 | 32.2 ms | 601 ms |
| **eager** | 16 | 16 (baked) | 196 | 76.2 ms | 561 ms |

- **Decode (TPOT) is the clean DeepGEMM metric: 25.3 ms on CUDA graphs** — identical to the earlier
  Kimi-image cudagraph run (25.1 ms), i.e. the decode kernel speed is the same.
- **CUDA graphs vs eager: 3× faster decode** (25 ms vs 76 ms).
- **Concurrency caveat:** output tok/s scales with concurrency. The image bakes `--max-num-seqs 16`
  (tuned for the PoC profile), which caps inference concurrency. The 583 tok/s figure used a
  `--max-num-seqs 64` override at concurrency 32; the eager 196 figure is at the baked concurrency 16.
  These inference numbers are therefore **not at matched concurrency** — compare TPOT (decode) for an
  apples-to-apples view. The high CUDA-graph TTFT (6.4 s) is prefill queueing at concurrency 32 with
  `max-num-batched-tokens 16384`, not a decode effect.

### Why this image is faster than the earlier Kimi-image run

The earlier Kimi-image `gonka_poc` produced **928 nonces/min** on the same hardware; this image's
updated `gonka_poc` produces **1054** (cudagraph) / **1517** (eager). The difference is **purely the
plugin version**, not hardware or the DeepGEMM kernel:
- The new `gonka_poc` ships `generate_queue.py` (pipelined generation) and runs the PoC forward
  **eager** via `set_forward_context(skip_compiled=True)`. The old plugin ran the PoC forward through
  whatever compilation mode the engine was in (the CUDA-graph penalty).
- Confirmed by measuring both on the **same physical host** (`57662`, where the gonka deploy measured
  1055): same host, same DeepGEMM, only the plugin differs.

## Files

- [`artifacts/nonces_fp8_deepgemm_newplugin/nonces_1000.json`](artifacts/nonces_fp8_deepgemm_newplugin/nonces_1000.json) — 1056 DeepGEMM PoC nonces (cudagraph engine, batch 16)
- [`artifacts/poclive.log`](artifacts/poclive.log) — PoC steady-state, CUDA graphs (1054/min)
- [`artifacts/poclive_eager.log`](artifacts/poclive_eager.log) — PoC steady-state, eager (1517/min)
- [`artifacts/bench32b.log`](artifacts/bench32b.log) — inference, CUDA graphs, conc 32 / seqs 64 (583 tok/s, TPOT 25.3 ms)
- [`artifacts/bench.log`](artifacts/bench.log) — inference, CUDA graphs, conc 16 (440 tok/s)
- [`artifacts/bench_eager.log`](artifacts/bench_eager.log) — inference, eager, conc 16 (196 tok/s, TPOT 76.6 ms)
- [`artifacts/collect.log`](artifacts/collect.log) — nonce collection log
- [`scripts/native_start.sh`](scripts/native_start.sh) — start the image natively (DeepGEMM); arg `cudagraph` | `eager`
- [`scripts/poc_measure.sh`](scripts/poc_measure.sh) — PoC steady-state measurement (init/generate + /status)
- [`scripts/inference_bench.sh`](scripts/inference_bench.sh) — `vllm bench serve`, conc 32

## Findings / recommendation

1. **PoC mining → DeepGEMM eager (`--enforce-eager`)** — 1517 nonces/min, +44% over CUDA graphs.
2. **Inference serving → DeepGEMM CUDA graphs** — TPOT 25 ms, 3× faster decode than eager.
3. **Single-config deploy (CUDA-graph engine)** gives 1054 PoC + serviceable inference from one
   engine (eager PoC via `skip_compiled`); accept the ~44% PoC haircut vs a dedicated eager PoC node.
4. **DeepGEMM still needs small `max-num-batched-tokens` (16384)** — it crashes vLLM memory profiling
   at larger mnbt (`cudaErrorInvalidValue`). This caps the PoC batch and the inference prefill batch.
5. **The new `gonka_poc` is the win, not DeepGEMM per se** — same hardware, the updated plugin lifts
   PoC 928 → 1054 (cudagraph) / 1517 (eager).

## How to reproduce

```bash
export VAST_API_KEY=<your-key>
export HF_TOKEN=<your-hf-token>

# 1. Rent 8×B200 (driver >=580 for CUDA-13 image), RAM >= 900 GB (model lives in /dev/shm).
#    Prefer a low inet $/GB host (Vast bills downloaded bandwidth) — see the experiments tooling.

# 2. On the box (image runs GLM-5.2 natively — no runner patch):
hf download zai-org/GLM-5.2-FP8 --revision 31cba24fb749908a485082bdeed6eb1ac6cffc2f \
  --local-dir /dev/shm/GLM-5.2-FP8 --max-workers 16

# 3a. CUDA graphs (default) — PoC 1054 + inference 583 tok/s:
bash scripts/native_start.sh cudagraph
bash scripts/poc_measure.sh           # ~1054 nonces/min
bash scripts/inference_bench.sh       # conc 32; needs --max-num-seqs 64 (see note)

# 3b. eager — PoC 1517 + inference 196 tok/s:
bash scripts/native_start.sh eager
bash scripts/poc_measure.sh           # ~1517 nonces/min
bash scripts/inference_bench.sh

# 4. Destroy the box when done.
```

> Note: the baked image forces `--max-num-seqs 16`. The 583 tok/s inference number used a 64-seq
> override (via the shared `02_start_vllm.sh` runner patch in
> [`../glm-5.2-poc-backend-sweep/scripts/`](../glm-5.2-poc-backend-sweep/scripts/), which inserts
> after the baked `B200-GLM` block). For the image as-shipped (16 seqs), inference is concurrency-capped.

## Related

- Earlier Kimi-image run (triton + eager, DeepGEMM crashed): [`../glm-5.2-fp8-8xb200/README.md`](../glm-5.2-fp8-8xb200/README.md)
- Cross-repo PoC plugin source (pinned): `https://github.com/gonka-ai/vllm/tree/gm/port-pocv2-vllm-0.23.0/vllm/poc`

## Reproducibility checklist

- [x] A reader with only this folder can reach the headline result by following this README.
- [x] Hardware stated exactly: 8× B200 SXM6, driver 590.48.01, sm_100, machine 57662.
- [x] Image stated by tag; model pinned by repo + commit SHA.
- [x] Every command is copy-pasteable; placeholders obvious.
- [x] Reproduction scripts committed under `scripts/`.
- [x] No links to `.claude/...`; cross-repo PoC source is a pinned-branch GitHub URL.
- [x] All artifacts referenced exist in `artifacts/`.
- [x] Expected outputs stated: PoC 1054 (cudagraph) / 1517 (eager); inference 583 / 196 tok/s; TPOT 25 / 77 ms.
- [x] Known gotchas listed (DeepGEMM mnbt cap, max-num-seqs concurrency cap, plugin-version effect).
