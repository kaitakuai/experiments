# Inference Validation: DeepSeek-V4-Flash-0731 — NVFP4 quant vs stale-checkpoint fraud

**Date:** 2026-08-01
**Model honest:** [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (284B total / 13B active, 43 layers, 256 routed experts, FP4 experts + FP8 rest, ~167 GB)
**Fraud A (quantization):** [`MJPansa/DeepSeek-V4-Flash-0731-NVFP4`](https://huggingface.co/MJPansa/DeepSeek-V4-Flash-0731-NVFP4) (~176 GB)
**Fraud B (stale checkpoint):** [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) — the previous release, replayed against the 0731 validator
**vLLM:** `0.25.1+gonka.sampler1` (image `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash:0.2.13-vllm0.25.1-overlay-k4`)
**Methodology:** gonka `enforced_tokens` replay + `distance2`, 1000 prompts × 5 languages (sp/en/ch/ar/hi, 200 each), both logprobs modes

## Experiment matrix

| Experiments | GPU Honest | Version Honest | Model Honest | GPU Fraud | Version Fraud | Model Fraud | GPU Validator | Version Validator | Model Validator | Link to artifacts | graph |
|---|---|---|---|---|---|---|---|---|---|---|---|
| processed_logprobs | 1×B300 SXM6 (TP=1) | 0.25.1 | DeepSeek-V4-Flash-0731 | 1×B300 SXM6 (TP=1) | 0.25.1 | DeepSeek-V4-Flash-0731-NVFP4 | 2×H200 (TP=2) | 0.25.1 | DeepSeek-V4-Flash-0731 | [artifacts](exp1-processed-logprobs/artifacts/) | [graph](exp1-processed-logprobs/artifacts/length_vs_distance.png) |
| processed_logprobs | 1×B300 SXM6 (TP=1) | 0.25.1 | DeepSeek-V4-Flash-0731 | 1×B300 SXM6 (TP=1) | 0.25.1 | DeepSeek-V4-Flash (previous release) | 2×H200 (TP=2) | 0.25.1 | DeepSeek-V4-Flash-0731 | [artifacts](exp1-processed-logprobs/artifacts/) | [graph](exp1-processed-logprobs/artifacts/length_vs_distance.png) |
| raw_logprobs | 1×B300 SXM6 (TP=1) | 0.25.1 | DeepSeek-V4-Flash-0731 | 1×B300 SXM6 (TP=1) | 0.25.1 | DeepSeek-V4-Flash-0731-NVFP4 | 2×H200 (TP=2) | 0.25.1 | DeepSeek-V4-Flash-0731 | [artifacts](exp2-raw-logprobs/artifacts/) | [graph](exp2-raw-logprobs/artifacts/length_vs_distance.png) |
| raw_logprobs | 1×B300 SXM6 (TP=1) | 0.25.1 | DeepSeek-V4-Flash-0731 | 1×B300 SXM6 (TP=1) | 0.25.1 | DeepSeek-V4-Flash (previous release) | 2×H200 (TP=2) | 0.25.1 | DeepSeek-V4-Flash-0731 | [artifacts](exp2-raw-logprobs/artifacts/) | [graph](exp2-raw-logprobs/artifacts/length_vs_distance.png) |

Honest and both fraud variants share the same executor hardware and the same validator, so the only variable is what the executor serves.

## Results

Full sample (1000 per group):

| Mode | Fraud | Honest mean | Fraud mean | Separation | F1 | TP | FP | TP@FP=1% | TP@FP=5% |
|---|---|---|---|---|---|---|---|---|---|
| processed | NVFP4-0731 | 0.02539 | 0.03454 | 1.36× | 0.664 | 98.1% | 97.2% | 11.9% | 31.7% |
| processed | old V4-Flash | 0.02539 | 0.07455 | **2.94×** | 0.869 | 78.8% | 2.5% | 75.5% | 79.8% |
| raw | NVFP4-0731 | 0.02913 | 0.03918 | 1.34× | 0.742 | 78.4% | 32.9% | 17.9% | 34.7% |
| raw | old V4-Flash | 0.02913 | 0.08603 | **2.95×** | 0.911 | 84.9% | 1.5% | 84.2% | 86.0% |

Restricted to answers with ≥100 generated tokens:

| Mode | Fraud | n honest / fraud | Honest mean ± std | Fraud mean ± std | F1 | TP | FP | TP@FP=1% | TP@FP=5% |
|---|---|---|---|---|---|---|---|---|---|
| processed | NVFP4-0731 | 754 / 764 | 0.02866 ± 0.00825 | 0.03943 ± 0.01127 | 0.723 | 84.3% | 49.6% | 14.0% | 37.3% |
| processed | old V4-Flash | 754 / 711 | 0.02866 ± 0.00825 | 0.08935 ± 0.02523 | 0.974 | 97.0% | 2.1% | 94.2% | 97.7% |
| raw | NVFP4-0731 | 754 / 757 | 0.03212 ± 0.00657 | 0.04341 ± 0.00841 | 0.808 | 92.7% | 37.0% | 20.2% | 37.6% |
| raw | old V4-Flash | 754 / 716 | 0.03212 ± 0.00657 | 0.09973 ± 0.01210 | **1.000** | **100%** | **0.0%** | **100%** | **100%** |

**Token mismatches: 0 out of 6000 replays.**

## Key findings

1. **A stale checkpoint is ~3× easier to detect than a quantization swap.** Serving the previous V4-Flash release instead of 0731 lands at 2.94–2.95× the honest mean; NVFP4 lands at 1.34–1.36×. In raw mode restricted to answers ≥100 tokens the stale-checkpoint case separates **perfectly** — F1 = 1.000, zero false positives, zero misses. The two fraud classes need different thresholds; one number cannot serve both.

2. **NVFP4 remains close to undetectable, and the community quant reproduces NVIDIA's exactly.** MJPansa's conversion receipt names `reference_nvfp4_revision = e3cd60e7de…`, the same snapshot as `nvidia/DeepSeek-V4-Flash-NVFP4` we measured in the previous run, and the numbers agree: 1.34–1.36× here vs 1.31–1.34× there. Both quantize only `layers.*.ffn.experts` routed projections and leave attention and shared experts alone.

3. **The honest noise floor is a property of the hardware pair, not the checkpoint.** processed 0.02539 here vs 0.02516 for the previous model (+0.9%), raw 0.02913 vs 0.02758 (+5.6%) — measured on a *different physical B300*. Box-to-box and release-to-release variation is inside 1–6%, so the baseline transfers across model versions.

4. **raw beats processed, again via variance rather than means.** Separation ratios barely move between modes (1.36× → 1.34×), but honest std drops from 0.01078 to 0.00975 (and to 0.00657 excluding short answers), which is what actually lifts F1.

5. **The NVFP4 checkpoint is larger than the FP8 original** — 175.6 GB vs 166.9 GB. With `group_size: 16`, the per-group scale metadata outweighs the saving from 8→4 bits on the expert weights.

## Provenance check

The honest repo was modified after the fraud quant was published, which would confound the comparison if the weights had changed. Commit history rules it out:

```
2026-08-01T03:07  7872f01b1d  add sglang cookbook to model card (#20)   <- README only
2026-07-31T12:02  9e165c30e2  Release DeepSeek-V4-Flash-0731            <- weights final here
```

`conversion-receipt.json` in the fraud repo records `source_revision = 9e165c30e2704aec5d9d593cce3eebd58bbef1cb` — the release commit. No revision mismatch.

## Configuration

Identical on both sides except TP and model path:

```
--kv-cache-dtype fp8          # mandatory: FlashMLA V4 coerces KV to fp8_ds_mla and asserts without it
--max-model-len 32768
--gpu-memory-utilization 0.92
--max-num-seqs 64
--trust-remote-code
--logprobs-mode processed_logprobs
```

Executor B300 selects `FLASHINFER_TRTLLM_MXFP4_MXFP8` (honest) / `FLASHINFER_TRTLLM` NvFp4 (fraud); validator H200 selects `MARLIN` Mxfp4. `tokenizer_mode='deepseek_v4'` auto-selected, torch.compile disabled on both sides via auto-enabled `VLLM_USE_BREAKABLE_CUDAGRAPH`. No speculative decoding.

Request parameters, identical everywhere:

```python
{"temperature": 0.7, "seed": 1, "max_tokens": 1000, "top_logprobs": 4,
 "repetition_penalty": 1.2, "top_k": 40, "top_p": 0.95,
 "skip_special_tokens": False,
 "logprobs_mode": "processed_logprobs" | "raw_logprobs"}
```

`logprobs_mode` is a per-request override in this build, so both modes were collected against one loaded model with no restart.

## Known image bug (still present)

The image ships `libnvrtc.so.13` without the unversioned dev symlink; the FlashInfer JIT links `-lnvrtc` and the engine dies on Hopper with `/usr/bin/ld: cannot find -lnvrtc`. Fixed at container start with:

```bash
ln -sf /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so.13 /usr/local/cuda/lib64/libnvrtc.so && ldconfig
```

Does not surface on Blackwell.

## Artifacts

```
exp1-processed-logprobs/artifacts/
├── honest_b300_h200.jsonl              1000 items
├── fraud_nvfp4_b300_h200.jsonl         1000 items
├── fraud_oldmodel_b300_h200.jsonl      1000 items
├── length_vs_distance.png
├── length_vs_distance_min100tok.png
├── metrics.json
└── metrics_min100tok.json

exp2-raw-logprobs/artifacts/            same layout
logs/                                   vLLM startup on both sides + all six replay logs
honest_0731.png                         honest-only baseline, both modes side by side
```

Each JSONL row: `prompt`, `language`, `inference_result` (executor text + per-token top-4 logprobs), `validation_result` (validator logprobs on the enforced token sequence), `distance`, `n_compared`, `len_mismatch`, `sentinel_frac`, `val_sentinel_frac`.

## Comparison with previous runs

| Model | Fraud | Mode | Separation | F1 | Source |
|---|---|---|---|---|---|
| MiniMax-M2.7 FP8 | AWQ-4bit | raw | 2.1× | 0.980 | 2026-04 |
| GLM-5.2 FP8 | AWQ-INT4 | raw | — | 0.740 | 2026-06 |
| DeepSeek-V4-Flash | NVFP4 | raw | 1.31× | 0.737 | 2026-07 |
| DeepSeek-V4-Flash | W4A16-AutoRound | raw | 1.62× | 0.852 | 2026-07 |
| **DeepSeek-V4-Flash-0731** | **NVFP4** | **raw** | **1.34×** | **0.742** | this run |
| **DeepSeek-V4-Flash-0731** | **stale checkpoint** | **raw** | **2.95×** | **0.911** | this run |
| DeepSeek-V4-Flash-0731 | stale checkpoint | raw, ≥100 tok | 3.11× | **1.000** | this run |
