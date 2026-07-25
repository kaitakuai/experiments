# Inference Validation: DeepSeek-V4-Flash — NVFP4 and W4A16-AutoRound fraud

**Date:** 2026-07-25
**Model honest:** [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) (284B total / 13B active, MoE experts FP4 + rest FP8, 43 layers, 256 routed experts, ~149 GiB)
**Models fraud:** [`nvidia/DeepSeek-V4-Flash-NVFP4`](https://huggingface.co/nvidia/DeepSeek-V4-Flash-NVFP4), [`Intel/DeepSeek-V4-Flash-W4A16-AutoRound`](https://huggingface.co/Intel/DeepSeek-V4-Flash-W4A16-AutoRound)
**vLLM:** `0.25.1+gonka.sampler1` (image `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash:0.2.13-vllm0.25.1-overlay-k4`)
**Methodology:** gonka `enforced_tokens` replay + `distance2`, 1000 prompts × 5 languages (sp/en/ch/ar/hi, 200 each)

## Experiment matrix

| Experiments | GPU Honest | Version Honest | Model Honest | GPU Fraud | Version Fraud | Model Fraud | GPU Validator | Version Validator | Model Validator | Link to artifacts | graph |
|---|---|---|---|---|---|---|---|---|---|---|---|
| processed_logprobs | 1×B300 SXM6 (TP=1) | 0.25.1 | DeepSeek-V4-Flash | 1×B300 SXM6 (TP=1) | 0.25.1 | DeepSeek-V4-Flash-NVFP4 | 2×H200 (TP=2) | 0.25.1 | DeepSeek-V4-Flash | [artifacts](exp1-processed-logprobs/artifacts/) | [graph](exp1-processed-logprobs/artifacts/length_vs_distance.png) |
| processed_logprobs | 1×B300 SXM6 (TP=1) | 0.25.1 | DeepSeek-V4-Flash | 1×B300 SXM6 (TP=1) | 0.25.1 | DeepSeek-V4-Flash-W4A16-AutoRound | 2×H200 (TP=2) | 0.25.1 | DeepSeek-V4-Flash | [artifacts](exp1-processed-logprobs/artifacts/) | [graph](exp1-processed-logprobs/artifacts/length_vs_distance.png) |
| raw_logprobs | 1×B300 SXM6 (TP=1) | 0.25.1 | DeepSeek-V4-Flash | 1×B300 SXM6 (TP=1) | 0.25.1 | DeepSeek-V4-Flash-NVFP4 | 2×H200 (TP=2) | 0.25.1 | DeepSeek-V4-Flash | [artifacts](exp2-raw-logprobs/artifacts/) | [graph](exp2-raw-logprobs/artifacts/length_vs_distance.png) |
| raw_logprobs | 1×B300 SXM6 (TP=1) | 0.25.1 | DeepSeek-V4-Flash | 1×B300 SXM6 (TP=1) | 0.25.1 | DeepSeek-V4-Flash-W4A16-AutoRound | 2×H200 (TP=2) | 0.25.1 | DeepSeek-V4-Flash | [artifacts](exp2-raw-logprobs/artifacts/) | [graph](exp2-raw-logprobs/artifacts/length_vs_distance.png) |

Honest and fraud share the same executor hardware (1×B300) and the same validator (2×H200), so the only variable between rows of a mode is the executor's quantization.

## Results

Full sample (1000 honest / 1000 per fraud model):

| Experiment | Fraud model | Honest mean | Fraud mean | Separation | F1 | TP | FP | TP@FP=1% | TP@FP=5% |
|---|---|---|---|---|---|---|---|---|---|
| exp1 processed | NVFP4 | 0.02516 | 0.03376 | 1.34× | 0.666 | 98.7% | 97.7% | 10.5% | 27.3% |
| exp1 processed | W4A16-AutoRound | 0.02516 | 0.04341 | 1.73× | 0.736 | 68.1% | 16.9% | 34.2% | 51.4% |
| exp2 raw | NVFP4 | 0.02758 | 0.03624 | 1.31× | 0.737 | 71.8% | 23.0% | 20.9% | 44.7% |
| exp2 raw | W4A16-AutoRound | 0.02758 | 0.04472 | 1.62× | **0.852** | 78.6% | 5.9% | 64.6% | 77.6% |

Restricted to answers with ≥100 generated tokens (see *Short answers* below):

| Experiment | Fraud model | n honest / fraud | Honest mean ± std | Fraud mean ± std | F1 | TP | FP | TP@FP=1% | TP@FP=5% |
|---|---|---|---|---|---|---|---|---|---|
| exp1 processed | NVFP4 | 711 / 692 | 0.02932 ± 0.00903 | 0.04006 ± 0.01139 | 0.719 | 86.0% | 51.9% | 10.3% | 30.2% |
| exp1 processed | W4A16-AutoRound | 711 / 707 | 0.02932 ± 0.00903 | 0.05120 ± 0.01521 | 0.824 | 80.2% | 14.5% | 39.9% | 61.8% |
| exp2 raw | NVFP4 | 716 / 691 | 0.03122 ± 0.00506 | 0.04183 ± 0.00651 | 0.834 | 86.8% | 20.5% | 26.5% | 53.4% |
| exp2 raw | W4A16-AutoRound | 716 / 711 | 0.03122 ± 0.00506 | 0.05146 ± 0.00643 | **0.969** | 97.6% | 3.8% | **83.4%** | **98.3%** |

**Token mismatches: 0 out of 6000 replays.** `enforced_tokens` reproduced the executor's token sequence exactly in every run, across Blackwell→Hopper and across all three quantizations.

## Key findings

1. **NVFP4 is close to undetectable, W4A16 is not.** NVFP4 sits 1.31–1.34× above the honest mean; INT4 AutoRound sits 1.62–1.75×. In the best configuration (raw, ≥100 tokens) W4A16 reaches F1 = 0.969 / TP 98.3% at 5% FP, while NVFP4 only reaches F1 = 0.834 / TP 53.4%. NVIDIA's own card claims near-lossless quality (GPQA Diamond 0.894 → 0.891) and the logprob distance agrees — this quantization largely survives inference validation.

2. **raw_logprobs beats processed_logprobs, but not via the means.** The mean ratios are nearly identical between modes (1.31× vs 1.34× for NVFP4). What changes is spread: honest std drops from 0.01136 to 0.00921 (and to 0.00506 once short answers are excluded). Less overlap → better detection. Same conclusion as MiniMax-M2.7 and GLM-5.2, different mechanism than "the means separate more".

3. **In processed mode 50.3% of all logprob values are `-9999` sentinels and another 13.3% are exactly `0.0`.** They are deterministic and cancel between executor and validator, which is why processed compresses the signal. In raw mode there are zero sentinels and only 0.54% exact zeros — those are legitimate near-probability-1 tokens.

4. **Short answers are dead weight for `distance2`.** 28.4% of Alpaca prompts produce answers under 100 tokens (some 2–3 tokens: *"Identify the tone" → "Negative"*). The `max(100, n)` term in `distance2 = (Σd + 1) / (max(100, n)·top_k + 1)` pins the denominator at 401 for those, so both honest and fraud collapse toward the floor 1/401 = 0.002494 and contribute only noise. Excluding them lifts raw-mode F1 from 0.737 → 0.834 (NVFP4) and 0.852 → 0.969 (W4A16). Worth considering as a methodology change: either raise `max_tokens`-weighted sampling of prompts, or apply a minimum-length filter before thresholding.

5. **Cross-vendor MoE backends differ and cannot be equalised for this model.** Executor B300 selects `FLASHINFER_TRTLLM_MXFP4_MXFP8`, validator H200 selects `MARLIN` Mxfp4 — Triton has no SILU support for this path and DeepGEMM MXFP4 is sm_100+ only. This is baked into the honest baseline (0.0252 / 0.0276) and is identical for honest and fraud, so it does not bias the comparison.

## Two patches were required

### 1. `libnvrtc.so` missing in the image (validator, blocking)

FlashInfer JIT-compiles `fp8_blockscale_gemm_90` on Hopper and links with `-lnvrtc`. The image ships `libnvrtc.so.13` but not the unversioned dev symlink, so the engine dies at startup:

```
/usr/bin/ld: cannot find -lnvrtc
collect2: error: ld returned 1 exit status
RuntimeError: Engine core initialization failed.
```

Fix:
```bash
ln -sf /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so.13 /usr/local/cuda/lib64/libnvrtc.so
ldconfig
```

Never surfaced on B300 because the Blackwell path does not JIT this kernel.

### 2. BF16 `wo_a` fallback for AutoRound checkpoints (executor, blocking)

`Intel/DeepSeek-V4-Flash-W4A16-AutoRound` keeps `wo_a` at 16 bits (`extra_config: {'wo_a': {'bits': 16}}`), but `vllm/models/deepseek_v4/nvidia/flashmla.py::_o_proj` routes it unconditionally through `deep_gemm_fp8_o_proj`, which reads `wo_a.weight_scale_inv`:

```
File "vllm/models/deepseek_v4/nvidia/ops/o_proj.py", line 66, in deep_gemm_fp8_o_proj
    (wo_a.weight, wo_a.weight_scale_inv),
AttributeError: 'ColumnParallelLinear' object has no attribute 'weight_scale_inv'
```

Fix — fall back to the inverse-RoPE + bf16 einsum path the AMD backend already uses (`rocm_inv_rope_einsum` is pure torch/Triton, no AITER import at module level, and `_get_cached_wo_a_bf16` already has a branch for weights without `weight_scale_inv`):

```python
def _o_proj(self, o, positions):
    if not hasattr(self.wo_a, "weight_scale_inv"):
        from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_inv_rope_einsum
        z = rocm_inv_rope_einsum(self.rotary_emb, o, positions, self.rope_head_dim,
                                 self.n_local_groups, self.o_lora_rank, self.wo_a)
        return self.wo_b(z.flatten(1))
    return deep_gemm_fp8_o_proj(...)
```

This is the same idea as [vllm-project/vllm#45645](https://github.com/vllm-project/vllm/pull/45645) (closed unmerged, 2026-07-14). That PR's *dequantization* half is **not** needed: the June 2026 re-upload of the Intel checkpoint fixed [intel/auto-round#1776](https://github.com/intel/auto-round/issues/1776) — router gates and embeddings are no longer quantized (`ROUTER-GATE QUANTIZED TENSORS: 0`), and `wo_a` is plain 16-bit rather than MXFP4. Only the fallback branch is required.

## Configuration

Identical vLLM flags on both sides; only `--tensor-parallel-size` and the model path differ:

```
--kv-cache-dtype fp8          # mandatory: FlashMLA V4 coerces KV to fp8_ds_mla and asserts without it
--max-model-len 32768
--gpu-memory-utilization 0.92
--max-num-seqs 64
--trust-remote-code
--logprobs-mode processed_logprobs
```

`tokenizer_mode='deepseek_v4'` is auto-selected. `VLLM_USE_BREAKABLE_CUDAGRAPH=1` is auto-enabled on both sides (torch.compile pipeline disabled) — verified identical, so it is not a confounder. No speculative decoding (MTP off).

Request parameters, identical on executor and validator:

```python
{"temperature": 0.7, "seed": 1, "max_tokens": 1000, "top_logprobs": 4,
 "repetition_penalty": 1.2, "top_k": 40, "top_p": 0.95,
 "skip_special_tokens": False,
 "logprobs_mode": "processed_logprobs" | "raw_logprobs"}
```

`logprobs_mode` is a **per-request** override in this build, so both modes were collected against a single loaded model — no restart between passes. Sampling is identical across modes, so the generated token sequences are byte-identical between the processed and raw pass; only the reported logprobs differ.

## Artifacts

```
exp1-processed-logprobs/artifacts/
├── honest_b300_h200.jsonl              1000 items, 74 MB
├── fraud_nvfp4_b300_h200.jsonl         1000 items, 69 MB
├── fraud_w4a16_b300_h200.jsonl         1000 items, 72 MB
├── length_vs_distance.png
├── length_vs_distance_min100tok.png
├── metrics.json
└── metrics_min100tok.json

exp2-raw-logprobs/artifacts/
├── honest_b300_h200.jsonl              1000 items, 96 MB
├── fraud_nvfp4_b300_h200.jsonl         1000 items, 91 MB
├── fraud_w4a16_b300_h200.jsonl         1000 items, 93 MB
├── length_vs_distance.png
├── length_vs_distance_min100tok.png
├── metrics.json
└── metrics_min100tok.json
```

Each JSONL row: `prompt`, `language`, `inference_result` (executor text + per-token top-4 logprobs), `validation_result` (validator logprobs on the enforced token sequence), `distance`, `n_compared`, `len_mismatch`, `sentinel_frac`, `val_sentinel_frac`.

## Comparison with previous models

| Model | Fraud | Mode | Threshold | F1 | TP | Source |
|---|---|---|---|---|---|---|
| MiniMax-M2.7 FP8 | AWQ-4bit | raw | 0.092 | 0.980 | 96.6% @ 0.6% FP | 2026-04 |
| GLM-5.2 FP8 | AWQ-INT4 | raw | — | 0.740 | 79% @ 35% FP | 2026-06 |
| GLM-5.2 FP8 | AWQ-INT4 | processed | — | 0.540 | 85% @ 63% FP | 2026-06 |
| **DeepSeek-V4-Flash** | **NVFP4** | **raw** | 0.0339 | **0.737** | 71.8% @ 23% FP | this run |
| **DeepSeek-V4-Flash** | **W4A16-AutoRound** | **raw** | — | **0.852** | 78.6% @ 5.9% FP | this run |
| DeepSeek-V4-Flash | W4A16-AutoRound | raw, ≥100 tok | 0.0351 | 0.969 | 97.6% @ 3.8% FP | this run |
