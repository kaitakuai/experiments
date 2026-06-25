# GLM-5.2 FP8 vs AWQ-INT4 — Inference Validation (processed vs raw logprobs)

**Date:** 2026-06-25
**Model (honest):** [`zai-org/GLM-5.2-FP8`](https://huggingface.co/zai-org/GLM-5.2-FP8) @ `31cba24fb749908a485082bdeed6eb1ac6cffc2f`
  FP8 block-wise e4m3 [128,128], arch `GlmMoeDsaForCausalLM`, 753B / ~40B active, 256 experts × top-8, DSA sparse attn.
**Model (fraud):** [`cyankiwi/GLM-5.2-AWQ-INT4`](https://huggingface.co/cyankiwi/GLM-5.2-AWQ-INT4) @ `431c1cd297c7a2f38d17c7b9520b10c15101df25` (W4A16 compressed-tensors, Marlin)
**Image:** `ghcr.io/kaitakuai/mlnode-b200-glm-5-2:0.2.13-vllm0.23.0-k1` (vLLM 0.23.0; has working `enforced_tokens`)
**Methodology:** Gonka inference validation — enforced_tokens replay + `distance2` metric (per-token L2-style on log-prob vectors).

## Summary

Cross-hardware honest baseline (8×B200 ↔ 8×H200) and AWQ-INT4 fraud, validated with the official enforced_tokens
method, in BOTH log-prob modes. **`raw_logprobs` separates honest/fraud better than `processed_logprobs`**
(F1 0.54 → 0.74, FP 63% → 35%), but the AWQ-INT4 quant is close enough to FP8 that **neither mode gives clean
detection** on this 753B MoE — consistent with the proposal pattern (Qwen3-235B fraud accuracy only 24%).

| Experiment | honest mean | fraud mean | threshold (F1-opt) | F1 | TP | FP | n (honest/fraud) |
|---|---|---|---|---|---|---|---|
| **exp1 — processed_logprobs** | 0.046 | 0.057 | 0.039 | 0.54 | 85% | 63% | 1000 / 477 |
| **exp2 — raw_logprobs** | 0.059 | 0.067 | 0.060 | **0.74** | 79% | **35%** | 1000 / 1000 |

Reference (MiniMax-M2.7, 2026-04): processed F1=0.767, raw F1=0.980 — a much worse quant, so raw cleanly separated.
Here the quant is high-fidelity, so the raw gain is real but partial.

## Why raw helps (and why it's not enough)

- **processed_logprobs** pads empty top-k slots with sentinels (`-9999`, ~60% of slots). These are deterministic and
  cancel between inference and validation, compressing `distance2` ~6× and making the F1-optimal threshold collapse to
  a low value with high false-positive rate.
- **raw_logprobs** has no sentinels (0% measured) — every top-k slot carries a real value, so the F1-optimal threshold
  sits between the distributions (lower FP).
- But the *gap between means* stays small (~0.008–0.011 in both modes): the AWQ-INT4 distribution is intrinsically
  close to FP8. Removing sentinels lifts honest and fraud roughly equally.

## KV-cache ablation (not the cause)

The wider-than-minimax honest baseline is **not** caused by the KV-cache dtype mismatch (B200 `fp8_e4m3` inference vs
H200 `bf16` validation). Paired ablation on 200 honest prompts: bf16-KV mean **0.0448** vs fp8_ds_mla-KV mean **0.0450**
(Δ +0.3%, noise). The honest spread comes from the model/quant + cross-hardware (MoE backend, attention kernels), not KV.

## Hardware & config

| | exp1 (processed) | exp2 (raw) |
|---|---|---|
| inference box | 8×B200 (TP=8, DeepGEMM, kv fp8_e4m3) | 8×H200 (TP=8, kv auto/bf16, forward-compat) |
| validation box | 8×H200 (TP=8, kv auto/bf16) | 8×B200 (TP=8, DeepGEMM, kv fp8_e4m3) |
| logprobs-mode | `processed_logprobs` | `raw_logprobs` |
| compilation | `{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}` | same |
| sampling | temperature 0.99, seed 42, top_logprobs 4, max_tokens 1000, repetition_penalty 1.2 | same |
| prompts | 1000 = 5 langs × 200 (sp/en/ch/ar/hi), Alpaca | same |

(Topology is mirrored between exp1/exp2 — the box that infers in one is the validator in the other. `distance2` is a
cross-hardware agreement metric, so direction does not change the conclusion; the KV ablation confirms it.)

## Artifacts

```
exp1-processed-logprobs/artifacts/
  honest_b200_h200.jsonl       (1000) inference B200-FP8 -> validation H200-FP8
  fraud_b200awq_h200.jsonl     (477)  inference B200-AWQ -> validation H200-FP8  [partial: diverted to KV ablation]
  length_vs_distance.png
exp2-raw-logprobs/artifacts/
  honest_h200_b200.jsonl       (1000) inference H200-FP8 -> validation B200-FP8
  fraud_h200awq_b200.jsonl     (1000) inference H200-AWQ -> validation B200-FP8
  length_vs_distance.png
```

Each JSONL row: `prompt`, `language`, `inference_result` (text + per-token logprobs), `validation_result`,
`inference_model`, `validation_model`, `request_params`. `distance2` is computed at analysis time
(`validation.analysis.process_data`).

## Key findings

1. **raw_logprobs > processed_logprobs** for fraud detection here (F1 0.54 → 0.74, FP 63% → 35%), confirming the
   sentinels compress signal — but the effect is partial, not the 0.77→0.98 jump seen on the weaker MiniMax quant.
2. **GLM-5.2 AWQ-INT4 is near-indistinguishable from FP8** under the chain method (processed): honest 0.046 vs fraud
   0.057. Expected for a high-quality W4A16 quant on a 753B MoE (cf. Qwen3-235B 24% fraud accuracy).
3. **KV-cache dtype is not the limiter** (ablation: Δ 0.3%).
4. enforced_tokens works correctly on the `mlnode-b200-glm-5-2` image (0 token mismatches across all runs); the
   kimi image lacks the serving-side enforced_tokens and crashes on it.
