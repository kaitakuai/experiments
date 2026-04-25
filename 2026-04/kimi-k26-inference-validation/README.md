# Kimi-K2.6 INT4 — Honest Cross-GPU Inference Validation

**Date:** 2026-04-24
**Model:** [`moonshotai/Kimi-K2.6`](https://huggingface.co/moonshotai/Kimi-K2.6) (native INT4 W4A16 QAT, compressed-tensors)
**Methodology:** Gonka inference validation via `run_inference_validation.py` + `distance2` metric

## Summary

Cross-GPU honest baseline for Kimi-K2.6 on gonka's inference-validation pipeline. No fraud variant (no AWQ/GPTQ exists for K2.6 yet), so this is baseline-only.

Two experiments to disentangle the effect of M2.7-style `-9999` sentinels on distance2:

| | exp1 (no sentinels) | exp2 (top_k=40, sentinels) |
|---|---|---|
| `top_k` in request | none | **40** |
| `top_p` in request | none | **0.95** |
| Sentinel positions | 0.0% | **90.5%** |
| distance2 mean | **0.1440** | **0.0232** |
| distance2 median | 0.1457 | 0.0231 |
| distance2 p95 | 0.1591 | 0.0351 |

**Ratio exp1/exp2 ≈ 6.2×** — sentinels collapse measured distance by ~6× because `-9999` tokens are deterministic and cancel out between identically-sampled inference & validator.

## Hardware

| Role | GPU | TP | VRAM |
|------|-----|-----|------|
| Inference | 4× NVIDIA B200 183GB | 4 | 732 GB |
| Validator | 8× NVIDIA H200 143GB | 8 | 1144 GB |

## Software

- vLLM: **0.19.0**
- torch: 2.10.0+cu129
- Python: 3.12.13
- gonka-ai/gonka branch: `tg/qwen_models`

## Why two experiments?

M2.7 inference validation JSONLs show ~70% positions with `{chosen: 0.0, "2": -9999, "0": -9999, "1": -9999}` sentinels. Same vLLM version (0.19.0), same script, same flags — but K2.6 out-of-the-box produces **no** sentinels.

Root cause: `generation_config.json` differs.

| Model | `generation_config.json` sampling defaults |
|-------|-------------------------------------------|
| M2.7 | `top_k=40, top_p=0.95, temperature=1.0, do_sample=true` |
| K2.6 | `max_length=262144, eos_token_id=163586` (no sampling filters) |

vLLM auto-loads these defaults. M2.7's `top_k=40` truncates non-top-40 logits to `-inf` **before** softmax → `max(lp, -9999)` clamp → sentinels. K2.6 has no such truncation → all top-4 logprobs are finite.

**exp2** adds `top_k=40, top_p=0.95` to the request payload explicitly, reproducing M2.7 behaviour on K2.6 data.

## Experiment structure

```
kimi-k26-inference-validation/
├── README.md                          (this file)
├── compare_exp1_exp2.py               (statistics dump)
├── plot_exp1_vs_exp2.py               (2-panel + overlay comparison)
├── plot_exp1_only.py                  (exp1 standalone scatter)
├── plot_exp2_only.py                  (exp2 standalone scatter)
├── length_vs_distance_exp1_vs_exp2.png (2-panel)
├── length_vs_distance_overlay.png     (overlay)
├── exp1-no-sentinels/
│   ├── length_vs_distance.png
│   ├── inspect_data.py
│   └── artifacts/
│       ├── config.json
│       └── honest_b200_h200.jsonl     (1000 items, ~170 MB)
└── exp2-top-k-40-sentinels/
    ├── length_vs_distance.png
    └── artifacts/
        ├── config.json
        └── honest_b200_h200.jsonl     (1000 items)
```

## Request parameters

### exp1
```python
{
  "model": "moonshotai/Kimi-K2.6",
  "temperature": 0.7,
  "seed": 1,
  "max_tokens": 1000,
  "logprobs": True,
  "top_logprobs": 4,
  "repetition_penalty": 1.2,
}
```

### exp2
Same as exp1 + `"top_k": 40, "top_p": 0.95`.

Both: 5 languages × 200 prompts = 1000 total. Prompts from Alpaca datasets (en, sp, ch, ar, hi).

## Per-language distance2 mean

| Language | exp1 | exp2 |
|----------|------|------|
| Spanish (sp) | 0.1474 | 0.0217 |
| English (en) | 0.1477 | 0.0226 |
| Chinese (ch) | 0.1486 | 0.0265 |
| Arabic (ar) | 0.1410 | 0.0248 |
| Hindi (hi) | 0.1353 | 0.0207 |

Both experiments are highly language-uniform (< 10% variation across languages).

## Notes & caveats

- **Token mismatches: 0/1000 in both experiments.** Enforced tokens via `enforced_tokens` API parameter work correctly across B200 TP=4 ↔ H200 TP=8.
- **Cross-TP baseline:** inference (TP=4 on B200) vs validator (TP=8 on H200). Same-TP would likely give slightly lower variance.
- No fraud data — K2.6 has no AWQ/GPTQ variant on HuggingFace (only MXFP4, GGUF, MLX). Candidate frauders if later needed:
  - [`wafer-ai/Kimi-K2.6-MXFP4`](https://huggingface.co/wafer-ai/Kimi-K2.6-MXFP4) — 4-bit microscaling, B200 only
  - [`unsloth/Kimi-K2.6-GGUF` Q4_K_M](https://huggingface.co/unsloth/Kimi-K2.6-GGUF) — GGUF; vLLM supports formally, but questionable at 1T MoE scale
- Expected fraud distance range (extrapolated from M2.7): exp2 sentinel-mode → ~0.08-0.15 for fraud; exp1 raw-mode → ~0.25-0.35.

## Reproduction

See [REPRODUCTION_GUIDE.md](../minimax-m27-inference-validation/REPRODUCTION_GUIDE.md) for the full step-by-step. Key differences for K2.6:

1. Model ID: `moonshotai/Kimi-K2.6` (native INT4, ~540 GB, no dtype flag needed)
2. TP: 4 on B200 or 8 on H200 (540 GB doesn't fit on <4 B200 or <8 H100)
3. For M2.7-style sentinels (exp2), patch `mlnode/packages/benchmarks/src/validation/utils.py`:
   ```python
   payload = {
       ...
       "repetition_penalty": 1.2,
       "top_k": 40,        # <-- ADD
       "top_p": 0.95,      # <-- ADD
   }
   ```
4. `--logprobs-mode processed_logprobs` on both sides (matches gonka production).

## Key findings

1. **K2.6 distribution is more diffuse than M2.7** → no natural sentinels even with `processed_logprobs`. INT4 QAT + MoE routing spreads probability mass across more tokens.
2. **Sentinels can be forced** via explicit `top_k`/`top_p` in the request payload — bit-for-bit identical output pattern to M2.7.
3. **Sentinels distort distance2 downward by ~6×** because they're deterministic across inference & validator. For fraud detection, raw-mode (exp1) is more discriminative, but processed-with-sentinels (exp2) is what gonka production uses.
4. **Cross-TP (4↔8) honest baseline is clean** — zero token mismatches, tight distribution, no language bias.
