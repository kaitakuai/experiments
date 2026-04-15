# Inference Validation: MiniMax-M2.7 FP8 (H100 vs A100)

**Date:** 2026-04-15
**Model Honest:** `MiniMaxAI/MiniMax-M2.7` (FP8)
**Model Fraud:** `demon-zombie/MiniMax-M2.7-AWQ-4bit` (AWQ INT4)
**vLLM:** 0.19.0
**Validation framework:** gonka-ai/gonka `tg/qwen_models` branch

## Experiment table

| Exp | GPU Honest | Version | Model Honest | GPU Fraud | Version | Model Fraud | GPU Validator | Version | Model Validator | Logprobs mode |
|-----|-----------|---------|-------------|-----------|---------|-------------|--------------|---------|----------------|---------------|
| 1 | H100 | 0.19.0 | MiniMax-M2.7 FP8 | H100 | 0.19.0 | MiniMax-M2.7-AWQ-4bit | A100 | 0.19.0 | MiniMax-M2.7 FP8 | processed_logprobs |
| 2 | H100 | 0.19.0 | MiniMax-M2.7 FP8 | H100 | 0.19.0 | MiniMax-M2.7-AWQ-4bit | A100 | 0.19.0 | MiniMax-M2.7 FP8 | raw_logprobs |

## Infrastructure

- **Inference (H100):** 4×NVIDIA H100 80GB HBM3, TP=4, vLLM 0.19.0
- **Validator/Fraud (A100):** 4×NVIDIA A100 SXM4 80GB, TP=4, vLLM 0.19.0
- **Attention backend:** FLASHINFER (both)
- **Connection:** SSH tunnel through local machine (Vast.ai servers)

## Parameters

```yaml
max_tokens: 1000
temperature: 0.7
seed: 1
top_logprobs: 4
n_prompts: 500-1000
batch_size: 10
max_workers: 5
languages: [sp, en, ch, ar, hi] (200 per language)
```

## Results

### Experiment 1: processed_logprobs

| Metric | Honest | Fraud |
|--------|--------|-------|
| Samples | 2000 (1000 H100→A100 + 1000 A100→H100) | 1000 |
| Mean distance | 0.0380 | 0.0701 |
| P95 | 0.0602 | 0.1083 |
| Separation | 1.8x | — |
| Best F1 | 0.767 | — |
| TP at 5% FP | 65.7% | — |

**Observation:** `processed_logprobs` mode produces sentinel values (-9999) at ~75% of positions. These cancel out in distance calculation when both sides have same sentinels (honest), but reduce sensitivity to real logprob differences (fraud).

### Experiment 2: raw_logprobs (no processed_logprobs flag)

| Metric | Honest | Fraud |
|--------|--------|-------|
| Samples | 510 | 500 |
| Mean distance | 0.0666 | 0.1365 |
| P95 | 0.0849 | 0.1680 |
| Separation | **2.1x** | — |
| **Best F1** | **0.980** | — |
| **TP at 5% FP** | **99.0%** | — |
| TP at 0% FP | 91.8% | — |

**Recommended threshold:** 0.092 → FP=0.6%, TP=96.6%

### Detection rates (raw_logprobs)

| Threshold | FP% | TP% |
|-----------|-----|-----|
| 0.080 | 15.9 | 99.2 |
| 0.090 | 1.4 | 97.2 |
| **0.092** | **0.6** | **96.6** |
| 0.095 | 0.4 | 95.4 |
| 0.100 | 0.0 | 91.8 |

### Per-language (raw_logprobs, honest)

| Lang | N | Mean | P95 |
|------|---|------|-----|
| Spanish | 200 | 0.0736 | — |
| English | 200 | 0.0642 | — |
| Chinese | 110 | 0.0582 | — |

## Comparison with other models

| Model | Threshold | Best F1 | Detection (TP) | Mode |
|-------|-----------|---------|----------------|------|
| Gemma-3-27B (gonka) | 0.05+0.10 | — | 99% | — |
| DeepSeek-R1-0528 (gonka) | 0.053+0.02 | — | 83% | — |
| Qwen3-30B (gonka) | 0.023 | — | 52% | — |
| Qwen3-235B (gonka) | 0.042 | — | 24% | — |
| **MiniMax-M2.7 (ours, raw)** | **0.092** | **0.980** | **96.6%** | raw_logprobs |
| MiniMax-M2.7 (ours, processed) | 0.055 | 0.767 | 73.8% | processed_logprobs |

## Key findings

1. **raw_logprobs dramatically improves fraud detection** — F1 jumps from 0.767 to 0.980. The `processed_logprobs` mode injects sentinel values (-9999) that mask real logprob differences.

2. **MiniMax-M2.7 with raw_logprobs achieves 96.6% detection** at 0.6% false positive — comparable to Gemma-3-27B (99%) and better than DeepSeek-R1 (83%).

3. **Recommended configuration:** raw_logprobs (no `--logprobs-mode processed_logprobs` flag), threshold=0.092.

4. **Cross-GPU validation works** — H100 inference + A100 validation with FLASHINFER attention on both sides.

## Artifacts

### exp1-processed-logprobs/
- `artifacts/honest_h100_a100.jsonl` — 1000 honest samples (H100→A100)
- `artifacts/fraud_h100_a100.jsonl` — 1000 fraud samples (H100 FP8 → A100 AWQ)
- `artifacts/length_vs_distance.png` — scatter plot

Additional honest data (same directory level):
- `honest_a100_h100.jsonl` — 1000 honest samples (A100→H100)

### exp2-raw-logprobs/
- `artifacts/honest_h100_a100.jsonl` — 510 honest samples (H100→A100)
- `artifacts/fraud_h100_a100.jsonl` — 500 fraud samples (H100 FP8 → A100 AWQ)
- `artifacts/length_vs_distance.png` — scatter plot

## Reproduction

### Validation script
From `gonka-ai/gonka` branch `tg/qwen_models`:
```bash
cd mlnode/packages/benchmarks/scripts
python3 run_inference_validation.py
```

### vLLM launch (raw_logprobs — recommended)
```bash
# NO --logprobs-mode processed_logprobs
--served-model-name MiniMaxAI/MiniMax-M2.7
--tensor-parallel-size 4
--gpu-memory-utilization 0.92
--max-num-seqs 64
--max-model-len 131072
--trust-remote-code
--attention-backend FLASHINFER
```

### vLLM launch (processed_logprobs — for comparison)
```bash
# Same as above plus:
--logprobs-mode processed_logprobs
```
