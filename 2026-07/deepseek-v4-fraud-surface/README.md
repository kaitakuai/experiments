# DeepSeek-V4-Flash — PoC fraud surface: which cheats are even possible, and which are detectable

**Date:** 2026-07-24
**Model:** `deepseek-ai/DeepSeek-V4-Flash` (FP8, sparse-MLA, hash-routed MoE)
**Image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash:0.2.13-vllm0.25.1-overlay-k4`
**Digest:** `sha256:2af898fa516424ea2884b77e40ae480ce61b19a7d0112f8c9b8cc866c8bcb28a`
**Hardware:** 2× B200 SXM (TP=2) for the fraud runs; honest reference sets from B200 / B300 / H200 SXM
**PoC:** v2 plugin, `--worker-extension-cls gonka_poc.worker.PoCWorkerExtension`, seq_len 1024, k_dim 12, batch 32

> Thresholds for V4 are **not calibrated yet**. This report deliberately contains **no
> PASS/FRAUD verdicts** — only measured L2 distributions, so limits can be fitted later.

## Summary

Two questions: *what can a cheating node actually run*, and *does per-nonce L2 catch it*.

1. **The cheat menu is short.** Of the quantized V4 checkpoints available, only **NVFP4**
   loads. Every INT4/GPTQ variant fails at weight loading — not at inference, not on
   performance, but because the V4 weight loader in vLLM only understands FP8 blocks and
   FP4 experts. So the realistic fraud surface is **FP8 (honest) + NVFP4 (cheaper) +
   checkpoint substitution**.
2. **The two cheats are not equally detectable.** Substituting a different checkpoint
   (V4-Base) is trivially caught — median L2 0.44, 61.5 % of nonces beyond 0.4. Running
   **NVFP4 instead of FP8 is nearly invisible** — median L2 0.21 against honest 0.187,
   6.4 % beyond 0.4 against an honest-noise floor of 2.8–3.8 %. That is a ~2× separation
   on a tail statistic, not the order-of-magnitude gap checkpoint substitution gives.

**Practical consequence:** a per-nonce L2 threshold tuned to admit honest cross-hardware
noise will also admit an NVFP4 node. Catching NVFP4 needs either a distributional test
over many nonces rather than a per-nonce cutoff, or an architecture-pinned validator (see
`../deepseek-v4-flash-2xh200/README.md` — on Blackwell honest repeats are ~97 % bit-identical,
which collapses the honest floor and re-opens the gap).

## What loads and what does not

| Checkpoint | Format | Size | Loads? | Evidence |
|---|---|---:|:---:|---|
| `deepseek-ai/DeepSeek-V4-Flash` | FP8 | — | ✅ | honest reference |
| `nvidia/DeepSeek-V4-Flash-NVFP4` | NVFP4 | 156.7 GiB | ✅ | `artifacts/nvfp4_load_evidence.txt` |
| `Intel/DeepSeek-V4-Flash-W4A16-AutoRound` | INT4/GPTQ | 144.9 GiB | ❌ | `artifacts/int4_quantization_attempts.log` |

### NVFP4 loads, but the first start is slow

78.93 GiB per GPU, weights in 91 s on 2× B200 TP=2, then
`nvfp4.py: Using MoEPrepareAndFinalizeNoDPEPModular`. The first run then spends **~10–15
minutes with GPUs at 0 % and a CPU core pinned at 100 %** while `ptxas`/`ninja` JIT-build
`fused_moe_trtllm_sm100`. This is compilation, not a hang — do not restart it.

### INT4 cannot load, by construction

Tried four ways — auto-detect, `--quantization moe_wna16`, `gptq_marlin`, `auto_gptq` —
all fail identically:

```
AttributeError: 'ColumnParallelLinear' object has no attribute 'weight_scale_inv'
```

The cause is in `vllm/models/deepseek_v4/nvidia/model.py`, which assumes FP8 block scales
and FP4 experts and has no branch for GPTQ tensors (`qweight`/`qzeros`/`scales`/`g_idx`):

```python
block_size = getattr(self.quant_config, "weight_block_size", None)
step = 1 if name.endswith("weight_scale_inv") else block_size[0]
...
if expert_dtype == "fp4":
```

`--quantization` cannot help: the failure is in model weight loading, before any
quantization backend is selected. Supporting INT4 V4 would require patching the model
itself. Full trace and the quoted source region: `artifacts/int4_loader_error_and_code.txt`.

## L2 matrix — 1000 nonces per pair, same block_hash and public_key

Distance is per-nonce L2 between decoded 12-dim fp16 vectors. `>0.4` is the fraction of
nonces past the reference threshold used for other models — shown for scale only, not as
a verdict for V4.

| Pair | N | median L2 | >0.4 |
|---|---:|---:|---:|
| **honest** B200 vs B300 | 1000 | 0.1869 | 3.60 % |
| **honest** B200 vs H200 SXM | 1000 | 0.1895 | 3.80 % |
| **honest** B300 vs H200 SXM | 1000 | 0.1891 | 2.80 % |
| NVFP4 vs honest B200 | 1000 | **0.2102** | **6.40 %** |
| NVFP4 vs honest B300 | 1000 | **0.2138** | **5.70 %** |
| NVFP4 vs honest H200 SXM | 1000 | **0.2123** | **5.80 %** |
| V4-Base vs honest B200 | 1000 | **0.4384** | **61.50 %** |
| V4-Base vs honest B300 | 1000 | **0.4437** | **61.70 %** |
| V4-Base vs honest H200 SXM | 1000 | **0.4419** | **61.50 %** |
| V4-Base vs NVFP4 | 1000 | 0.4390 | 61.50 % |

Machine-readable: `artifacts/l2_fraud_matrix.json`.

### Reading the matrix

- The **honest floor is flat at 0.187** regardless of which pair of honest machines or
  architectures is compared — the distribution shape matches, not just the median.
- **NVFP4 sits at 0.21**, i.e. +12 % on the median and roughly double the tail. Against a
  single honest reference this is within the range a threshold generous enough for honest
  cross-hardware noise would accept.
- **V4-Base sits at 0.44** with 61.5 % of nonces past 0.4 — an order of magnitude above the
  honest floor and separable on every topology tested.
- The last row is a sanity check: V4-Base is equally far from NVFP4 as from honest FP8,
  confirming it is a genuinely different model rather than a shifted one.

## Files

| Path | What |
|---|---|
| `artifacts/nonces_honest_{b200,b300,h200}.json` | honest FP8 reference sets, 1000 nonces each |
| `artifacts/nonces_fraud_nvfp4.json` | NVFP4 run on 2× B200, 1000 nonces |
| `artifacts/nonces_fraud_v4base.json` | V4-Flash-Base (checkpoint substitution), 1000 nonces |
| `artifacts/l2_fraud_matrix.json` | the table above, machine-readable |
| `artifacts/nvfp4_load_evidence.txt`, `nvfp4_run.log` | NVFP4 load path + full run log |
| `artifacts/int4_quantization_attempts.log` | all four INT4 attempts |
| `artifacts/int4_loader_error_and_code.txt` | trace + the vLLM source region that rejects it |
| `scripts/compare_nonces.py` | canonical L2 comparison (decode → per-nonce L2 → binomtest) |
| `scripts/fraud_fresh.sh` | starts a fraud model and collects 1000 nonces |
| `scripts/collect_artifacts.py` | pulls nonces from the PoC endpoint |

## Findings

1. **INT4 V4 is not a threat vector today** — it cannot be served at all without patching
   vLLM's V4 model code. Threshold calibration does not need to account for it.
2. **NVFP4 is the hard case.** It loads, it serves, and per-nonce L2 barely separates it
   from honest FP8 on a Hopper-noise-tolerant threshold.
3. **Checkpoint substitution is easy to catch** and stays catchable across all topologies.
4. **The honest floor is architecture-driven, not topology-driven.** Since all honest cross
   pairs land at 0.187 while same-Blackwell repeats are ~97 % bit-identical, pinning
   validation to Blackwell shrinks the honest floor far more than it shrinks the NVFP4
   signal — that is the lever that makes NVFP4 detectable.

## Reproduce

```bash
# honest reference (any SXM box, ≥2 GPUs, driver ≥580 for the CUDA-13 image)
./scripts/fraud_fresh.sh deepseek-ai/DeepSeek-V4-Flash 2
# NVFP4 — expect ~15 min of silent JIT on first start, do not restart
./scripts/fraud_fresh.sh nvidia/DeepSeek-V4-Flash-NVFP4 2
# compare (use distinct filenames — the script labels pairs by basename)
python3 scripts/compare_nonces.py nonces_honest_b200.json nonces_fraud_nvfp4.json
```

## Reproducibility checklist

- [x] Image referenced by tag **and** digest
- [x] All scripts used are committed in `scripts/`
- [x] All raw nonce sets committed in `artifacts/`
- [x] L2 numbers regenerated from committed artifacts with the committed script
- [x] Failure evidence (INT4) committed as logs + traces, not summarized from memory
- [x] No links to `.claude/`, no absolute local paths, no sibling-repo references
- [x] No verdicts asserted — thresholds for V4 are not calibrated yet
