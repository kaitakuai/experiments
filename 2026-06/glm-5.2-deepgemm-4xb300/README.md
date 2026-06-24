# GLM-5.2 FP8 — B300 — TP=4 honest+fraud, DeepGEMM (753B + full 1M context on 4 cards)

**Date:** 2026-06-24
**Model:** `zai-org/GLM-5.2-FP8` @ `31cba24fb749908a485082bdeed6eb1ac6cffc2f`
  FP8 block-wise e4m3 [128,128], arch `GlmMoeDsaForCausalLM` (DeepSeek-style MLA + DSA sparse
  attention), 753B total / ~40B active, 256 experts × top-8.
  Fraud model: `cyankiwi/GLM-5.2-AWQ-INT4` @ `431c1cd297c7a2f38d17c7b9520b10c15101df25`
  (compressed-tensors W4A16, ~390 GB).
**Hardware:** 8× NVIDIA **B300 SXM6 AC** (**275 GiB HBM/GPU**, driver **595.71.05**, sm_100).
  Vast.ai Virginia US. Each engine uses **TP=4 (4 GPUs)**; the box runs **2 engines**.
**Image:** `ghcr.io/kaitakuai/mlnode-b200-glm-5-2:0.2.13-vllm0.23.0-k1`
  (GLM-5.2 image reused; TP overridden 8→4 via the shared runner patch)

## Summary

B300's **275 GiB HBM** changes the deployment math: GLM-5.2 FP8 (753B, ~704 GB) fits on **just
4×B300 at TP=4** (176 GB weights/card), so an 8-GPU box runs **two independent TP=4 engines**.
This nearly **doubles PoC throughput per box** vs the single-TP=8 engine B200/H200 are forced into:

| Config | engines | GPUs | PoC nonces/min |
|---|---:|---:|---:|
| 1× TP=4 | 1 | 4 | **896** |
| **2× TP=4** (box) | 2 | 8 | **1792** |
| 1× TP=8 | 1 | 8 | 1078 |

**2× TP=4 = 1792/box, +66% over TP=8 (1078)** — smaller TP means less all-reduce overhead per engine.

And the headline capacity result: **the full 1M context fits on 4×B300 at TP=4 — and PoC still runs
at full speed simultaneously** (864 vs 896 nonces/min, i.e. no meaningful context↔PoC tradeoff).

## Environment

| Parameter | Value |
|---|---|
| GPU | 8× B300 SXM6 AC, 275 GiB HBM, sm_100 |
| NVIDIA driver | 595.71.05 |
| vLLM | 0.23.0 (in-image system python) |
| MoE/linear (honest) | **DeepGEMM** (`DeepGemmFp8BlockScaledMMKernel`, E8M0) |
| MoE (fraud AWQ) | **Marlin** (compressed-tensors W4A16, auto-detected) |
| Plugin | `gonka_poc` with `generate_queue` (eager PoC via `skip_compiled`) |

## Config

Per-engine honest config (TP=4, DeepGEMM, CUDA graphs):

```bash
--tensor-parallel-size 4              # 753B FP8 fits on 4×B300 (176 GB/card of 275)
--gpu-memory-utilization 0.85         # 0.92 for the 1M-context run (still leaves PoC headroom)
--max-model-len 32768                 # PoC; 1048576 for the full-context run (both fit)
--max-num-batched-tokens 16384        # DeepGEMM survives profiling only at small mnbt
--max-num-seqs 64
--kv-cache-dtype fp8_e4m3             # B300 sm_100 accepts e4m3 (no fp8_ds_mla needed, unlike H200)
--tool-call-parser glm47  --reasoning-parser glm45  --trust-remote-code
--worker-extension-cls gonka_poc.worker.PoCWorkerExtension
--compilation-config '{"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"}'
VLLM_USE_DEEP_GEMM=1  VLLM_MOE_USE_DEEP_GEMM=1
```

The MLNode runner **auto-detects** that FP8 fits TP=4 on a spare-GPU box and launches **2 instances**
(GPUs 0-3 → port 5001, GPUs 4-7 → port 5002); the PoC proxy fans out across both. To force a single
engine for a clean per-engine number, restrict visibility: `CUDA_VISIBLE_DEVICES=0,1,2,3`.

## Results

### PoC v2 throughput (honest FP8, DeepGEMM, batch sweep)

Single TP=4 engine (4 GPUs), via [`../glm-5.2-poc-backend-sweep/scripts/03_poc_sweep.sh`](../glm-5.2-poc-backend-sweep/scripts/03_poc_sweep.sh):

| batch | nonces/min |
|---:|---:|
| 8 | 832 |
| **16** | **896** ★ |
| 32 | 0 (mnbt cap) |
| 64 | 0 |

- **Per-engine TP=4: 896/min** (batch 16). Box (2 engines): **1792/min** = 2 × 896 (verified).
- Batch 32/64 = 0: the PoC forward exceeds `max-num-batched-tokens 16384` (DeepGEMM can't profile at
  higher mnbt → `cudaErrorInvalidValue`), so the batch is capped at 16. Same as B200.

### Full 1M context (the B300 advantage)

`--max-model-len 1048576`, TP=4 (4 GPUs), fp8_e4m3 KV, gmu 0.92:

| Metric | Value |
|---|---|
| **GPU KV cache size** | **1,124,928 tokens** (> 1M ✓) |
| Max concurrency @ 1,048,576 tokens | 1.07x |
| **PoC at 1M context** | **864 nonces/min** (batch 16) |

**The full 1,048,576-token context fits on 4×B300, and PoC still runs at ~864/min** — essentially the
same as the 32K-context 896/min. There is **no meaningful context↔PoC tradeoff** on B300 (the DSA
MLA-compressed KV is small, and 275 GiB/card leaves ample room). Contrast H200, where fp8_ds_mla is
required and large context noticeably slows PoC.

> Note: an initial 1M-context PoC reading of 448/min was a measurement artifact (sweep run before the
> engine was fully warm); the re-checked steady-state is 864/min.

### Inference (honest FP8, DeepGEMM, CUDA graphs)

`vllm bench serve`, 200 prompts, 1024→256, concurrency 32, box (2 engines via proxy):

| Metric | Value |
|---|---|
| Output tok/s | 556 |
| TPOT median | 42 ms |

### L2 chain validation — fraud detection

Honest FP8 (DeepGEMM) vs fraud AWQ-INT4 (Marlin), canonical Gonka L2 via
[`../../tools/gonka-l2-validate/`](../../tools/gonka-l2-validate/), k_dim 12, gate dist_threshold=0.4 / p_mismatch=0.02:

| Pair | N | mean L2 | mismatch | p_value | verdict |
|---|---:|---:|---:|---:|---|
| **honest FP8 ↔ fraud AWQ-INT4** | 980 | 0.348 | 279/980 (28.5%) | 3.47e-228 | **FRAUD** |
| honest B300 ↔ honest B200 | 1000 | 0.193 | 8/1000 (0.8%) | 0.999 | **PASS** |
| honest B300 ↔ honest H200 | 1000 | 0.189 | 9/1000 (0.9%) | 0.998 | **PASS** |

The gate catches the AWQ-INT4 fraud on B300 (mean ~0.35, ~28% mismatch), and **B300 honest
cross-validates against B200 and H200 honest** (mean ~0.19, <1% mismatch) — the PoC vectors are
hardware-independent across all three generations. Honest pairs cluster at mean ~0.19 / <1%; fraud
pairs at ~0.35 / ~28% — a clean ~1.8× / ~40× separation.

## Files

- [`artifacts/honest_fp8.json`](artifacts/honest_fp8.json) — honest FP8 PoC nonces (DeepGEMM, TP=4)
- [`artifacts/fraud_awq.json`](artifacts/fraud_awq.json) — fraud AWQ-INT4 PoC nonces (Marlin, TP=4)
- [`artifacts/l2_honest_vs_fraud.json`](artifacts/l2_honest_vs_fraud.json) — L2 fraud verdict (FRAUD)
- [`artifacts/l2_honest_b300_vs_b200.json`](artifacts/l2_honest_b300_vs_b200.json) — honest B300↔B200 cross-hardware (PASS)
- [`artifacts/l2_honest_b300_vs_h200.json`](artifacts/l2_honest_b300_vs_h200.json) — honest B300↔H200 cross-hardware (PASS)
- [`artifacts/sweep_fp8_deepgemm_auto.log`](artifacts/sweep_fp8_deepgemm_auto.log) — PoC batch sweep
- [`scripts/collect_nonces.sh`](scripts/collect_nonces.sh) — direct `collect_artifacts.py` driver
  (the in-image `05_collect_nonces.sh` wrapper hangs on `import vllm` in this image; call the tool directly)

## Findings / recommendation

1. **B300 → run TP=4, two engines per 8-GPU box.** 1792 PoC nonces/min/box vs 1078 at TP=8 (+66%).
   The 275 GiB HBM lets the 753B FP8 model fit on 4 cards.
2. **Full 1M context fits on 4×B300 AND PoC runs at full speed** (864/min) — no context tradeoff.
   This is unique to B300; B200 needs all 8 GPUs for 1M, H200 caps lower and fp8_ds_mla slows PoC.
3. **KV dtype `fp8_e4m3`** works natively on B300 (sm_100) — no `fp8_ds_mla` workaround (unlike H200).
4. **DeepGEMM is the honest backend; AWQ fraud uses Marlin.** L2 gate catches the fraud (p≈3e-228).
5. **DeepGEMM caps the PoC batch at 16** (mnbt≤16384 to survive profiling) — same constraint as B200.

## How to reproduce

```bash
export VAST_API_KEY=<key>; export HF_TOKEN=<hf-token>
# 1. Rent B300 (>=4 GPUs; an 8-GPU box runs 2 engines). 275 GiB HBM per card.
# 2. Download both models (to /dev/shm — needs >=1.1 TB RAM, or to disk):
hf download zai-org/GLM-5.2-FP8 --revision 31cba24fb749908a485082bdeed6eb1ac6cffc2f --local-dir /dev/shm/GLM-5.2-FP8 --max-workers 16
hf download cyankiwi/GLM-5.2-AWQ-INT4 --revision 431c1cd297c7a2f38d17c7b9520b10c15101df25 --local-dir /dev/shm/GLM-5.2-AWQ-INT4 --max-workers 16

# 3. Honest FP8 @ TP=4 + DeepGEMM (the shared 02_start_vllm.sh applies the runner patch, TP override):
MODELS_DIR=/dev/shm MODEL_ROLE=fp8 MOE_BACKEND=deepgemm TP=4 GPU_MEM_UTIL=0.85 \
  MAX_MODEL_LEN=32768 MAX_NUM_SEQS=64 MAX_NUM_BATCHED_TOKENS=16384 KV_CACHE_DTYPE=fp8_e4m3 \
  COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  bash ../glm-5.2-poc-backend-sweep/scripts/02_start_vllm.sh
MODEL_ROLE=fp8 MOE_BACKEND=deepgemm bash ../glm-5.2-poc-backend-sweep/scripts/03_poc_sweep.sh   # → 896 (1 engine) / 1792 (2)
bash scripts/collect_nonces.sh fp8 /dev/shm/GLM-5.2-FP8 artifacts/nonces_honest               # honest nonces

# 4. Full 1M context check: rerun step 3's start with MAX_MODEL_LEN=1048576 GPU_MEM_UTIL=0.92.

# 5. Fraud AWQ @ TP=4 (Marlin auto): start with MODEL_ROLE=awq, then:
bash scripts/collect_nonces.sh awq /dev/shm/GLM-5.2-AWQ-INT4 artifacts/nonces_fraud

# 6. L2: honest vs fraud (distinct basenames!):
cp artifacts/nonces_honest/nonces_1000.json artifacts/honest_fp8.json
cp artifacts/nonces_fraud/nonces_1000.json  artifacts/fraud_awq.json
python3 ../../tools/gonka-l2-validate/compare_nonces.py artifacts/honest_fp8.json artifacts/fraud_awq.json --scenario "glm chain gate,0.4,0.02"

# 7. Destroy the box.
```

## Related

- B200 DeepGEMM (cudagraph vs eager): [`../glm-5.2-deepgemm-8xb200/README.md`](../glm-5.2-deepgemm-8xb200/README.md)
- B200 FP8 / AWQ fraud baseline: [`../glm-5.2-fp8-8xb200/README.md`](../glm-5.2-fp8-8xb200/README.md)

## Reproducibility checklist

- [x] Reader with this folder + `../glm-5.2-poc-backend-sweep/scripts/` reaches the result.
- [x] Hardware stated: 8× B300 SXM6 AC, 275 GiB HBM, driver 595.71.05, sm_100; engine = TP=4 (4 GPUs).
- [x] Image stated by tag; models pinned by repo + commit SHA.
- [x] Commands copy-pasteable; the wrapper gotcha (05 hangs on import vllm → call collect_artifacts directly) is noted.
- [x] All artifacts referenced exist in `artifacts/`.
- [x] Expected outputs: PoC 896/engine (1792/box), 1M context fits + PoC 864, L2 FRAUD (p≈3e-228).
- [x] Known gotchas listed (DeepGEMM mnbt cap, auto-2-instance split, fp8_e4m3 on B300, 1M-context warm-up artifact).
