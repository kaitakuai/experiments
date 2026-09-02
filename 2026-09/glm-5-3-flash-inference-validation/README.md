# GLM-5.3-Flash — inference validation — cross-hardware 2×B300 → 4×H200

**Date:** 2026-09-02
**Honest:** `zai-org/GLM-5.3-Flash` — `Glm5NextForConditionalGeneration` (multimodal: text +
vision), 62 shards, **305.8 GiB**. FP8 `e4m3` with `activation_scheme: dynamic`;
`modules_to_not_convert` keeps `attn_mha`, `attn_mqa`, `dt_bias`, `hyper_connection`,
`lm_head`, `mapping_proj` and `embed_tokens` at 16 bits.
**Fraud:** `LibertAIDAI/GLM-5.3-Flash-NVFP4` — 121 shards, **181.3 GiB**. Packed NVFP4
(155.8 B values in `U8`) plus 9.67 B parameters left in BF16, i.e. roughly 6 % of the model
unquantised.
**Images:** generation `ghcr.io/kaitakuai/mlnode-b300-glm-5-3-flash:0.2.14-vllm0.28-glm53-test-k3`,
validation `ghcr.io/kaitakuai/mlnode-h100-glm-5-3-flash:0.2.14-vllm0.28-glm53-test-k3`.
Both carry vLLM `0.28.0.dev0+glm53.gonka.sampler1`.

## Experiments

| Experiments | GPU Honest | Version Honest | Model Honest | GPU Fraud | Version Fraud | Model Fraud | GPU Validator | Version Validator | Model Validator | Link to artifacts | graph |
|---|---|---|---|---|---|---|---|---|---|---|---|
| processed_logprobs | 2×B300 | 0.28 | zai-org/GLM-5.3-Flash | 2×B300 | 0.28 | LibertAIDAI/GLM-5.3-Flash-NVFP4 | 4×H200 | 0.28 | zai-org/GLM-5.3-Flash | [artifacts](artifacts/) | [png](artifacts/glm53_processed_honest_vs_nvfp4.png) |
| raw_logprobs | 2×B300 | 0.28 | zai-org/GLM-5.3-Flash | 2×B300 | 0.28 | LibertAIDAI/GLM-5.3-Flash-NVFP4 | 4×H200 | 0.28 | zai-org/GLM-5.3-Flash | [artifacts](artifacts/) | [png](artifacts/glm53_raw_honest_vs_nvfp4.png) |

Both arms were generated on the same 2×B300 host and validated on the same 4×H200 host, so
honest and fraud share one GPU pair. 1000 multilingual prompts (en/es/zh/ar/hi),
`max_tokens=1000`, `temperature=0.7`, `seed=1`, `top_k=40`, `top_p=0.95`,
`repetition_penalty=1.2`, `top_logprobs=4`, 16 concurrent workers. 4000 generations, 4000
replays, **0 length mismatches**.

Both sides ran the same engine configuration — `--kv-cache-dtype fp8`, `--block-size 2304`,
`--max-num-seqs 256`, `--no-enable-flashinfer-autotune`, `--logprobs-mode processed_logprobs`
with the per-request override. The Hopper image bakes those flags; the Blackwell image does
not, so they were passed explicitly there. Without that the two sides would differ in KV
blocking and kernel selection, and part of the measured distance would be configuration
rather than hardware.

## Results

| mode | arm | distance2 | ×floor | best F1 | TP@FP5% |
|---|---|---:|---:|---:|---:|
| processed | honest | 0.024969 | — | — | — |
| processed | NVFP4 | 0.054328 | 2.18× | 0.918 | 88.5 % |
| raw | honest | 0.029738 | — | — | — |
| raw | **NVFP4** | 0.076863 | **2.58×** | **0.998** | **99.8 %** |

Restricted to answers of ≥ 100 tokens — 991 of 1000 qualify, so the filter barely changes
anything on this model:

| mode | arm | distance2 | ×floor | best F1 | TP@FP5% |
|---|---|---:|---:|---:|---:|
| processed | honest | 0.025055 | — | — | — |
| processed | NVFP4 | 0.054524 | 2.18× | 0.920 | 89.0 % |
| raw | honest | 0.029777 | — | — | — |
| raw | **NVFP4** | 0.077111 | **2.59×** | **0.999** | **100.0 %** |

## Findings

1. **The fraud is caught essentially every time.** In `raw`, one replay of one nonce gives
   F1 0.999 and catches 100 % of fraudulent answers at a 5 % false-positive rate. On
   [Hy3](../2026-08/hy3-inference-validation/) the best any arm reached was 1.60× the floor
   with F1 0.918 and 88 % recall.
2. **The noise floor is low and the cross-hardware penalty is small.** Same-machine
   self-validation measured 0.0222 (processed) / 0.0256 (raw); going Blackwell → Hopper moves
   it only to 0.0250 / 0.0297, i.e. 1.12×–1.16×. Hy3 paid 1.21× for the same crossing.
3. **`processed` is usable here, unlike on Hy3.** It reaches F1 0.918 rather than collapsing
   to the degenerate 0.667. The reason is answer length: GLM-5.3 answers have a median of 796
   tokens and 99 % exceed 100, so almost nothing sits on the `1/401` floor that `distance2`
   imposes via `max(100, n)`. `raw` is still clearly better.
4. **Length filtering is not needed for this model** — 991 of 1000 answers already qualify.
   On Hy3 it was worth +0.11 F1 because a quarter of answers were short.
5. **Replay is exact across architectures.** 0 length mismatches in 4000 replays between
   Blackwell and Hopper.
6. **Fraud economics are the same shape as Hy3.** Honest needs 305.8 GiB and does not fit one
   B300 (~242 GiB usable at `gmu 0.90`); the NVFP4 build needs 181.3 GiB and does.

## Operational notes

Collected while getting this to run; all of them cost real time.

- **`NCCL_NVLS_ENABLE=0` is required on these Blackwell hosts.** Without it NCCL aborts with
  `Failed to bind NVLink SHARP (NVLS) Multicast memory … CUDA error 401` (or, on some
  allocations, hangs silently on the first collective). NCCL's own message recommends the
  same flag. Verify with a 2-rank all-reduce *before* downloading half a terabyte.
- **Cold engine bring-up on Blackwell takes ~22 minutes**, most of it silent JIT of
  TileLang/CUTLASS kernels — no log output, no compile-cache growth. Judge liveness by GPU
  memory, which climbs steadily, not by log activity. A 12-minute timeout aborts a healthy
  start one minute short.
- **Hopper hosts with driver 565/570 still run these CUDA-13 images** via
  `LD_LIBRARY_PATH=/usr/local/cuda/compat`, which the image ships but disables by default.
- **The mlnode API can die while vLLM keeps serving** on port 5001; the replay driver talks
  to the OpenAI-compatible endpoint directly, so a dead proxy does not require a reload.
- `--model` must be a local snapshot path: `Glm5NextProcessor` looks for `processor_config.json`
  as a file, so an HF id does not work.

## Files

```
artifacts/
  val_{honest,fraud}_{processed,raw}_logprobs.jsonl   validator replays, 1000 rows each
  glm53_{raw,processed}_honest_vs_nvfp4.png
scripts/
  v4val.py           generate / replay driver (enforced_tokens + distance2)
  genarm.sh          generation for one arm, both logprobs modes
  validate_all.sh    replays every set against the honest model
  analyze.py         the tables above, recomputed from the artifacts
  plot_gonka.py      the scatter plots, gonka format (answer length in characters)
```

The plots omit 149 of 1000 answers whose `message.content` came back empty: GLM-5.3 is a
reasoning model and with `--reasoning-parser glm45` those responses carry their text in the
reasoning field, which the driver does not record, so they have no character length to plot.
They are present in every number in the tables above — only the scatter drops them.

Each replay row keeps the executor's `inference_result`, the validator's
`validation_result`, the per-row `distance`, `len_mismatch` and sentinel fractions, so every
number above is recomputable with `scripts/analyze.py`.

## Reproducibility checklist

- [x] Images pinned by tag; quantisation described from `config.json`
- [x] Every script referenced above committed under `scripts/`
- [x] All tables recomputable from committed artifacts
- [x] Honest and fraud share one GPU pair; no same-GPU floor substituted
- [x] 0 length mismatches reported per arm, not aggregated away
- [x] Engine flags equalised across the two images, and the difference documented
