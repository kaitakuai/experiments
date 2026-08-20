# Hy3 — inference validation — cross-hardware B300 → 4×H200

**Date:** 2026-08-20
**Honest:** `tencent/Hy3-FP8` — 295B total / 21B active MoE, 192 experts × top-8, 80 layers
+ 1 MTP layer, GQA 64 heads / 8 KV heads × 128, 256K context. `quant_method: fp8`,
`activation_scheme: static`, `kv_cache_scheme: static`; only `lm_head` and `embed_tokens`
excluded. **276 GiB in VRAM.**
**Fraud A:** `RedHatAI/Hy3-NVFP4-FP8` — llm-compressor, `format: mixed-precision`. Attention
FP8 block `[128,128]`, **MoE NVFP4** `tensor_group` 16 with `float8_e4m3` scales. `ignore` has
81 entries: router gates of layers 1–79, `lm_head`, and the whole MTP layer. **160 GiB.**
**Fraud B:** `cyankiwi/Hy3-AWQ-INT4` — compressed-tensors `pack-quantized`, 4 bits on
`Linear`, group_size 32, kernels `MarlinLinearKernel` + `CompressedTensorsWNA16MarlinMoEMethod`.
The `ignore` list keeps the MTP layer, layer 0, every router gate, `lm_head` and all dense
MLPs at 16 bits. **164 GiB.**
**Image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash-0731:3.0.16-overlay-k5`
(vLLM 0.25.1, mlnode 3.0.16) on both sides; `runner.py` flags replaced by `scripts/patch_hy3.py`.

## Experiments

| Experiments | GPU Honest | Version Honest | Model Honest | GPU Fraud | Version Fraud | Model Fraud | GPU Validator | Version Validator | Model Validator | Link to artifacts | graph |
|---|---|---|---|---|---|---|---|---|---|---|---|
| processed_logprobs | 2×B300 | 0.25.1 | tencent/Hy3-FP8 | 1×B300 | 0.25.1 | cyankiwi/Hy3-AWQ-INT4 | 4×H200 | 0.25.1 | tencent/Hy3-FP8 | [exp1](exp1-processed-logprobs/artifacts/) | [png](exp1-processed-logprobs/artifacts/hy3_processed_honest_vs_int4.png) |
| raw_logprobs | 2×B300 | 0.25.1 | tencent/Hy3-FP8 | 1×B300 | 0.25.1 | cyankiwi/Hy3-AWQ-INT4 | 4×H200 | 0.25.1 | tencent/Hy3-FP8 | [exp2](exp2-raw-logprobs/artifacts/) | [png](exp2-raw-logprobs/artifacts/hy3_raw_honest_vs_int4.png) |
| processed_logprobs | 2×B300 | 0.25.1 | tencent/Hy3-FP8 | 1×B300 | 0.25.1 | RedHatAI/Hy3-NVFP4-FP8 | 4×H200 | 0.25.1 | tencent/Hy3-FP8 | [exp1](exp1-processed-logprobs/artifacts/) | [png](exp1-processed-logprobs/artifacts/hy3_processed_honest_vs_nvfp4.png) |
| raw_logprobs | 2×B300 | 0.25.1 | tencent/Hy3-FP8 | 1×B300 | 0.25.1 | RedHatAI/Hy3-NVFP4-FP8 | 4×H200 | 0.25.1 | tencent/Hy3-FP8 | [exp2](exp2-raw-logprobs/artifacts/) | [png](exp2-raw-logprobs/artifacts/hy3_raw_honest_vs_nvfp4.png) |

Honest is cross-hardware on the same GPU pair as the fraud arms — generated on B300,
validated on H200. There is no same-GPU floor run anywhere in this table.

1000 multilingual prompts (en/es/zh/ar/hi), `max_tokens=1000`, `temperature=0.7`, `seed=1`,
`top_k=40`, `top_p=0.95`, `repetition_penalty=1.2`, `top_logprobs=4`, 16 concurrent workers.
6000 replays total, **0 length mismatches**.

## Results

| mode | arm | distance2 | ×floor | best F1 | TP@FP5% |
|---|---|---:|---:|---:|---:|
| processed | honest | 0.022386 | — | — | — |
| processed | NVFP4 | 0.034283 | 1.53× | 0.667 | 32.5 % |
| processed | INT4 | 0.029294 | 1.31× | 0.667 | 19.6 % |
| raw | honest | 0.049917 | — | — | — |
| raw | NVFP4 | 0.079605 | 1.59× | 0.807 | 70.0 % |
| raw | INT4 | 0.067413 | 1.35× | 0.732 | 47.8 % |

Restricted to answers of **≥ 100 tokens** — `distance2` divides by `max(100, n) · top_k + 1`,
so shorter answers are pinned near `1/401` for honest and fraud alike and only add noise:

| mode | arm | distance2 | ×floor | best F1 | TP@FP5% |
|---|---|---:|---:|---:|---:|
| processed | honest | 0.025528 | — | — | — |
| processed | NVFP4 | 0.040204 | 1.57× | 0.725 | 32.1 % |
| processed | INT4 | 0.034106 | 1.34× | 0.686 | 19.9 % |
| raw | honest | 0.058192 | — | — | — |
| raw | **NVFP4** | 0.093383 | **1.60×** | **0.918** | **88.2 %** |
| raw | **INT4** | 0.078774 | **1.35×** | 0.824 | 59.5 % |

About a quarter of the set (≈ 260 of 1000) is shorter than 100 tokens.

## Findings

1. **`raw_logprobs` is the only usable mode here.** Best F1 0.918 against 0.725 for the same
   arm. In `processed` both frauds land at F1 0.667 with precision exactly 0.500 and recall
   1.000 — a degenerate threshold at 0.0025, i.e. the `1/401` floor catching everything.
   Sentinels cancel on both sides and take the signal with the noise.
2. **INT4 is the quieter fraud, not the louder one — the reverse of DeepSeek-V4.** 1.35× and
   F1 0.824, against NVFP4's 1.60× and 0.918. On DeepSeek-V4 the same comparison ran the other
   way (INT4 1.62×, NVFP4 1.31×). Thresholds must be calibrated against INT4.
3. **More aggressive quantisation did not mean more detectable.** The INT4 build takes all
   `Linear` layers to 4 bits but keeps the MTP layer, layer 0, every router gate, `lm_head` and
   all dense MLPs at 16 bits, and MARLIN dequantises to BF16 before the matmul. The NVFP4 build
   takes the whole MoE — the bulk of the compute — to real FP4 tensor cores.
4. **Filtering by answer length is worth more than any threshold tuning.** In `raw` it moves
   F1 from 0.807 to 0.918 and TP@FP5% from 70 % to 88 %.
5. **The replay is exact across architectures.** 0 length mismatches in 6000 replays between
   Blackwell and Hopper, with different kernels on each side (`FLASHINFER_TRTLLM` NvFp4 /
   MARLIN WNA16 on B300; `FLASHINFER_CUTLASS` Fp8 + `CutlassFP8ScaledMMLinearKernel` +
   `FLASH_ATTN` on H200).
6. **Fraud economics.** Honest FP8 needs 276 GiB and does not fit one B300 (~242 GiB usable at
   `gmu 0.90`); both fraud builds do, at 160 and 164 GiB.

## Speculative decoding

Hy3 speculates via **MTP** (`num_nextn_predict_layers: 1`, `HYV3MTPModel`,
`--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`).

| configuration | distance2 | length mismatches |
|---|---:|---:|
| no speculation (floor, processed) | 0.018489 | 0 |
| executor speculates, validator does not | 0.018008 (0.97×) | 0 |
| validator speculates, stock image | 0.2072 | 82/100 |
| validator speculates, with `scripts/patch_v1_sched_nospec.py` | 0.0229 | **0** |

Speculation **on the executor is invisible to validation** (0.97× of the floor). On the
validator it destroys validation on the stock image — and silently, with no error anywhere.

Cause: `use_v2_model_runner` forces the V2 runner only for `dspark`; everything else falls to
`_is_default_v2_model_runner_model()`, false for any MoE architecture outside
`DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES`. Hy3 + MTP therefore runs V1, whose `RejectionSampler`
carries no enforced-token hook — only the bonus token reaches the plain `Sampler`. An accepted
draft also books two emitted tokens while the reply carries one, so the replay index runs ahead
of the output. Fix: keep replaying requests out of speculation at scheduling time
([kaitakuai/vllm#21](https://github.com/kaitakuai/vllm/pull/21)); other requests in the batch
keep speculating (acceptance length 1.65).

With speculation on, per-request `logprobs_mode` is ignored on the executor — `raw` silently
returns `processed`. Unfixed, and untouched by that PR.

## Files

```
exp1-processed-logprobs/artifacts/
  val_{honest,nvfp4,int4}_processed_logprobs.jsonl   validator replays, 1000 each
  hy3_processed_honest_vs_{nvfp4,int4}.png
exp2-raw-logprobs/artifacts/
  val_{honest,nvfp4,int4}_raw_logprobs.jsonl
  hy3_raw_honest_vs_{nvfp4,int4}.png
scripts/
  v4val.py                    generate / replay driver (enforced_tokens + distance2)
  patch_hy3.py                replaces the image's V4 flags with Hy3 ones
  analyze.py                  the tables above, from the artifacts
  plot_distance_vs_length.py  the scatter plots
  patch_v1_sched_nospec.py    the speculation fix, as applied to the running image
```

Each replay row keeps the executor's `inference_result`, the validator's `validation_result`,
the per-row `distance`, `len_mismatch` and sentinel fractions, so every number above is
recomputable from the artifacts with `scripts/analyze.py`.

## Reproducibility checklist

- [x] Image pinned by tag; quantisation described from `config.json` and loader output
- [x] Every script referenced above committed under `scripts/`
- [x] All tables recomputable from committed artifacts via `scripts/analyze.py`
- [x] Honest arm is cross-hardware on the same GPU pair as the fraud arms
- [x] 0 length mismatches reported per arm, not aggregated away
- [x] Negative results kept (processed mode, and the two failed speculation patches)
