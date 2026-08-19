# Hy3 fraud arm — NVFP4 (NVIDIA ModelOpt) on 2×B300 and 4×B200 — +48 % nonces by dropping tensor parallelism

**Date:** 2026-08-19
**Fraud model:** `r0b0tlab/Hy3-295B-NVFP4` — 186 GB, 100 shards, built with NVIDIA ModelOpt.
vLLM reports `quantization=modelopt_mixed` and detects a **mix** of `FP8`, `NVFP4`,
`W4A16_NVFP4` and `MXFP8` quant algos, so the repo's "W4A4" label is a simplification.
MoE kernel: **`FLASHINFER_TRTLLM` NvFp4** — real FP4 tensor cores, not MARLIN emulation.
Blackwell-only (sm_100); this arm cannot run on Hopper.

**Honest reference:** `tencent/Hy3-FP8` on each host — see
[`../hy3-fp8-honest-baseline/`](../hy3-fp8-honest-baseline/). Reference sets are duplicated
here as `ref_nonces_*` so the folder is self-contained.

**Hardware:**
- 2× **B300 SXM6** (1100 W, 275 GB, NV18, driver 610.57.04, sm_100) — Vast 48124506
- 4× **B200 SXM** (1000 W, 183 GB, NV18, driver 580.126.20, sm_100) — Vast 48135501

**Image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash-0731:3.0.16-overlay-k5`
**Digest:** `sha256:8ce8830b4182b3dfd642c5e00f548f50a830611e4bc64ec4fbd84fe91070e3aa`

## Summary

Two findings, both about how the attack actually pays and how reliably it can be measured.

**1. The gain is not from quantisation — it is from not paying for tensor parallelism.**
At TP=2 this arm and the honest FP8 arm produce *identical* throughput (1599 vs 1599
nonces/min). But 4-bit weights fit in a single B300 (166.48 GiB against ~242 GiB usable),
while honest FP8 does not (276 GiB). Two independent TP=1 instances on the same two cards
therefore produce **2366 nonces/min against the honest 1599**.

| Scenario on 2× B300 | nonces/min | per card |
|---|---:|---:|
| honest FP8, TP=2 | 1599 | 800 |
| this arm, TP=2 | 1599 | 800 |
| **this arm, 2 × TP=1** | **2366 (+48 %)** | **1183** |

**2. The fingerprint is a property of the build, and it reproduces across machines.**
Measured against honest FP8 on two different hosts, this checkpoint gives the same distance
to three decimal places — while a different NVFP4 build of the same scheme
([llm-compressor arm](../hy3-fraud-nvfp4-llmcompressor/)) sits a third lower at 0.367.

| host | L2 median vs honest (s1 / s2 / s3) |
|---|---|
| 2×B300 | 0.4926 / 0.4897 / 0.5038 |
| 4×B200 | 0.4909 / 0.4890 / 0.5002 |

## Environment

| Parameter | Value |
|---|---|
| CUDA | 13.0.2 (image); drivers 610.57.04 (B300), 580.126.20 (B200) |
| vLLM | 0.25.1, build `752a3a5` |
| Python | 3.12.13 |
| mlnode | 3.0.16 |

## Config

Identical to the honest baseline (`scripts/patch_hy3.py`); only the model id and `TP` vary:

```bash
TP=2 python3 scripts/patch_hy3.py      # TP=1 for the single-card economics arm
curl -X POST http://127.0.0.1:8081/api/v1/inference/up/async \
  -H 'Content-Type: application/json' \
  -d '{"model":"r0b0tlab/Hy3-295B-NVFP4","dtype":"auto","additional_args":[]}'
```

Measurement window **120 s**, batches 16/32/64.

### What changed vs the honest arm

| Parameter | Honest | This arm |
|---|---|---|
| model | `tencent/Hy3-FP8` (276 GiB) | `r0b0tlab/Hy3-295B-NVFP4` (166 GiB at TP=1) |
| MoE kernel | FP8 path | `FLASHINFER_TRTLLM` NvFp4 |
| TP | 2 (B300) / 4 (B200) | same, **plus a TP=1 arm impossible for honest** |

## Validation

### Throughput (120 s window, corrected accounting)

| batch | B300 honest TP=2 | B300 this arm TP=2 | B300 this arm **TP=1** | B200 honest TP=4 | B200 this arm TP=4 |
|---:|---:|---:|---:|---:|---:|
| 16 | 1439 | 1304 | 1120 | 1736 | 1776 |
| 32 | 1535 | 1552 | 1168 | 1840 | 1872 |
| 64 | **1599** | **1599** | **1183** | **1888** | **1920** |

TP=2 scales only 1.35× over TP=1 — the arithmetic that makes the attack profitable. On
B200 the arm is 1.7 % ahead of honest at the same topology.

### Fingerprint

```bash
python3 scripts/l2_matrix.py artifacts ref_nonces_b300_fp8_s1.json nonces_b300_nvfp4_s1.json
python3 scripts/l2_matrix.py artifacts ref_nonces_b200_fp8_s1.json nonces_b200_r0_s1.json
```

| Pair | bit-identical | L2 median | p95 | >0.40 |
|---|---:|---:|---:|---:|
| this arm ↔ itself, B300 repeat | **100.0 %** | 0.0000 | 0.0000 | 0.0 % |
| this arm ↔ itself, B200 repeat | **100.0 %** | 0.0000 | 0.0000 | 0.0 % |
| honest ↔ this arm, B300 s1 | 0.0 % | 0.4926 | 0.8580 | 73.3 % |
| honest ↔ this arm, B300 s2 | 0.0 % | 0.4897 | 0.8543 | 70.6 % |
| honest ↔ this arm, B300 s3 | 0.0 % | 0.5038 | 0.8783 | 75.0 % |
| honest ↔ this arm, B200 s1 | 0.0 % | 0.4909 | 0.8731 | 72.7 % |
| honest ↔ this arm, B200 s2 | 0.0 % | 0.4890 | — | 70.7 % |
| honest ↔ this arm, B200 s3 | 0.0 % | 0.5002 | — | 75.8 % |
| honest H200 ↔ this arm (cross-arch) | 0.0 % | 0.4906 | 0.8543 | 73.7 % |

Both the honest arm and this one are **bit-exact on repeat** within a single machine, so on
matched hardware a single differing nonce is already a verdict. That property does not
survive a change of host — see the honest baseline for the B200↔B300 floor of 0.2028.

### Inference

Warm-engine numbers, throughput recomputed from `measurements`.

| Scenario | B300 honest | B300 this arm | Δ | B200 this arm |
|---|---:|---:|---:|---:|
| s1 long, sequential | 60.3 | 114.3 | **+90 %** | — |
| s2 short, 30 runners | 840.8 | 1294.5 | **+54 %** | — |
| s3 very long, sequential | 87.1 | 93.4 | +7 % | — |
| s4 very long, 20 runners | 269.2 | 356.7 | +33 % | — |

### Resources

| Arm | weights | KV tokens | bringup |
|---|---:|---:|---:|
| honest FP8 TP=2 (B300) | 276 GiB | 1 172 144 | 508 s |
| this arm TP=2 (B300) | — | 1 914 384 | 363 s |
| **this arm TP=1 (one B300)** | **166.48 GiB** | 415 104 | — |

A batch-64 sweep needs 65 536 KV tokens, so the single-card arm still has a 6× margin.

## Findings

1. **Quantisation buys no PoC throughput; escaping TP does.** 1599 = 1599 at TP=2, +48 %
   only via two TP=1 instances. The right economic metric is nonces **per card at the
   minimum viable topology**, not nonces per box.
2. **The fingerprint reproduces across machines to three decimals** (0.4909 on B200 vs
   0.4926 on B300), which both validates the methodology and shows the distance belongs to
   the *build*, not the hardware.
3. **This build is loud.** At 0.49 it is the most detectable fraud measured for Hy3; the
   llm-compressor build of the same scheme sits at 0.367. Threshold calibration must target
   the quietest known build, not this one.
4. **Bit-exactness holds for quantised weights too** — it is an architecture property.
5. **`r0b0tlab` is not a pure W4A4 checkpoint**; vLLM reports a mixed FP8/NVFP4/MXFP8 config.

## Files

```
artifacts/
  nonces_b300_nvfp4_{s1,s1_r2,s2,s3}.json   this arm on 2×B300
  nonces_b200_r0_{s1,s1_r2,s2,s3}.json      this arm on 4×B200
  ref_nonces_b300_fp8_*.json / ref_nonces_b200_fp8_*.json   honest references, same hosts
  sweep_b300_nvfp4_120s.log                 TP=2
  sweep_b300_nvfp4_tp1_120s.log             TP=1 — the economics number
  sweep_b200_nvfp4_120s.log                 TP=4
  serving_b300_nvfp4.sqlite / serving_b200_nvfp4.sqlite
scripts/
  patch_hy3.py  run_pow_generation.py  collect_artifacts.py  l2_matrix.py  poc_seeds.json
```

## Reproducibility checklist

- [x] Image pinned by digest; quantisation described from vLLM's own loader output
- [x] Honest references committed alongside — folder is self-contained
- [x] All scripts committed under `scripts/`
- [x] 120 s window for every quoted throughput number
- [x] L2 tables reproducible via `scripts/l2_matrix.py`
- [x] 3 seeds per claim; the key result repeated on two independent hosts
- [x] No internal-tooling links, absolute paths or sibling-repo references
