# DeepSeek-V4-Flash NVFP4 on 2× B200: +42 % PoC, and CUDA graphs stop mattering

**Date:** 2026-07-25
**Models:** `nvidia/DeepSeek-V4-Flash-NVFP4` vs `deepseek-ai/DeepSeek-V4-Flash` (FP8)
**Hardware:** 2× NVIDIA B200 SXM (183,359 MiB, 1000 W, NV18), driver 595.71.05, CUDA 13, TP=2
**Image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash:0.2.13-vllm0.25.1-overlay-k4` (`sha256:2af898fa…8cbb28a`)
**PoC:** v2 plugin, seq_len 1024, k_dim 12, `--max-model-len 400000`

> V4 thresholds are **not calibrated**. L2 values are distances only — no PASS/FRAUD verdicts.

## Why this run exists

NVFP4 on B200 had only ever been *collected*, never *swept*: 1000 nonces at a continuous
batch of 32, no batch curve. That gap is why an earlier report quoted "+129 %" for NVFP4 —
a figure obtained by comparing a B200 collection against a B300 collection, folding
hardware and TP differences into what was presented as a quantisation effect.

This run measures the batch sweep in both compilation modes on one box, against an honest
FP8 reference measured on the same hardware.

## Result 1 — PoC throughput: +42 %

| batch | honest FP8 eager | honest FP8 graphs | NVFP4 eager | NVFP4 graphs |
|------:|-----------------:|------------------:|------------:|-------------:|
| 8 | 1232 | 2160 | 2320 | **2944** |
| 16 | 1344 | 2272 | 3167 | 3168 |
| **32** | **1344** | **2304** | **3263** | **3264** |

**NVFP4 delivers 3263 nonces/min against honest FP8's 2304 — +42 %.**

This supersedes the "+129 %" figure. It is consistent with the +37 % measured on 1× B300
TP=1, so the honest range for NVFP4's PoC advantage is **≈ 37–42 %**, not a doubling.

## Result 2 — CUDA graphs do nothing for NVFP4, while they are worth +71 % to honest FP8

Same box, same TP, same day:

| model | eager → graphs at batch 32 | gain |
|---|---|---:|
| honest FP8 | 1344 → 2304 | **+71 %** |
| **NVFP4** | **3263 → 3264** | **+0.03 %** |

At batch 8 NVFP4 does gain (2320 → 2944, +27 %); by batch 16 the gain is gone.

This refines the model proposed in `../deepseek-v4-flash-1xb300-cudagraph-ab/`, which said
throughput is `min(launch limit, compute limit)` and graphs lift the launch limit. That
model predicted NVFP4 at TP=2 should gain like honest FP8 does. It does not.

The resolution: **NVFP4 is fast enough that it never reaches the launch limit.** Honest FP8
sits at 1344 in eager because it is launch-bound; graphs raise it to 2304. NVFP4 already
produces 3263 in eager — above the ceiling graphs pull honest FP8 to — so there is nothing
for graphs to remove. Graphs help whoever computes slowly enough to be waiting on kernel
launches.

That statement now covers every V4 configuration measured in this series: 2×B200 FP8
(+71 %), 1×B300 FP8 (+3.8 %), 2×H200 FP8 (<5 %), 4×H100 FP8 (<5 %), 1×B300 NVFP4 (0 %),
4×H100 INT4 (0 %), 2×B200 NVFP4 (0 %).

## Result 3 — the same flag: nothing for PoC, 20× for serving

Same box, same model, same day. CUDA graphs, eager → on:

| Scenario | out tok/s | Δ |
|---|---|---:|
| s1 long prompt, sequential | 6.2 → **133.4** | **+2039 %** |
| s2 short prompt, concurrent | 145.5 → **921.9** | +533 % |
| s3 very long, sequential | 5.7 → **106.6** | +1769 % |
| s4 very long, max concurrency | 52.8 → **455.9** | +763 % |

Zero failed requests in all eight runs.

**PoC: +0.03 %. Serving: up to +2039 %.** One flag, one model, one machine, opposite
outcomes — which is the cleanest possible demonstration that the question was never
"does this model/architecture benefit from graphs" but "is *this workload* waiting on
kernel launches". The PoC forward at batch 32 is not; the decode loop, paying a full launch
round per token, very much is.

Practical consequence for the fleet: `--enforce-eager` costs almost nothing on the PoC side
here, and costs an order of magnitude on the serving side. A node that serves inference
should never run with it.

## Result 4 — reproducibility: 96.8 % is a constant of Blackwell

Three independent slices, all on B200, all giving the same number:

| what changed between the two collections | bit-identical | median L2 |
|---|---:|---:|
| compilation mode (eager ↔ graphs), same box | **968/1000** | 0.0000 |
| **the physical machine** (two separate rentals, a day apart) | **968/1000** | 0.0000 |
| nothing (same-box repeat, measured earlier in this series) | 968/1000 | 0.0000 |

**The honest floor of 0.188 is not "machine noise" — it is *GPU-model* noise.** Two
different rented B200s reproduce each other bit-for-bit; a B200 against a B300 gives
0.19–0.22 even though both are Blackwell.

Consequence for validation design: a validator does **not** need the same physical card, and
does not need to match the prover's compilation mode. It needs **the same GPU model**. Under
that condition the honest floor collapses toward zero while NVFP4 stays at 0.210 — which
turns the one genuinely dangerous fraud vector from undetectable into cleanly separable.

Also reproducible: the collection rate was 2509 nonces/min in both B200 rentals.

## Result 5 — L2

| Pair | N | bit-identical | median L2 | >0.4 |
|---|---:|---:|---:|---:|
| NVFP4 B200 vs NVFP4 B200 (other rental) | 1000 | 968 | 0.000 | 0.1 % |
| NVFP4 B200 vs honest B200 | 1000 | 0 | **0.210** | 6.4 % |
| NVFP4 B200 vs honest B300 | 1000 | 0 | 0.211 | 5.7 % |
| NVFP4 B300 vs honest B300 | 1000 | 0 | 0.210 | 6.1 % |
| NVFP4 B200 vs NVFP4 B300 | 1000 | 0 | 0.221 | 7.0 % |
| honest B200 vs honest B300 | 1000 | 0 | 0.188 | 3.7 % |

NVFP4's offset against honest is **0.209–0.212 across five independent pairs** — different
rentals, different GPU models, different directions. It is a stable property of the
quantisation.

Note the second-to-last row: two NVFP4 runs on *different GPU models* are further from each
other (0.221) than either is from an honest run on its own card (0.210). Distance is
dominated by the GPU model, not by whether the model is honest or quantised.

Machine-readable: `artifacts/l2_matrix.json`.

## Startup cost — NVFP4 is slow to come up

| phase | duration |
|---|---:|
| weight load (78.93 GiB/GPU) | ~2 min |
| `ptxas` JIT (GPU idle, one core at 100 %) | ~10 min |
| **DeepGEMM warmup, 4145 configurations** | ~10 min |
| **total to `is_running: true`** | **1395 s eager / 1155 s graphs** |

For comparison: INT4 on H100 comes up in 210 s, honest FP8 on B300 in 915 s. Any wait loop
must allow **at least an hour**, or it will abandon a healthy engine mid-warmup. Healthy
progress looks like `ptxas` at 100 % CPU with GPUs at 0 %, then `DeepGEMM warmup: N/4145`
climbing, with zero errors in the log.

## Files

| Path | What |
|---|---|
| `artifacts/nvfp4_{eager,graphs}_poc_sweep.log` | both sweeps |
| `artifacts/nvfp4_{eager,graphs}_nonces.json` | 1152 nonces per mode |
| `artifacts/nvfp4_prior_run_nonces.json` | the earlier B200 rental, for the reproducibility check |
| `artifacts/honest_fp8_*` | the honest reference, same hardware |
| `artifacts/nvfp4_{eager,graphs}_serving.json` | serving metrics, both modes |
| `artifacts/nvfp4_{eager,graphs}_backends.txt` | resolved backends |
| `artifacts/l2_matrix.json` | the table above |
| `scripts/` | box prep, the run driver, sweep, collection, L2, metrics |

## Reproduce

```bash
./scripts/prep.sh 2                                  # deps, runner args (TP=2), API on :8081
hf download nvidia/DeepSeek-V4-Flash-NVFP4
./scripts/run_model.sh nvidia/DeepSeek-V4-Flash-NVFP4 b200-nvfp4
python3 scripts/compare_nonces.py artifacts/nvfp4_eager_nonces.json artifacts/honest_fp8_eager_nonces.json
```

Rent by `gpu_ram` and check `nvidia-smi topo -m` for `NV#`; require `cuda_max_good >= 13.0`
or the CUDA-13 image dies at `init_device` on a 570-series driver. Pass `MODEL=` to
`run_pow_generation.py` — it is pinned to the default checkpoint and otherwise answers
`409 params mismatch`, surfaced as a bare `502`.

## Reproducibility checklist

- [x] Image by tag and digest; both models on the same box, same day, both modes
- [x] Raw sweeps, nonce sets and backend records committed
- [x] L2 regenerated from committed artifacts
- [x] A superseded figure ("+129 %") named and corrected
- [x] A prediction of the earlier throughput model stated, tested, and shown to fail, with the refinement
- [x] No links to `.claude/`, no absolute local paths, no host addresses
- [x] No verdicts asserted — V4 thresholds are not calibrated
