# Are the V4 L2 findings a property of one seed? No — and the batch artifact is a knob, not noise

**Date:** 2026-07-27
**Models:** `deepseek-ai/DeepSeek-V4-Flash` (FP8) vs `nvidia/DeepSeek-V4-Flash-NVFP4`
**Hardware:** 1× NVIDIA B300 SXM6 AC (275,040 MiB, 1100 W), driver 580.126.09, CUDA 13.0, TP=1
**Image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash:0.2.13-vllm0.25.1-overlay-k4` (`sha256:2af898fa…8cbb28a`)
**PoC:** v2 plugin, seq_len 1024, k_dim 12, `--max-model-len 400000`

> V4 thresholds are **not calibrated**. L2 values are distances only — no PASS/FRAUD verdicts.

## Why this run exists

Every L2 number in this series — the 0.188 honest floor, NVFP4's 0.210 offset, INT4's 0.296 —
was measured on a **single** `block_hash`/`public_key` pair (`artifact_collection_block_v1` /
`artifact_collection_pk_v1`). Since all 1000 vectors are derived from
`murmur3(block_hash + public_key + nonce_index)`, the entire dataset is a function of that one
pair. Nothing proved those offsets were not artefacts of that specific seed.

This run repeats the key measurement on **five independent seeds**, and — prompted by a
reviewer's hypothesis — also tests whether the residual non-reproducibility tracks the batch
size.

The seed set (`scripts/poc_seeds.json`) is derived deterministically by sha256 from the phrase
`"kaitakuai V4 PoC validation seed set, generated 2026-07-25"`, so it is reproducible and
demonstrably not cherry-picked. It is fixed: comparisons across runs are only meaningful with
the same seeds.

## Result 1 — NVFP4's offset does not depend on the seed

NVFP4 vs honest FP8, same card, same seed on both sides:

| seed | n | median L2 | p95 | mismatch >0.4 | p-value |
|---|---:|---:|---:|---:|---:|
| №1 `bce35dd7…` | 1000 | 0.2066 | 0.397 | 50 (5.0 %) | 1.000 |
| №2 `013161c6…` | 1000 | 0.1993 | 0.401 | 51 (5.1 %) | 1.000 |
| №3 `ab21f384…` | 1000 | 0.1997 | 0.394 | 45 (4.5 %) | 1.000 |
| №4 `1a6c3ce1…` | 1000 | 0.2012 | 0.384 | 41 (4.1 %) | 1.000 |
| №5 `b9bfb5ff…` | 1000 | 0.2098 | 0.399 | 48 (4.8 %) | 1.000 |
| **spread** | | **0.1993–0.2098** | | 4.1–5.1 % | |
| earlier measurement (July, old seed) | 1000 | 0.2100 | — | 5.7–6.4 % | 1.000 |

**The spread across seeds is ±0.005 — smaller than the spread previously seen across machines
(0.209–0.216).** The July figure of 0.210 sits inside the new range. The offset is a property
of the quantisation, not of the input seed.

The `p-value` is **1.000 on every seed**: under the chain rule with `p_mismatch = 0.10` and
`dist_threshold = 0.40`, NVFP4 is indistinguishable from an honest node. This was already known
from the single-seed data; it is now confirmed to be seed-independent, which removes the last
hope that a different seed might expose it.

## Result 2 — the control: independent seeds saturate the metric

Honest vs honest across **different** seeds — the vectors come from different inputs, so this
should reach the asymptote for uncorrelated 12-dim vectors:

| pair | median L2 | | pair | median L2 |
|---|---:|---|---|---:|
| №1 ↔ №2 | 1.418 | | №2 ↔ №4 | 1.410 |
| №1 ↔ №3 | 1.415 | | №2 ↔ №5 | 1.416 |
| №1 ↔ №4 | 1.430 | | №3 ↔ №4 | 1.406 |
| №1 ↔ №5 | 1.414 | | №3 ↔ №5 | 1.409 |
| №2 ↔ №3 | 1.414 | | №4 ↔ №5 | 1.409 |

All ten pairs land in **1.406–1.430**, matching what completely different *models* give
(Kimi-K2.6 1.420, MiniMax-M2.7 1.413, Qwen3-235B 1.390). Two independent routes to "maximally
distant" agree, which confirms both that the seeds are mutually uncorrelated and that ~1.41 is
the scale's ceiling rather than a property of any particular model.

**Practical warning:** comparing nonce sets collected under *different* seeds yields ~1.41 — the
same value as a foreign model. Always carry the seed in the filename; a mismatched comparison
looks exactly like fraud.

## Result 3 — the residual 3.2 % is the first nonce of every batch, and the period *is* the batch size

Collecting twice on the same card, changing only the compilation mode:

| batch size | differing nonces | positions | step |
|---:|---:|---|---:|
| **8** | **125 / 1000 (12.5 %)** | every `n mod 8 == 0` | 8 (124 times, no exception) |
| **32** | **32 / 1000 (3.2 %)** | every `n mod 32 == 0` | 32 (31 times, no exception) |

The divergence sits **exactly on the first element of each batch**, and its frequency is
`1 / batch_size`. This confirms a reviewer's hypothesis that the effect comes from
recompilation / path selection at the batch boundary rather than from inherent
nondeterminism.

Per seed at batch 32, the structure is identical:

| seed | bit-identical | differing | first-in-batch median L2 | of those, >0.4 | all others, max L2 |
|---|---:|---:|---:|---:|---:|
| №1 | 968 | 32 (all `mod 32 = 0`) | 0.188 | 2 | **0.0000** |
| №2 | 968 | 32 (all `mod 32 = 0`) | 0.168 | 0 | **0.0000** |
| №3 | 968 | 32 (all `mod 32 = 0`) | 0.172 | 1 | **0.0000** |

**968 of 1000 nonces are identical to the last bit — max distance exactly zero.** The artefact
is fully deterministic in *where* it occurs and seed-independent; only how many of the 32 cross
0.4 varies (0 to 2).

### What this changes

**"Blackwell reproduces 96.8 % bit-identical" is not a hardware property.** It is
`1 − 1/32`, i.e. an artefact of the batch size we happened to use. At batch 8 the same hardware
gives 87.5 %; at batch 128 it would give 99.2 %. Earlier reports in this series quote 96.8 % as
a characteristic of Blackwell — that number is only meaningful alongside the batch size.

**For threshold calibration this matters more.** An honest node compared against an honest
validator produces 0–2 mismatches per 1000 purely from this artefact at batch 32, and would
produce proportionally more at smaller batches. If prover and validator run *different* batch
sizes, the structural mismatch rate rises toward 12.5 % — close enough to the 10 % baseline that
the direction of the null hypothesis starts deciding verdicts. The artefact should be subtracted
explicitly, not absorbed into the honest-noise budget.

## Result 4 — throughput is seed-independent

Collection rate, nonces/min: honest FP8 gives 1428 / 1426 / 1426 / 1425 / 1425 across the five
seeds — a spread of 0.2 %. NVFP4 gives 2175 / 2245 / 2246 / 2245 / 2245. Batch size does move
it (honest: 1310 at b8, 1401 at b16, 1425 at b32), as expected.

## Files

| Path | What |
|---|---|
| `artifacts/honest_s{1..5}_b32.json` | honest FP8, five seeds |
| `artifacts/honest_s1_b{8,16}.json` | honest FP8, seed №1, smaller batches |
| `artifacts/honestg_s{1,2,3}_b32.json`, `honestg_s1_b8.json` | same with CUDA graphs |
| `artifacts/nvfp4_s{1..5}_b32.json` | NVFP4, five seeds |
| `artifacts/summary.json` | every table above, machine-readable |
| `scripts/poc_seeds.json` | the fixed seed set with its provenance |
| `scripts/seedrun.sh` | the run driver |
| `scripts/collect_artifacts.py` | collector, now accepting `--block-hash` / `--public-key` |

## Reproduce

```bash
# the collector takes the seed explicitly; the sweep takes it from the environment
python3 collect_artifacts.py --url $API --model "$M" --output-dir out \
  --nonces 1000 --batch-size 32 --block-hash <bh> --public-key <pk>
POC_BLOCK_HASH=<bh> POC_PUBLIC_KEY=<pk> python3 run_pow_generation.py --phase 3 --skip-check
```

Before trusting a seeded run, verify two things: the collector prints
`PoC seeds: block_hash=…` at startup (proof the parameter reached the computation), and two
different seeds give ~1.41 (proof they are independent). Both checks are cheap at 64 nonces.

## Reproducibility checklist

- [x] Five independent seeds, provenance of the set documented and reproducible
- [x] Instrument validated before the run (seed reaches the computation; seeds are uncorrelated)
- [x] Raw nonce sets for every cell committed
- [x] All tables regenerated from the committed artifacts
- [x] A previously reported figure (96.8 % as a hardware property) identified as batch-dependent
- [x] No links to `.claude/`, no absolute local paths, no host addresses
- [x] No verdicts asserted — V4 thresholds are not calibrated
