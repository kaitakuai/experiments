# NVFP4 against 0731 on 2×B200: +38.9 % PoC — the hole is topology-dependent, the invisibility is not

**Date:** 2026-08-02
**Honest baseline:** `../deepseek-v4-flash-0731-2xb200` (same box, same seeds, same instrument)
**Candidate:** `MJPansa/DeepSeek-V4-Flash-0731-NVFP4` (175.6 GB, `quant_method: fp8, group_size: 16`)
**Hardware:** 2× NVIDIA B200 SXM 183 GB (1000 W, NV18), TP=2, driver 580.126.09, CUDA 13
**Image:** `mlnode-b300-deepseek-v4-flash:0.2.14-vllm0.25.1-overlay-k10`
**Digest:** `sha256:a6213dac621c1634a82940533190c9a5149b6535a5690c69ca6d3919c74c8138`

> V4 thresholds are **not calibrated**. L2 values are distances; the p-value is quoted under the
> parameters this series has used throughout (`dist_threshold 0.40, p_mismatch 0.10`), not as a
> network policy.

The second topology for this fraud vector, after `../deepseek-v4-flash-0731-nvfp4-1xb300`.
The question it answers: does the hole found on one card survive a different tensor split.

## Summary

- **Yes for the invisibility, no for the size of the prize.** NVFP4 earns **+38.9 %** here
  (3200 nonces/min against 2304 honest) versus **+63 %** on 1×B300. The profit is a property of
  the topology; the detectability is not.
- **Median L2 0.197–0.203 at 1.7–3.2 % mismatches**, p-value 1.000 — the same numbers as on
  1×B300 (0.196–0.200 at 2.5–3.0 %). Against the honest floor between two Blackwell boxes
  (0.177–0.181), the gap is ~0.020 on both topologies.
- The July measurements on the previous checkpoint showed the same topology split: +41.6 % on
  2×B200 against +63 % on 1×B300. **The refresh changed neither number.**

## Result 1 — what the fraud earns

Nonces/min, batch sweep:

| batch | honest | **NVFP4** | gain |
|---:|---:|---:|---:|
| 8 | 2032 | 2416 | +18.9 % |
| 16 | 2272 | 3104 | +36.6 % |
| 32 | **2304** | **3200** | **+38.9 %** |

| topology | honest | NVFP4 | gain | July, previous checkpoint |
|---|---:|---:|---:|---:|
| 1×B300 TP=1 | 1728 | 2816 | **+63.0 %** | +63 % (1664 → 2720) |
| **2×B200 TP=2** | 2304 | 3200 | **+38.9 %** | +41.6 % (2304 → 3263) |

A node on a single B300 gains far more from this fraud than one on two B200s. That is worth
knowing for whoever sets policy: the incentive is not uniform across the fleet, and it is
strongest exactly where the honest baseline is weakest.

## Result 2 — is it detectable?

NVFP4 against honest, same box, same three seeds, both without speculation:

| seed | n | median L2 | mismatches > 0.4 | p-value |
|---|---:|---:|---:|---:|
| s1 | 1000 | 0.2026 | 17 (1.7 %) | 1.000 |
| s2 | 1000 | 0.1980 | 32 (3.2 %) | 1.000 |
| s3 | 1000 | 0.1965 | 20 (2.0 %) | 1.000 |
| *1×B300, for comparison* | | *0.196–0.200* | *2.5–3.0 %* | *1.000* |
| *honest floor, 2×B200 ↔ 1×B300* | | *0.177–0.181* | *0.8–2.1 %* | |

**The two topologies agree to the third decimal.** Whatever a validator would threshold on, it
sees the same thing at TP=1 and TP=2 — a distribution sitting about 0.020 above the honest
floor with tails that run into it. The binomial test at `p_mismatch = 0.10` returns 1.000 on
every seed, on both topologies.

So the finding is not an artefact of one machine: **NVFP4 is profitable and, under the current
parameters, invisible, regardless of how the model is split across cards.**

*What is still not measured:* NVFP4 is compared against honest on the *same* box, which isolates
the quantisation effect. A real validator runs elsewhere and sees quantisation summed with
cross-machine noise. That pairing is absent from both NVFP4 runs and should be measured before
any threshold is set.

## Result 3 — serving

| scenario | honest | honest + DSpark | **NVFP4** |
|---|---:|---:|---:|
| s1 — 20k prompt, sequential, 300 tok | 60.2 | 232.7 | 88.4 |
| s2 — 2k prompt, concurrency 30 | 1478.6 | 855.9 | 1652.3 |
| s3 — 45k prompt, sequential, 1000 tok | 104.5 | 344.0 | 116.2 |
| s4 — 45k prompt, concurrency 20 | 1383.5 | 2963.4 | 1521.9 |

Tokens/s, zero failures. Raw NVFP4 beats raw honest by 12–47 % — the quantisation genuinely is
faster. But against an honest node **that speculates**, it loses by 3× in the two single-stream
scenarios, exactly as on 1×B300, because DSpark does not work on the quantised checkpoint
(measured there: acceptance collapses to 1.14–1.24 tokens per step against 3.6–6.0).

The fraudster therefore trades away the single largest serving improvement the new checkpoint
offers, in exchange for PoC weight. **A reward that prices delivered service, not only PoC,
charges for that trade.** Note also s2, where the honest node's best move is to leave DSpark
*off* — there the fraudster's raw speed advantage stands unopposed.

## Files

| path | what |
|---|---|
| `artifacts/summary.json` | every table above, machine-readable |
| `artifacts/nonces_nvfp4_{s1,s2,s3}.json` | 3 × 1000 nonces, batch 32, three fixed seeds |
| `artifacts/serving_nvfp4.json` | four scenarios |
| `artifacts/logs/sweep_nvfp4_dspark_off.log` | the PoC sweep |
| `artifacts/logs/b200_all.log`, `artifacts/logs/env_b200.txt` | campaign log and environment |
| `../deepseek-v4-flash-0731-2xb200/artifacts/` | the honest baseline this is measured against |
| `scripts/` | the same instruments as the honest run |

`run_pow_generation.py` hardcodes the model name; the NVFP4 arm needs it edited, otherwise the
PoC plugin answers `409 params mismatch` and the proxy surfaces a bare `502`.

## Reproduce

```bash
# after the honest run in ../deepseek-v4-flash-0731-2xb200
hf download MJPansa/DeepSeek-V4-Flash-0731-NVFP4
sed -i 's|^MODEL_NAME = .*|MODEL_NAME = "MJPansa/DeepSeek-V4-Flash-0731-NVFP4"|' scripts/run_pow_generation.py
MODEL=MJPansa/DeepSeek-V4-Flash-0731-NVFP4 TAGPREFIX=nvfp4 SKIP_DSPARK=1 bash scripts/b200_run.sh
```

`SKIP_DSPARK=1` because the NVFP4 + DSpark combination was already measured on 1×B300 and
collapses to ~1.2 accepted tokens per step; repeating it here would confirm a known result at
the cost of another cold bring-up.

## Reproducibility checklist

- [x] Fraud and honest measured on the same box, same seeds, same instrument, back to back
- [x] Image pinned by digest
- [x] Three independent seeds behind the detection claim
- [x] Compared against the same vector on a second topology, and against July's numbers
- [x] p-value quoted with its parameters; no calibrated threshold asserted
- [x] The unmeasured pairing (fraud on one machine vs validator on another) stated explicitly
- [x] A deliberately skipped arm named, with the reason
- [x] No links to `.claude/`, no absolute local paths, no host addresses
