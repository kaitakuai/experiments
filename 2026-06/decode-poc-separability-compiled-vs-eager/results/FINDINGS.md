# 7-case compiled/eager separability — result

Cross-hardware fraud separability of the chained-decode seal: prover = B300 (FP8,
`TP=1`), validator = 4×H100 (`TP=4`), MiniMax-M2.7. honest = validator runs FP8
(real); fraud = validator runs AWQ (compute-downgrade forgery). Per-step k-id
mismatch, teacher-forced against the prover reference. `gap = fraud − honest`
(separability; higher = better). 8 nonces × (1 prefill + 64 decode).

| # | case                                 | honest | fraud | **gap**   |
| - | ------------------------------------ | ------ | ----- | --------- |
| 1 | compiled prefill + compiled decode   | 0.277  | 0.581 | **+0.304** |
| 2 | compiled prefill only (step 0)       | 0.250  | 0.750 | **+0.500** |
| 3 | compiled decode only (steps 1..)     | 0.277  | 0.578 | **+0.301** |
| 4 | eager prefill + eager decode         | 0.444  | 0.467 | +0.023    |
| 5 | eager prefill only (step 0)          | 0.250  | 0.750 | **+0.500** |
| 6 | eager decode only (steps 1..)        | 0.447  | 0.463 | +0.016    |
| 7 | eager prefill + compiled decode      | 0.552  | 0.575 | +0.023    |

## Findings

1. **Decode separates fraud only in the fully-compiled pipeline.** Compiled
   prefill+decode (case 3) gives a decode gap of **+0.30**. Eager decode (case 6)
   and compiled-decode-under-eager-prefill (case 7) collapse to **~+0.02** — no
   separation. It is the *whole* pipeline being compiled (cc) that matters, not
   the decode step alone.

2. **Mechanism.** Compiled execution makes the honest cross-hardware decode
   mismatch *tight* (0.277); eager leaves it *noisy* (0.444–0.557). A fraud raises
   the mismatch to ~0.58 in both, but only the tight compiled baseline leaves
   headroom to detect it. (Matches the prior compiled-vs-eager cross-hw figure:
   27.7% vs 41.9%.) In eager, honest noise (0.44) already sits on top of fraud
   (0.46) — the signal is drowned.

3. **The prefill step is the strongest, compile-robust discriminator** (gap ≈
   +0.50, cases 2 & 5). The AWQ forgery diverges on 75% of prefill seeds vs 25%
   for honest, regardless of compile mode. (At 8 nonces the step-0 numbers carry
   ±0.125 quantisation noise — the 0.750 vs 0.625 split in cases 2/5/7 is one
   nonce.)

4. **This unlocks separability eager could not achieve.** The earlier eager
   real-KV result had honest 0.44 ≈ fraud 0.46 (no separation). Compiling the
   pipeline opens honest 0.28 vs fraud 0.58 — a usable gap.

## N=64 confirmation (per-nonce, tight CIs)

The 7-case table above is the N=8 per-step study. To put proper confidence
intervals on the core compiled-vs-eager decode comparison, the `cc` and `ee`
configs were rerun at **N=64 nonces** (same B300 prover / 4×H100 validator),
measured **per-nonce** (the nonce is the independent unit), decode-only,
teacher-forced. Raw data and analyzer: `results/n64/`,
`code/analyze_crosshw_n64.py`, `code/crosshw-validator.sh`.

| config | honest decode | fraud decode | gap ± 95% CI | separates? |
| ------ | ------------- | ------------ | ------------ | ---------- |
| cc (compiled) | 0.199 ± 0.020 | 0.569 ± 0.017 | **+0.369 ± 0.027** | yes (CI excludes 0) |
| ee (eager)    | 0.441 ± 0.021 | 0.465 ± 0.020 | **+0.024 ± 0.029** | no (CI includes 0) |

Difference cc − ee = **+0.345 ± 0.039** — compiled separates significantly more.

1. The qualitative finding is statistically solid at N=64 — compiled separates
   (gap CI well clear of 0), eager does not (gap CI spans 0).
2. It *strengthens* vs N=8: the compiled honest floor dropped 0.28 → 0.20, so the
   gap grew 0.30 → 0.37.
3. N=8 was directionally correct but imprecise — the ~0.08 shift in the compiled
   honest floor is exactly the ±0.06 CI expected from 8 nonces.

## Convergent conclusion with the Goodhart test

Both experiments point the same way — **run decode-PoC compiled**:

- **Goodhart** (`../goodhart/`): compiled decode-PoC does not reward DeepGEMM and
  agrees with compiled serving on the backend choice.
- **Separability** (this): only the compiled pipeline detects the AWQ
  compute-downgrade fraud *in this TP-mismatched cross-hardware setup* (see Caveats).

## Caveats

- The core cc-vs-ee decode comparison is now **N=64** with tight per-nonce CIs
  (above). The full 7-case slicing (prefill-only, `ec`) remains N=8 — step-0
  figures there are coarse (±0.125).
- **Tensor-parallelism confound (important).** The prover runs `TP=1` (one B300,
  ~288 GB, fits the FP8 model) and the validator runs `TP=4` (the ~215 GB model
  does not fit one 80 GB H100), so the cross-hardware test also mixes `TP=1` vs
  `TP=4`. MoE expert-routing flips between TP=1 and TP=4 are a major divergence
  source, so the high eager honest floor (~0.44) is plausibly TP-routing-driven,
  and compilation may specifically dampen it. Thus "only the compiled pipeline
  separates" may be partly a TP1↔TP4 effect rather than a pure compiled-vs-eager
  law. A clean control needs **matched TP on both sides** (e.g. 4×H100 TP=4 vs
  4×B200 TP=4); it has not been run.
- Single prover/validator pair, single model. The mechanism (compiled tightens the
  honest cross-hw floor → headroom for fraud) generalises; the exact gaps do not.
- "Fraud" here is AWQ as a stand-in for a compute-downgrade forgery, not an
  adaptive adversary.
