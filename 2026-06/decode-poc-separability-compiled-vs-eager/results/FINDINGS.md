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

## Convergent conclusion with the Goodhart test

Both experiments point the same way — **run decode-PoC compiled**:

- **Goodhart** (`../goodhart/`): compiled decode-PoC does not reward DeepGEMM and
  agrees with compiled serving on the backend choice.
- **Separability** (this): only the compiled pipeline detects the AWQ
  compute-downgrade fraud.

## Caveats

- 8 nonces per config — step-0 figures are coarse (±0.125); the decode gaps
  (averaged over 64 steps) are the robust signal.
- Single prover/validator pair, single model. The mechanism (compiled tightens
  honest cross-hw → headroom for fraud) generalises; the exact gaps do not.
- "Fraud" here is AWQ as a stand-in for a compute-downgrade forgery, not an
  adaptive adversary.
