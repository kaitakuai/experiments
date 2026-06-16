# Partial results — same-hardware config coupling (prover side)

Measured on the **prover** alone (single GPU, FP8, `TP=1`, MiniMax-M2.7), from the
three free-run reference trajectories `refs/ref_{cc,ee,ec}.json` (8 nonces ×
1+64 steps, sphere codebook 16 points). Reproduce: `python3 code/analyze_samehw.py results/refs`.

This is the **floor** of how much the compile config alone perturbs the seal,
before adding cross-hardware heterogeneity or honest-vs-fraud — i.e. it is *not*
the separability number. The cross-hw honest/fraud matrix (the 7 cases) needs the
validator and is pending.

## Controls (passed)

- **Determinism / shared-prefill:** `ee` and `ec` use identical eager prefill and
  produce a **bit-identical step 0** (`agree = 1.000`). This validates the harness
  and lets `ee` vs `ec` isolate the decode path.
- **Compiled prefill is non-degenerate:** `cc` step-0 k-ids are input-dependent
  (5/8 distinct across nonces; all 16 codebook cells used over the trajectory).
  The compiled prefill produces real outputs — just in different cells than eager.

## Findings

| Component | Comparison (same GPU)        | Disagreement | Notes                          |
| --------- | ---------------------------- | ------------ | ------------------------------ |
| Prefill   | compiled vs eager (`cc`↔`ee`, step 0) | **100%** | every nonce lands in a different cell |
| Decode    | compiled vs eager (`ee`↔`ec`, step 1) | **37.5%** | clean (identical state into step 1) |
| Decode    | compiled vs eager (`ee`↔`ec`, step 2) | 62.5%        | cascade has begun — not a clean step |

`cc` vs `ee` / `cc` vs `ec` diverge already at step 0 (prefill differs), so their
whole trajectories cascade (mean first-divergence = step 0). `ee` vs `ec` agree
through step 0 and first diverge around step 2.

## Reading

On **identical hardware**, switching the compile mode is a hard partition of the
seal output:

- **Prefill is maximally compile-sensitive** — compiled and eager prefill disagree
  on 100% of nonces at the first quantized index. A validator running a different
  compile mode than the prover will reject every honest prefill.
- **Decode is moderately compile-sensitive** — ~37.5% of nonces flip at the first
  clean decode step under compiled vs eager.

Operational consequence (independent of the cross-hw result): **all validators
must run the exact same compile configuration as the prover**, and the prefill
step is the component that makes mixing compile modes fatal. This is the prover-
side, same-hardware confirmation; the cross-validator separability (honest FP8 on
other hardware vs AWQ forgery) is measured by the pending validator run.
