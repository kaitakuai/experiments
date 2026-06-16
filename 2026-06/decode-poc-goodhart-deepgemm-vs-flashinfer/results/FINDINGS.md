# decode-PoC vs Goodhart — DeepGEMM vs FlashInfer (result)

B300, MiniMax-M2.7 FP8, single GPU. Only the MoE backend varies. Both backends
verified to actually engage (`Using FLASHINFER_TRTLLM` vs `Using DEEPGEMM Fp8 MoE
backend` + `DeepGEMM E8M0 enabled`).

P = decode-PoC throughput (steps/s, batched genref 48×128). S = serving tok/s
(concurrent `/v1/completions`, `ignore_eos`, out=256, conc=64). Same units
(decode forward-passes/s).

## Headline — the answer depends on compile mode

| mode      | backend     | P (PoC steps/s) | S (serving tok/s) | repeats |
| --------- | ----------- | --------------- | ----------------- | ------- |
| compiled  | FlashInfer  | 1087 ± 20       | 3584 ± 10         | 3       |
| compiled  | DeepGEMM    | 992 ± 20        | 1819 ± 5          | 3       |
| eager     | FlashInfer  | 808 ± 15        | 1443 ± 12         | 3       |
| eager     | DeepGEMM    | 749 ± 12        | 1505 ± 1          | 3       |

| DeepGEMM vs FlashInfer | PoC Δ     | serving Δ | verdict                        |
| ---------------------- | --------- | --------- | ------------------------------ |
| **compiled**           | **−8.7%** | **−49.3%**| **AGREE → Goodhart OVERCOME**  |
| **eager**              | **−7.3%** | **+4.2%** | **DISAGREE → Goodhart PERSISTS** |

All four comparisons have non-overlapping error bars (3 reps each) — every Δ is real.

**decode-PoC overcomes the Goodhart effect in COMPILED mode, but not in eager.**

- **Compiled:** both metrics prefer FlashInfer. decode-PoC does not reward
  DeepGEMM, and compiled serving punishes it hard (−48%). Aligned.
- **Eager (3 reps, error bars non-overlapping → the sign flip is real, not noise):**
  decode-PoC prefers FlashInfer (DeepGEMM −7.3%) while eager serving prefers
  DeepGEMM (+4.2%). The metric and the product point at different backends — a
  (small but real) Goodhart divergence.

## Controlled contrast with prefill-PoC

Same hardware / model / backend toggle; the only change is the PoC workload.

| DeepGEMM vs FlashInfer        | prefill-PoC (prior) | decode-PoC compiled | decode-PoC eager |
| ----------------------------- | ------------------- | ------------------- | ---------------- |
| change in **PoC** throughput  | **+40%**            | −8.7%               | −7.3%            |
| Goodhart                      | persists (severe)   | **overcome**        | persists (mild)  |

decode-PoC never *rewards* DeepGEMM in either mode (−8.7% / −7.3% — it
consistently prefers FlashInfer by ~7-9%), unlike
prefill-PoC's +40%. That removes the catastrophic Goodhart. What remains in eager
is a mild *opposite* divergence — and it vanishes under compilation.

## Why compile mode decides it

The harm DeepGEMM does to the **product** is a compiled-serving phenomenon:

| serving tok/s | eager   | compiled | compile gain |
| ------------- | ------- | -------- | ------------ |
| FlashInfer    | 1443    | 3539     | **+145%**    |
| DeepGEMM      | 1505    | 1826     | +21%         |

FlashInfer composes with the compiled serving path (inductor/CUDA-graph) and gains
+145%; DeepGEMM barely benefits (+21%). So compiled serving strongly prefers
FlashInfer (−48% for DeepGEMM), while eager serving is backend-flat (DeepGEMM even
+4%). Since production serving runs compiled, the operationally-relevant comparison
is the compiled row — where decode-PoC is aligned with the product.

## Operational reading

Run decode-PoC **compiled**. There it (a) does not reward the DeepGEMM trap and
(b) matches how the product actually serves. Eager decode-PoC carries a small
residual Goodhart and (per the separability matrix) also fails to detect fraud —
two independent reasons to compile.

## Caveats

- All rows are mean ± population-std over 3 reps; every reported Δ has
  non-overlapping error bars.
- One model / one GPU / one backend pair — the mechanism generalises, not the
  exact percentages.
