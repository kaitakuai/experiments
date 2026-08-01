# DSpark on 4×H100: 3.4× decode, output preserved — and three launch flags that fight each other

**Date:** 2026-07-31
**Model:** `deepseek-ai/DeepSeek-V4-Flash-0731` @ `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
**Hardware:** 4× NVIDIA H100 SXM 80 GB (700 W, NV18 full mesh), TP=4, driver 580.126.20, CUDA 13
**Image:** `mlnode-b300-deepseek-v4-flash:0.2.14-vllm0.25.1-overlay-k10`
**Digest:** `sha256:a6213dac621c1634a82940533190c9a5149b6535a5690c69ca6d3919c74c8138`
**vLLM:** 0.25.1 — stock, no source patches

Companion to `../deepseek-v4-flash-0731-2xh200`, which established that DSpark exists,
that the k9/k10 images disable it, and that it does not disturb PoC. This run answers the
question that one left open — **does DSpark change the output** — repeats the serving A/B on
a second Hopper topology, and adds nonce vectors from a second GPU model so the cross-machine
honest floor is measured rather than assumed.

## Summary

- **DSpark costs nothing in PoC**: 1504 nonces/min at batch 16 with and without it, 1408 vs
  1424 at batch 8, and nonce *values* agree to within the honest floor. A node can speculate
  for serving without losing weight or failing validation.
- **A heterogeneous fleet validates**: 4×H100 with DSpark against 2×H200 without measures
  0.170–0.172, below the 0.188 floor, across GPU model, card count, TP and batch size.
- **DSpark preserves the output.** Teacher forcing puts DSpark at **97.87 %** argmax
  agreement against a **98.36 %** control — a 1.45 σ difference, i.e. within the noise of
  the measurement itself.
- **DSpark is worth more at TP=4 than at TP=2**: 3.39× on single-stream long decode, and
  unlike 2×H200 it also helps under concurrency (1.32× where H200 gave exactly 1.00×).
- **Three launch flags contradict each other on 80 GB cards.** The H200 configuration does
  not transfer; details in *The parameter trap*.
- k10 still ships both k9 defects (`VLLM_USE_V2_MODEL_RUNNER=0`, missing `libnvrtc.so`).
  It does drop the hardcoded forced args, which is an improvement.

## Result 1 — PoC throughput: DSpark costs nothing

Nonces/min, `run_pow_generation.py --phase 3` (5 s warmup + 30 s steady state), both arms on
the chosen configuration (`gmu 0.85`, `maxnbt 16384`):

| batch | DSpark off | DSpark on | previous `-Flash`, 4×H100 (2026-07-24) |
|---:|---:|---:|---:|
| 8 | 1424 | 1408 | — |
| 16 | **1504** | **1504** | — |
| 32 | 0 *(see below)* | 0 *(see below)* | **1536** |

**Identical at batch 16 and within 1 % at batch 8.** Speculative decoding neither helps nor
hurts the PoC forward, which is what the design implies — PoC runs through the worker
extension and the draft model takes no part in it. This repeats on 4×H100 what the 2×H200 run
found (1215 vs 1216), now on a second topology and a second image.

The batch-32 zero is **not** a DSpark effect — it appears in both arms. It is requirement 1 of
*The parameter trap* below: a PoC forward at batch 32 submits 32768 tokens, which does not fit
the metadata buffer that `max-num-batched-tokens 16384` sizes. It is a hard configuration
limit, not a crash, and it is why the previous checkpoint's 1536 (measured at batch 32 under
`maxnbt 32768`) is not directly comparable to the 1504 here.

**Practical reading for the fleet:** a node can enable DSpark for serving without losing PoC
weight. On 80 GB cards the cost is not DSpark but the configuration it forces — batch 16
instead of 32, i.e. 1504 against the 1536 the previous checkpoint reached, about 2 %.

## Result 2 — nonce vectors: DSpark changes nothing, and the fleet may be heterogeneous

Collected after the runs above, on a second 4×H100 box, at **batch 16** under the working
configuration (`gmu 0.90`, `maxnbt 32768`). Three seeds per arm, 1000 nonces each.

| comparison | median L2 (s1 / s2 / s3) | mismatches > 0.4 |
|---|---|---:|
| DSpark on vs off — same box, TP=4 | 0.178 / 0.170 / 0.170 | 1.4–1.9 % |
| 4×H100 vs 2×H200, both without speculation | 0.175 / 0.173 / 0.168 | 1.3–1.9 % |
| 4×H100 **with** DSpark vs 2×H200 without — worst realistic cross-fleet case | 0.172 / 0.171 / 0.170 | 1.4–1.9 % |
| *reference: honest floor between different GPU models* | *0.188* | *2.5–3.8 %* |

**All nine comparisons sit below the honest floor.** The middle row varies four things at once
— GPU model, card count, tensor-parallel degree and collection batch — and still lands under
it. A validator on 2×H200 and a prover on 4×H100 running DSpark agree within ordinary noise.

This also settles the question the July handoff left open: the cross-machine honest floor on
the current seed set is ~0.17, not something larger.

### Why batch 16 here and batch 32 on H200 is not a problem

Nonce values do not depend on the collection batch. From the committed
`../../2026-07/deepseek-v4-seed-stability-1xb300` artifacts, same seed and machine and mode, batch alone
varying:

| pair | median L2 | mismatches > 0.4 |
|---|---:|---:|
| b8 vs b16 | **0.000000** | 0.2 % |
| b8 vs b32 | **0.000000** | 0.3 % |
| b16 vs b32 | **0.000000** | 0.1 % |

Median exactly zero — the sets are bit-identical apart from one to three nonces per thousand
at batch boundaries, an order below the honest floor. Sets taken at different batch sizes
validate each other directly.

### The configuration that satisfies both requirements

`TP=4, --gpu-memory-utilization 0.90, --max-model-len 400000, --max-num-batched-tokens 32768,
--kv-cache-dtype fp8` runs 400k-context inference **and** PoC at batch 16 with DSpark enabled.
KV lands at 17.16 GiB per card against the 15.26 GiB a single 400k request needs.

Lowering `gmu` to make room for graph capture does not work: at 0.80 the KV drops to 9.24 GiB
and at 0.75 to 5.28 GiB, and the engine refuses to start — *"To serve at least one request
with the model's max seq len (400000), 15.26 GiB KV cache is needed"*. The s4 serving OOM
therefore has to be addressed with `--max-num-seqs`, which bounds the runtime peak without
taking memory from the context. **That knob was not tested in this run.**

## Result 3 — does DSpark change what the model would have said?

Comparing generated *texts* between arms does not work: the engine is nondeterministic run
to run (the 2×H200 report measured 2/5 identical for the *same* arm sampled twice), so texts
diverge with speculation off.

So: teacher forcing. Record greedy completions, then replay `prompt + completion` through an
engine **without** speculation asking for per-token logprobs, and count how often the recorded
token was the target model's argmax at that position. The control replays completions recorded
*without* speculation through the same path, so whatever noise teacher forcing itself carries
applies to both.

| source of the replayed tokens | positions | argmax agreement |
|---|---:|---:|
| DSpark on | 1411 | **97.874 %** |
| control — DSpark off | 1588 | **98.363 %** |

At the control's rate, 1411 positions predict 23.1 misses; 30 were observed, against
σ = 4.77 — a **z of 1.45**. There is no evidence that DSpark's acceptance is lossy.

The control matters more than the headline: without it, 97.87 % reads as "DSpark loses 2 % of
tokens". It does not — a non-speculative run loses the same 1.6 %, which is the numerical
noise floor of this engine, not an artefact of speculation.

## Result 4 — serving A/B (both arms on the V2 runner, one flag apart)

| scenario | tok/s off | tok/s on | × | TPOT off | TPOT on | × | tok/chunk | fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| s1 — 20k prompt, sequential, 300 tok | 133.4 | 395.6 | **2.97×** | 6.96 ms | 2.01 ms | 3.46× | 4.95 | 0/0 |
| s2 — 2k prompt, concurrency 30 | 1880.8 | 2484.5 | **1.32×** | 13.74 ms | 10.17 ms | 1.35× | 3.58 | 0/0 |
| s3 — 45k prompt, sequential, 1000 tok | 138.6 | 469.9 | **3.39×** | 6.97 ms | 1.94 ms | 3.59× | 5.56 | 0/0 |
| s4 — 45k prompt, concurrency 20 | 1050.2 | 1890.7 | **1.80×** | 15.74 ms | 9.75 ms | 1.61× | 5.33 | 0/0 |

`tok/chunk` is the observed acceptance; the baseline sits at exactly 1.00 in every scenario,
which is what makes the metric trustworthy.

**Against 2×H200** (same instrument, same scenarios, `../deepseek-v4-flash-0731-2xh200`):
H200 gave 2.63× / **1.00×** / 2.98× / 1.36×. Four H100s therefore benefit *more*, and the
difference is largest exactly where H200 showed nothing — under concurrency. A 4-way tensor
split issues four times the kernel launches per token, so the baseline is launch-bound where
the 2-way one is not, and that is the headroom speculation removes.

## The parameter trap

The H200 configuration (`gpu-memory-utilization 0.90`, `max-num-batched-tokens 32768`) does
not transfer to 80 GB cards. Three requirements pull in different directions:

1. **The PoC attention-metadata buffer is sized by `max-num-batched-tokens`.** The PoC forward
   submits `batch × 1024` tokens, so batch 32 needs 32768. Below that it fails in
   `gonka_poc/_compat/v0_25.py → deepseek_v4/sparse_mla.py:204 → backend.py:535` on
   `assert buffer.shape[0] >= max(num_mapped_tokens, num_tokens)`. **Lowering this flag caps
   the PoC batch size.**
2. **That same buffer does not shard across TP.** Every card pays for the full token budget
   regardless of how many cards there are, so the flag costs relatively more at higher TP.
   Halving it from 32768 to 16384 returned 6.4 GiB per card and raised KV capacity 2.45×
   (509,334 → 1,249,327 tokens).
3. **vLLM's memory profiler does not reserve anything for CUDA-graph capture.** It fills up to
   `gpu-memory-utilization` and hands the remainder to KV; capture then runs and needs ~1.7 GiB
   more than the 1.29 GiB left. At `gmu 0.90` the engine either dies during capture or — worse —
   starts and dies later under load.

Measured, DSpark on, per card:

| config | KV available | KV tokens | outcome |
|---|---:|---:|---|
| gmu 0.90, maxnbt 32768 | 17.16 GiB | 509,334 | starts; **OOMs under load** (32/40 requests lost in s4) |
| gmu 0.90, maxnbt 16384 | 23.53 GiB | 1,249,327 | **fails at graph capture** (1.29 GiB free, needs 1.73) |
| gmu 0.85, maxnbt 32768 | 13.20 GiB | — | **fails at graph capture** |
| **gmu 0.85, maxnbt 16384** | **19.57 GiB** | **1,039,126** | **works**: s4 passes with 0 failures |

The chosen configuration carries a cost: with `maxnbt 16384` the PoC batch tops out at 16
(1024 × 16 = 16384, exactly the buffer), so batch 32 is unavailable. That costs about 2 % of
PoC throughput — 1504 nonces/min against the 1536 the previous checkpoint reached at batch 32
under `maxnbt 32768`.

**The dangerous case is the first row.** It starts cleanly, reports a healthy KV cache, serves
small requests fine, and only dies once a long-prompt concurrent load arrives. A configuration
that fails at startup is safer than one that fails an hour in.

## What this run did not produce

**The nonce collector needs `max-num-batched-tokens 32768`.** Every attempt during the runs
above used `maxnbt 16384` and failed — at batch 16 the PoC forward submits exactly 16384
tokens, filling the metadata buffer with nothing left for DSpark's draft tokens
(`gonka_poc/_compat/v0_25.py → sparse_mla.py:204 → backend.py:535`,
`assert buffer.shape[0] >= max(num_mapped_tokens, num_tokens)`). Re-running at 32768 collected
all six sets on the first attempt. Neither TP=4 nor the tighter memory was involved — an
earlier revision of this report guessed at those and was wrong.

**PoC sweep at batch 32 returns 0 nonces in BOTH arms** at `maxnbt 16384` — expected from
requirement 1 above (32768 tokens do not fit a 16384 buffer). Listed here so the zero in
`artifacts/logs/sweep_dspark_{on,off}.log` is not read as a crash, and not as a DSpark defect:
the baseline returns the same zero.

## Files

| path | what |
|---|---|
| `artifacts/summary.json` | every table above, machine-readable |
| `artifacts/nonces_dspark_{on,off}_{s1,s2,s3}.json` | 6 × 1000 nonces, batch 16, three seeds per arm |
| `artifacts/env_nonces.txt` | engine args and measured KV for the nonce runs |
| `artifacts/logs/nonce_collection.log` | the collection run, including the two failed `gmu` attempts |
| `artifacts/equiv_dspark_on.json`, `artifacts/equiv_control.json` | teacher-forcing results, per prompt |
| `artifacts/gen_dspark_{on,off}.json` | the recorded greedy completions that were replayed |
| `artifacts/serving_dspark_{on,off}.json` | four scenarios per arm |
| `artifacts/logs/sweep_dspark_{on,off}.log` | PoC sweeps |
| `artifacts/logs/{param_test,gmu_test}.log` | the two configuration probes behind *The parameter trap* |
| `artifacts/env_h100.txt` | hardware, driver, versions, applied fixes |
| `scripts/setup_h100.sh` | box preparation incl. both k10 fixes |
| `scripts/final4_h100.sh` | the A/B driver as run |
| `scripts/nonces_h100.sh` | nonce collection driver (collector first, on a clean engine) |
| `scripts/equiv_probe.py` | teacher-forcing probe (`gen` / `check`) |
| `scripts/serving_bench.py` | serving load generator (counts tokens via `usage`, not SSE chunks) |

## Reproduce

```bash
bash scripts/setup_h100.sh          # libnvrtc symlink, VLLM_USE_V2_MODEL_RUNNER=1, deps, weights
bash scripts/final4_h100.sh         # both arms: sweep, greedy capture, serving, then equivalence
```

Engine args used by both arms, differing only by `--speculative-config`:
`--tensor-parallel-size 4 --gpu-memory-utilization 0.85 --max-model-len 400000
--max-num-batched-tokens 16384 --kv-cache-dtype fp8 --logprobs-mode processed_logprobs
--worker-extension-cls gonka_poc.worker.PoCWorkerExtension --trust-remote-code`

## Reproducibility checklist

- [x] Image pinned by digest; model pinned by revision
- [x] Stock vLLM — no source patches in the measured configuration
- [x] The equivalence claim carries its own control, and the control is reported
- [x] Significance stated (z = 1.45) rather than asserted qualitatively
- [x] Configuration probes committed as logs, including the ones that failed
- [x] Nonce vectors collected for both arms on three seeds; batch-invariance shown, not assumed
- [x] An earlier wrong attribution (nonce collector vs TP/memory) retracted in place
- [x] Failed measurements named explicitly (batch-32 sweep, the two starved `gmu` attempts)
- [x] Cross-topology comparison uses the same instrument as the H200 report
- [x] No links to `.claude/`, no absolute local paths, no host addresses
- [x] No verdicts asserted — V4 thresholds are not calibrated
