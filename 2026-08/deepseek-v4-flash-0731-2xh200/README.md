# DeepSeek-V4-Flash-0731 on 2×H200: PoC unchanged, DSpark up to 2.98× — once the image stops disabling it

**Date:** 2026-07-31
**Model:** `deepseek-ai/DeepSeek-V4-Flash-0731` @ `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` (FP8 blocks + FP4 experts)
**Hardware:** 2× NVIDIA H200 SXM (700 W, NV18), TP=2, driver 590.48.01, CUDA 13
**Image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash:0.2.14-vllm0.25.1-overlay-k9`
**Digest:** `sha256:ef5260aefa5cc46fbbdee6783d3492d399ae0a8fb649c03a12f731707edcc11b`
**vLLM:** 0.25.1 — **stock, no source patches**
**PoC:** gonka-poc plugin, seq_len 1024, k_dim 12, `--max-model-len 400000`

> V4 thresholds are **not calibrated**. L2 values below are distances only — no PASS/FRAUD verdicts.

## Summary

The 0731 refresh keeps the model body untouched and adds DSpark, a speculative decoder
(`mtp.*` tensors, +7.3 GB over the previous checkpoint). Three questions follow: does PoC
change, is the new checkpoint distinguishable from the old one, and what does DSpark buy.

- **PoC throughput is unchanged: 1215 vs 1216 nonces/min** at batch 32, against the previous
  checkpoint measured on the same topology. A node's weight does not move on upgrade.
- **DSpark is worth up to 2.98×** on single-stream long decode — and **nothing** (1.00×) for
  aggregate throughput at concurrency 30.
- **DSpark is invisible to PoC**: it changes neither nonce values (L2 0.165–0.173, below the
  0.188 honest floor) nor PoC throughput (1215 vs 1216).
- **The k9 image disables all speculative decoding** by shipping
  `VLLM_USE_V2_MODEL_RUNNER=0`, and does not start on Hopper at all without a `libnvrtc.so`
  symlink. Both are one-line fixes; details in *Image defects*.
- Old vs new checkpoint: **median L2 0.648**, 88.6 % of nonces beyond 0.4 — a node serving the
  wrong version is detectable with a wide margin.

## Environment and config

`artifacts/env.txt` carries the full capture. What we changed relative to the image defaults:

| setting | image default (k9) | this run | why |
|---|---|---|---|
| `--tensor-parallel-size` | 1 | **2** | match the earlier 2×H200 `-Flash` run |
| `--max-model-len` | 200000 | **400000** | same |
| `--max-num-batched-tokens` | 16384 | **32768** | same |
| `VLLM_USE_V2_MODEL_RUNNER` | **0** (`/etc/environment`) | **1** | without it DSpark cannot run at all |
| `libnvrtc.so` | absent | symlinked | without it the engine dies on the first forward |

Unchanged: `--gpu-memory-utilization 0.90`, `--kv-cache-dtype fp8`,
`--logprobs-mode processed_logprobs`, `--worker-extension-cls gonka_poc.worker.PoCWorkerExtension`.
DSpark arm adds exactly one flag:
`--speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}'`.

## Image defects found (all reproducible, all one-line)

**1. No `libnvrtc.so` — the image does not start on Hopper.** FlashInfer JIT-compiles the
sm90 kernel `fp8_blockscale_gemm_sm90` on the first forward and the link step fails with
`/usr/bin/ld: cannot find -lnvrtc`; the worker dies after the KV cache is already allocated.
The image ships only the versioned `libnvrtc.so.13`. Blackwell never takes this code path, so
the defect is invisible on B200/B300. Fix: symlink `libnvrtc.so` → `libnvrtc.so.13`.

**2. `VLLM_USE_V2_MODEL_RUNNER=0` kills every form of speculative decoding.** vLLM enables the
V2 runner for `method="dspark"` by itself, but the explicit `0` short-circuits that decision
(`config/vllm.py::use_v2_model_runner` returns the env value before reaching the DSpark
branch). The engine then silently runs V1, where DSpark is routed to `EagleProposer` and fails
with a cascade of unrelated-looking errors — a missing `mask_hidden` attribute, a
`parallel_drafting` token lookup that does not know `dspark_noise_token_id`, and a
`forward() got an unexpected keyword argument 'hidden_states'` signature mismatch. **None of
these are vLLM bugs.** With `VLLM_USE_V2_MODEL_RUNNER=1` the correct `DSparkSpeculator`
(`v1/worker/gpu/spec_decode/dspark/`) is selected and stock vLLM works unpatched.

**3. Forced args changed between image revisions.** k4 forced TP=2 / 400k context, k9 forces
TP=1 / 200k. Anyone comparing numbers across the two images without checking will be comparing
two different configurations.

**Not applicable to this checkpoint:** `--speculative-config '{"method":"mtp",...}'` fails with
`KeyError: 'model.layers.43.mtp_block.main_norm.weight'` — the MTP loader expects the module as
layer 43, while 0731 ships it under the `mtp.*` namespace. MTP is for the preview variants.

## Result 1 — PoC throughput does not move

Nonces/min, `run_pow_generation.py --phase 3` (5 s warmup + 30 s steady state):

| batch | V1 runner, no spec | V2 runner, no spec | V2 + DSpark | previous `-Flash` (2026-07-24) |
|---:|---:|---:|---:|---:|
| 8 | 1120 | 1120 | 1104 | 1104 |
| 16 | 1184 | 1184 | 1184 | 1184 |
| 32 | **1215** | **1216** | **1215** | **1216** |

Four independent measurements land within 0.1 %, against a ~5 % measurement quantum (the sweep
counts whole batches over 30 s). Neither the checkpoint refresh, nor the model runner, nor
DSpark changes what a node earns from PoC.

## Result 2 — DSpark on serving

Both arms on the V2 runner, same box, same instrument, differing only in the speculative flag.
`tokens/chunk` is the directly observed acceptance: how many tokens arrive per streamed chunk.

| scenario | tok/s off | tok/s on | ×  | TPOT off | TPOT on | × | tok/chunk |
|---|---:|---:|---:|---:|---:|---:|---:|
| s1 — 20k prompt, sequential, 300 tok | 121.0 | 318.1 | **2.63×** | 7.48 ms | 2.54 ms | 2.95× | 4.53 |
| s2 — 2k prompt, concurrency 30 | 1039.2 | 1041.1 | **1.00×** | 40.7 ms | 16.1 ms | 2.53× | 3.52 |
| s3 — 45k prompt, sequential, 1000 tok | 125.7 | 375.1 | **2.98×** | 7.47 ms | 2.27 ms | 3.29× | 5.15 |
| s4 — 45k prompt, concurrency 20 | 776.6 | 1058.3 | **1.36×** | 21.1 ms | 13.2 ms | 1.60× | 5.41 |

Zero failed requests in all eight runs.

**The gain is a property of the load profile, not of the model.** Single-stream decode is
launch-bound, and DSpark removes most of that: s3 triples. At concurrency 30 the GPU is already
saturated, so per-request latency still improves 2.5× but aggregate throughput does not move at
all. In s4 DSpark also **costs TTFT** — 2.26 s → 4.53 s — because the draft model competes with
prefill.

DSpark also costs KV cache: **970,879 tokens vs 1,126,674** (−13.8 %) at the same
`--gpu-memory-utilization`.

## Result 3 — DSpark does not disturb PoC vectors

The risk: if speculative decoding changed nonce values, a prover running DSpark and a validator
without it would disagree, and honest nodes would be flagged. It does not.

| comparison | median L2 (s1 / s2 / s3) | mismatches > 0.4 |
|---|---|---:|
| DSpark on vs off, same runner | 0.169 / 0.173 / 0.165 | 1.2–1.7 % |
| DSpark on vs the V1 baseline (worst case) | 0.173 / 0.170 / 0.167 | 1.1–1.3 % |
| V1 runner vs V2 runner, no spec | 0.170 / 0.169 / 0.166 | 1.4–2.2 % |
| *reference: honest floor between different GPU models* | *0.188* | *2.5–3.8 %* |
| *control: two independent seeds* | *1.423* | *100 %* |

Every cross-configuration distance sits **below** the known honest floor, on three independent
seeds. Speculative decoding and the runner version are both safe to vary across the fleet.

## Result 4 — old vs new checkpoint is far outside honest noise

Same seed pair, same 2×H200 topology, `nonces_v1_off_legacyseed.json` against the committed
`../../2026-07/deepseek-v4-flash-2xh200/artifacts/nonces_eager.json`:

**median L2 0.648, 88.6 % of nonces beyond 0.4.** For scale: the honest floor is 0.188, a
V4-Base checkpoint substitution measured 0.443, and a foreign model saturates at ~1.41. Serving
the previous `-Flash` while claiming 0731 is detectable at any sane threshold; this fraud vector
needs no dedicated calibration.

*Caveat:* the reference set was collected on image k4 and this one on k9. The two share the same
vLLM version, and the V1↔V2 comparison above shows that runtime changes of this kind move
vectors by ~0.17 — an order below 0.648 — but the image is not held constant in this particular
row.

## Validation

- Instrument checked before the long runs: the collector prints the seed it received, and two
  independent seeds give 1.423 (the expected ceiling for uncorrelated 12-dim vectors).
- L2 arithmetic matches the chain (`vllm/poc/data.py`): fp16 LE → fp32, fp64 norm, strict `>`.
- The engine configuration is taken from the engine's own log (`artifacts/engine_args.txt`,
  `artifacts/logs/api_v2_dspark.log`: `non-default args`, `Using V2 Model Runner`,
  `Capturing model for DSpark speculator`, KV cache size), not assumed.
- Every table above is regenerated from the committed artifacts by `scripts/summarize.py`.

**Output equivalence is NOT established.** Greedy completions differ between the DSpark arms
(0/5 prompts identical), but the control — the same arm sampled twice — is only 2/5 identical,
so the engine is nondeterministic run to run and the text diff cannot decide the question.
Whether DSpark's acceptance is lossy (it has a `confidence_head`) needs a per-token logprob
comparison, which this run does not provide.

**Serving was not measured on the V1 runner.** An earlier V1 pass used an instrument that counted
SSE chunks as tokens, which undercounts by ~4× whenever speculation packs several tokens into a
chunk. Those numbers are discarded rather than published; the serving section is V2-only, where
both arms share the corrected instrument.

## Files

| path | what |
|---|---|
| `artifacts/summary.json` | every table above, machine-readable |
| `artifacts/nonces_v1_off_{s1,s2,s3}.json` | V1 runner, no spec, three seeds |
| `artifacts/nonces_v1_off_legacyseed.json` | V1 runner, the 2026-07-24 seed pair |
| `artifacts/nonces_v2_off_{s1,s2,s3}.json` | V2 runner, no spec |
| `artifacts/nonces_v2_dspark_{s1,s2,s3}.json` | V2 runner, DSpark on |
| `artifacts/serving_dspark_{on,off}.json` | four scenarios per arm |
| `artifacts/greedy_dspark_{on,off,off_run2}.json` | greedy probes incl. the determinism control |
| `artifacts/logs/sweep_*.log` | full batch sweeps, three configurations |
| `artifacts/logs/api_v2_dspark.log` | engine log: V2 runner, DSpark speculator, KV cache |
| `artifacts/env.txt` | hardware, driver, versions, applied fixes |
| `scripts/setup_box.sh` | box preparation incl. both image fixes |
| `scripts/ab_v2.sh`, `scripts/poc_v2_ab.sh` | the two A/B drivers |
| `scripts/serving_bench.py` | serving load generator (counts tokens via `usage`) |
| `scripts/greedy_probe.py` | greedy completion capture |
| `scripts/collect_artifacts.py`, `scripts/run_pow_generation.py` | PoC tooling, patched (see below) |
| `scripts/l2_crossval.py`, `scripts/summarize.py` | analysis |
| `scripts/poc_seeds.json` | the fixed seed set with its provenance |

Both PoC scripts are committed **as patched**, without which the run does not reproduce:
`run_pow_generation.py` needed `API_PREFIX = "/api/v1/inference"` (the bare `/api/v1/pow/*`
family is the legacy PoW v1 service and answers 409 while inference is up), `MLNODE_URL` on port
8081, and `HOST_IP=127.0.0.1` (its default falls back to a Docker gateway that does not exist on
a bare host, so every batch is silently lost and the sweep reports 0). `collect_artifacts.py`
needed its seed arguments promoted to module globals.

## Reproduce

```bash
bash scripts/setup_box.sh                 # deps, libnvrtc symlink, forced args, weights
export VLLM_USE_V2_MODEL_RUNNER=1         # otherwise DSpark cannot start
bash scripts/ab_v2.sh                     # serving A/B, both arms
bash scripts/poc_v2_ab.sh                 # PoC sweep + nonces, both arms
python3 scripts/summarize.py artifacts > artifacts/summary.json
```

## Reproducibility checklist

- [x] Image pinned by tag **and** digest; model pinned by revision
- [x] Stock vLLM — no source patches in the measured configuration
- [x] Three independent seeds behind every L2 claim; seed set provenance documented
- [x] Instrument validated before the long runs (seed reaches the computation; seeds uncorrelated)
- [x] Engine configuration read from the engine's own log, not assumed
- [x] Raw nonce sets for every cell committed (10 × 1000 nonces)
- [x] All tables regenerated from the committed artifacts by an in-tree script
- [x] Discarded measurements named as discarded (V1 serving) rather than quietly dropped
- [x] Unresolved question stated (output equivalence) rather than glossed
- [x] No links to `.claude/`, no absolute local paths, no host addresses
- [x] No verdicts asserted — V4 thresholds are not calibrated
