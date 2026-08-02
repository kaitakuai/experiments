# DeepSeek-V4-Flash-0731 on 1×B300: PoC unchanged from July, DSpark up to 3.7× — and negative under load

**Date:** 2026-08-01
**Model:** `deepseek-ai/DeepSeek-V4-Flash-0731` @ `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
**Hardware:** 1× NVIDIA B300 SXM6 AC 275 GB (1100 W), TP=1, driver 580.126.09, CUDA 13, 30 CPU cores
**Image:** `mlnode-b300-deepseek-v4-flash:0.2.14-vllm0.25.1-overlay-k10`
**Digest:** `sha256:a6213dac621c1634a82940533190c9a5149b6535a5690c69ca6d3919c74c8138`
**vLLM:** 0.25.1 — stock, no source patches
**PoC:** gonka-poc plugin, seq_len 1024, k_dim 12

> V4 thresholds are **not calibrated**. L2 values elsewhere in this series are distances only.

The honest baseline for this card: the 0731 checkpoint with DSpark off and on. The NVFP4
fraud candidate measured against these numbers lives in
`../deepseek-v4-flash-0731-nvfp4-1xb300`, and reuses the nonce sets committed here.

This is the third topology in the 0731 campaign, after `../deepseek-v4-flash-0731-2xh200`
(TP=2) and `../deepseek-v4-flash-0731-dspark-4xh100` (TP=4). One card at 275 GB makes it the
most favourable PoC configuration we have measured, and the only one where DSpark turns out
mildly *negative* in one scenario.

## Summary

- **PoC is unchanged, twice over.** 1728 nonces/min at batch 32 — identical with and without
  DSpark, and identical to the July reference for this same card on the *previous* checkpoint.
  Neither the refresh nor speculation moves node weight.
- **DSpark gives 3.74× and 3.00×** in the two single-stream scenarios, 2.10× at concurrency 20
  with long prompts, and **0.98×** — a small loss — at concurrency 30 with short prompts.
- **KV cost of DSpark: −16.9 %** (2,660,974 → 2,211,880 tokens), in line with the −13.8 %
  measured on 2×H200.
- One card at 275 GB carries **2.66 M tokens of KV**, against 1.04 M on 4×H100 and 0.97 M on
  2×H200. Scenario s4, which OOM'd both arms on 80 GB cards, passes here with zero failures.

## Result 1 — PoC throughput

Nonces/min, `run_pow_generation.py --phase 3` (5 s warmup + 30 s steady state):

| batch | DSpark off | DSpark on | previous `-Flash`, same card (2026-07) |
|---:|---:|---:|---:|
| 8 | 1504 | 1648 | 1184 eager / 1648 graphs |
| 16 | 1696 | 1696 | 1664 / 1696 |
| 32 | **1728** | **1728** | 1664 eager / **1728** graphs |

Two readings. **Speculation is free for PoC** — the same 1728 either way, which repeats what
2×H200 (1215/1216) and 4×H100 (1504/1504) showed. And **the checkpoint refresh changed
nothing**: 1728 now against 1728 in July on this card.

The PoC forward runs through the worker extension and the draft model takes no part in it, so
this is the expected result — but it is the result a node operator cares about most, and it now
holds on three topologies.

## Result 2 — DSpark on serving

Both arms differ by one flag: `--speculative-config '{"method":"dspark",
"num_speculative_tokens":7,"draft_sample_method":"greedy"}'`.

| scenario | off | on | × | acceptance (tok/chunk) |
|---|---:|---:|---:|---:|
| s1 — 20k prompt, sequential, 300 tok | 81.4 | 304.7 | **3.74×** | 5.37 |
| s2 — 2k prompt, concurrency 30 | 1328.9 | 1301.2 | **0.98×** | 3.64 |
| s3 — 45k prompt, sequential, 1000 tok | 143.0 | 429.4 | **3.00×** | 6.04 |
| s4 — 45k prompt, concurrency 20 | 1039.8 | 2180.0 | **2.10×** | 5.69 |

Tokens/s, zero failed requests in all eight measurements. `acceptance` is the observed tokens
per streamed chunk; the baseline sits at exactly 1.00 everywhere, which is what makes the
metric trustworthy.

**s2 is the first negative cell in this series.** At concurrency 30 with short prompts the card
is already saturated — there is no kernel-launch headroom left for speculation to recover — and
the draft model still costs cycles. On 2×H200 the same scenario gave exactly 1.00×; on 4×H100,
1.32×. The pattern across topologies is consistent: the more the baseline is launch-bound, the
more DSpark returns.

**Against the other topologies**, single-stream long decode: 3.00× here, 2.98× on 2×H200,
3.39× on 4×H100. The effect is a property of the workload, not of the hardware.

## Result 3 — memory

| | KV cache | per-card |
|---|---:|---|
| DSpark off | 2,660,974 tokens | 275 GB card, TP=1 |
| DSpark on | 2,211,880 tokens | −16.9 % |

For scale, the same measurement elsewhere in the campaign: 2×H200 gave 1,126,674 → 970,879
(−13.8 %), and 4×H100 at the working configuration 1,039,126. A single B300 therefore carries
more than twice the KV of either multi-card configuration, which is why scenario s4 — twenty
concurrent 45k-token prompts — runs clean here and OOM'd on 80 GB cards regardless of
speculation.

## Environment

`artifacts/logs/env_b300.txt`. Engine args, identical across both arms except the speculative
flag: `--tensor-parallel-size 1 --gpu-memory-utilization 0.90 --max-model-len 400000
--max-num-batched-tokens 32768 --kv-cache-dtype fp8 --logprobs-mode processed_logprobs
--worker-extension-cls gonka_poc.worker.PoCWorkerExtension --trust-remote-code`.

Two fixes are needed before the k10 image runs at all, both carried in `scripts/b300_setup.sh`
and diagnosed in `../deepseek-v4-flash-0731-2xh200`: a `libnvrtc.so` symlink, without which
FlashInfer cannot link its Hopper kernel, and `VLLM_USE_V2_MODEL_RUNNER=1`, because the image
ships `0` and speculative decoding exists only in the V2 runner.

**This is our own box, not a rented one**, so everything runs inside the container: the host
has no `uvicorn` and no `hf`. Engine bring-up took **17 minutes** against ~4 on a rented
192-core machine — kernel JIT is CPU-bound and this host has 30 cores. Later bring-ups on the
warm cache took about 12.

## What is missing from this folder, and why

The host went offline immediately after the runs, during artifact retrieval:

- **Nonce sets for the DSpark-on arm** were not retrieved. The three committed sets are the
  no-speculation arm, which is what the fraud comparison uses. DSpark's neutrality on nonce
  *values* was established with six sets each on 2×H200 and 4×H100.
- **Raw serving JSON for the DSpark-on arm.** Its numbers are recovered from the run log; the
  tables above come from `artifacts/summary.json`, regenerated from
  `artifacts/logs/run_honest.log`. `artifacts/serving_dspark_off.json` is the one raw file that
  transferred, for cross-checking the reconstruction.
- **TTFT and TPOT** are therefore available only for the off arm.

## Files

| path | what |
|---|---|
| `artifacts/summary.json` | every table above, regenerated from the run log and nonce sets |
| `artifacts/nonces_dspark_off_{s1,s2,s3}.json` | 3 × 1000 nonces, batch 32, three fixed seeds |
| `artifacts/serving_dspark_off.json` | raw serving JSON, off arm |
| `artifacts/logs/run_honest.log` | both arms: sweeps and serving inline |
| `artifacts/logs/api_b300.log` | engine log across the bring-ups |
| `artifacts/logs/env_b300.txt` | hardware, versions, engine args, measured KV |
| `scripts/b300_setup.sh` | container preparation on our own box, incl. both k10 fixes |
| `scripts/b300_run.sh` | the two-arm driver (`MODEL` / `TAGPREFIX` select the checkpoint) |
| `scripts/serving_bench.py` | serving load generator (counts tokens via `usage`, not SSE chunks) |
| `scripts/collect_artifacts.py`, `scripts/run_pow_generation.py` | PoC tooling, patched — see below |
| `scripts/l2_crossval.py`, `scripts/poc_seeds.json` | analysis and the fixed seed set |

The PoC scripts are committed **as patched**: `run_pow_generation.py` needs
`API_PREFIX = "/api/v1/inference"` (the bare `/api/v1/pow/*` family is the legacy PoW v1
service), `MLNODE_URL` on port 8081, `HOST_IP=127.0.0.1`, and its hardcoded `MODEL_NAME` edited
for any non-default checkpoint. `collect_artifacts.py` needs its seed arguments promoted to
module globals.

## Reproduce

```bash
bash scripts/b300_setup.sh     # container, libnvrtc fix, V2 runner, deps, weights, API
MODEL=deepseek-ai/DeepSeek-V4-Flash-0731 TAGPREFIX=official bash scripts/b300_run.sh
```

## Reproducibility checklist

- [x] Image pinned by digest; model pinned by revision
- [x] Stock vLLM — no source patches in the measured configuration
- [x] Both arms differ by exactly one flag
- [x] Acceptance reported directly, so "speculation is off" is observed rather than assumed
- [x] Three independent seeds behind the committed nonce sets
- [x] Tables regenerated from the committed log by an in-tree step, not transcribed
- [x] Unretrieved artifacts named explicitly, with what they would and would not have changed
- [x] Compared against the July reference for the same card, and against the other two topologies
- [x] No links to `.claude/`, no absolute local paths, no host addresses
