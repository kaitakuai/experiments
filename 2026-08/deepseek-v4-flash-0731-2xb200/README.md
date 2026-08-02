# DeepSeek-V4-Flash-0731 on 2×B200: DSpark is bit-identical for PoC — and costs 42 % of throughput under load

**Date:** 2026-08-02
**Model:** `deepseek-ai/DeepSeek-V4-Flash-0731` @ `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
**Hardware:** 2× NVIDIA B200 SXM 183 GB (1000 W, NV18), TP=2, driver 580.126.09, CUDA 13, 192 CPU cores
**Image:** `mlnode-b300-deepseek-v4-flash:0.2.14-vllm0.25.1-overlay-k10`
**Digest:** `sha256:a6213dac621c1634a82940533190c9a5149b6535a5690c69ca6d3919c74c8138`
**vLLM:** 0.25.1 — stock, no source patches
**PoC:** gonka-poc plugin, seq_len 1024, k_dim 12

The fourth and last topology of the 0731 campaign, after
`../deepseek-v4-flash-0731-1xb300`, `../deepseek-v4-flash-0731-2xh200` and
`../deepseek-v4-flash-0731-dspark-4xh100`. The NVFP4 fraud candidate measured against this
baseline is in `../deepseek-v4-flash-0731-nvfp4-2xb200`.

## Summary

- **PoC unchanged for the fourth time.** 2304 nonces/min at batch 32, with and without DSpark,
  and equal to the July reference for this topology on the *previous* checkpoint (2304 with
  graphs). Across all four topologies the refresh has not moved node weight by a single nonce.
- **DSpark leaves PoC vectors bit-identical.** 968 of 1000 nonces per seed match exactly; the
  32 that differ are precisely the first of each batch — the known batch-boundary artefact,
  `1 − 1/32`. On Hopper the same comparison measured 0.17, which was **the hardware's own
  non-determinism, not an effect of speculation**.
- **DSpark's serving effect is the widest we have measured: 3.87× and 0.58×**, on the same
  machine, in the same run. Short prompts at concurrency 30 lose 42 % of throughput.
- KV cost of DSpark: −12.2 %, the mildest of the four topologies.

## Result 1 — PoC throughput

Nonces/min, `run_pow_generation.py --phase 3` (5 s warmup + 30 s steady state):

| batch | DSpark off | DSpark on | previous `-Flash`, same topology (2026-07) |
|---:|---:|---:|---:|
| 8 | 2032 | 2192 | 1184 eager / 2160 graphs |
| 16 | 2272 | 2272 | 1664 / 2272 |
| 32 | **2304** | **2304** | 1344 eager / **2304** graphs |

The campaign's PoC picture is now complete, and it is the same picture on every topology:

| topology | 0731 | previous checkpoint |
|---|---:|---:|
| 1×B300 TP=1 | 1728 | 1728 |
| 2×H200 TP=2 | 1216 | 1216 |
| 4×H100 TP=4 | 1504 | 1536 |
| **2×B200 TP=2** | **2304** | **2304** |

Three exact matches and one within the 5 % measurement quantum. **The refresh does not
redistribute weight between nodes**, and 2×B200 remains the most productive PoC configuration
we have measured.

## Result 2 — DSpark does not touch the PoC vectors

DSpark on against DSpark off, same box, same seeds, batch 32:

| seed | bit-identical | differing | median L2 | mismatches > 0.4 |
|---|---:|---:|---:|---:|
| s1 | **968 / 1000** | 32 | **0.000000** | 1 (0.1 %) |
| s2 | **968 / 1000** | 32 | **0.000000** | 0 |
| s3 | **968 / 1000** | 32 | **0.000000** | 0 |

968 is `1 − 1/32`: the differing nonces are the first of each batch, the artefact documented in
`../../2026-07/deepseek-v4-seed-stability-1xb300`, where the period was shown to equal the batch
size exactly. Everything else matches to the last bit.

**This corrects the reading of the Hopper measurements.** On 4×H100 the same comparison gave a
median of 0.170–0.178, which invited the conclusion that speculation perturbs the vectors a
little. It does not: Blackwell reproduces its own forward bit-identically while Hopper does not,
so what was measured there was the hardware's non-determinism. Speculation contributes nothing.

Cross-machine, both honest, this box against 1×B300 (both Blackwell, different card counts and
TP): median **0.177–0.181** at 0.8–2.1 % mismatches — below the 0.188 honest floor, so the two
validate each other.

## Result 3 — DSpark on serving: +287 % and −42 % on one machine

Both arms differ by one flag: `--speculative-config '{"method":"dspark",
"num_speculative_tokens":7,"draft_sample_method":"greedy"}'`.

| scenario | off | on | × | TPOT off → on | acceptance |
|---|---:|---:|---:|---|---:|
| s1 — 20k prompt, sequential, 300 tok | 60.2 | 232.7 | **3.87×** | 6.65 → 2.00 ms | 4.70 |
| s2 — 2k prompt, concurrency 30 | 1478.6 | 855.9 | **0.58×** | 16.58 → 29.16 ms | 3.56 |
| s3 — 45k prompt, sequential, 1000 tok | 104.5 | 344.0 | **3.29×** | 6.65 → 1.73 ms | 5.72 |
| s4 — 45k prompt, concurrency 20 | 1383.5 | 2963.4 | **2.14×** | 12.13 → 5.56 ms | 5.80 |

Tokens/s, zero failed requests in all eight measurements.

**s2 is the campaign's worst cell and it is not marginal**: throughput falls 42 % and time per
token nearly doubles. Acceptance there is healthy at 3.56 tokens per step, so the draft model is
guessing well — it simply does not pay. At concurrency 30 with short prompts the cards are
saturated, and every cycle the drafter takes comes straight out of useful work.

Across topologies, the same scenario: 1.32× on 4×H100, 1.00× on 2×H200, 0.98× on 1×B300,
**0.58×** here. The ordering tracks how launch-bound the baseline is — the more parallel the
configuration and the more saturated the load, the less speculation returns, until it goes
negative.

**The operational rule is therefore not "enable DSpark".** It is: enable it where long
single-stream generation dominates, and measure before enabling it on a node serving short
requests at high concurrency — on this topology the mistake costs 42 % of capacity.

## Result 4 — memory

| | KV cache | change |
|---|---:|---|
| DSpark off | 2,359,599 tokens | |
| DSpark on | 2,071,252 tokens | **−12.2 %** |

The mildest speculation tax of the four topologies: −12.2 % here, −13.8 % on 2×H200, −16.9 % on
1×B300.

## Environment

`artifacts/logs/env_b200.txt`. Engine args, identical across both arms except the speculative
flag: `--tensor-parallel-size 2 --gpu-memory-utilization 0.90 --max-model-len 400000
--max-num-batched-tokens 32768 --kv-cache-dtype fp8 --logprobs-mode processed_logprobs
--worker-extension-cls gonka_poc.worker.PoCWorkerExtension --trust-remote-code`.

Two fixes are needed before the k10 image runs at all, both in `scripts/setup_box.sh` and
diagnosed in `../deepseek-v4-flash-0731-2xh200`: a `libnvrtc.so` symlink and
`VLLM_USE_V2_MODEL_RUNNER=1` (the image ships `0`, and speculative decoding exists only in the
V2 runner).

**Driver matters on this hardware.** An earlier attempt at this same topology on driver
`595.71.05` hung twice: the TP=2 workers spun at 100 % CPU with zero bytes read and 1 GB of GPU
memory allocated, never reaching weight loading, through a 40-minute and a 20-minute window.
The run here, on `580.126.09`, loaded weights within seconds. Offers should be filtered on
driver family, not only on CUDA version.

Cold bring-up took 26.75 min for the first arm and 19.5 min for the second — kernel JIT plus,
in the DSpark arm, a FlashInfer autotuner pass over the fp4 MoE kernels.

## Files

| path | what |
|---|---|
| `artifacts/summary.json` | every table above, machine-readable |
| `artifacts/nonces_dspark_{off,on}_{s1,s2,s3}.json` | 6 × 1000 nonces, batch 32, three fixed seeds |
| `artifacts/serving_dspark_{off,on}.json` | four scenarios per arm |
| `artifacts/logs/sweep_official_dspark_{off,on}.log` | PoC sweeps |
| `artifacts/logs/api_b200.log` | engine log across both bring-ups |
| `artifacts/logs/b200_all.log` | campaign driver output |
| `artifacts/logs/env_b200.txt` | hardware, driver, versions, engine args, measured KV |
| `scripts/setup_box.sh` | box preparation incl. both k10 fixes |
| `scripts/b200_run.sh` | the two-arm driver (`MODEL` / `TAGPREFIX` select the checkpoint) |
| `scripts/serving_bench.py` | serving load generator (counts tokens via `usage`, not SSE chunks) |
| `scripts/collect_artifacts.py`, `scripts/run_pow_generation.py` | PoC tooling, patched |
| `scripts/l2_crossval.py`, `scripts/poc_seeds.json` | analysis and the fixed seed set |

## Reproduce

```bash
bash scripts/setup_box.sh      # libnvrtc fix, V2 runner, deps, weights, API
MODEL=deepseek-ai/DeepSeek-V4-Flash-0731 TAGPREFIX=official bash scripts/b200_run.sh
```

## Reproducibility checklist

- [x] Image pinned by digest; model pinned by revision
- [x] Stock vLLM — no source patches in the measured configuration
- [x] Both arms differ by exactly one flag
- [x] Three independent seeds behind every L2 claim
- [x] Bit-identity reported as a count, not inferred from a median
- [x] Acceptance reported directly, so "speculation is off" is observed rather than assumed
- [x] A previous reading of the Hopper numbers corrected in light of this measurement
- [x] The driver that failed named, with the evidence that distinguished hang from slow start
- [x] No links to `.claude/`, no absolute local paths, no host addresses
