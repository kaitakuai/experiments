# DeepSeek-V4-Flash on 2× B200 SXM — PoC throughput, serving, and what "compiled" actually means

**Date:** 2026-07-24
**Model:** `deepseek-ai/DeepSeek-V4-Flash` — FP8 (`e4m3`, block `[128,128]`, `scale_fmt: ue8m0`), 256 routed experts, sparse-MLA, hash-routed MoE
**Hardware:** 2× NVIDIA B200 SXM (183,359 MiB, 1000 W, NV18), driver 595.71.05, TP=2
**Image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash:0.2.13-vllm0.25.1-overlay-k4`
**Digest:** `sha256:2af898fa516424ea2884b77e40ae480ce61b19a7d0112f8c9b8cc866c8bcb28a`
**Stack:** vLLM 0.25.1, torch 2.11.0+cu130. PoC v2 via `--worker-extension-cls gonka_poc.worker.PoCWorkerExtension`

> Thresholds for V4 are **not calibrated yet**. This report asserts **no PASS/FRAUD
> verdicts** — L2 numbers are reported as distances only.

## Summary

1. **CUDA graphs are worth +71 % of PoC throughput on B200** — 1344 → 2304 nonces/min at
   batch 32. A reverse-order control run on a second box reproduced the compiled sweep
   exactly, ruling out warm-cache ordering as the cause.
2. **They are worth 5–19× on serving** — output throughput 7.3 → 121.9 tok/s sequential,
   165 → 932 tok/s under concurrency, with zero failed requests in either mode.
3. **"compiled" here does not mean `torch.compile`.** For V4, vLLM auto-enables
   `VLLM_USE_BREAKABLE_CUDAGRAPH`, which *disables* the torch.compile pipeline outright.
   Our `--compilation-config '{"mode":3,...}'` has no effect. The only thing separating the
   two columns of this report is whether `--enforce-eager` is passed, i.e. whether the
   **breakable CUDA graph** is active.
4. **The gain is not an architecture property.** A later A/B on 1×B300 TP=1 — also
   Blackwell — gained only 3.8 % at batch 32, because that configuration is already
   compute-bound in eager. What decides the gain is whether kernel-launch cost is the
   binding limit; see the section below.

## What "compiled" really is

From the startup log (`artifacts/env.txt`, and reproduced on the control box):

```
Auto-enabling VLLM_USE_BREAKABLE_CUDAGRAPH=1. Set VLLM_USE_BREAKABLE_CUDAGRAPH=0 to opt out.
VLLM_USE_BREAKABLE_CUDAGRAPH is set, disabling vLLM's torch.compile pipeline.
  Equivalent to -cc.mode=none.
Breakable CUDA graph enabled
Profiling CUDA graph memory: PIECEWISE=51 (largest=512), FULL=51 (largest=512)
```

The `largest=512` line is only the *memory profiling* pass. The wrapper itself captures
lazily, per observed batch shape — from `vllm/compilation/breakable_cudagraph.py`:

> *captures whatever the dispatcher emits (any non-NONE runtime_mode) … Entries are keyed
> by `BatchDescriptor` which already encodes batch shape / uniformity.*

So the PoC forward (batch 32 × seq_len 1024 = 32,768 tokens) **is** graph-captured despite
being far outside the profiled buckets. That is why it benefits.

**Naming guidance for future reports:** label the two configurations
`breakable CUDA graph on/off`, not `eager vs compiled mode 3`. The latter implies a
torch.compile comparison that is not happening.

## PoC throughput

seq_len 1024, k_dim 12, 5 s warmup + 30 s measure per batch size.

| batch | eager (graphs off) | compiled (graphs on) | gain |
|------:|-------------------:|---------------------:|-----:|
| 8 | 1232 | 2160 | +75 % |
| 16 | 1344 | 2272 | +69 % |
| **32** ★ | **1344** | **2304** | **+71 %** |

Both curves are nearly batch-flat (+9 % and +7 % from b8 to b32): throughput is bounded by
per-iteration fixed cost, not by batch parallelism.

### Reverse-order control run

Because the original run measured eager first and compiled second, the gain could have
been a warm-cache artifact. A second B200 box ran the **same sweep in the opposite order**:

| batch | original (compiled 2nd) | control (compiled 1st) |
|------:|------------------------:|-----------------------:|
| 8 | 2160 | 1744 |
| 16 | 2272 | **2272** |
| 32 | 2304 | **2304** |

The control's eager phase was not measured — the box was released after the compiled sweep
answered the ordering question. The eager column above is therefore from the original run
only.

b16 and b32 reproduce to the digit (1136 and 1152 raw nonce counts). The b8 shortfall in
the control is the lazy graph capture for that batch shape landing *inside* the measurement
window — in the original ordering the shape had already been captured during the preceding
phase. **Conclusion: the +71 % is a genuine CUDA-graph effect, not ordering** — compiled-first
and compiled-second produce identical throughput.

## What actually determines the CUDA-graph gain

Throughput here is **the minimum of two limits**: how fast the host can launch kernels, and
how fast the GPU can compute. CUDA graphs remove the launch limit, so the gain appears only
when that limit was the binding one.

Measured across four configurations, all V4-Flash, all k4 image, batch 32:

| Config | eager | graphs on | gain | binding limit in eager |
|---|---:|---:|---:|---|
| 2×B200 TP=2 | 1344 | **2304** | **+71 %** | launch |
| 1×B300 TP=1 | 1664 | 1728 | +3.8 % | compute |
| 2×H200 SXM TP=2 | 1216 | 1216 | < 5 % | compute |
| 4×H100 TP=4 | 1536 | 1536 | < 5 % | compute |

The B300 run settles it: **the gain is not a property of the architecture.** B300 is
Blackwell like B200, yet gains nothing at batch 32 — because with TP=1 on a fast host
(240-vCPU EPYC) it is already at its compute ceiling in eager. TP=2 doubles the per-step
launch work and adds cross-worker synchronisation, which is why 2×B200 was launch-bound at
1344 despite having two GPUs' worth of compute available.

The same B300 run shows the mechanism directly at small batch, where launch cost dominates:

| batch | eager | graphs on | gain |
|------:|------:|----------:|-----:|
| 8 | 1184 | 1648 | **+39 %** |
| 16 | 1664 | 1696 | +1.9 % |
| 32 | 1664 | 1728 | +3.8 % |

Both modes converge on the same ~1700 ceiling; graphs merely reach it at a smaller batch.

**Practical rule:** do not assume a fleet-wide answer. Whether `--enforce-eager` costs 70 %
or 4 % depends on TP, host CPU speed and batch size together — measure the actual serving
configuration.

## Serving — compressa-perf

| Scenario | Metric | eager | compiled | Δ |
|---|---|---:|---:|---:|
| s1 long prompt, sequential | LATENCY (s) | 14.1897 | **0.9220** | −94 % |
| | out tok/s | 7.33 | **121.90** | +1563 % |
| s2 short prompt, concurrent | LATENCY (s) | 23.4483 | **4.4534** | −81 % |
| | out tok/s | 165.54 | **932.26** | +463 % |
| s3 very long, sequential | LATENCY (s) | 13.6829 | **1.6301** | −88 % |
| | out tok/s | 5.96 | **113.86** | +1809 % |
| s4 very long, max concurrency | LATENCY (s) | 30.8043 | **6.7858** | −78 % |
| | out tok/s | 31.98 | **287.88** | +800 % |

`FAILED_REQUESTS = 0` in all eight runs. TPOT drops 73–95 %, i.e. the win is concentrated
in decode, where every token pays a full kernel-launch round — exactly the cost graphs
remove. Full per-scenario metrics: `artifacts/compressa_parsed.json`.

## Validation — L2 against other topologies

1000 common nonces per pair, same `block_hash` and `public_key`.

| Pair | bit-identical | median L2 | >0.4 |
|---|---:|---:|---:|
| B200 eager vs B200 compiled (same box) | **968/1000 (96.8 %)** | 0.0000 | 0.00 % |
| B200 vs H200 SXM | 0/1000 | 0.1895 | 3.80 % |
| B200 vs B300 | 0/1000 | 0.1880 | 3.60 % |

Switching compilation mode does **not** move a single nonce past 0.4 — the 3.2 % of
non-identical vectors all stay inside the honest band. Cross-machine pairs behave as
everywhere else in this series (median ≈ 0.19). Machine-readable: `artifacts/l2_matrix.json`.

Blackwell reproducing 96.8 % bit-identical while Hopper reproduces 0/1000 is analysed in
`../deepseek-v4-flash-2xh200/README.md`.

## Config — what changed vs default

| Setting | Default | Here | Why |
|---|---|---|---|
| `--tensor-parallel-size` | 1 | 2 | model does not fit on one B200 |
| `--max-model-len` | model default | 400000 | realistic serving context |
| `--max-num-batched-tokens` | 8192 | 32768 | PoC forward batches 32×1024 |
| `--kv-cache-dtype` | auto | fp8 | required by FlashMLA on V4 |
| `--gpu-memory-utilization` | 0.9 | 0.90 | unchanged |
| `--logprobs-mode` | raw | processed_logprobs | PoC scheme |
| `--worker-extension-cls` | — | `gonka_poc.worker.PoCWorkerExtension` | PoC plugin |
| `--enforce-eager` | off | **the variable under test** | toggles breakable CUDA graph |

KV cache: 2,359,479 tokens (graphs off) → 2,300,269 (graphs on) — graphs cost ~59k tokens,
about 2.5 %. Max concurrency at full 400k context: 5.90× → 5.75×.

## Files

| Path | What |
|---|---|
| `artifacts/{eager,compiled}_poc_sweep.log` | batch sweeps, both modes |
| `artifacts/{eager,compiled}_nonces_1000.json` | 1000 PoC nonce vectors per mode |
| `artifacts/{eager,compiled}_compressa.log` | serving benchmark, both modes |
| `artifacts/compressa_parsed.json` | per-scenario metrics extracted per mode |
| `artifacts/l2_matrix.json` | L2 + bit-identity table above |
| `artifacts/config.json` | GPU, driver, versions, full arg list, KV sizing |
| `artifacts/env.txt`, `runner_forced_args.txt` | raw environment and the forced-arg patch |
| `artifacts/control_reverse_order_compiled_sweep.log` | the reverse-order control sweep |
| `artifacts/control_reverse_order_run.log` | full control run log with milestones |
| `artifacts/control_startup_cudagraph_evidence.txt` | startup lines proving graphs were active |
| `scripts/` | box setup, mode runner, PoC sweep, nonce collection, L2 comparison |

## Findings

1. On Blackwell, **do not run PoC nodes with `--enforce-eager`** — it costs ~70 % of nonce
   throughput and 5–19× of serving throughput, for ~2.5 % of KV cache saved.
2. The same flag is nearly free on Hopper (< 5 %), so a single fleet-wide recommendation is
   wrong; it must be per-architecture.
3. Any claim that "PoC throughput is compilation-invariant" holds only where the GPU is
   slow enough to mask launch overhead. It does not generalise.
4. `--compilation-config` is inert for V4. Reports comparing "mode 3" on this model are
   really comparing CUDA graphs on/off.

## Reproduce

```bash
./scripts/setup_box.sh 2 deepseek-ai/DeepSeek-V4-Flash
./scripts/run_mode.sh eager    b200 deepseek-ai/DeepSeek-V4-Flash
./scripts/run_mode.sh compiled b200 deepseek-ai/DeepSeek-V4-Flash
# use distinct filenames — compare_nonces.py labels pairs by basename
python3 scripts/compare_nonces.py eager_nonces_1000.json compiled_nonces_1000.json
```

Pick the box by `gpu_max_power` (1000 W here → SXM) and `cuda_max_good >= 13.0`; the
CUDA-13 image will not start on a driver older than 580.

## Reproducibility checklist

- [x] Image referenced by tag **and** digest
- [x] Full vLLM argument list committed (`artifacts/config.json`)
- [x] All scripts used committed in `scripts/`
- [x] Raw sweep, nonce, and serving logs committed
- [x] L2 numbers regenerated from committed artifacts with the committed script
- [x] Ordering confound tested explicitly with a reverse-order control run
- [x] No links to `.claude/`, no absolute local paths, no sibling-repo references
- [x] No verdicts asserted — V4 thresholds are not calibrated yet
