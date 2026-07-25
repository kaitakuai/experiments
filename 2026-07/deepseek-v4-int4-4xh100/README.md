# DeepSeek-V4-Flash INT4 on 4× H100: matches honest FP8 exactly — and saves nothing

**Date:** 2026-07-25
**Models:** `Intel/DeepSeek-V4-Flash-W4A16-AutoRound` (INT4, auto-round / auto_gptq, group 128) vs `deepseek-ai/DeepSeek-V4-Flash` (FP8)
**Hardware:** 4× NVIDIA H100 80GB HBM3 SXM (700 W, NV18 full mesh), driver 595.71.05, TP=4
**Base image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash:0.2.13-vllm0.25.1-overlay-k4` (`sha256:2af898fa…8cbb28a`)
**Patch:** [vllm-project/vllm#45645](https://github.com/vllm-project/vllm/pull/45645), applied in place (`scripts/pr45645_vllm_only.diff`)
**PoC:** v2 plugin, seq_len 1024, k_dim 12, `--max-model-len 400000`

> V4 thresholds are **not calibrated**. L2 values are distances only — no PASS/FRAUD verdicts.

## Why this run exists

The B300 measurement (`../deepseek-v4-int4-autoround-1xb300/`) found INT4 running at **half**
the honest throughput and concluded it was economically pointless as a cheat. That
conclusion was drawn on one architecture, where the PR's BF16 fallback is the only path
available. This run repeats it on Hopper, where a real W4A16 kernel exists.

The answer changes for throughput — and then a second measurement kills the cheat anyway,
for a different reason.

## Result 1 — on Hopper, INT4 ties honest FP8 exactly

PoC batch sweep, same card count, same TP, same args:

| batch | honest FP8 | INT4 eager | INT4 graphs |
|------:|-----------:|-----------:|------------:|
| 8 | — | 1423 | 1423 |
| 16 | — | 1503 | 1503 |
| **32** | **1536** | **1535** | **1535** |

**1535 vs 1536 — a one-nonce difference, i.e. zero within the measurement quantum.**
Compare with B300, where the same checkpoint gave 832 against 1664 honest.

The cause is visible in the engine log (`artifacts/int4_eager_backends.txt` and the
`MARLIN` / `W4A16` lines): Hopper selects a genuine **W4A16 MARLIN** kernel, so the
computation runs in 4 bits. On Blackwell no such path exists and the patch routes
`_o_proj` to the BF16 reference, dequantising on every forward — hence the halving.

**So "INT4 is slow" was an artifact of the architecture, not of the quantisation.**

CUDA graphs change nothing (1535 in both modes), consistent with every other V4 topology
measured in this series.

| | value |
|---|---|
| **MoE backend** | `W4A16 MARLIN` |
| **ATTN backend** | `sparse_mla` (C128A) |
| quantization as seen by the engine | `inc` |

## Result 2 — INT4 saves no memory, so it needs the same 4 GPUs

This is what actually settles the question. The Intel INT4 checkpoint is **144.9 GiB**
against the honest FP8's **148.6 GiB** — 2.5 % smaller, not 4× smaller, because its own
config keeps `wo_a`, `embed`, `gate` and others at 16 bits (`extra_config: {"wo_a":
{"bits": 16}}`); only the experts are 4-bit.

Measured directly by retrying the load with fewer GPUs:

| TP | weights per worker | total | outcome |
|---:|---:|---:|---|
| 1 | — | ~145 GiB | **OOM** — 77.34 GiB in use on a 79.18 GiB card, 2 GiB short |
| 2 | 72.65 GiB | ~145 GiB | loads, then **fails allocating KV** at `gmu 0.96` |
| 4 | 37.63 GiB | ~150 GiB | works — 1535 nonces/min |

Evidence: `artifacts/mingpu_tp1_oom.log`, `artifacts/mingpu_tp2_failure.log`. Both arms
were verified to actually use the requested TP (`tensor_parallel_size=` in the log plus
`CUDA_VISIBLE_DEVICES`), after a first attempt silently ran TP=4 — see *Gotchas*.

**An INT4 node needs exactly the same four H100s an honest node needs.** There is no
hardware saving to be had.

## Result 3 — serving

eager, TP=4, zero failed requests in all four scenarios:

| Scenario | out tok/s | TPOT |
|---|---:|---:|
| s1 long prompt, sequential | 13.3 | 0.0725 |
| s2 short prompt, concurrent | 219.0 | 0.1193 |
| s3 very long, sequential | 10.5 | 0.0758 |
| s4 very long, max concurrency | 64.9 | 0.2097 |

Full metrics: `artifacts/int4_{eager,graphs}_serving.json`, recomputed from the raw
per-request table by `scripts/metrics.py` (the public `compressa-perf` 0.2.7 aggregates
only TTFT).

## Result 4 — L2 does not depend on the kernel

| Pair | median L2 | >0.4 |
|---|---:|---:|
| INT4 (H100, **Marlin**) vs honest H100 | **0.295** | 20.8 % |
| INT4 (H100, Marlin) vs honest B300 | 0.295 | 19.2 % |
| INT4 (B300, **BF16 fallback**) vs honest B300 | **0.296** | 19.0 % |
| INT4 (H100) vs INT4 (B300) | 0.142 | 1.2 % |
| honest vs honest, different cards | 0.189 | 3.0 % |
| NVFP4 vs honest | 0.210 | 6.1 % |

Two **completely different compute paths** — a real 4-bit Marlin kernel and a
dequantise-to-BF16 fallback — land on the same distance from honest, 0.295 and 0.296,
agreeing to the third decimal.

**This retires a caveat from the B300 report.** That report flagged 0.296 as "an upper
bound that may contain a patch component" and proposed a control run of honest FP8 on the
patched image. The control is no longer needed: if the patch were generating the offset,
the Marlin path would not reproduce it. **0.296 is a property of the quantisation.**

It also explains the B300 report's "INT4 does not reproduce itself" (0/1000 bit-identical,
0.1386 between its own two modes): that 0.14 is the distance between *kernels*, the same
value seen here between H100-Marlin and B300-BF16, not run-to-run instability.

## The verdict on INT4 as a fraud vector

| axis | INT4 vs honest |
|---|---|
| PoC throughput | **equal** on Hopper (1535 vs 1536), half on Blackwell |
| memory | **equal** (145 vs 149 GiB) |
| GPUs required | **equal** (4× H100) |
| detectability | **3× more visible than NVFP4** (0.295 vs 0.210) |

INT4 offers a cheater nothing on any axis while making them easier to catch. Combined with
the earlier results, **NVFP4 remains the only vector that is simultaneously profitable
(+37 % PoC) and near-invisible (0.210 against a 0.189 honest floor).**

## Gotchas that cost time here

- **Editing `runner.py` while the API is running does nothing.** The mlnode API imports the
  module at startup; a later edit is not picked up. The first min-GPU attempt reported
  "TP=1 works on one card" while the engine had actually launched `tensor_parallel_size=4`
  — same four workers, same 1535 nonces. Always restart the API after patching, verify
  `tensor_parallel_size=` in the engine log, and pin `CUDA_VISIBLE_DEVICES` so a wrong TP
  fails loudly instead of silently succeeding.
- `pkill -f "VLLM::"` and friends **match your own SSH command line** and kill the session.
  Kill by PID obtained from `ps` with a pattern that cannot match itself.
- Rent by `gpu_ram`, not by name: `gpu_name: "A100 SXM4"` matches the **40 GB** part, on
  which V4 does not fit at all. Check `nvidia-smi topo -m` for `NV#` after renting — one
  "SXM4" offer had `NODE`/`PHB`, i.e. no NVLink.
- sm_90 needs a `libnvrtc.so` symlink or FlashInfer's JIT fails to link `-lnvrtc` and every
  worker dies at startup (`scripts/prep.sh` does this).

## Files

| Path | What |
|---|---|
| `artifacts/int4_{eager,graphs}_poc_sweep.log` | both sweeps |
| `artifacts/int4_{eager,graphs}_nonces.json` | 1088 nonces per mode |
| `artifacts/int4_{eager,graphs}_serving.json` | full serving metrics |
| `artifacts/int4_eager_backends.txt` | resolved MoE / attention backends |
| `artifacts/honest_fp8_eager_nonces.json` | the honest 4×H100 reference set |
| `artifacts/mingpu_tp{1,2}_*.log`, `mingpu_run.log` | the minimum-GPU experiment |
| `scripts/pr45645_vllm_only.diff` | the patch, test file stripped |
| `scripts/prep.sh`, `run_model.sh`, `min_gpus2.sh` | box prep, the run, the TP experiment |
| `scripts/metrics.py` | serving metrics from raw measurements |

## Reproduce

```bash
./scripts/prep.sh 4 --patch          # deps, libnvrtc, PR patch, runner args, API on :8081
hf download Intel/DeepSeek-V4-Flash-W4A16-AutoRound
./scripts/run_model.sh Intel/DeepSeek-V4-Flash-W4A16-AutoRound h100-int4
./scripts/min_gpus2.sh               # TP=1 and TP=2, each with a fresh API
python3 scripts/compare_nonces.py artifacts/int4_eager_nonces.json artifacts/honest_fp8_eager_nonces.json
```

`run_pow_generation.py` is pinned to the default checkpoint — pass `MODEL=` or the PoC
plugin answers `409 params mismatch`, surfaced as a bare `502` in the sweep log.

## Reproducibility checklist

- [x] Image by tag and digest; patch committed in-tree
- [x] TP verified from the engine's own log in every arm
- [x] Raw sweeps, nonce sets, serving metrics and failure logs committed
- [x] L2 regenerated from committed artifacts
- [x] A superseded conclusion from the B300 report explicitly retired, with the evidence
- [x] No links to `.claude/`, no absolute local paths, no host addresses
- [x] No verdicts asserted — V4 thresholds are not calibrated
