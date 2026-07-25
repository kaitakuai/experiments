# DeepSeek-V4-Flash: NVFP4 vs honest FP8 on one B300 — how much does the cheap quantisation actually buy?

**Date:** 2026-07-25
**Models:** `deepseek-ai/DeepSeek-V4-Flash` (FP8 `e4m3`, block `[128,128]`, `scale_fmt: ue8m0`) vs `nvidia/DeepSeek-V4-Flash-NVFP4`
**Hardware:** 1× NVIDIA B300 SXM6 AC (275,040 MiB, 1100 W), driver 580.126.09, CUDA 13.0, TP=1
**Image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash:0.2.13-vllm0.25.1-overlay-k4`
**Digest:** `sha256:2af898fa516424ea2884b77e40ae480ce61b19a7d0112f8c9b8cc866c8bcb28a`
**PoC:** v2 plugin, seq_len 1024, k_dim 12, `--max-model-len 400000`

> V4 thresholds are **not calibrated**. L2 values are distances only — no PASS/FRAUD verdicts.

## Why this run exists

NVFP4 is the only quantised V4 checkpoint that loads at all: GPTQ/AutoRound and
compressed-tensors 4-bit both fail in the V4 weight loader, which understands only FP8
block scales and FP4 experts, and expert-pruned variants fail in the MoE router CUDA
kernel, which hard-codes the allowed expert counts. It is therefore *the* realistic cheat:
a node that runs
NVFP4 instead of FP8 produces nonces that per-nonce L2 barely distinguishes from honest
ones. The open question was how much such a node actually gains.

Previous numbers compared an NVFP4 collection on 2×B200 against an honest collection on a
different box — so hardware and TP differences were folded into the "gain". **This run puts
both models on the same single GPU, same image, same context length, in both compilation
modes**, so the difference is the model and nothing else.

## Result 1 — PoC throughput: NVFP4 gains a lot

Batch sweep, 5 s warmup + 30 s measure, nonces/min:

| batch | honest FP8 eager | honest FP8 graphs | NVFP4 eager | NVFP4 graphs |
|------:|-----------------:|------------------:|------------:|-------------:|
| 8 | 1184 | 1648 | 2192 | 2192 |
| 16 | 1664 | 1696 | **2720** | 2304 |
| 32 | 1664 | **1728** | 2368 | 2368 |

**NVFP4 advantage: +37 %** comparing like-for-like at batch 32 (2368 vs 1728). Comparing
each model's best configuration gives +57 % (2720 vs 1728), but see the caveat below.

**Caveat on the 2720 figure.** The NVFP4 eager curve is non-monotonic (2192 → 2720 → 2368).
Each batch size is measured once for 30 s, so a single point can be an outlier; 2720 vs
2368 is 5.5 measurement quanta, i.e. not rounding noise, but it deserves a re-measure
before being relied on. **The conservative +37 % at equal batch is the number to use.**

CUDA graphs buy NVFP4 nothing (2192/2368 identical at b8 and b32) — the same
compute-bound behaviour the honest model shows on this box, analysed in
`../deepseek-v4-flash-1xb300-cudagraph-ab/`.

## Result 2 — inference: NVFP4's gain depends entirely on the load

All four cells were measured: both models × both compilation modes, same card, same image,
**zero failed requests in all sixteen runs**. Full metrics recomputed from the raw
per-request `Measurements` table (see *Metrics* below).

**Eager mode** — NVFP4 gains almost nothing:

| Scenario | out tok/s honest → NVFP4 | Δ | TPOT Δ |
|---|---|---:|---:|
| s1 long prompt, sequential | 14.17 → 15.17 | +7 % | −5 % |
| s2 short prompt, concurrent | 346.63 → 351.50 | **+1 %** | −2 % |
| s3 very long, sequential | 14.03 → 14.63 | +4 % | −4 % |
| s4 very long, max concurrency | 110.13 → 122.81 | +12 % | −11 % |

**Graphs on (the production configuration)** — three scenarios unchanged, one very much not:

| Scenario | out tok/s honest → NVFP4 | Δ | TPOT Δ |
|---|---|---:|---:|
| s1 long prompt, sequential | 109.45 → 107.04 | **−2 %** | 0 % |
| s2 short prompt, concurrent | 1548.65 → 1607.89 | +4 % | −6 % |
| s3 very long, sequential | 96.84 → 104.71 | +8 % | 0 % |
| **s4 very long, max concurrency** | 256.67 → **363.81** | **+42 %** | **−47 %** |

So the honest summary is *not* "NVFP4 barely helps inference". It is:

- **Sequential or lightly loaded serving: no meaningful gain** (−2 % to +8 %). Decode there
  is not limited by weight bandwidth, so four-bit weights buy nothing.
- **Heavy concurrency at long context: +42 %**, with TPOT halved. With twenty sequences in
  flight the weight traffic starts to dominate, and that is exactly where a 4× smaller model
  pays off.

### Consequence for fraud economics

An NVFP4 node gains **+37 % on PoC** unconditionally, and on inference it gains anywhere
between nothing and +42 % **depending on how loaded it is**. A lightly-loaded node cheats
mainly for the PoC reward; a heavily-loaded one gains on both. Threshold calibration and
reward weighting are therefore coupled, and the inference-side incentive cannot be dismissed
as negligible without knowing the fleet's real concurrency profile.

## Result 3 — CUDA graphs help inference on both models, PoC on neither

NVFP4, eager → graphs on, same card:

| Scenario | LATENCY | TPOT | out tok/s |
|---|---|---|---|
| s1 | 8.48 → **0.65** s (−92 %) | −89 % | 15.2 → **107.0** |
| s2 | 11.77 → **2.93** s (−75 %) | −77 % | 351.5 → **1607.9** |
| s3 | 10.24 → **1.28** s (−87 %) | −89 % | 14.6 → **104.7** |
| s4 | 11.45 → **6.81** s (−40 %) | −65 % | 122.8 → **363.8** |

The same flag that does nothing for PoC on this box transforms serving. Both facts follow
from one rule: graphs remove per-step kernel-launch cost, and only workloads bound by that
cost benefit. PoC at batch 32 on a B300 is compute-bound; the decode loop, which pays a
full launch round per token, is not.

## Nonce reproducibility

| Pair | bit-identical | median L2 | >0.4 |
|---|---:|---:|---:|
| NVFP4 eager vs NVFP4 graphs (same card) | 968/1000 (96.8 %) | 0.0000 | 0.10 % |
| NVFP4 vs honest FP8 (same card) | 0/1000 | 0.2100 | 6.10 % |
| NVFP4 (B300) vs honest FP8 (B200) | 0/1000 | 0.2088 | 5.70 % |
| NVFP4 (B200) vs honest FP8 (B300) | 0/1000 | 0.2121 | 5.70 % |
| honest FP8 across cards (3 pairs) | 0/1000 | 0.188–0.190 | 2.5–3.8 % |

Two things worth noting. NVFP4 reproduces itself bit-for-bit across a compilation-mode
change exactly as the honest model does — determinism on Blackwell is not a property of the
quantisation. And the NVFP4 offset is **identical whether measured on the same card or
across cards** (0.2088 / 0.2100 / 0.2121): it is a model-level signal, whereas the honest
0.188 floor is entirely machine noise.

That is the lever for detection: pin validation to one machine and the honest floor
collapses toward zero while NVFP4 stays at 0.21. Across machines the two distributions
overlap and no per-nonce threshold separates them.

## Metrics — how they were computed

The public `compressa-perf` 0.2.7 on PyPI is **not** the internal Gonka build: it rejects
`--no-sign/--node_url/--model_name`, uses `openai_url` instead of `node_url` in the YAML,
crashes after each scenario in report generation (`FileNotFoundError: 'logo.png'`), and
aggregates only TTFT.

Workarounds used here, all committed in `scripts/`:

- `compressa_config_027.yml` — config ported to the 0.2.7 schema.
- `serving.sh` — splits the config into one file per scenario and runs them separately, so
  the report-generation crash cannot abort the remaining scenarios.
- `metrics.py` — recomputes the full metric set (latency, TPOT, throughput, RPS,
  percentiles) from the raw per-request `Measurements` table, which 0.2.7 does populate.

## Files

| Path | What |
|---|---|
| `artifacts/{nvfp4,honest}_{eager,graphs}_poc_sweep.log` | four PoC sweeps |
| `artifacts/{nvfp4_eager,nvfp4_graphs,honest_eager}_nonces.json` | nonce sets |
| `artifacts/{honest,nvfp4}_{eager,graphs}_full.json` | full serving metrics, all four cells |
| `scripts/v4ab.sh`, `next_cell.sh`, `serving.sh` | the run drivers |
| `scripts/patch_runner.py` | the only modification to the image |
| `scripts/metrics.py` | metric recomputation from raw measurements |

## Gotchas that cost time here

- **`run_pow_generation.py` is pinned to the default model** via
  `MODEL_NAME = os.environ.get("MODEL", "deepseek-ai/DeepSeek-V4-Flash")`. Running it
  against NVFP4 makes the PoC plugin answer `409 params mismatch`, which the mlnode proxy
  turns into a bare `502 Bad Gateway` in the sweep log. Nonce collection keeps working
  (it takes `--model` as an argument), so the failure looks like "collection fine, sweep
  broken" and misleads toward warmup/JIT. Pass `MODEL=` explicitly.
- **`is_running: true` does not mean the engine can compute** — part of the kernels compile
  lazily on the first forward. Warm up with a real request before measuring, or the first
  scenario measures JIT.
- `docker exec` without `-i` silently swallows heredocs; the runner patch appears to succeed
  while the file is untouched. Use `docker cp` and run by path.
- The image needs mlnode's Python deps installed before the API starts (`toml` first of all).

## Reproducibility checklist

- [x] Both models on the same GPU, same image, same context length, both modes
- [x] Image referenced by tag **and** digest
- [x] All drivers and the metric recomputation committed in `scripts/`
- [x] Raw sweeps, nonce sets and per-cell metrics committed
- [x] Non-monotonic data point flagged rather than silently used
- [x] No links to `.claude/`, no absolute local paths, no host addresses
- [x] No verdicts asserted — V4 thresholds are not calibrated
