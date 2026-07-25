# DeepSeek-V4-Flash on 1× B300 — CUDA-graph A/B, and why the B200 result does not generalise

**Date:** 2026-07-25
**Model:** `deepseek-ai/DeepSeek-V4-Flash` — FP8 (`e4m3`, block `[128,128]`, `scale_fmt: ue8m0`), 256 routed experts
**Hardware:** 1× NVIDIA B300 SXM6 AC (275,040 MiB, 1100 W), driver 580.126.09, CUDA 13.0, TP=1
**Host:** AMD EPYC, 240 vCPU (8-GPU node; one GPU borrowed for this run)
**Image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash:0.2.13-vllm0.25.1-overlay-k4`
**Digest:** `sha256:2af898fa516424ea2884b77e40ae480ce61b19a7d0112f8c9b8cc866c8bcb28a`
**Stack:** vLLM 0.25.1. PoC v2 via `--worker-extension-cls gonka_poc.worker.PoCWorkerExtension`

> Thresholds for V4 are **not calibrated yet** — L2 values below are distances only, no verdicts.

## Why this run exists

The 2×B200 report measured **+71 %** PoC throughput from CUDA graphs and explained it as
Blackwell being fast enough that per-step kernel-launch cost stops being hidden behind
compute. That explanation predicts B300 — also Blackwell, and faster — should gain at least
as much.

It does not. **This run falsifies the architecture-based explanation.**

An earlier 1×B300 number (`../deepseek-v4-flash-poc-1xb300/`) claimed compilation had no
effect, but that was never a controlled A/B: it compared a 400k compiled sweep against an
8192 eager one, on a different image and a different PoC integration. This run changes one
variable only — `--enforce-eager` — on one image, one box, one context length.

## Result

Batch sweep, seq_len 1024, 5 s warmup + 30 s measure, `--max-model-len 400000`:

| batch | eager (graphs off) | graphs on | gain |
|------:|-------------------:|----------:|-----:|
| 8 | 1184 | **1648** | **+39 %** |
| 16 | 1664 | 1696 | +1.9 % |
| **32** ★ | **1664** | **1728** | **+3.8 %** |

At batch 32 the difference is 832 vs 864 nonces — **exactly one batch, one measurement
quantum**. Indistinguishable from zero at this resolution.

Nonce collection (batch 32, continuous) agrees: 1360 vs 1375 nonces/min, +1.1 %.

## The model that fits all four topologies

Throughput is **the minimum of two limits**: how fast the host can launch kernels, and how
fast the GPU can compute. CUDA graphs remove the launch limit, so a gain appears only where
that limit was binding.

| Config | eager | graphs on | gain | binding limit in eager |
|---|---:|---:|---:|---|
| 2×B200 TP=2 | 1344 | **2304** | **+71 %** | launch |
| **1×B300 TP=1** | **1664** | **1728** | **+3.8 %** | **compute** |
| 2×H200 SXM TP=2 | 1216 | 1216 | < 5 % | compute |
| 4×H100 TP=4 | 1536 | 1536 | < 5 % | compute |

Two things fall out of this:

- **The batch sweep shows the mechanism directly.** Both modes on B300 converge on the same
  ~1700 ceiling; graphs only reach it at a smaller batch. At batch 8, where launch cost
  dominates, even this box gains +39 %.
- **TP is a cost, not just a capacity.** 1×B300 does 1664 nonces/min in eager while 2×B200
  does 1344 — one GPU beating two. That is not B300 being faster than two B200s; TP=2
  doubles per-step launch work and adds cross-worker synchronisation, so the B200 pair was
  pinned at its launch limit while its compute sat idle. Graphs unpinned it, hence +71 %.

**Practical rule:** there is no fleet-wide answer to "should PoC nodes run with
`--enforce-eager`". It costs 71 % in one configuration and 4 % in another. Measure the
configuration you actually deploy — TP, host CPU and batch size together decide it.

## Reproducibility of nonces

| Pair | bit-identical | median L2 | >0.4 |
|---|---:|---:|---:|
| B300 eager vs B300 graphs (same box) | **968/1000 (96.8 %)** | 0.0000 | 0.00 % |
| B300 vs 2×B200 | 0/1000 | 0.1882 | 3.70 % |
| B300 vs 2×H200 SXM | 0/1000 | 0.1889 | 2.70 % |

Repeating on the same Blackwell box reproduces 96.8 % of nonces bit-for-bit — identical to
the B200 figure, and unlike Hopper's 0/1000. Note this supersedes the 87.5 % measured on
B300 with the older in-tree fork image: on the k4 plugin image B300 and B200 agree exactly.

Switching graph mode moves no nonce past 0.4. Cross-machine pairs sit at the usual ≈0.19.

## Config — what changed vs the image defaults

Only `runner.py` parameters were touched (`scripts/patch_runner.py`):

| Setting | Image default | Here |
|---|---|---|
| `--tensor-parallel-size` | 1 | 1 (unchanged) |
| `--max-model-len` | 200000 | **400000** |
| `--max-num-batched-tokens` | 16384 | **32768** (batch 32 × seq_len 1024) |

Everything else the image forces is untouched: `--gpu-memory-utilization 0.90`,
`--kv-cache-dtype fp8`, `--logprobs-mode processed_logprobs`, `--trust-remote-code`,
`--worker-extension-cls gonka_poc.worker.PoCWorkerExtension`.

Engine confirmed at `max_seq_len=400000`, `tensor_parallel_size=1`, GPU KV cache
**2,661,034 tokens**, max concurrency at 400k context **6.65×**.

Mode toggle, verified in the startup log (`artifacts/graph_evidence.txt`):

| Mode | args | log evidence |
|---|---|---|
| eager | `--enforce-eager` | no `Breakable CUDA graph enabled` line |
| graphs on | `--compilation-config '{"mode":3,…}'` | `BREAKABLE_CUDAGRAPH=1`, `Breakable CUDA graph enabled` |

Note the `mode:3` argument is **inert** for V4 — vLLM auto-enables
`VLLM_USE_BREAKABLE_CUDAGRAPH`, which disables the torch.compile pipeline outright. The only
effective variable is `--enforce-eager`.

## Files

| Path | What |
|---|---|
| `artifacts/{eager,compiled}_poc_sweep.log` | the two sweeps |
| `artifacts/{eager,compiled}_nonces_1000.json` | 1088 nonces per mode |
| `artifacts/{eager,compiled}_collect.log` | collection logs |
| `artifacts/graph_evidence.txt` | startup lines proving which mode was active |
| `artifacts/run.log` | full run log with milestones |
| `artifacts/l2_matrix.json` | the table above, machine-readable |
| `scripts/v4ab.sh` | the whole A/B, container-based |
| `scripts/patch_runner.py` | the only modification to the image |
| `scripts/{run_pow_generation,collect_artifacts,compare_nonces}.py` | sweep, collection, L2 |

## Reproduce

```bash
# on a box with the image and the weights, one free GPU:
bash scripts/v4ab.sh
python3 scripts/compare_nonces.py artifacts/eager_nonces_1000.json artifacts/compiled_nonces_1000.json
```

Two failure modes cost a restart each and are already fixed in `v4ab.sh`, but they bite any
container-based run of this image:

- `docker exec` **without `-i` silently swallows a heredoc** — the runner patch appeared to
  succeed while the file was untouched. Copy the script in with `docker cp` and run it by path.
- The image needs mlnode's Python deps installed before the API will start
  (`toml accelerate fire fastrlock h2 termcolor typer-slim setuptools-scm tenacity`);
  without `toml` the API dies with `ModuleNotFoundError` and the engine never comes up.

Cold JIT takes ~15 minutes per mode (`ptxas` at 100 % CPU, GPU idle). That is compilation,
not a hang — do not restart it.

## Findings

1. **The CUDA-graph gain is configuration-specific, not architectural.** Same vendor, same
   generation, same image, same model: +71 % on 2×B200 TP=2, +3.8 % on 1×B300 TP=1.
2. **Throughput = min(launch limit, compute limit).** Graphs raise the first. Batch 8 on
   B300 (+39 %) shows the launch limit binding on the very same box where batch 32 shows it
   not binding.
3. **TP has a launch-side cost** that can leave a multi-GPU configuration slower than a
   single GPU in eager mode.
4. **B300 reproduces 96.8 % of nonces bit-for-bit across a mode change**, matching B200 and
   superseding the 87.5 % measured on the older fork image.

## Reproducibility checklist

- [x] Image referenced by tag **and** digest
- [x] Only one variable changed between the two arms (`--enforce-eager`)
- [x] Mode actually in effect verified from the engine's own startup log
- [x] All scripts committed, including the exact image modification
- [x] Raw sweep, collection and nonce artifacts committed
- [x] L2 numbers regenerated from committed artifacts
- [x] No links to `.claude/`, no absolute local paths, no sibling-repo references
- [x] No verdicts asserted — V4 thresholds are not calibrated yet
