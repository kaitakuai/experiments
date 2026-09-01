# GLM-5.3-Flash — 4×H200 — honest FP8 baseline (floor 0.0000; the 0.40 gate works on Hopper)

**Date:** 2026-09-01
**Model:** [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) @
`03eb5366286afd40d2221b1d9c63a6dd1ba4832e` — native FP8 (`weight_block_size [128,128]`),
288 routed experts, `num_experts_per_tok` 8, 45 layers, 306 GB, 62 shards.
**Hardware:** 4× NVIDIA H200 SXM (143 771 MiB each, 700 W, NV18 full mesh), TP=4,
driver 595.71.05, CUDA 13.
**Image:** `ghcr.io/kaitakuai/mlnode-h100-glm-5-3-flash:0.2.14-vllm0.28-glm53-test-k3`
**Digest:** `sha256:b92b8cc6fbccd59f60d283bc979510d6bd519009864c2e60e83cff8852be5f3a`
**vLLM:** `0.28.0.dev0+glm53.gonka.sampler1`, FlashInfer **0.6.18 stable** — no source patches
**PoC:** `gonka_poc` 0.1.4, seq_len 1024, k_dim 12, collection batch 16

## Summary

The honest baseline for GLM-5.3-Flash on Hopper, and the reference set for the fraud arm in
[`../glm53-flash-reap50-patrickbdevaney-4xh200/`](../glm53-flash-reap50-patrickbdevaney-4xh200/).

- **The 0.40 gate is usable on Hopper.** Honest floor puts **1 nonce in 1000** past it. Earlier
  guidance said the threshold was inapplicable to Hopper; that came from the previous image
  (FlashInfer 0.6.17), where an honest run alone produced 15–17 % mismatches.
- **Honest runs are bit-exact except one nonce per batch.** Same box, same seed, two consecutive
  runs: **937 of 1000 vectors identical to the bit**, median L2 **0.0000**.
- **That exception is a defect, not noise.** The 63 differing nonces are exactly those at
  `index % 16 == 0`. Against a different GPU generation their median L2 is **1.07** versus 0.25
  for the other 937, and they alone raise an honest node's cross-hardware mismatch rate from
  11 % to 17 %. DeepSeek-V4 shows no such artifact.
- **Cross-generation honest-vs-honest is 17 % past the gate**, so at the chain's `p_mis = 0.001`
  a healthy mixed fleet would be called fraudulent. The mismatch tolerance, not the distance
  threshold, is what needs calibrating.
- **PoC saturates at batch 8** (1439 nonces/min at both 8 and 16). Batch 24 kills the engine
  with XID 31 out of DeepGEMM — but that is **almost certainly self-inflicted**: this run capped
  `--max-num-batched-tokens` at 16384 and batch 24 needs 24576. On B200 the identical overrun
  produces a clean shape error, and raising the budget makes batch 32 work. See *Corrections*.

## Environment

| Parameter | Value |
|---|---|
| GPU | 4× NVIDIA H200 SXM, 143 771 MiB each, 700 W, NV18 full mesh |
| NVIDIA driver | 595.71.05 (CUDA 13 native; the image's `compat` layer is not used) |
| Python | 3.12 |
| vLLM | `0.28.0.dev0+glm53.gonka.sampler1` |
| FlashInfer | 0.6.18 (stable, released 2026-08-29) |
| gonka_poc | 0.1.4 |
| Attention backend | `FLASHINFER_MLA_SPARSE_SM90` |
| Weights per GPU | 75.95 GiB |
| KV cache | ≈ 6.1–6.2 M tokens |

FlashInfer 0.6.18 is what makes this configuration possible: it opens the
`FLASHINFER_MLA_SPARSE_SM90` path, the only one that handles fp8 KV at `qk_rope_head_dim = 0`.
On 0.6.17 fp8 is unavailable on Hopper.

## Config

```bash
ulimit -n 524288                     # Vast ships a soft limit of 1024; NCCL needs more

gonka-vllm-serve \
  --model <SNAPSHOT PATH> --served-model-name glm53 \
  --tensor-parallel-size 4 \
  --kv-cache-dtype fp8 \
  --block-size 2304 \
  --max-num-seqs 256 \
  --no-enable-flashinfer-autotune \
  --logprobs-mode processed_logprobs \
  --worker-extension-cls gonka_poc.worker.PoCWorkerExtension \
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
  --host 0.0.0.0 --port 8081
```

`--model` must be a **local snapshot path**, not an HF id: `Glm5NextProcessor` looks for
`processor_config.json` as a file and fails on `zai-org/GLM-5.3-Flash`.

### What changed vs the default

The image's own `runner.py` ships this exact configuration — verified by reading it on the box
before patching anything (`scripts/setup_mlnode.sh` prints it):

| Parameter | Image as shipped | This run | Why |
|---|---|---|---|
| `--tensor-parallel-size` | **8** | 4 | the box has 4 GPUs |
| everything else | `--kv-cache-dtype fp8`, `--block-size 2304`, `--max-num-seqs 256`, `--logprobs-mode processed_logprobs`, `--worker-extension-cls`, `--no-enable-flashinfer-autotune` | unchanged | — |

Launching through mlnode (`scripts/start_mlnode_api.sh`) and reading the engine's own config
line confirms the chain does **not** override these: `tensor_parallel_size=4`,
`kv_cache_dtype=fp8`, `quantization=fp8`, `enforce_eager=False`. The one value this run changes
is the GPU count, and on 141 GB cards that is a legitimate production topology rather than a
reduction: the weights fit at TP=4 with room for KV. The image's `TP=8` default is sized for
80 GB cards.

## Validation

### L2

| metric | value |
|---|---:|
| median L2 | **0.0000** |
| mean | 0.0101 |
| p95 / p99 | 0.1163 / 0.1986 |
| max | 0.5737 |
| differing nonces | 63 / 1000 |
| past 0.40 | **1 / 1000 (0.1 %)** → PASS |

937 of 1000 vectors are identical to the bit. The previous finding "0 % bit-identical on
Hopper" belonged to the FlashInfer 0.6.17 image and is superseded.

### Cross-hardware L2

Against the published 2×B300 set, [`../../2026-08/glm53-flash-fp8-2xb300/`](../../2026-08/glm53-flash-fp8-2xb300/),
same three seeds. Gate thresholds quoted from the chain default: `threshold = 0.40`,
`p_mis = 0.001`.

| seed | median L2 | p95 | past 0.40 | past 0.40, excluding first-in-batch |
|---|---:|---:|---:|---:|
| s1 | 0.2559 | 0.8971 | 16.8 % | 11.2 % |
| s2 | 0.2674 | 0.8606 | 17.0 % | 11.4 % |
| s3 | 0.2658 | 0.8916 | 16.3 % | 10.7 % |

Both sides are honest, so **at `p_mis = 0.001` the chain would call a healthy mixed fleet
fraudulent**. Distinguishing power is intact — 17 % honest against 90 % for the fraud arm is a
5× gap — but the mismatch tolerance has to be calibrated to roughly 0.20 for cross-generation
validation, not 0.001.

Three levels measured across this folder and its fraud counterpart:

| comparison | past 0.40 |
|---|---:|
| honest vs itself, same box | 0.1 % |
| honest 2×B300 vs honest 4×H200 | 17 % |
| REAP50 fraud vs honest, same box | 90 % |

### The batch-boundary artifact

Nonces are indexed 0…999 and collected in batches of 16. Splitting every comparison by
`index % 16`:

| comparison | first-in-batch (63 nonces) | the other 937 |
|---|---|---|
| honest floor, same box | median 0.1453, 2 % past 0.40 | median **0.0000**, 0 % past |
| honest 2×B300 vs honest 4×H200 | median **1.07**, **100 % past 0.40** | median 0.25, 11 % past |

All three seeds give exactly 63 first-in-batch nonces and exactly 63 flagged — no near misses.
Every nonce whose index is a multiple of the collection batch size fails cross-hardware
comparison **unconditionally**, regardless of whether the node is honest.

The same split applied to DeepSeek-V4 (honest H100 vs honest B300, committed sets under
[`../../2026-07/`](../../2026-07/)) shows **nothing**: first-in-batch median 0.168–0.178 against
0.187–0.188 for the rest, at every candidate boundary (8, 16, 32, 64). The artifact belongs to
GLM-5.3, not to the PoC tooling.

**Working hypothesis, untested:** GLM-5.3-Flash is a hybrid — 34 KDA linear-attention layers
carrying Mamba state, 11 sparse-MLA layers. DeepSeek-V4 is pure MLA with no carried state. The
one structural difference between the two models is exactly what would make the first sequence
of a batch depend on what ran before it. Nobody has yet checked whether that state is reset
before each PoC batch.

### Throughput

PoC, `run_pow_generation.py --phase 3`, 5 s warmup + **120 s** measurement, launched through
mlnode (the production path):

| batch | tokens/pass | nonces/min |
|---:|---:|---:|
| 8 | 8 192 | **1439** |
| 16 | 16 384 | **1439** |
| 24 | 24 576 | XID 31, engine dead |
| 32 | 32 768 | not reached |

Batch 8 and 16 are identical, so PoC is compute-bound here, not launch-bound: a larger batch
buys nothing even where it survives.

Batch 24 raises `CUDA_ERROR_ILLEGAL_ADDRESS` on all four GPUs from
`deepgemm-src/csrc/.../jit_kernels/impls/runtime_utils.hpp:145`, called out of
`gonka_poc/poc/poc_model_runner.py`.

**This is a configuration limit, not a hardware ceiling.** The PoC forward builds
batch × 1024 tokens, and this run ran with `--max-num-batched-tokens 16384`; batch 24 needs
24576. The same overrun on B200 surfaces as
`RuntimeError: The size of tensor a (16384) must match the size of tensor b (49152)` and, with
the budget raised to 65536, batch 32 runs at 2727 nonces/min. The real kernel limit is higher —
batch 48 dies in the Triton sparse-MLA indexer. Hopper was not re-tested with a raised budget,
so the DeepGEMM crash above is attributed, not proven; treat batch 16 as the verified figure for
this configuration.

Per-seed collection runs (1000 nonces, ~42 s windows) agree with the 120 s sweep to within 1 %:
1414 / 1422 / 1423.

**Discard the first run after engine start.** It is reproducibly ~11 % low (1265 vs 1414 on the
same seed); three subsequent runs agree within 1 %.

### Serving

Single pass per concurrency level, 800 max tokens, one prompt:

| concurrency | tok/s | median latency |
|---:|---:|---:|
| 1 | 125.8 | 6.4 s |
| 8 | 619.5 | 10.3 s |
| 20 | 1231.2 | 13.0 s |

Zero failed requests.

### Integrity checks

- 4000 nonces across 4 sets: **100 % non-empty, 100 % unique** (`artifacts/summary.json`).
- Each seed's `block_hash` matches the fixed set in `scripts/poc_seeds.json`.
- Control: two different seeds give median **1.4116** — the expected ceiling for uncorrelated
  12-dim vectors, confirming the collector reads the seeds it is given. A silent empty-seed
  failure would produce this value everywhere.
- `grep -c "illegal memory"` = **0** across all collection runs (the XID 31 above is the batch
  sweep, a separate engine instance).
- L2 arithmetic follows the chain (`vllm/poc/data.py`): fp16 LE → fp32, fp64 norm, strict `>`.

## What this does not settle

**~~Architecture and image are confounded in the cross-generation section.~~ Settled.**
A 4×B200 run on this same image, same TP=4 and same seeds gives 16.4 / 17.0 / 16.2 % — matching
the confounded B300 pairing within noise. The build contributes nothing measurable; the 17 % is
the GPU generation. See [`../glm53-flash-fp8-4xb200/`](../glm53-flash-fp8-4xb200/).

**The batch-boundary artifact has no root cause yet.** The hybrid-state hypothesis is untested.

**8×H100 is unmeasured, and it is the only topology that still matters.** H200 carries 141 GB
per card and the weights fit at TP=4 with room to spare, so this run is a full production
topology rather than a reduced one; the image's baked `TP=8` targets 80 GB cards, where TP=4
would leave 76 GB of weights on an 80 GB card. No H100 arm of any width has been measured.

**Serving is a single pass** per concurrency level, not a compressa-perf run.

### Corrections to the first version of this report

Two claims in the first version of this report were wrong and are corrected above:

1. **"The tokens-per-pass ceiling lies between 16 384 and 24 576, and DeepGEMM is the failing
   component."** The binding constraint was `--max-num-batched-tokens 16384`, a flag this run
   set. Batch 32 works once the budget allows it (verified on B200).
2. **"Architecture and image are confounded."** Resolved by the B200 run: the gap is
   architectural.

## Files

| path | what |
|---|---|
| [`artifacts/summary.json`](artifacts/summary.json) | floor, control and the batch split, machine-readable |
| [`artifacts/cross_arch.json`](artifacts/cross_arch.json) | the 2×B300 ↔ 4×H200 comparison, per seed |
| `artifacts/nonces_honest_{s1,s2,s3}.json` | three seeds |
| `artifacts/nonces_honest_repeat_s1.json` | s1 repeated on the same engine — this pair is the floor |
| [`scripts/setup_h200.sh`](scripts/setup_h200.sh) | box prep, weights, shard verification |
| [`scripts/serve_h200_cell.sh`](scripts/serve_h200_cell.sh) | engine launch, parameterised by model and TP |
| [`scripts/cell_seeds.sh`](scripts/cell_seeds.sh) | three seeds + load test against a running engine |
| [`scripts/start_mlnode_api.sh`](scripts/start_mlnode_api.sh) | production path: mlnode API, then vLLM through it |
| [`scripts/setup_mlnode.sh`](scripts/setup_mlnode.sh) | mlnode deps; prints the image's own arguments before anything is patched |
| [`scripts/run_sweep_mlnode.sh`](scripts/run_sweep_mlnode.sh) | the 120 s batch sweep |
| [`scripts/collect_artifacts.py`](scripts/collect_artifacts.py), [`scripts/run_pow_generation.py`](scripts/run_pow_generation.py) | PoC tooling, committed **as patched** (see below) |
| [`scripts/summarize.py`](scripts/summarize.py), [`scripts/cross_arch.py`](scripts/cross_arch.py) | regenerate every table from the artifacts |
| [`scripts/poc_seeds.json`](scripts/poc_seeds.json) | the fixed seed set with its provenance |

Both PoC scripts are committed **as patched**, without which the run does not reproduce:

- `collect_artifacts.py` — routes are `/api/v1/pow/*` **without** the `inference` prefix for
  this image, and the collector's 600 s timeout is made configurable via `POC_COLLECT_TIMEOUT`.
- `run_pow_generation.py` — same route change (the URL is assembled from fragments, so the
  string to replace is `/inference/pow/`, not the whole path), and `MLNODE_URL` read from the
  environment so the sweep can target vLLM directly.

## Reproduce

```bash
bash scripts/setup_h200.sh                       # deps, route fixes, weights
TP=4 bash scripts/serve_h200_cell.sh &
BATCH=16 LABEL=honest bash scripts/cell_seeds.sh # three seeds + load test

# the 120 s sweep goes through mlnode, i.e. the production path
bash scripts/start_mlnode_api.sh
bash scripts/run_sweep_mlnode.sh

python3 scripts/summarize.py artifacts > artifacts/summary.json
python3 scripts/cross_arch.py            > artifacts/cross_arch.json
```

Success criteria: attention backend `FLASHINFER_MLA_SPARSE_SM90`; `/api/v1/pow/versions`
reports `poc_validation_inference: true`; 100 % non-empty and unique nonces; floor median
0.0000 with 63 differing nonces, all at `index % 16 == 0`.

## Gotchas

- **`ulimit -n 524288` before launching.** The default soft limit of 1024 is not enough for
  NCCL's P2P/IPC channels. On Blackwell this hangs the engine outright.
- **Kill the collector as soon as it prints `Nonces saved`.** Its post-processing phases
  (logprobs, 5-language probe) crash the engine and invalidate whatever runs next.
- **`POST /api/v1/pow/stop` before each collection**, otherwise a stale `GENERATING` session
  answers 409.
- **Guard against an empty seed.** An empty `block_hash` produces plausible-looking nonces whose
  L2 is ≈ 1.41 against everything — indistinguishable from "wrong model" unless checked.
- **After an illegal-memory error a process keeps ~130 GB on the GPU** and `pkill` by name does
  not find it. Kill by `nvidia-smi --query-compute-apps=pid`.
- **mlnode is split across four packages** (`api`, `common`, `pow`, `train`) and `api` imports
  `common`. Put the paths in a `site-packages/*.pth` rather than `PYTHONPATH` — the vLLM
  subprocess has to inherit them too.
- **`/api/v1/pow/*` on the mlnode port (8081) is the legacy standalone PoW service** and answers
  409 while inference is up. PoC-v2 lives inside vLLM, behind mlnode's proxy on port **5000**.
  Point the sweep there.
- **mlnode registers the model under its full snapshot path.** The sweep's default model name
  yields `400 expected string or buffer`; pass `MODEL=<snapshot path>`.

## Related

- fraud arm on this same box: [`../glm53-flash-reap50-patrickbdevaney-4xh200/README.md`](../glm53-flash-reap50-patrickbdevaney-4xh200/README.md)
- Blackwell counterpart, same image and TP: [`../glm53-flash-fp8-4xb200/README.md`](../glm53-flash-fp8-4xb200/README.md)
- honest 2×B300 on the previous image: [`../../2026-08/glm53-flash-fp8-2xb300/README.md`](../../2026-08/glm53-flash-fp8-2xb300/README.md)
- NVFP4 fraud arm on B300: [`../../2026-08/glm53-flash-nvfp4-libertai-2xb300/README.md`](../../2026-08/glm53-flash-nvfp4-libertai-2xb300/README.md)

## Reproducibility checklist

- [x] A reader with only this folder can reach the headline result by following the README
      top to bottom.
- [x] Hardware is stated exactly: GPU model, count, driver version, interconnect.
- [x] Image is pinned by tag + digest (`sha256:b92b8cc6…`).
- [x] Every command is copy-pasteable; the only placeholder is `<SNAPSHOT PATH>`.
- [x] Every script the steps invoke is committed under `scripts/`.
- [x] No links to `.claude/...` and no paths into sibling repos.
- [x] All artifacts referenced in the report exist in `artifacts/`.
- [x] Expected outputs / success criteria are stated.
- [x] Known gotchas and their fixes are listed.
- [ ] Engine logs and `env.txt` are **not** committed — the box was released before they were
      pulled. Versions and KV cache sizes are transcribed from the session, not from a
      committed log.
