# GLM-5.3-Flash on 4×H200 — honest FP8 vs REAP50 expert pruning

**Date:** 2026-09-01
**Model:** `zai-org/GLM-5.3-Flash` @ `03eb5366286afd40d2221b1d9c63a6dd1ba4832e` (FP8 blocks, 288 experts, top-k 8)
**Hardware:** 4× NVIDIA H200 SXM (700 W, NV18 full mesh), TP=4, driver 595.71.05, CUDA 13
**Image:** `ghcr.io/kaitakuai/mlnode-h100-glm-5-3-flash:0.2.14-vllm0.28-glm53-test-k3`
**Digest:** `sha256:b92b8cc6fbccd59f60d283bc979510d6bd519009864c2e60e83cff8852be5f3a`
**vLLM:** `0.28.0.dev0+glm53.gonka.sampler1`, FlashInfer **0.6.18 stable** — no source patches
**PoC:** `gonka_poc` 0.1.4, seq_len 1024, k_dim 12, collection batch 16

## Summary

Two arms on one box, differing **only in weights**: honest `zai-org/GLM-5.3-Flash` (288 experts)
against `patrickbdevaney/GLM-5.3-Flash-REAP50-FP8` (144 experts, same top-k 8, same FP8 block
scheme, same kernels). Everything else — engine, flags, attention backend, seeds — is held fixed.

- **The production threshold 0.40 works on Hopper.** Honest floor puts **1 nonce in 1000** past
  it; the pruned build puts **90 %**. Prior guidance said 0.40 was inapplicable to Hopper; that
  came from the previous image (FlashInfer 0.6.17) and no longer holds.
- **Honest runs are bit-exact except for one nonce per batch.** Same box, same seed, two runs:
  937 of 1000 vectors identical to the bit, median L2 **0.0000**.
- **That exception is not noise — it is a defect.** The 63 differing nonces are exactly those
  with `nonce % 16 == 0`, and against a different GPU generation their median L2 is **1.07**
  (vs 0.25 for the other 937). They alone drive honest cross-hardware comparison from 11 % to
  17 % past threshold. **DeepSeek-V4 shows no such artifact**, so this is specific to GLM-5.3.
- **Expert pruning buys memory, not speed.** Identical PoC throughput (1427 vs 1439 nonces/min),
  identical single-stream serving, but **2.1× the KV cache** and half the weights per GPU.
- **PoC throughput saturates at batch 8.** Batch 8 and 16 both give 1439 nonces/min; batch 24
  kills the engine with XID 31 out of **DeepGEMM** — not the FlashInfer MLA kernel, as
  previously assumed.

## Environment

| Parameter | Value |
|---|---|
| GPU | 4× NVIDIA H200 SXM, 143 771 MiB each, 700 W, NV18 full mesh |
| NVIDIA driver | 595.71.05 (CUDA 13 native; no `compat` layer needed) |
| CUDA | 13 |
| Python | 3.12 |
| vLLM | `0.28.0.dev0+glm53.gonka.sampler1` |
| FlashInfer | 0.6.18 (stable, released 2026-08-29) |
| gonka_poc | 0.1.4 |
| Attention backend | `FLASHINFER_MLA_SPARSE_SM90` |
| KV cache | 6 230 570 tokens (direct launch) / 6 090 760 (via mlnode) |
| Weights per GPU | 75.95 GiB |

## Config

```bash
# both arms, identical except --model
ulimit -n 524288                     # Vast ships a soft limit of 1024; NCCL needs more

gonka-vllm-serve \
  --model <SNAPSHOT PATH> --served-model-name glm53 \
  --tensor-parallel-size 4 \
  --kv-cache-dtype fp8 \
  --block-size 2304 \
  --max-num-seqs 256 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --no-enable-flashinfer-autotune \
  --logprobs-mode processed_logprobs \
  --worker-extension-cls gonka_poc.worker.PoCWorkerExtension \
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
  --host 0.0.0.0 --port 8081
```

`--model` must be a **local snapshot path**, not an HF id: `Glm5NextProcessor` looks for
`processor_config.json` as a file and fails on `zai-org/GLM-5.3-Flash`.

### What changed vs the default

The image's own `runner.py` ships this exact configuration, verified by reading it on the box
(`scripts/setup_mlnode.sh` prints it before anything is patched):

| Parameter | Image as shipped | This run | Why |
|---|---|---|---|
| `--tensor-parallel-size` | **8** | 4 | the box has 4 GPUs |
| `--limit-mm-per-prompt` | absent | `{"image":0,"video":0}` | vision tower is unused here; on B300 it also kills the worker during memory profiling |
| everything else | `--kv-cache-dtype fp8`, `--block-size 2304`, `--max-num-seqs 256`, `--logprobs-mode processed_logprobs`, `--worker-extension-cls`, `--no-enable-flashinfer-autotune` | unchanged | — |

Launching through mlnode (`scripts/start_mlnode_api.sh`) and reading the engine's own config
line confirms the chain does **not** override these: `tensor_parallel_size=4`,
`kv_cache_dtype=fp8`, `quantization=fp8`, `enforce_eager=False`. The only deviation of this run
from a production node is the GPU count.

## Validation

### Honest floor — same box, same seed, two consecutive runs

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

### Fraud arm — REAP50 (144 of 288 experts) vs honest, three seeds

| seed | median L2 | mean | p25 | past 0.40 | verdict |
|---|---:|---:|---:|---:|---|
| s1 | 0.6003 | 0.6530 | 0.4881 | 913 / 1000 (91.3 %) | **FRAUD**, p ≈ 0 |
| s2 | 0.6047 | 0.6532 | 0.4904 | 900 / 1000 (90.0 %) | **FRAUD**, p ≈ 0 |
| s3 | 0.6247 | 0.6603 | 0.4977 | 904 / 1000 (90.4 %) | **FRAUD**, p ≈ 0 |

Separation is complete: the fraud arm's **lower quartile** (0.488) is already above the
threshold, while the honest floor puts 0.1 % past it. There is no overlap to argue about.

Gate thresholds quoted from the chain default: `threshold = 0.40`, `p_mis = 0.001`.

### The batch-boundary artifact

Nonces are indexed 0…999 and collected in batches of 16. Splitting every comparison by
`nonce % 16`:

| comparison | first-in-batch (63 nonces) | the other 937 |
|---|---|---|
| honest floor, same box | median 0.1453, 2 % past 0.40 | median **0.0000**, 0 % past |
| honest B300 vs honest H200 | median **1.07**, **100 % past 0.40** | median 0.25, 11 % past |
| fraud REAP50 vs honest | median 1.28–1.31, 100 % past | median 0.58–0.60, 90 % past |

All three seeds give exactly 63 first-in-batch nonces and exactly 63 flagged — no near misses.

Every nonce whose index is a multiple of the batch size therefore fails cross-hardware
comparison **unconditionally**, regardless of whether the node is honest. These 63 nonces are
6.3 % of the set and they raise honest cross-hardware mismatch from 11 % to 17 %.

The same split applied to DeepSeek-V4 (honest H100 vs honest B300, committed sets from
`../../2026-07/`) shows **nothing**: first-in-batch median 0.168–0.178 against 0.187–0.188 for
the rest, at every candidate batch boundary (8, 16, 32, 64). The artifact is a property of
GLM-5.3, not of the PoC tooling.

**Working hypothesis:** GLM-5.3-Flash is a hybrid — 34 KDA linear-attention layers with Mamba
state, 11 sparse-MLA layers. DeepSeek-V4 is pure MLA with no carried state. The one structural
difference between the two models is exactly the thing that would make the first sequence of a
batch depend on what ran before it. This is untested; see below.

### Cross-hardware floor (honest vs honest, different GPU generations)

| seed | median L2 | p95 | past 0.40 | past 0.40, excluding first-in-batch |
|---|---:|---:|---:|---:|
| s1 | 0.2559 | 0.8971 | 16.8 % | 11.2 % |
| s2 | 0.2674 | 0.8606 | 17.0 % | 11.4 % |
| s3 | 0.2658 | 0.8916 | 16.3 % | 10.7 % |

Both sides are honest, so **at `p_mis = 0.001` the chain would call a healthy mixed fleet
fraudulent**. Distinguishing power is intact — 17 % honest vs 90 % fraud is a 5× gap — but the
mismatch tolerance has to be calibrated to roughly 0.20 for cross-generation validation, not
0.001.

For scale, three levels measured here:

| comparison | past 0.40 |
|---|---:|
| honest vs itself, same box | 0.1 % |
| honest B300 vs honest H200 | 17 % |
| fraud REAP50 vs honest, same box | 90 % |

### Throughput

PoC, `run_pow_generation.py --phase 3`, 5 s warmup + **120 s** measurement, launched through
mlnode (production path):

| batch | tokens/pass | nonces/min |
|---:|---:|---:|
| 8 | 8 192 | **1439** |
| 16 | 16 384 | **1439** |
| 24 | 24 576 | XID 31, engine dead |
| 32 | 32 768 | not reached |

Batch 8 and 16 are identical, so PoC is compute-bound here, not launch-bound — a larger batch
buys nothing even where it survives.

Batch 24 raises `CUDA_ERROR_ILLEGAL_ADDRESS` on all four GPUs from
`deepgemm-src/csrc/.../jit_kernels/impls/runtime_utils.hpp:145`, called out of
`gonka_poc/poc/poc_model_runner.py`. The failing component is **DeepGEMM**, not the FlashInfer
MLA kernel; the tokens-per-pass ceiling lies between 16 384 and 24 576.

Per-seed collection runs (1000 nonces each, 42 s windows) agree with the 120 s sweep to within
1 %: 1414 / 1422 / 1423 honest, 1427 / 1425 / 1423 fraud.

**Discard the first run after engine start.** It is reproducibly ~11 % low (1265 vs 1414 on the
same seed); three subsequent runs agree within 1 %. `scripts/floor_then_fraud.sh` therefore
collects a throwaway seed before each series.

### Serving

Both arms, same box, 800 max tokens, one prompt:

| concurrency | honest tok/s | REAP50 tok/s | Δ |
|---:|---:|---:|---:|
| 1 | 125.8 | 124.6 | −1 % |
| 8 | 619.5 | 691.4 | +12 % |
| 20 | 1231.2 | 1279.0 | +4 % |

| | honest | REAP50 |
|---|---:|---:|
| KV cache | 6 230 570 tokens | 13 042 932 tokens |
| weights per GPU | 75.95 GiB | ~38 GiB |

Pruning half the experts does not speed up computation because `top_k = 8` activates eight
experts per token regardless of pool size. The attacker's gain is **density** — the same
network weight from half the fleet — which is also why the fraud is easy to catch: the model
computes a different function rather than the same one less precisely.

### Integrity checks

- 7000 nonces across 7 sets: **100 % non-empty, 100 % unique** (`artifacts/summary.json`).
- Each seed's `block_hash` matches the fixed set in `scripts/poc_seeds.json`.
- Control: two different seeds give median **1.4116** — the expected ceiling for uncorrelated
  12-dim vectors. The instrument is therefore reading the seeds it is given; a silent
  empty-seed failure would produce this value everywhere.
- `grep -c "illegal memory"` = **0** across all collection runs (the XID 31 above is the batch
  sweep, a separate engine instance).
- L2 arithmetic follows the chain (`vllm/poc/data.py`): fp16 LE → fp32, fp64 norm, strict `>`.

## What this does not settle

**Architecture and image are confounded in the cross-hardware section.** The B300 reference was
collected on the previous image (FlashInfer 0.6.17), this run on `test-k3` (0.6.18). The 17 %
therefore mixes "different silicon" with "different build", and the split is unknown. Given
that changing FlashInfer turned "0 % bit-identical on Hopper" into 94 %, the build share could
be large. Settling it needs a B300 run on `test-k3` at both TP=4 (isolating architecture) and
TP=2 (isolating the build); it was attempted and the box was released before the engine came
up.

**The batch-boundary artifact has no root cause yet.** The hybrid-state hypothesis above is
untested — nobody has yet checked whether the KDA/Mamba state is reset before each PoC batch.

**8×H200 is unmeasured**, and that is the topology the image is actually built for (`TP=8`).

**Serving numbers are a single pass** per concurrency level, not a compressa-perf run.

## Files

| path | what |
|---|---|
| [`artifacts/summary.json`](artifacts/summary.json) | every table above, machine-readable |
| [`artifacts/cross_arch.json`](artifacts/cross_arch.json) | the B300↔H200 comparison, per seed |
| `artifacts/nonces_honest_{s1,s2,s3}.json` | honest arm, three seeds |
| `artifacts/nonces_honest_repeat_s1.json` | honest arm, s1 repeated — this is the floor |
| `artifacts/nonces_fraud_reap50_{s1,s2,s3}.json` | expert-pruned arm, three seeds |
| [`scripts/setup_h200.sh`](scripts/setup_h200.sh) | box prep, both weight sets, shard verification |
| [`scripts/serve_h200_cell.sh`](scripts/serve_h200_cell.sh) | engine launch, parameterised by model and TP |
| [`scripts/cell_seeds.sh`](scripts/cell_seeds.sh) | three seeds + load test against a running engine |
| [`scripts/floor_then_fraud.sh`](scripts/floor_then_fraud.sh) | floor measurement, then switch weights and repeat |
| [`scripts/start_mlnode_api.sh`](scripts/start_mlnode_api.sh) | production path: mlnode API, then vLLM through it |
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
bash scripts/setup_h200.sh                       # deps, route fixes, both weight sets
MODEL_REPO=models--zai-org--GLM-5.3-Flash TP=4 bash scripts/serve_h200_cell.sh &
BATCH=16 LABEL=honest bash scripts/cell_seeds.sh # three seeds + load test
bash scripts/floor_then_fraud.sh                 # floor, then REAP50 arm

# the 120 s sweep goes through mlnode, i.e. the production path
bash scripts/start_mlnode_api.sh
bash scripts/run_sweep_mlnode.sh

python3 scripts/summarize.py artifacts > artifacts/summary.json
python3 scripts/cross_arch.py            > artifacts/cross_arch.json
```

Success criteria: attention backend `FLASHINFER_MLA_SPARSE_SM90`; `/api/v1/pow/versions`
reports `poc_validation_inference: true`; 100 % non-empty and unique nonces; honest floor
median 0.0000 with 63 differing nonces, all at `nonce % 16 == 0`; fraud arm ≈ 90 % past 0.40.

## Gotchas

- **`ulimit -n 524288` before launching.** The default soft limit of 1024 is not enough for
  NCCL's P2P/IPC channels. On Blackwell this hangs the engine outright; see
  the B300 notes in *Related*.
- **Kill the collector as soon as it prints `Nonces saved`.** Its post-processing phases
  (logprobs, 5-language probe) crash the engine and invalidate whatever runs next.
- **`POST /api/v1/pow/stop` before each collection**, otherwise a stale `GENERATING` session
  answers 409.
- **Guard against an empty seed.** An empty `block_hash` produces plausible-looking nonces whose
  L2 is ≈ 1.41 against everything — indistinguishable from "wrong model" unless checked.
- **After an illegal-memory error a process keeps ~130 GB on the GPU** and `pkill` by name does
  not find it. Kill by `nvidia-smi --query-compute-apps=pid`.
- **`--limit-mm-per-prompt` is not optional on Blackwell.** The vision tower kills the worker
  during memory profiling with no Python traceback.
- **mlnode is split across four packages** (`api`, `common`, `pow`, `train`) and `api` imports
  `common`. Put the paths in a `site-packages/*.pth` rather than `PYTHONPATH` — the vLLM
  subprocess has to inherit them too.
- **`/api/v1/pow/*` on the mlnode port (8081) is the legacy standalone PoW service** and
  answers 409 while inference is up. PoC-v2 lives inside vLLM, behind mlnode's proxy on port
  **5000**. Point the sweep there.
- **mlnode registers the model under its full snapshot path.** The sweep's default model name
  yields `400 expected string or buffer`; pass `MODEL=<snapshot path>`.

## Related

- honest 2×B300, previous image: [`../../2026-08/glm53-flash-fp8-2xb300/README.md`](../../2026-08/glm53-flash-fp8-2xb300/README.md)
- NVFP4 fraud arm on B300: [`../../2026-08/glm53-flash-nvfp4-libertai-2xb300/README.md`](../../2026-08/glm53-flash-nvfp4-libertai-2xb300/README.md)
- DeepSeek-V4 cross-hardware sets used for the artifact control: [`../../2026-07/`](../../2026-07/)
- fraud build: `https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8`

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
      pulled. Versions and KV cache sizes in this report are transcribed from the session, not
      from a committed log.
