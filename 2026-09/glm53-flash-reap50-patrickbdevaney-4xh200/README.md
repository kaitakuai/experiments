# GLM-5.3-Flash — 4×H200 — REAP50 expert pruning (L2 0.60, 90 % past 0.40; buys memory, not speed)

**Date:** 2026-09-01
**Model (fraud):** [`patrickbdevaney/GLM-5.3-Flash-REAP50-FP8`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8)
— half the experts removed: `n_routed_experts` **144** vs 288, `num_experts_per_tok` **8** in
both. Same FP8 block scheme (`weight_block_size [128,128]`), same 45 layers, same
`index_kpool 4`, same `qk_rope_head_dim 0`, 157 GB, 62 shards.
**Model (reference):** `zai-org/GLM-5.3-Flash`, native FP8 — measured on this same box,
see [`../glm53-flash-fp8-4xh200/`](../glm53-flash-fp8-4xh200/).
**Hardware:** 4× NVIDIA H200 SXM (143 771 MiB each, 700 W, NV18 full mesh), TP=4,
driver 595.71.05, CUDA 13.
**Image:** `ghcr.io/kaitakuai/mlnode-h100-glm-5-3-flash:0.2.14-vllm0.28-glm53-test-k3`
**Digest:** `sha256:b92b8cc6fbccd59f60d283bc979510d6bd519009864c2e60e83cff8852be5f3a`

## Summary

Structural fraud — deleting half the expert pool — is **loud and unprofitable in the way the
attacker wants**. Median L2 **0.60**, with 90 % of nonces past the 0.40 threshold on all three
seeds, against an honest floor of 0.1 % measured on the same box.

The attack buys **no PoC throughput at all** (1427 vs 1439 nonces/min) because `top_k = 8`
activates eight experts per token regardless of how many are in the pool. What it does buy is
**density**: half the weights per GPU and 2.1× the KV cache, i.e. the same network weight from
half the fleet.

| Arm (batch 16, TP=4) | nonces/min | Δ vs honest | median L2 vs honest | past 0.40 |
|---|---:|---:|---:|---:|
| honest FP8 (288 experts) | 1439 | — | — | 0.1 % (floor) |
| **REAP50 (144 experts)** | 1427 | **−1 %** | **0.60** | **90 %** |

This is a different fraud class from quantisation: the model computes a *different function*
rather than the same one less precisely, which is why the signal is so clean.

## Environment

Identical to the honest arm — same box, same image, same engine flags, same seeds. Only the
checkpoint differs. Attention backend `FLASHINFER_MLA_SPARSE_SM90` in both arms.

| Parameter | honest | REAP50 |
|---|---:|---:|
| weights per GPU | 75.95 GiB | ~38 GiB |
| KV cache | 6 230 570 tokens | **13 042 932 tokens** |
| engine start (weights cached) | ~12 min | ~4 min |

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

`--model` must be a local snapshot path, not an HF id. `scripts/serve_h200_cell.sh` takes
`MODEL_REPO=models--patrickbdevaney--GLM-5.3-Flash-REAP50-FP8` to select this arm.

### What changed vs the default

| Parameter | Image as shipped | This run |
|---|---|---|
| `--tensor-parallel-size` | 8 | 4 (the box has 4 GPUs) |
| checkpoint | `zai-org/GLM-5.3-Flash` | `patrickbdevaney/GLM-5.3-Flash-REAP50-FP8` |
| everything else | — | unchanged |

## Validation

### L2 against the honest arm

Gate thresholds quoted from the chain default: `threshold = 0.40`, `p_mis = 0.001`.

| seed | median | mean | p25 | p95 | max | past 0.40 | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| s1 | 0.6003 | 0.6530 | 0.4881 | 1.2185 | 1.8186 | 913 / 1000 (91.3 %) | **FRAUD**, p ≈ 0 |
| s2 | 0.6047 | 0.6532 | 0.4908 | 1.1960 | 1.6915 | 900 / 1000 (90.0 %) | **FRAUD**, p ≈ 0 |
| s3 | 0.6247 | 0.6603 | 0.4977 | 1.1248 | 1.6286 | 904 / 1000 (90.4 %) | **FRAUD**, p ≈ 0 |

The **lower quartile** already sits above the threshold. For scale, on this same box the honest
floor is median 0.0000 with 1 nonce past 0.40, and an unrelated seed pair saturates at 1.41.
One nonce settles a verdict; the three-seed agreement (0.600 / 0.605 / 0.625) shows the
fingerprint is a property of the build, not of a seed.

### Excluding the batch-boundary artifact

The honest arm has a defect — nonces at `index % 16 == 0` are unreliable, see
[`../glm53-flash-fp8-4xh200/`](../glm53-flash-fp8-4xh200/). They are broken out here so the
fraud signal cannot be accused of resting on them:

| seed | first-in-batch (63) | the other 937 |
|---|---|---|
| s1 | median 1.3103, 100 % past | median 0.5820, **90.7 % past** |
| s2 | median 1.2801, 100 % past | median 0.5904, **89.3 % past** |
| s3 | median 1.2749, 100 % past | median 0.6017, **89.8 % past** |

Removing the artifact changes the verdict by less than two points. The detection does not
depend on it.

### Serving

Same box, same instrument, 800 max tokens, one prompt, single pass per concurrency level:

| concurrency | honest tok/s | REAP50 tok/s | Δ |
|---:|---:|---:|---:|
| 1 | 125.8 | 124.6 | −1 % |
| 8 | 619.5 | 691.4 | **+12 %** |
| 20 | 1231.2 | 1279.0 | +4 % |

The gain appears only under batching, where the doubled KV cache and halved weights matter, and
it is modest and inconsistent. Zero failed requests in all six runs.

### PoC throughput

| arm | s1 | s2 | s3 |
|---|---:|---:|---:|
| honest | 1414 | 1422 | 1423 |
| REAP50 | 1427 | 1425 | 1423 |

Within 1 % — pruning the expert pool does not change PoC cost, because `top_k` is unchanged.
The first run after an engine start is reproducibly ~11 % low and is discarded;
`scripts/floor_then_fraud.sh` collects a throwaway seed before each series.

### Integrity checks

- 6000 nonces across 6 sets: **100 % non-empty, 100 % unique** (`artifacts/summary.json`).
- Each seed's `block_hash` matches the fixed set in `scripts/poc_seeds.json`.
- Control: two different seeds give median **1.4116** — the expected ceiling for uncorrelated
  12-dim vectors, confirming the collector reads the seeds it is given.
- `grep -c "illegal memory"` = **0** across all runs in this folder.
- L2 arithmetic follows the chain (`vllm/poc/data.py`): fp16 LE → fp32, fp64 norm, strict `>`.

## What this does not settle

- **Only one pruned build was measured.** The fingerprint is calibrated for REAP50 at 50 %
  pruning; a 25 % or 75 % variant is unmeasured, and the signal presumably scales with the
  fraction removed.
- **Serving is a single pass per concurrency level**, not a compressa-perf run.
- **Cross-hardware detection is not covered here.** Both arms ran on the same box; the
  cross-generation floor is in the honest folder and is the constraint that actually limits
  the threshold in production.

For contrast, quantisation fraud on the same model measures **0.37 / 42 %** on 4×B200
([`../glm53-flash-nvfp4-libertai-4xb200/`](../glm53-flash-nvfp4-libertai-4xb200/)) — half the
distance of expert pruning, and with a fingerprint that does not transfer between platforms.
Structural fraud is the loud, stable case; quantisation is the awkward one.

## Files

| path | what |
|---|---|
| [`artifacts/summary.json`](artifacts/summary.json) | every table above, machine-readable |
| `artifacts/nonces_reap50_{s1,s2,s3}.json` | fraud arm, three seeds |
| `artifacts/ref_nonces_honest_{s1,s2,s3}.json` | honest reference, byte-identical copies of the honest folder's sets |
| [`scripts/setup_h200.sh`](scripts/setup_h200.sh) | box prep, downloads **both** checkpoints, verifies shards |
| [`scripts/serve_h200_cell.sh`](scripts/serve_h200_cell.sh) | engine launch, `MODEL_REPO` selects the arm |
| [`scripts/floor_then_fraud.sh`](scripts/floor_then_fraud.sh) | honest floor, then switch weights and collect the fraud arm |
| [`scripts/collect_artifacts.py`](scripts/collect_artifacts.py) | PoC collector, committed **as patched** (routes `/api/v1/pow/*` without the `inference` prefix; `POC_COLLECT_TIMEOUT`) |
| [`scripts/summarize.py`](scripts/summarize.py) | regenerates every table from the artifacts |
| [`scripts/poc_seeds.json`](scripts/poc_seeds.json) | the fixed seed set with its provenance |

## Reproduce

```bash
bash scripts/setup_h200.sh          # deps, route fixes, both checkpoints
bash scripts/floor_then_fraud.sh    # honest floor, then the REAP50 arm, three seeds each
python3 scripts/summarize.py artifacts > artifacts/summary.json
```

Success criteria: attention backend `FLASHINFER_MLA_SPARSE_SM90`; KV cache ≈ 13.0 M tokens on
the pruned arm against ≈ 6.2 M honest; 100 % non-empty and unique nonces; median L2 ≈ 0.60 with
≈ 90 % past 0.40 on all three seeds.

## Gotchas

- **Kill the collector as soon as it prints `Nonces saved`.** Its post-processing phases crash
  the engine and invalidate whatever runs next.
- **`POST /api/v1/pow/stop` before each collection**, otherwise a stale `GENERATING` session
  answers 409.
- **Guard against an empty seed.** An empty `block_hash` produces plausible nonces whose L2 is
  ≈ 1.41 against everything — indistinguishable from "wrong model" unless checked.
- **After an illegal-memory error a process keeps ~130 GB on the GPU** and `pkill` by name does
  not find it. Kill by `nvidia-smi --query-compute-apps=pid`.
- **Disk:** both checkpoints together are 463 GB. Vast will not grow a running instance's disk
  (the API returns `success: true` and nothing changes), so order ≥ 600 GB up front or the
  second arm needs its own rental.

## Related

- honest arm, same box: [`../glm53-flash-fp8-4xh200/README.md`](../glm53-flash-fp8-4xh200/README.md)
- quantisation fraud on the same model (NVFP4, 2×B300): [`../../2026-08/glm53-flash-nvfp4-libertai-2xb300/README.md`](../../2026-08/glm53-flash-nvfp4-libertai-2xb300/README.md)
- honest 2×B300 on the previous image: [`../../2026-08/glm53-flash-fp8-2xb300/README.md`](../../2026-08/glm53-flash-fp8-2xb300/README.md)
- quantisation fraud on 4×B200, same image: [`../glm53-flash-nvfp4-libertai-4xb200/README.md`](../glm53-flash-nvfp4-libertai-4xb200/README.md)

## Reproducibility checklist

- [x] A reader with only this folder can reach the headline result by following the README
      top to bottom.
- [x] Hardware is stated exactly: GPU model, count, driver version, interconnect.
- [x] Image is pinned by tag + digest (`sha256:b92b8cc6…`); both checkpoints named with their
      expert counts.
- [x] Every command is copy-pasteable; the only placeholder is `<SNAPSHOT PATH>`.
- [x] Every script the steps invoke is committed under `scripts/`.
- [x] No links to `.claude/...` and no paths into sibling repos.
- [x] All artifacts referenced in the report exist in `artifacts/`, including the honest
      reference sets.
- [x] Expected outputs / success criteria are stated.
- [x] Known gotchas and their fixes are listed.
- [ ] Engine logs and `env.txt` are **not** committed — the box was released before they were
      pulled. Versions and KV cache sizes are transcribed from the session, not from a
      committed log.
