# GLM-5.3-Flash — 8×H100 — honest FP8 at TP=8 (the batch artifact nearly vanishes: 1 nonce, not 63)

**Date:** 2026-09-02
**Model:** [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) @
`03eb5366286afd40d2221b1d9c63a6dd1ba4832e` — native FP8 (`weight_block_size [128,128]`),
288 routed experts, `num_experts_per_tok` 8, 45 layers, 306 GB, 62 shards.
**Hardware:** 8× NVIDIA H100 80GB HBM3 (81 559 MiB each, 700 W, NV18 full mesh), **TP=8**,
driver 595.71.05, CUDA 13.
**Image:** `ghcr.io/kaitakuai/mlnode-h100-glm-5-3-flash:0.2.14-vllm0.28-glm53-test-k3`
**Digest:** `sha256:b92b8cc6fbccd59f60d283bc979510d6bd519009864c2e60e83cff8852be5f3a`
**vLLM:** `0.28.0.dev0+glm53.gonka.sampler1`, FlashInfer 0.6.18 — no source patches
**PoC:** `gonka_poc` 0.1.4, seq_len 1024, k_dim 12, collection batch 16

## Summary

The third architecture in the campaign and the only arm at `TP=8` — the topology the image
ships, and the one that is actually required on 80 GB cards (at TP=4 the weights alone would
take 76 GB of an 80 GB card).

- **The batch-boundary artifact is not universal.** Two honest runs of the same seed on this box
  differ in **1 nonce out of 1000** — nonce 0, the very first of the run. On 4×H200 and 4×B200
  the same test differs in **63**, every `index % 16 == 0`. Whatever causes it is
  configuration-dependent, and this configuration almost avoids it.
- **Same architecture does not mean agreement.** H100 against H200 — both Hopper, both
  `FLASHINFER_MLA_SPARSE_SM90`, same image — still gives **15.1 %** past the 0.40 gate, barely
  below the 16.5 % measured across GPU *generations*. The mismatch is a property of differing
  hardware and topology, not of crossing an architecture boundary.
- **The batch ceiling here is memory, not kernels.** Batch 32 dies with
  `CUDA out of memory` — a third distinct mechanism, after the token-budget limit and the
  Triton indexer limit seen elsewhere.
- **This box needs `--gpu-memory-utilization 0.95` to start at all**: at the default the engine
  refuses, needing 5.98 GiB of KV cache with 5.14 GiB available.

## Environment

| Parameter | Value |
|---|---|
| GPU | 8× NVIDIA H100 80GB HBM3, 81 559 MiB each, 700 W, NV18 full mesh |
| NVIDIA driver | 595.71.05, CUDA 13 |
| Attention backend | `FLASHINFER_MLA_SPARSE_SM90` (same as 4×H200) |
| Weights per GPU | 39.37 GiB |
| KV cache | 2 482 757 tokens (4×H200: 6 230 570 · 4×B200: 13 076 757) |
| Max concurrency at 1M context | 2.37× (4×B200: 12.44×) |

Eight 80 GB cards give **less** KV headroom than four 141 GB cards: the weights are replicated
in per-GPU shards but the fixed overheads are not, and CUDA graphs cost 4.55 GiB per GPU.

## Config

```bash
ulimit -n 524288

gonka-vllm-serve \
  --model <SNAPSHOT PATH> --served-model-name glm53 \
  --disable-custom-all-reduce \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 --block-size 2304 --max-num-seqs 256 \
  --gpu-memory-utilization 0.95 \
  --max-num-batched-tokens 65536 \
  --no-enable-flashinfer-autotune --logprobs-mode processed_logprobs \
  --worker-extension-cls gonka_poc.worker.PoCWorkerExtension \
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
  --host 0.0.0.0 --port 8081
```

### What changed vs the default

| Parameter | Image as shipped | This run | Why |
|---|---|---|---|
| `--tensor-parallel-size` | 8 | **8 — unchanged** | this is the topology the image is built for |
| `--gpu-memory-utilization` | 0.92 | **0.95** | at 0.92 the engine refuses to start: 5.98 GiB of KV cache needed, 5.14 GiB available |
| `--max-num-batched-tokens` | 16384 | 65536 | must be ≥ batch × 1024 to sweep past batch 16 |

This is the only arm in the campaign whose TP matches the image default.

## Validation

### L2

Honest floor — same box, same seed, two consecutive runs:

| metric | value |
|---|---:|
| median L2 | **0.0000** |
| differing nonces | **1 / 1000** (nonce 0) |
| L2 of that one nonce | 0.1547 |
| past 0.40 | **0 / 1000 (0 %)** → PASS |

999 of 1000 vectors are identical to the bit. For contrast, the same test gives 63 differing
nonces on 4×H200 and on 4×B200.

### Cross-hardware L2

Gate defaults: `threshold = 0.40`, `p_mis = 0.001`.

**vs honest 4×H200 TP=4 (`k3`)** — varies: GPU model, TP

| seed | mean | median | p25 | p75 | p95 | max | past 0.40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| s1 | 0.3261 | 0.2517 | 0.1915 | 0.3397 | 1.0834 | 1.7286 | 153 (15.3 %) |
| s2 | 0.3318 | 0.2552 | 0.1964 | 0.3366 | 1.1292 | 1.8788 | 157 (15.7 %) |
| s3 | 0.3209 | 0.2522 | 0.1891 | 0.3324 | 1.1215 | 1.8320 | 144 (14.4 %) |
| **all three, 3000 nonces** | **0.3262** | **0.2524** | | | | | **454 (15.1 %)** |

**vs honest 4×B200 TP=4 (`k3`)** — varies: architecture, TP

| seed | mean | median | p25 | p75 | p95 | max | past 0.40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| s1 | 0.3384 | 0.2671 | 0.2033 | 0.3540 | 1.1380 | 1.7101 | 160 (16.0 %) |
| s2 | 0.3389 | 0.2635 | 0.2040 | 0.3470 | 1.1564 | 1.7882 | 162 (16.2 %) |
| s3 | 0.3375 | 0.2667 | 0.2058 | 0.3532 | 1.0929 | 1.7434 | 168 (16.8 %) |
| **all three, 3000 nonces** | **0.3383** | **0.2657** | | | | | **490 (16.3 %)** |

**vs honest 2×B300 TP=2 (August image)** — varies: architecture, image, TP

| seed | mean | median | p25 | p75 | p95 | max | past 0.40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| s1 | 0.3253 | 0.2618 | 0.2002 | 0.3458 | 0.9899 | 1.6767 | 162 (16.2 %) |
| s2 | 0.3345 | 0.2635 | 0.1997 | 0.3565 | 1.0600 | 1.6563 | 170 (17.0 %) |
| s3 | 0.3325 | 0.2681 | 0.2062 | 0.3486 | 1.0224 | 1.7221 | 164 (16.4 %) |
| **all three, 3000 nonces** | **0.3308** | **0.2647** | | | | | **496 (16.5 %)** |

**Same architecture buys almost nothing.** H100 ↔ H200 are both Hopper, both on
`FLASHINFER_MLA_SPARSE_SM90`, both on this image — 15.1 %, against 16.3–16.5 % for the
cross-generation pairs. Whatever drives the mismatch is not the architecture boundary; it is
differing hardware and topology in general. (TP also differs here, 8 vs 4, so the SKU and the
parallelism cannot be separated from this pair alone.)

### The batch-boundary artifact

This is where this arm differs from every other:

| arm | differing nonces between two honest runs | which |
|---|---:|---|
| 4×H200 TP=4 | 63 / 1000 | every `index % 16 == 0` |
| 4×B200 TP=4 | 63 / 1000 | every `index % 16 == 0` |
| **8×H100 TP=8** | **1 / 1000** | **nonce 0 only** |

In the cross-hardware pairs above, the 63 positions still read 100 % past the gate — but the
corruption comes from the *other* side. Checking which side moves: of the 63 first-in-batch
positions in the H100 ↔ H200 pair, **63 are unstable on the H200 side and 1 on the H100 side**.

That nonce 0 is the single survivor is itself informative: it is the only sequence in the run
with no predecessor, consistent with a buffer whose content is inherited rather than written.
Whether TP=8 or the H100 itself suppresses the rest is not separable from this run.

### Throughput

Per-seed collection, batch 16:

| run | nonces/min |
|---|---:|
| s4 (warm-up, discarded) | 1416 |
| s1 | 1610 |
| s2 | 1612 |
| s3 | 1612 |
| s1 repeat | 1613 |

The warm-up effect is confirmed again on Hopper: the first run after engine start is 12 % low,
then three runs agree within 0.2 %.

Sweep, 5 s warmup + 120 s measurement, `--max-num-batched-tokens 65536`:

| batch | tokens/pass | nonces/min |
|---:|---:|---:|
| 8 | 8 192 | 1438 |
| 16 | 16 384 | **1775** |
| 32 | 32 768 | 0 — `CUDA out of memory` |

**A third distinct batch limit.** Batch 32 needs 1024 MiB and only 548 MiB is free, because
`--gpu-memory-utilization 0.95` — required for the engine to start at all on 80 GB cards —
leaves nothing spare. Elsewhere the same batch fails for entirely different reasons: a token
budget below batch × 1024 (any arm at 16384) or a Triton illegal access in the sparse-MLA
indexer (4×B200 at batch 48). Three mechanisms, one symptom.

Eight H100 at TP=8 reach 1775 nonces/min against 1439 for four H200 — **+23 % for twice the
cards**.

### Serving

Measured after a discarded warm-up request:

| concurrency | tok/s | median latency |
|---:|---:|---:|
| 1 | 115.5 | 6.9 s |
| 8 | 662.5 | 9.6 s |
| 20 | 1309.3 | 12.2 s |

Zero failed requests.

### Integrity checks

- 4000 nonces across 4 sets: 100 % non-empty, 100 % unique (`artifacts/summary.json`).
- Each seed's `block_hash` matches `scripts/poc_seeds.json`.
- Control: two different seeds give median **1.4099** — the expected ceiling.
- `illegal memory` = **0** across every run in this folder, including the failed batch 32
  (it was an allocation failure, not a memory fault).

## What this does not settle

- **SKU and TP are confounded** in the H100 ↔ H200 pair (8 cards vs 4). An H100 run at a lower
  TP is impossible — the weights do not fit — so this particular pair cannot be decomposed.
- **Why the batch artifact nearly disappears here is unknown.** TP=8, the H100 itself, and the
  much smaller KV cache all change together.
- **No fraud arm on this hardware.** Only the honest baseline was collected.
- **`--gpu-memory-utilization 0.95` is a deviation** from the other arms, forced by the card
  size; it is also what makes batch 32 impossible here.

## Files

| path | what |
|---|---|
| [`artifacts/summary.json`](artifacts/summary.json) | floor, control, batch split |
| `artifacts/nonces_honest_{s1,s2,s3}.json` | three seeds |
| `artifacts/nonces_honest_repeat_s1.json` | s1 repeated — this pair is the floor |
| [`artifacts/seeds.log`](artifacts/seeds.log) | the collection run: backend, KV cache, every seed, load test |
| [`scripts/cell_h100_tp8.sh`](scripts/cell_h100_tp8.sh) | full driver: weights, engine, seeds, load test |
| [`scripts/seeds_only.sh`](scripts/seeds_only.sh) | collection against an already-running engine |
| [`scripts/collect_artifacts.py`](scripts/collect_artifacts.py), [`scripts/run_pow_generation.py`](scripts/run_pow_generation.py) | PoC tooling, committed as patched |
| [`scripts/summarize.py`](scripts/summarize.py), [`scripts/matrix.py`](scripts/matrix.py) | regenerate every table |
| [`scripts/poc_seeds.json`](scripts/poc_seeds.json) | the fixed seed set |

## Reproduce

```bash
bash scripts/cell_h100_tp8.sh          # weights, engine, seeds, load test
# if the engine is already up (the cell script does not survive an ssh disconnect):
BATCH=16 LABEL=honest bash scripts/seeds_only.sh
python3 scripts/summarize.py artifacts > artifacts/summary.json
```

Success criteria: backend `FLASHINFER_MLA_SPARSE_SM90`; KV cache ≈ 2.48 M tokens; floor median
0.0000 with **exactly one** differing nonce (nonce 0); ~1610 nonces/min at batch 16.

## Gotchas

- **`--gpu-memory-utilization 0.95` is mandatory on 80 GB cards.** At the shipped 0.92 the
  engine aborts with `To serve at least one request with the model's max seq len (1048576),
  5.98 GiB KV cache is needed, which is larger than the available (5.14 GiB)`.
- **Batch 32 then fails for lack of memory**, not for the reasons seen on other arms. Batch 16
  is the working value here.
- **A cell script launched in the background over ssh does not survive the disconnect**, while
  the engine it starts (its own `nohup`) does. This run lost 25 minutes to an engine sitting
  ready with nothing collecting from it; `seeds_only.sh` exists for exactly that situation.
- **Weight loading is slow on first touch** — 741 s here versus 74 s once the files are in page
  cache. Budget for it or expect a restart to look much faster than the first attempt.
- **Check `cuda_max_good ≥ 13.0` before renting.** A CUDA-13 image will not run on a 12.x host,
  and the image's `compat` layer does not save it: with driver 560 it fails with
  `Error 803: system has unsupported display driver / cuda driver combination`.

## Related

- 4×H200, same architecture and image, TP=4: [`../glm53-flash-fp8-4xh200/README.md`](../glm53-flash-fp8-4xh200/README.md)
- 4×B200: [`../glm53-flash-fp8-4xb200/README.md`](../glm53-flash-fp8-4xb200/README.md)
- 2×B300 (August data): [`../glm53-flash-fp8-2xb300/README.md`](../glm53-flash-fp8-2xb300/README.md)
- campaign summary: [`../glm53-flash-cross-hardware-summary/README.md`](../glm53-flash-cross-hardware-summary/README.md)

## Reproducibility checklist

- [x] A reader with only this folder can reach the headline result.
- [x] Hardware is stated exactly: GPU model, count, driver version, interconnect.
- [x] Image is pinned by tag + digest.
- [x] Every command is copy-pasteable; the only placeholder is `<SNAPSHOT PATH>`.
- [x] Every script the steps invoke is committed under `scripts/`.
- [x] No links to `.claude/...` and no paths into sibling repos.
- [x] All artifacts referenced exist in `artifacts/`, including the raw collection log.
- [x] Expected outputs / success criteria are stated.
- [x] Known gotchas and their fixes are listed.
- [x] Sections absent for a reason say so instead of being omitted.
- [ ] The engine log itself is not committed; backend, KV cache size and the failure messages
      quoted above are transcribed from it via `artifacts/seeds.log` and the session.
