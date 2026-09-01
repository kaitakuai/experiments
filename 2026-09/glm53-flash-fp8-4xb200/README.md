# GLM-5.3-Flash — 4×B200 — honest FP8 baseline (floor 0.0000; the Hopper↔Blackwell gap is architectural)

**Date:** 2026-09-01
**Model:** [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) @
`03eb5366286afd40d2221b1d9c63a6dd1ba4832e` — native FP8 (`weight_block_size [128,128]`),
288 routed experts, `num_experts_per_tok` 8, 45 layers, 306 GB, 62 shards.
**Hardware:** 4× NVIDIA B200 (183 359 MiB each, 1000 W, NV18 full mesh), TP=4,
driver 595.71.05, CUDA 13.
**Image:** `ghcr.io/kaitakuai/mlnode-b300-glm-5-3-flash:0.2.14-vllm0.28-glm53-test-k3`
**Digest:** `sha256:4a255793457229bceae1eb13643101be2c0375e9bf1b3e770d0a1aeea26c2f9b`
**vLLM:** `0.28.0.dev0+glm53.gonka.sampler1`, FlashInfer 0.6.18 — no source patches
**PoC:** `gonka_poc` 0.1.4, seq_len 1024, k_dim 12, collection batch 16

## Summary

The Blackwell counterpart of [`../glm53-flash-fp8-4xh200/`](../glm53-flash-fp8-4xh200/), run on
the **same image, same TP, same seeds** specifically so that the two can be differenced.

- **The 17 % cross-generation mismatch is architectural, not a build artifact.** B200 against
  H200 on identical software gives **16.4 / 17.0 / 16.2 %** past the 0.40 gate — the same as the
  earlier B300-on-0.6.17 against H200-on-0.6.18 comparison (16.8 / 17.0 / 16.3 %). The confound
  flagged as unresolved in the H200 report is now resolved: FlashInfer version is not the cause.
- **The honest floor on B200 is perfect: 0 of 1000 past the gate**, median L2 0.0000, max 0.2614.
- **The batch-boundary artifact reproduces exactly.** 63 of 1000 nonces differ between two
  honest runs, all at `index % 16 == 0`. Cross-generation they are **100 % past the gate**
  (median 1.23–1.28) against 0.25 and ~11 % for the other 937.
- **No warm-up effect on Blackwell.** Four honest runs: 2217 / 2196 / 2213 / 2215 nonces/min,
  spread under 1 %. On Hopper the first run after engine start is reproducibly ~11 % low.
- **The "batch ceiling" was our own flag.** `--max-num-batched-tokens` must be ≥ batch × 1024;
  at 16384 anything above batch 16 silently yields zero nonces. Raised to 65536, batch 32 runs
  fine. The real kernel limit is higher: batch 48 dies in the Triton sparse-MLA indexer.

## Environment

| Parameter | Value |
|---|---|
| GPU | 4× NVIDIA B200, 183 359 MiB each, 1000 W, NV18 full mesh |
| NVIDIA driver | 595.71.05, CUDA 13 |
| Attention backend | `FLASHINFER_MLA_SPARSE` (Hopper uses `..._SM90`) |
| Weights per GPU | 75.85 GiB |
| KV cache | 13 076 757 tokens (H200: 6 230 570) |
| Engine start, cold | ~23 min; with kernels cached ~3 min |

## Config

```bash
ulimit -n 524288
export NCCL_MNNVL_ENABLE=0 NCCL_NVLS_ENABLE=0 NCCL_CUMEM_ENABLE=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

gonka-vllm-serve \
  --model <SNAPSHOT PATH> --served-model-name glm53 \
  --disable-custom-all-reduce \
  --tensor-parallel-size 4 \
  --kv-cache-dtype fp8 --block-size 2304 --max-num-seqs 256 \
  --max-num-batched-tokens 65536 \
  --no-enable-flashinfer-autotune --logprobs-mode processed_logprobs \
  --worker-extension-cls gonka_poc.worker.PoCWorkerExtension \
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
  --host 0.0.0.0 --port 8081
```

### What changed vs the default

| Parameter | Image as shipped | This run | Why |
|---|---|---|---|
| `--tensor-parallel-size` | 8 | 4 | the box has 4 GPUs |
| `--max-num-batched-tokens` | 16384 | **65536** | must be ≥ batch × 1024; 16384 caps PoC at batch 16 |
| NCCL env + `--disable-custom-all-reduce` | absent | set | without them the engine never starts in a container — see *Gotchas* |

## Validation

### L2

| metric | value |
|---|---:|
| median L2 | **0.0000** |
| mean | 0.0088 |
| p95 / p99 | 0.0983 / 0.1786 |
| max | 0.2614 |
| differing nonces | 63 / 1000 |
| past 0.40 | **0 / 1000 (0 %)** → PASS |

Cleaner than Hopper, where one nonce crossed the gate.

### Cross-hardware L2

Against [`../glm53-flash-fp8-4xh200/`](../glm53-flash-fp8-4xh200/): same image, same FlashInfer
0.6.18, same TP=4, same seeds. Gate defaults: `threshold = 0.40`, `p_mis = 0.001`.

| seed | median L2 | p95 | past 0.40 | past 0.40, excluding first-in-batch |
|---|---:|---:|---:|---:|
| s1 | 0.2604 | 1.0710 | 16.4 % | 10.8 % |
| s2 | 0.2606 | 1.1176 | 17.0 % | 11.4 % |
| s3 | 0.2626 | 1.0879 | 16.2 % | 10.6 % |

Compare the earlier, confounded pairing (B300 on FlashInfer 0.6.17 against H200 on 0.6.18):
16.8 / 17.0 / 16.3 %. Identical within noise, so **the build contributes nothing measurable**
and the whole gap is the GPU generation.

Both sides are honest, so at `p_mis = 0.001` the chain would call a healthy mixed fleet
fraudulent. The mismatch tolerance needs to be ≈ 0.20 for cross-generation validation.

### The batch-boundary artifact

| comparison | first-in-batch (63 nonces) | the other 937 |
|---|---|---|
| honest floor, same box | 63 differ | 937 bit-identical |
| B200 vs H200, s1 | median 1.2828, **100 % past 0.40** | median 0.2525, 10.8 % past |
| B200 vs H200, s2 | median 1.2530, **100 % past** | median 0.2516, 11.4 % past |
| B200 vs H200, s3 | median 1.2275, **100 % past** | median 0.2549, 10.6 % past |

Identical structure to Hopper. These 63 nonces are 6.3 % of the set and account for ~6 points
of the 17 %.

### Throughput

Per-seed collection, batch 16, `--max-num-batched-tokens 16384`:

| run | nonces/min |
|---|---:|
| s1 | 2217 |
| s2 | 2196 |
| s3 | 2213 |
| s1 repeat | 2215 |

No warm-up effect, unlike Hopper. For scale, H200 on the same image gives 1439 — Blackwell is
~1.5× faster at this batch.

Sweep, `run_pow_generation.py --phase 3`, 5 s warmup + 120 s measurement,
`--max-num-batched-tokens 65536`:

| batch | tokens/pass | nonces/min |
|---:|---:|---:|
| 8 | 8 192 | 2487 |
| 16 | 16 384 | 2549 |
| 32 | 32 768 | **2727** |
| 48 | 49 152 | 48, then `Triton Error: illegal memory access` |

**Two distinct limits, and only the second is hardware.** With the shipped
`--max-num-batched-tokens 16384`, batches 32 and 48 produce **zero nonces** — the PoC forward
builds batch × 1024 tokens and the mismatch surfaces as
`RuntimeError: The size of tensor a (16384) must match the size of tensor b (49152)`. The
plugin catches it, deactivates the gate and resets the prefix cache, so the engine stays up and
healthy while the node earns nothing — a silent failure, not a crash. Raising the budget to
65536 makes batch 32 work. Batch 48 then fails for a real reason: an illegal memory access
inside the Triton sparse-MLA indexer.

### Serving

Honest arm, measured after a discarded warm-up request (the first request after a PoC run pays
for cold decode kernels — an un-warmed measurement read 13.1 tok/s and is not a performance
number):

| concurrency | tok/s | median latency |
|---:|---:|---:|
| 1 | 147.6 | 5.4 s |
| 8 | 782.6 | 8.2 s |
| 20 | 1535.8 | 10.4 s |

### Integrity checks

- 4000 nonces across 4 sets: 100 % non-empty, 100 % unique (`artifacts/summary.json`).
- Each seed's `block_hash` matches `scripts/poc_seeds.json`.
- Control: two different seeds give median 1.41 — the expected ceiling for uncorrelated 12-dim
  vectors.
- `illegal memory` = 0 for every collection run; the only occurrences are batch 48 of the sweep.

## What this does not settle

- **The batch-boundary artifact still has no root cause.** It reproduces on both architectures,
  so it is not hardware; the hybrid-state hypothesis (KDA/Mamba state not reset between PoC
  batches) is untested.
- **The Hopper batch-24 crash was not re-tested with a raised budget.** On Hopper, batch 24 with
  `--max-num-batched-tokens 16384` died with XID 31 out of DeepGEMM. 24 × 1024 = 24576 exceeds
  that budget, so the cause is almost certainly the same as here, but it manifested as a hard
  crash rather than a shape error and has not been re-run.
- **8-GPU topologies are not applicable here.** B200 carries 183 GB per card, so TP=4 leaves
  ample room for KV; the image's `TP=8` default targets 80 GB cards. The unmeasured case is
  **8×H100**, which is a different architecture, not a wider version of this one.

## Files

| path | what |
|---|---|
| [`artifacts/summary.json`](artifacts/summary.json) | floor, control, batch split |
| [`artifacts/cross_arch.json`](artifacts/cross_arch.json) | B200 ↔ H200, per seed |
| `artifacts/nonces_honest_{s1,s2,s3}.json` | three seeds |
| `artifacts/nonces_honest_repeat_s1.json` | s1 repeated — this pair is the floor |
| [`artifacts/sweep_both_arms.log`](artifacts/sweep_both_arms.log) | full sweep output, both arms |
| [`artifacts/cell.log`](artifacts/cell.log) | the whole run: hardware, versions, every seed |
| [`scripts/cell_b200_k3.sh`](scripts/cell_b200_k3.sh) | the run driver: weights, engine, seeds, load test |
| [`scripts/sweep_b200.sh`](scripts/sweep_b200.sh) | the 120 s sweep over both arms |
| [`scripts/collect_artifacts.py`](scripts/collect_artifacts.py), [`scripts/run_pow_generation.py`](scripts/run_pow_generation.py) | PoC tooling, committed as patched (routes `/api/v1/pow/*`, `MLNODE_URL` and `POC_COLLECT_TIMEOUT` from the environment) |
| [`scripts/summarize.py`](scripts/summarize.py), [`scripts/cross_arch.py`](scripts/cross_arch.py) | regenerate every table from the artifacts |
| [`scripts/poc_seeds.json`](scripts/poc_seeds.json) | the fixed seed set |

## Reproduce

```bash
bash scripts/cell_b200_k3.sh        # weights, engine, seeds, floor, load test
bash scripts/sweep_b200.sh          # 120 s sweep, both arms
python3 scripts/summarize.py artifacts > artifacts/summary.json
python3 scripts/cross_arch.py            > artifacts/cross_arch.json
```

Success criteria: backend `FLASHINFER_MLA_SPARSE`; KV cache ≈ 13.1 M tokens; floor median
0.0000 with exactly 63 differing nonces, all at `index % 16 == 0`; ~2200 nonces/min at batch 16.

## Gotchas

- **Six settings are mandatory in a container on Blackwell**, or the engine never starts —
  workers spin on one core each, GPUs and disk idle, log silent for hours:
  `ulimit -n 524288`, `NCCL_MNNVL_ENABLE=0`, `NCCL_NVLS_ENABLE=0`, `NCCL_CUMEM_ENABLE=0`,
  `VLLM_ALLREDUCE_USE_SYMM_MEM=0`, `--disable-custom-all-reduce`. Each was isolated against a
  specific hang: the fabric probe, multicast memory, a 268 GB symmetric window, and the IPC
  buffer layout of vLLM's own all-reduce.
- **Distinguish a hang from cold JIT** by CPU shape, not by log silence: JIT shows `ptxas`/`cicc`
  and growing file reads; a hang is exactly one core per worker and zero reads. Measure with tick
  deltas from `/proc/PID/stat` — `ps %CPU` reports a lifetime average and misleads here.
- **`--max-num-batched-tokens` must be ≥ batch × 1024.** Otherwise the node produces zero nonces
  while looking healthy.
- **After an illegal memory access the engine is poisoned**: `/health` still answers 200 but
  inference returns 500. Restart before measuring anything else.
- **Give nonce files distinct basenames before comparing.** `compare_nonces.py` labels pairs by
  basename; two files with the same name are compared against themselves and report a false
  `L2 = 0.0000`. Check the `PAIR:` / `vs:` lines in its output.
- **Kill the collector as soon as it prints `Nonces saved`** — its post-processing phases crash
  the engine.
- **`POST /api/v1/pow/stop` before each collection**, otherwise a stale session answers 409.

## Related

- Hopper counterpart, same image: [`../glm53-flash-fp8-4xh200/README.md`](../glm53-flash-fp8-4xh200/README.md)
- NVFP4 fraud arm on this same box: [`../glm53-flash-nvfp4-libertai-4xb200/README.md`](../glm53-flash-nvfp4-libertai-4xb200/README.md)
- honest 2×B300 on the previous image: [`../../2026-08/glm53-flash-fp8-2xb300/README.md`](../../2026-08/glm53-flash-fp8-2xb300/README.md)

## Reproducibility checklist

- [x] A reader with only this folder can reach the headline result.
- [x] Hardware is stated exactly: GPU model, count, driver version, interconnect.
- [x] Image is pinned by tag + digest (`sha256:4a255793…`); model pinned by snapshot SHA.
- [x] Every command is copy-pasteable; the only placeholder is `<SNAPSHOT PATH>`.
- [x] Every script the steps invoke is committed under `scripts/`.
- [x] No links to `.claude/...` and no paths into sibling repos.
- [x] All artifacts referenced in the report exist in `artifacts/`, including the raw run log.
- [x] Expected outputs / success criteria are stated.
- [x] Known gotchas and their fixes are listed.
- [ ] Engine logs beyond `cell.log` and a machine-readable `env.txt` are not committed; driver
      and library versions are transcribed from `cell.log`, which does record them.
