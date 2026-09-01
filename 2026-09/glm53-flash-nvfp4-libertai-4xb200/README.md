# GLM-5.3-Flash — 4×B200 — NVFP4 fraud (L2 0.37, 42 % past 0.40 — half as loud as on B300; the fingerprint is not portable)

**Date:** 2026-09-01
**Model (fraud):** [`LibertAIDAI/GLM-5.3-Flash-NVFP4`](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4)
— NVFP4 via ModelOpt (`quant_algo: NVFP4`), 121 shards. The **same checkpoint** measured on
2×B300 in August, see *Related*.
**Model (reference):** `zai-org/GLM-5.3-Flash`, native FP8 — measured on this same box,
see [`../glm53-flash-fp8-4xb200/`](../glm53-flash-fp8-4xb200/).
**Hardware:** 4× NVIDIA B200 (183 359 MiB each, 1000 W, NV18 full mesh), TP=4,
driver 595.71.05, CUDA 13.
**Image:** `ghcr.io/kaitakuai/mlnode-b300-glm-5-3-flash:0.2.14-vllm0.28-glm53-test-k3`
**Digest:** `sha256:4a255793457229bceae1eb13643101be2c0375e9bf1b3e770d0a1aeea26c2f9b`

## Summary

Quantisation fraud is **profitable and detectable, but its fingerprint is not a portable
constant** — which is the actionable finding here.

| Arm (batch 16, TP=4) | nonces/min | Δ vs honest | median L2 vs honest | past 0.40 |
|---|---:|---:|---:|---:|
| honest FP8 | 2196–2217 | — | — | 0 % (floor) |
| **NVFP4 (LibertAI / ModelOpt)** | **2693–2725** | **+23 %** | **0.369–0.379** | **42–44 %** |

- **The August measurement does not transfer.** The identical checkpoint on 2×B300 with the
  previous image reported median **0.711 / 97.3 %** — "the loudest fraud we have measured".
  Here it is **0.37 / 42 %**, half as loud.
- **The two NVFP4 runs disagree with each other more than fraud disagrees with honest.**
  August-B300 against September-B200, same checkpoint, same seeds: median **0.736–0.751,
  97.8–98.4 % past 0.40** — larger than this arm's distance from the honest baseline (0.37).
  A published NVFP4 signature is therefore a property of the platform it was measured on, not
  of the attack.
- **Detection must be framed as "far from honest", not "close to a known fraud fingerprint".**
- **The margin over honest cross-generation noise is 2.5×, not 5×.** An honest node on a
  different GPU generation already sits at 17 % past the gate
  (see [`../glm53-flash-fp8-4xb200/`](../glm53-flash-fp8-4xb200/)); this fraud is at 42 %.
  Compare structural fraud (expert pruning) at 90 %.

## Environment

Identical to the honest arm — same box, same image, same engine flags, same seeds. Only the
checkpoint differs. Attention backend `FLASHINFER_MLA_SPARSE` in both arms.

| Parameter | honest FP8 | NVFP4 |
|---|---:|---:|
| KV cache | 13 076 757 tokens | **18 434 642 tokens** |
| engine start (kernels cached) | ~3 min | ~5 min |

## Config

```bash
ulimit -n 524288
export NCCL_MNNVL_ENABLE=0 NCCL_NVLS_ENABLE=0 NCCL_CUMEM_ENABLE=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

gonka-vllm-serve \
  --model <NVFP4 SNAPSHOT PATH> --served-model-name glm53 \
  --disable-custom-all-reduce \
  --tensor-parallel-size 4 \
  --kv-cache-dtype fp8 --block-size 2304 --max-num-seqs 256 \
  --max-num-batched-tokens 16384 \
  --no-enable-flashinfer-autotune --logprobs-mode processed_logprobs \
  --worker-extension-cls gonka_poc.worker.PoCWorkerExtension \
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
  --host 0.0.0.0 --port 8081
```

`scripts/cell_b200_k3.sh` selects this arm by pointing `--model` at
`models--LibertAIDAI--GLM-5.3-Flash-NVFP4`.

## Validation

### L2 against the honest arm

Gate defaults: `threshold = 0.40`, `p_mis = 0.001`.

| seed | median | mean | p25 | max | past 0.40 | verdict |
|---|---:|---:|---:|---:|---:|---|
| s1 | 0.3692 | 0.3894 | 0.2888 | 1.1034 | 417 / 1000 (41.7 %) | **FRAUD**, p ≈ 0 |
| s2 | 0.3791 | 0.3965 | 0.3045 | 1.0226 | 437 / 1000 (43.7 %) | **FRAUD**, p ≈ 0 |
| s3 | 0.3793 | 0.3972 | 0.3078 | 0.9628 | 442 / 1000 (44.2 %) | **FRAUD**, p ≈ 0 |

Note the median sits **below** the 0.40 gate: unlike expert pruning, where the lower quartile
was already past it, here fewer than half the nonces cross. The verdict comes from the
mismatch *rate* against a `p_mis` of 0.001, not from any single nonce.

### The fingerprint does not transfer between platforms

Same checkpoint, same seeds, different platform and image:

| pair | median L2 | past 0.40 |
|---|---:|---:|
| NVFP4 (Aug, 2×B300, FlashInfer 0.6.17) ↔ NVFP4 (Sep, 4×B200, 0.6.18), s1 | 0.7356 | 97.8 % |
| same, s2 | 0.7368 | 97.8 % |
| same, s3 | 0.7507 | 98.4 % |

Two runs of one quantised checkpoint disagree **twice as much** as this arm disagrees with the
honest baseline. The August figure of 0.711 describes "B300 + the old image", not NVFP4.

Contrast the honest side, where the same cross-platform move costs only 0.26 — quantised
numerics are far more sensitive to which kernels execute them than FP8 numerics are.

*Caveat:* architecture (B300 vs B200) and image (0.6.17 vs 0.6.18) both differ in that pairing,
so the split between them is unknown. For calibration it does not matter: the quantity is not
stable, and that is enough to disqualify it as a signature.

### The batch-boundary artifact does not drive this verdict

The honest arm has a defect at `index % 16 == 0`
(see [`../glm53-flash-fp8-4xb200/`](../glm53-flash-fp8-4xb200/)). Broken out:

| seed | first-in-batch (63) | the other 937 |
|---|---|---|
| s1 | median 0.4212, 65 % past | median 0.3655, 40.1 % past |
| s2 | median 0.4004, 51 % past | median 0.3766, 43.2 % past |
| s3 | median 0.4286, 52 % past | median 0.3770, 43.6 % past |

Unlike the cross-generation honest comparison, where those nonces are 100 % past the gate, here
they behave much like the rest. Removing them changes the rate by ~1 point.

### Throughput

Per-seed collection, batch 16, `--max-num-batched-tokens 16384`:

| seed | honest | NVFP4 |
|---|---:|---:|
| s1 | 2217 | 2725 |
| s2 | 2196 | 2702 |
| s3 | 2213 | 2693 |

**+23 %** — the attack pays. For comparison, on 2×B300 in August the same checkpoint bought
+36 %; on 4×H200 structural pruning bought nothing at all.

Sweep, 120 s per batch, this arm at `--max-num-batched-tokens 16384`:

| batch | tokens/pass | nonces/min |
|---:|---:|---:|
| 8 | 8 192 | 2596 |
| 16 | 16 384 | 3431 |
| 32 | 32 768 | 0 — exceeds the token budget |
| 48 | 49 152 | 0 — exceeds the token budget |

The zeros are a configuration limit, not a property of this arm: `--max-num-batched-tokens`
must be ≥ batch × 1024. The honest sweep was re-run at 65536 and batch 32 works there. **The two
arms' sweeps are therefore not directly comparable** — the budget differs. The per-seed
collection numbers above are the clean comparison; both arms ran at 16384.

### Serving

| concurrency | honest tok/s | NVFP4 tok/s | Δ |
|---:|---:|---:|---:|
| 1 | 147.6 | 141.4 | −4 % |
| 8 | 782.6 | 844.3 | +8 % |
| 20 | 1535.8 | 1734.3 | +13 % |

Quantisation is slightly *slower* single-stream and only pays under batching. Zero failed
requests.

### Integrity checks

- 6000 nonces across 6 sets: 100 % non-empty, 100 % unique (`artifacts/summary.json`).
- Each seed's `block_hash` matches `scripts/poc_seeds.json`.
- Control: two different seeds give median 1.4053 — the expected ceiling.
- `illegal memory` = 0 across this arm's collection runs.

## What this does not settle

- **One quantisation build only.** ModelOpt NVFP4 from LibertAI; other NVFP4 producers may sit
  elsewhere, as the Hy3 work already showed for a different model.
- **The B300↔B200 disagreement is not decomposed** into architecture versus image.
- **Sweeps across the two arms used different token budgets** (see above).

## Files

| path | what |
|---|---|
| [`artifacts/summary.json`](artifacts/summary.json) | every L2 table, machine-readable |
| `artifacts/nonces_nvfp4_{s1,s2,s3}.json` | fraud arm, three seeds |
| `artifacts/ref_nonces_honest_{s1,s2,s3}.json` | honest reference, byte-identical copies of the honest folder's sets |
| [`artifacts/sweep_both_arms.log`](artifacts/sweep_both_arms.log) | full sweep output, both arms |
| [`artifacts/cell.log`](artifacts/cell.log) | the whole run: hardware, versions, every seed |
| [`scripts/cell_b200_k3.sh`](scripts/cell_b200_k3.sh) | the run driver: both checkpoints, engine, seeds, load test |
| [`scripts/sweep_b200.sh`](scripts/sweep_b200.sh) | the 120 s sweep over both arms |
| [`scripts/collect_artifacts.py`](scripts/collect_artifacts.py), [`scripts/run_pow_generation.py`](scripts/run_pow_generation.py) | PoC tooling, committed as patched |
| [`scripts/summarize.py`](scripts/summarize.py) | regenerates every table from the artifacts |
| [`scripts/poc_seeds.json`](scripts/poc_seeds.json) | the fixed seed set |

## Reproduce

```bash
bash scripts/cell_b200_k3.sh        # both checkpoints, both arms, seeds, load tests
bash scripts/sweep_b200.sh          # 120 s sweep, both arms
python3 scripts/summarize.py artifacts > artifacts/summary.json
```

Success criteria: KV cache ≈ 18.4 M tokens on this arm against ≈ 13.1 M honest; median L2
0.37–0.38 with 42–44 % past 0.40 on all three seeds; +23 % nonces/min over honest.

## Gotchas

- **Give nonce files distinct basenames before comparing.** `compare_nonces.py` labels pairs by
  basename; two files with the same name are silently compared against themselves and report
  `L2 = 0.0000`. That trap produced a false "bit-identical" reading during this analysis before
  the `PAIR:` / `vs:` lines were checked.
- **`--max-num-batched-tokens` must be ≥ batch × 1024**, otherwise the node produces zero nonces
  while `/health` still answers 200.
- Six container settings are mandatory on Blackwell or the engine never starts; see the honest
  folder's *Gotchas*.

## Related

- honest arm, same box: [`../glm53-flash-fp8-4xb200/README.md`](../glm53-flash-fp8-4xb200/README.md)
- the same checkpoint on 2×B300, previous image: [`../../2026-08/glm53-flash-nvfp4-libertai-2xb300/README.md`](../../2026-08/glm53-flash-nvfp4-libertai-2xb300/README.md)
- structural fraud (expert pruning) on 4×H200: [`../glm53-flash-reap50-patrickbdevaney-4xh200/README.md`](../glm53-flash-reap50-patrickbdevaney-4xh200/README.md)

## Reproducibility checklist

- [x] A reader with only this folder can reach the headline result.
- [x] Hardware is stated exactly: GPU model, count, driver version, interconnect.
- [x] Image is pinned by tag + digest; both checkpoints named.
- [x] Every command is copy-pasteable; the only placeholder is `<NVFP4 SNAPSHOT PATH>`.
- [x] Every script the steps invoke is committed under `scripts/`.
- [x] No links to `.claude/...` and no paths into sibling repos.
- [x] All artifacts referenced exist in `artifacts/`, including the honest reference sets and
      the raw run log.
- [x] Expected outputs / success criteria are stated.
- [x] Known gotchas and their fixes are listed.
- [ ] A machine-readable `env.txt` is not committed; versions are transcribed from `cell.log`,
      which does record them.
