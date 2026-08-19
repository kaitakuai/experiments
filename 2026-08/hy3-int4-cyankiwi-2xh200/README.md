# Hy3 INT4 W4A16 by `cyankiwi` — 2×H200 — the attack turns unprofitable (−13 % against honest)

**Date:** 2026-08-19
**Model:** `cyankiwi/Hy3-AWQ-INT4` — 182 GB. **Not AWQ despite the name**: compressed-tensors
`pack-quantized`, INT4 **W4A16** asymmetric, `group_size 32`, activations BF16, MTP layer
unquantised. Kernels `MarlinLinearKernel` + `CompressedTensorsWNA16MarlinMoEMethod`.
Weights **166 GiB**, which is why two H200s are enough while honest FP8 (276 GiB) needs four.

**Hardware:** 2× NVIDIA **H200 SXM** (700 W, 143 GB, NV18, driver **590.48.01**, sm_90) — two
cards of a 4-card Vast.ai instance (48152962), the other two idle
(`CUDA_VISIBLE_DEVICES=0,1`, verified at 0 MiB).
**Honest reference:** [`../hy3-fp8-4xh200/`](../hy3-fp8-4xh200/) redo run, same host, same
session; its nonce sets and sweep are duplicated here as `ref_*`.
**Image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash-0731:3.0.16-overlay-k5`
**Digest:** `sha256:8ce8830b4182b3dfd642c5e00f548f50a830611e4bc64ec4fbd84fe91070e3aa`

> The DeepSeek-V4-Flash foundry image, reused only as a vLLM 0.25.1 + PoC-plugin runtime.
> Its `runner.py` hardcodes V4-specific flags that must be replaced — `scripts/patch_hy3.py`.

## Summary

This is the first configuration measured where **cheating does not pay**. The narrow-topology
trick that yields +48 % on B300 and +10 % on 8×H100 turns negative here:

| configuration on 4× H200 | cards | nonces/min | per card |
|---|---:|---:|---:|
| honest FP8, TP=4 | 4 | **1248** | **312** |
| this model, two instances at TP=2 | 4 | **1088 (−13 %)** | 272 |

The mechanism is the same everywhere — the fraud's gain is the honest arm's parallelism
penalty minus its own MARLIN dequantisation cost (~30 %). H200 carries 143 GB per card, so
honest FP8 fits at TP=4 and has paid only a modest parallelism penalty; the dequant cost then
outweighs it.

| host | honest topology | per card | fraud minimum | per card | outcome |
|---|---|---:|---|---:|---:|
| 2×B300 | TP=2 | 800 | TP=1 | 1183 | +48 % |
| 8×H100 | TP=8 | 168 | TP=4 | 184 | +10 % |
| **4×H200** | **TP=4** | **312** | **TP=2** | **272** | **−13 %** |

**More memory per card ⇒ narrower honest topology ⇒ the attack stops paying.**

Serving is worse still — see below — so on this hardware the fraud loses on every axis.

## Environment

| Parameter | Value |
|---|---|
| CUDA | 13.0.2 (image), driver 590.48.01 |
| vLLM | 0.25.1, build `752a3a504485790a2e8491cacbb35c137339ad34` |
| Python | 3.12.13 |
| mlnode | 3.0.16, `gonka_poc.entrypoint.api_router` |

## Config

```bash
TP=2 python3 scripts/patch_hy3.py
# uvicorn caches runner.py — restart the API or the change silently does nothing
CUDA_VISIBLE_DEVICES=0,1 python -m uvicorn api.app:app --host 0.0.0.0 --port 8081
curl -X POST http://127.0.0.1:8081/api/v1/inference/up/async \
  -H 'Content-Type: application/json' \
  -d '{"model":"cyankiwi/Hy3-AWQ-INT4","dtype":"auto","additional_args":[]}'
```

```
--tensor-parallel-size 2   --gpu-memory-utilization 0.90
--max-model-len 262144        --max-num-batched-tokens 65536
--kv-cache-dtype fp8          --logprobs-mode processed_logprobs
--worker-extension-cls gonka_poc.worker.PoCWorkerExtension
--tool-call-parser hy_v3      --reasoning-parser hy_v3
--trust-remote-code --enable-auto-tool-choice --enable-expert-parallel
```

Measurement window **120 s**, batches 16/32/64.

## Validation

### Throughput (120 s window, corrected accounting)

| batch | 16 | 32 | 64 |
|---:|---:|---:|---:|
| **this arm, TP=2 (2 cards)** | 512 | 528 | **544** |
| honest FP8, TP=4 (`ref_sweep_honest_tp4_120s.log`) | 1175 | 1232 | **1248** |

Two instances of this arm on the same four cards: 1088 nonces/min, against the honest 1248.

### Fingerprint

```bash
python3 scripts/l2_matrix.py artifacts ref_nonces_fp8_4card_s1.json nonces_int4_s1.json
```

| Pair | bit-identical | L2 median | >0.40 |
|---|---:|---:|---:|
| honest ↔ honest (floor, this host) | 0.0 % | 0.1996 | 4.1 % |
| **honest (TP=4) ↔ this arm (TP=2)** | 0.0 % | **0.3767** | 42.4 % |

Same checkpoint, five measurements across three machines, four topologies and two driver
versions: 0.3741 / 0.3767 / 0.3755 / 0.3753 (and 0.3745 pooled). **Spread 0.26 %** — the
fingerprint is invariant to chip, topology, host and driver, and belongs to the checkpoint.

### Inference

| Scenario | honest TP=4 | this arm TP=2 | Δ |
|---|---:|---:|---:|
| s1 long, sequential | 91.2 | 71.7 | −21 % |
| s2 short, 30 runners | 1108.1 | 873.0 | −21 % |
| s3 very long, sequential | 76.4 | 52.2 | −32 % |
| s4 very long, 20 runners | 270.2 | **109.5** | **−59 %** |

TTFT on s4: 4.441 s honest against **13.336 s** here — three times worse. A node narrowed for
mining is a markedly worse inference provider, and that is visible from outside without any
cryptography.

### Resources

| | value |
|---|---:|
| weights | 166 GiB across 2 cards (100 GB/card observed) |
| KV cache | 329 936 tokens |
| bring-up | 137 s (compilation 61 s) |
| idle cards 2–3 | 0 MiB (verified) |

A batch-64 sweep needs 65 536 KV tokens — a 5× margin remains.

## Findings

1. **First negative case for the attack**: −13 % against honest on the same four cards.
2. **The rule generalises**: gain = honest parallelism penalty − dequant cost. H200's large
   per-card memory keeps the honest arm narrow, so there is nothing left to capture.
3. **The fingerprint is driver-invariant too** — 0.3767 here (driver 590) against 0.3741 on
   driver 610.
4. **Serving collapses**: −59 % on s4 and a 3× worse TTFT.

## Files

```
artifacts/
  nonces_int4_{s1,s1_r2,s2,s3}.json          this arm at TP=2
  ref_nonces_fp8_4card_{s1,s1_r2,s2,s3}.json honest reference at TP=4 (it cannot run on 2 cards)
  sweep_120s.log, ref_sweep_honest_tp4_120s.log
  serving.sqlite
scripts/
  patch_hy3.py  run_pow_generation.py  collect_artifacts.py  l2_matrix.py  poc_seeds.json
```

Related: [honest FP8 on this host](../hy3-fp8-4xh200/) ·
[same model at TP=4 on H200](../hy3-int4-cyankiwi-4xh200/) ·
[the same manoeuvre on H100 (+10 %)](../hy3-int4-cyankiwi-4xh100/) ·
[on B300 (+48 %)](../hy3-nvfp4-r0b0tlab-1xb300/) · [campaign summary](../hy3-summary/)

## Reproducibility checklist

- [x] Image pinned by digest; quantisation described from `config.json` / loader output
- [x] Every script referenced above committed under `scripts/`
- [x] L2 tables reproducible from committed artifacts via `scripts/l2_matrix.py`
- [x] 3 seeds behind every fingerprint claim
- [x] Invalid or superseded measurements labelled in place, not dropped
- [x] No internal-tooling links, absolute paths, or sibling-repo references
