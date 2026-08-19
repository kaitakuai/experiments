# Hy3 INT4 W4A16 by `cyankiwi` — 4×H100 — the Hopper attack economics: +10 % over honest by halving the topology

**Date:** 2026-08-19
**Model:** `cyankiwi/Hy3-AWQ-INT4` — 182 GB. **Not AWQ despite the name**:
compressed-tensors `pack-quantized`, INT4 **W4A16** asymmetric, `group_size 32`, activations
BF16, MTP layer left unquantised. Kernels: `MarlinLinearKernel` +
`CompressedTensorsWNA16MarlinMoEMethod`. Weights: **166 GiB**, which is why four H100s are
enough and the honest FP8 checkpoint (276 GiB) is not an option here.
**Hardware:** 4× NVIDIA **H100 80GB HBM3 SXM** (700 W, NV18, driver **580.126.09**, sm_90) —
four cards of an 8-card host, the other four idle (`CUDA_VISIBLE_DEVICES=0,1,2,3`, verified
at 0 MiB).
**Image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash-0731:3.0.16-overlay-k5`
**Digest:** `sha256:8ce8830b4182b3dfd642c5e00f548f50a830611e4bc64ec4fbd84fe91070e3aa`

> The DeepSeek-V4-Flash foundry image, reused only as a vLLM 0.25.1 + PoC-plugin runtime.
> Its `runner.py` hardcodes V4-specific flags that must be replaced — `scripts/patch_hy3.py`.

## Summary

The same checkpoint that **loses** to honest FP8 at matched topology
([944 vs 1344 at TP=8](../hy3-int4-cyankiwi-8xh100/)) wins once it stops paying for wide
tensor parallelism:

| configuration on 8× H100 | cards | nonces/min | per card |
|---|---:|---:|---:|
| honest FP8, TP=8 | 8 | 1344 | 168 |
| this model, TP=8 | 8 | 944 | 118 |
| **this model, two instances at TP=4** | 8 | **1472 (+10 %)** | **184** |

Half the cards deliver **78 %** of the eight-card throughput (736 of 944), so two instances
beat one wide one — and beat the honest arm.

**+10 % is much less than the +48 % the same manoeuvre buys on B300**, and the reason is
instructive: there the honest arm ran at TP=2 and the fraud dropped to TP=1, so the fraud
captured almost the whole parallelism penalty. Here the honest arm is *forced* onto TP=8 by
its 276 GiB footprint and has already paid most of that penalty; the fraud can only recover
the difference between TP=8 and TP=4.

**The wider the honest configuration must be, the smaller the relative gain from cheating —
but it stays positive everywhere measured.**

## Environment

| Parameter | Value |
|---|---|
| CUDA | 13.0.2 (image), driver 580.126.09 |
| vLLM | 0.25.1, build `752a3a504485790a2e8491cacbb35c137339ad34` |
| Python | 3.12.13 |
| mlnode | 3.0.16, `gonka_poc.entrypoint.api_router` |

## Config

```bash
docker exec -e TP=4 hy3 python3 /root/patch_hy3.py
# uvicorn caches runner.py — the API must be restarted or the change silently does nothing
docker exec -d hy3 bash -c 'cd /app && source /app/packages/api/.venv/bin/activate && \
  WATCHER_MAX_UNHEALTHY_COUNT=9999 VLLM_RUNNER_TIMEOUT=3600 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  exec python -m uvicorn api.app:app --host 0.0.0.0 --port 8081'
docker exec hy3 curl -s -X POST http://127.0.0.1:8081/api/v1/inference/up/async \
  -H 'Content-Type: application/json' \
  -d '{"model":"cyankiwi/Hy3-AWQ-INT4","dtype":"auto","additional_args":[]}'
```

```
--tensor-parallel-size 4   --gpu-memory-utilization 0.90
--max-model-len 262144        --max-num-batched-tokens 65536
--kv-cache-dtype fp8          --logprobs-mode processed_logprobs
--worker-extension-cls gonka_poc.worker.PoCWorkerExtension
--tool-call-parser hy_v3      --reasoning-parser hy_v3
--trust-remote-code --enable-auto-tool-choice --enable-expert-parallel
```

Measurement window **120 s**, batches 16/32/64.

## Validation

### Throughput (120 s window)

| batch | 16 | 32 | 64 |
|---:|---:|---:|---:|
| **this arm, TP=4 (4 cards)** | 696 | 720 | **736** |
| same model, TP=8 (`ref_sweep_same_model_tp8_120s.log`) | 896 | 944 | 928 |
| honest FP8, TP=8 (`ref_sweep_honest_tp8_120s.log`) | 1232 | 1312 | **1344** |

Both reference sweeps are committed here, so the +10 % claim is checkable without leaving
this folder. Note also that at TP=4 the curve grows monotonically to batch 64, whereas at
TP=8 it peaked at 32 and fell — consistent with MARLIN saturating earlier when spread wider.

### Fingerprint

```bash
python3 scripts/l2_matrix.py artifacts ref_nonces_fp8_8card_s1.json nonces_int4_s1.json
```

| Pair | bit-identical | L2 median | >0.40 |
|---|---:|---:|---:|
| honest (8 cards) ↔ this arm (4 cards) | 0.0 % | **0.3753** | 42.7 % |
| honest (8 cards) ↔ same model at TP=8 | 0.0 % | 0.3755 | 43.0 % |
| **this arm ↔ same model at TP=8** (topology only) | 0.0 % | **0.1604** | 1.3 % |

The third row is the useful one. Changing *only* the topology of one checkpoint moves the
fingerprint by 0.16 — **less than the 0.20 honest floor between two different machines**. A
prover's tensor-parallel width is therefore not something a validator needs to know or
reproduce: 0.3753 at TP=4 versus 0.3755 at TP=8 against the same reference.

The honest reference here is the 8-card run, because honest FP8 **cannot** run on four
H100s (276 GiB against ~288 GiB raw, less KV and activations). That is not a methodological
compromise — it is precisely the asymmetry the attack exploits.

### Inference

| Scenario | honest TP=8 | this arm TP=4 | Δ |
|---|---:|---:|---:|
| s1 long, sequential | 90.8 | 80.8 | −11 % |
| s2 short, 30 runners | 1253.3 | 1189.3 | −5 % |
| s3 very long, sequential | 78.7 | 64.3 | −18 % |
| s4 very long, 20 runners | 333.5 | **181.5** | **−46 %** |

TTFT on s4: 3.394 s honest → 5.669 s at TP=8 → **8.079 s here**. A node tuned for mining
throughput becomes a markedly worse inference provider, and that is visible from outside
without any cryptography — a useful secondary signal for the network.

### Resources

| | value |
|---|---:|
| weights / card | 72.8 GiB (166 GiB total across 4) |
| KV cache | 361 120 tokens |
| bring-up | 157 s (compilation 107 s) |
| idle cards 4–7 | 0 MiB (verified) |

A batch-64 sweep needs 65 536 KV tokens, so even the narrow configuration keeps a ~5× margin.

## Findings

1. **The attack is a topology attack.** At equal topology the fraud loses by 30 %; two
   narrow instances beat the honest arm by 10 %.
2. **The gain shrinks as the honest configuration widens** — +48 % on B300 (honest at TP=2)
   against +10 % here (honest forced to TP=8).
3. **Topology does not move the fingerprint** (0.1604 between TP=8 and TP=4 of one
   checkpoint, below the honest inter-machine floor of 0.20).
4. **Mining-optimised fraud is a poor inference provider** — −46 % on s4 and 2.4× worse TTFT.
5. **`runner.py` edits require an API restart**; uvicorn caches the module.

## Files

```
artifacts/
  nonces_int4_{s1,s1_r2,s2,s3}.json          this arm at TP=4
  ref_nonces_fp8_8card_{s1,s1_r2,s2,s3}.json honest reference (8 cards — it cannot run on 4)
  ref_nonces_int4_tp8_s1.json                 same model at TP=8, for the topology-only pair
  sweep_120s.log                              this run
  ref_sweep_same_model_tp8_120s.log, ref_sweep_honest_tp8_120s.log
  serving.sqlite
scripts/
  patch_hy3.py  run_pow_generation.py  collect_artifacts.py  l2_matrix.py  poc_seeds.json
```

Related: [same model at TP=8](../hy3-int4-cyankiwi-8xh100/) ·
[honest FP8 on this host](../hy3-fp8-8xh100/) ·
[the same manoeuvre on B300 (+48 %)](../hy3-nvfp4-r0b0tlab-1xb300/)

## Reproducibility checklist

- [x] Image pinned by digest; quantisation described from `config.json` / loader output
- [x] Every script referenced above committed under `scripts/`
- [x] L2 tables reproducible from committed artifacts via `scripts/l2_matrix.py`
- [x] 3 seeds behind every fingerprint claim
- [x] Invalid or superseded measurements labelled in place, not dropped
- [x] No internal-tooling links, absolute paths, or sibling-repo references
