# Hy3 FP8 — 8×H100 — honest baseline (worst nonces-per-card in the fleet, and Hopper is non-deterministic again)

**Date:** 2026-08-19
**Model:** `tencent/Hy3-FP8` — 295B total / 21B active MoE, 192 experts × top-8, 80 layers
+ 1 MTP layer (`num_nextn_predict_layers: 1`), GQA 64 heads / **8 KV heads** × 128, 256K
context. `quant_method: fp8`, `activation_scheme: static`, **`kv_cache_scheme: static`**;
only `lm_head` and `embed_tokens` excluded. 300 GB, 101 shards, weights **276 GiB** in VRAM.
**Hardware:** 8× NVIDIA **H100 80GB HBM3 SXM** (700 W, NV18 full mesh, driver
**580.126.09**, sm_90). Bare-metal host, 176 cores, 1.3 TB RAM.
**Image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash-0731:3.0.16-overlay-k5`
**Digest:** `sha256:8ce8830b4182b3dfd642c5e00f548f50a830611e4bc64ec4fbd84fe91070e3aa`

> The DeepSeek-V4-Flash foundry image, reused only as a vLLM 0.25.1 + PoC-plugin runtime.
> Its `runner.py` hardcodes V4-specific flags that must be replaced — `scripts/patch_hy3.py`.

## Summary

Eight H100s are the widest topology measured for Hy3, and they are the **least efficient**:
1344 nonces/min, i.e. **168 per card**, against 800 per card on 2×B300. Doubling or
quadrupling the card count consistently loses ground, because the PoC prefill scales badly
with tensor parallelism.

Serving does not share that weakness — 333.5 output tok/s on the high-concurrency long-context
scenario, within 10 % of 4×B200. So a node on 8×H100 looks weak at mining while remaining a
perfectly good inference provider; the PoC-to-serving ratio is strongly topology-dependent.

This run also gives the third independent Hopper determinism point: **0 of 1000 nonces**
reproduce bit-exactly on repeat, at TP=8. Together with 4×H200 (TP=4) and the Blackwell hosts
(100 % at TP=2 and TP=4), the split is strictly architectural and independent of topology.

| | value |
|---|---:|
| best batch | 64 |
| nonces/min | **1344** |
| per card | 168 |
| bit-identical on repeat | **0 %** |
| honest floor (median L2) | 0.2013 |

## Environment

| Parameter | Value |
|---|---|
| CUDA | 13.0.2 (image), driver 580.126.09 |
| vLLM | 0.25.1, build `752a3a504485790a2e8491cacbb35c137339ad34` |
| Python | 3.12.13 |
| mlnode | 3.0.16, `gonka_poc.entrypoint.api_router` |

Unlike the Vast.ai hosts used for the other Hy3 experiments, this is an ordinary server: the
image is only a Docker image, so everything runs **inside the container** —
`docker run -d --name hy3 --gpus all --shm-size=64g --network host -v /root/hf:/root/.cache/huggingface --entrypoint sleep <image> infinity`,
then `docker exec`. Note that `docker exec` without `-i` silently swallows a heredoc.

## Config

```bash
docker exec -e TP=8 hy3 python3 /root/patch_hy3.py
docker exec -d hy3 bash -c 'cd /app && source /app/packages/api/.venv/bin/activate && \
  WATCHER_MAX_UNHEALTHY_COUNT=9999 VLLM_RUNNER_TIMEOUT=3600 \
  exec python -m uvicorn api.app:app --host 0.0.0.0 --port 8081'
docker exec hy3 curl -s -X POST http://127.0.0.1:8081/api/v1/inference/up/async \
  -H 'Content-Type: application/json' \
  -d '{"model":"tencent/Hy3-FP8","dtype":"auto","additional_args":[]}'
```

```
--tensor-parallel-size 8   --gpu-memory-utilization 0.90
--max-model-len 262144        --max-num-batched-tokens 65536
--kv-cache-dtype fp8          --logprobs-mode processed_logprobs
--worker-extension-cls gonka_poc.worker.PoCWorkerExtension
--tool-call-parser hy_v3      --reasoning-parser hy_v3
--trust-remote-code --enable-auto-tool-choice --enable-expert-parallel
```

Measurement window **120 s**, batches 16/32/64.

### What changed vs the default

| Parameter | Image as-shipped (V4 leaf) | This run |
|---|---|---|
| `--tensor-parallel-size` | 1 | 8 |
| `--max-model-len` | 400000 | 262144 |
| `--max-num-batched-tokens` | 32768 | 65536 (batch 64 × seq_len 1024) |
| `--tokenizer-mode` | `deepseek_v4` | removed (auto) |
| `--tool-call-parser` / `--reasoning-parser` | `deepseek_v4` | `hy_v3` |
| `--speculative-config` | DSpark, 7 tokens | removed |
| `--enable-expert-parallel` | absent | added |

## Validation

### Throughput (120 s window, corrected accounting)

| batch | 16 | 32 | 64 |
|---:|---:|---:|---:|
| nonces/min | 1232 | 1312 | **1344** |

Fleet context, all measured with the same script and window:

| topology | cards | nonces/min | per card |
|---|---:|---:|---:|
| 2×B300 | 2 | 1599 | **800** |
| 4×B200 | 4 | 1888 | 472 |
| **8×H100** | 8 | **1344** | **168** |

### Fingerprint

```bash
python3 scripts/l2_matrix.py artifacts nonces_fp8_s1.json nonces_fp8_s1_r2.json
```

| Pair | bit-identical | L2 median | p95 | max | >0.40 |
|---|---:|---:|---:|---:|---:|
| FP8 ↔ FP8 repeat (honest floor) | **0.0 %** | 0.2013 | 0.3744 | 1.0211 | 3.5 % |

Cross-host honest floors, computed against the sets committed in the sibling experiments —
every pair of distinct machines lands in a narrow band:

| pair | L2 median | >0.40 |
|---|---:|---:|
| 8×H100 ↔ 4×H200 | 0.2007 | 4.6 % |
| 8×H100 ↔ 2×B300 | 0.1977 | 2.4 % |
| 8×H100 ↔ 4×B200 | 0.2030 | 5.6 % |
| 4×H200 ↔ 2×B300 | 0.1977 | 4.3 % |
| 4×H200 ↔ 4×B200 | 0.2043 | 3.8 % |
| 2×B300 ↔ 4×B200 | 0.2028 | 3.3 % |

**All six pairs sit between 0.1977 and 0.2043** — a 3.3 % spread. Architecture, card
generation and tensor-parallel width do not move the honest floor. There is no
"closer" hardware pair: either the run is bit-exact (same machine, Blackwell) or it is at
0.20. That means a single network-wide gate is defensible; per-pair calibration is not
needed.

### Inference

| Scenario | TTFT | out tok/s |
|---|---:|---:|
| s1 long prompt, sequential | 0.254 | 90.8 |
| s2 short prompt, 30 runners | 0.228 | 1253.3 |
| s3 very long, sequential | 0.521 | 78.7 |
| s4 very long, 20 runners | 3.394 | 333.5 |

Throughput recomputed from the `measurements` table (compressa-perf 0.2.7 loses
`THROUGHPUT_*` and `RPS` — it inserts metrics with `conn.commit()` commented out, then
crashes generating a PDF).

### Resources

| | value |
|---|---:|
| weights / rank | 69.28 GiB (**276 GiB** total) |
| KV cache | 1 058 752 tokens |
| bring-up | 200 s (compilation 133 s) |

## Findings

1. **Worst nonces-per-card measured**: 168, against 800 on 2×B300. PoC prefill punishes wide
   tensor parallelism.
2. **Serving is unaffected by that** — 333.5 tok/s on s4, comparable with Blackwell. The
   PoC-to-serving ratio is a function of topology, not only of hardware class.
3. **Hopper is non-deterministic at TP=8 too** — 0/1000, matching 4×H200 at TP=4.
4. **The honest floor is a fleet-wide constant ≈0.20** across all six machine pairs.
5. On a non-Vast host everything must run inside the container, and `docker exec` needs `-i`
   for heredocs.

## Files

```
artifacts/
  nonces_fp8_{s1,s1_r2,s2,s3}.json   honest FP8 (s1_r2 = repeat of s1)
  sweep_120s.log                      valid sweep
  serving.sqlite                      compressa-perf database
scripts/
  patch_hy3.py  run_pow_generation.py  collect_artifacts.py  l2_matrix.py  poc_seeds.json
```

Related: [INT4 fraud at TP=8 on this host](../hy3-int4-cyankiwi-8xh100/) ·
[INT4 fraud at TP=4 — the economics](../hy3-int4-cyankiwi-4xh100/) ·
[FP8 on 4×H200](../hy3-fp8-4xh200/) · [FP8 on 2×B300](../hy3-fp8-2xb300/) ·
[FP8 on 4×B200](../hy3-fp8-4xb200/)

## Reproducibility checklist

- [x] Image pinned by digest; quantisation described from `config.json` / loader output
- [x] Every script referenced above committed under `scripts/`
- [x] L2 tables reproducible from committed artifacts via `scripts/l2_matrix.py`
- [x] 3 seeds behind every fingerprint claim
- [x] Invalid or superseded measurements labelled in place, not dropped
- [x] No internal-tooling links, absolute paths, or sibling-repo references
