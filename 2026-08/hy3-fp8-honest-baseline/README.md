# Hy3 FP8 — honest baseline on H200 / B300 / B200 (PoC is bit-exact on Blackwell, not on Hopper)

**Date:** 2026-08-19
**Model:** `tencent/Hy3-FP8` — 295B total / 21B active MoE, 192 experts × top-8, 80 layers
+ **1 MTP layer** (`num_nextn_predict_layers: 1`, tensors under `model.layers.80.*`),
GQA 64 heads / **8 KV heads** × 128, 256K context, vocab 120832.
`quant_method: fp8`, `activation_scheme: static`, **`kv_cache_scheme: static`**;
only `lm_head` and `embed_tokens` are excluded. 300 GB across 101 shards.

**Hardware:**
- 4× **H200 SXM** (700 W, 143 GB, NV18, driver 610.57.04, sm_90) — Vast 48115352
- 2× **B300 SXM6** (1100 W, 275 GB, NV18, driver 610.57.04, sm_100) — Vast 48124506
- 4× **B200 SXM** (1000 W, 183 GB, NV18, driver 580.126.20, sm_100) — Vast 48135501

**Image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash-0731:3.0.16-overlay-k5`
**Digest:** `sha256:8ce8830b4182b3dfd642c5e00f548f50a830611e4bc64ec4fbd84fe91070e3aa`

> The image is the DeepSeek-V4-Flash foundry image, reused as a vLLM 0.25.1 + PoC-plugin
> runtime. Its `runner.py` hardcodes V4 flags that must be replaced — `scripts/patch_hy3.py`.

## Summary

This is the reference arm every fraud experiment in `2026-08/hy3-fraud-*` compares against.
Three results define how Hy3 can be validated at all:

1. **Hy3 needs no vLLM port.** `HYV3ForCausalLM` and `HYV3MTPModel` are already registered
   in **0.25.1**; the recipe's "≥0.26" requirement concerns optimizations (PR #47433 +
   HPC-Ops kernels), not support.
2. **The PoC fingerprint is bit-exact on Blackwell and not on Hopper.** Re-running the same
   configuration on the same machine reproduces 1000/1000 nonces on B300 (TP=2) and B200
   (TP=4), and 0/1000 on H200 (TP=4). Topology is ruled out — H200 and B200 both ran TP=4.
3. **Bit-exactness does not survive a change of machine.** Honest FP8 on B200 vs honest FP8
   on B300 — both Blackwell — gives 0 % identical and a 0.2028 median, the same floor as
   Blackwell↔Hopper. Exact-match validation is therefore only usable inside a homogeneous
   pool or for self-checks.

| Honest ↔ honest comparison (seed s1) | bit-identical | L2 median | >0.40 |
|---|---:|---:|---:|
| B300 ↔ itself (repeat) | **100.0 %** | 0.0000 | 0.0 % |
| B200 ↔ itself (repeat) | **100.0 %** | 0.0000 | 0.0 % |
| H200 ↔ itself (repeat) | 0.0 % | 0.2025 | 4.1 % |
| B200 ↔ B300 | 0.0 % | 0.2028 | 3.3 % |
| B200 ↔ H200 | 0.0 % | 0.2043 | 3.8 % |
| B300 ↔ H200 | 0.0 % | 0.1977 | 4.3 % |

**Consequence for gating.** The 0.40 threshold inherited from earlier campaigns is not
usable for Hy3: an honest Hopper prover exceeds it on 4.1 % of nonces. Any gate must be
calibrated per hardware pair, and verdicts must be aggregate (see the INT4 experiment for
the separability analysis).

## Environment

| Parameter | Value |
|---|---|
| CUDA | 13.0.2 (image); drivers 610.57.04 (H200/B300), 580.126.20 (B200) |
| vLLM | 0.25.1, build `752a3a504485790a2e8491cacbb35c137339ad34` |
| Python | 3.12.13 |
| mlnode | 3.0.16, `gonka_poc.entrypoint.api_router` |

## Config

```bash
# mlnode API — must run from the image's own venv; system python cannot import `common`
cd /app && source /app/packages/api/.venv/bin/activate
WATCHER_MAX_UNHEALTHY_COUNT=9999 VLLM_RUNNER_TIMEOUT=3600 \
  python -m uvicorn api.app:app --host 0.0.0.0 --port 8081

python3 scripts/patch_hy3.py            # TP / SPEC / MML / MNBT / GMU via env
curl -X POST http://127.0.0.1:8081/api/v1/inference/up/async \
  -H 'Content-Type: application/json' \
  -d '{"model":"tencent/Hy3-FP8","dtype":"auto","additional_args":[]}'
```

Resulting forced flags:

```
--tensor-parallel-size {4|2|4}   --gpu-memory-utilization 0.90
--max-model-len 262144           --max-num-batched-tokens 65536
--kv-cache-dtype fp8             --logprobs-mode processed_logprobs
--worker-extension-cls gonka_poc.worker.PoCWorkerExtension
--tool-call-parser hy_v3         --reasoning-parser hy_v3
--trust-remote-code --enable-auto-tool-choice --enable-expert-parallel
```

MTP variant adds `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`.
No draft checkpoint is required: `hf_config_override` maps `hy_v3` → `hy_v3_mtp`
(`HYV3MTPModel`) and reuses the target weights.

### What changed vs the default

| Parameter | Image as-shipped (V4 leaf) | This run |
|---|---|---|
| `--tensor-parallel-size` | 1 | 4 / 2 / 4 |
| `--max-model-len` | 400000 | 262144 |
| `--max-num-batched-tokens` | 32768 | 65536 (batch 64 × seq_len 1024) |
| `--tokenizer-mode` | `deepseek_v4` | removed (auto) |
| parsers | `deepseek_v4` | `hy_v3` |
| `--speculative-config` | DSpark, 7 tokens | removed (or MTP-2) |
| `--enable-expert-parallel` | absent | added |
| sweep window | 30 s | **120 s** (B300/B200) |

## Validation

### Throughput

| batch | H200 TP=4 ⚠️ | B300 TP=2 | B200 TP=4 |
|---:|---:|---:|---:|
| 16 | 1376 | 1439 | 1736 |
| 32 | **1408** | 1535 | 1840 |
| 64 | 1408 | **1599** | **1888** |

⚠️ The H200 column was taken with a 30 s window **and** the pre-fix accounting bug
(`kaitakuai/experiments` PR #7) — inflated by an unknown 0–40 %. It is published only to
document the MTP comparison below; **do not cite it as Hy3 throughput on H200.**

Per card: B300 800, B200 472. Doubling the card count buys +18 % — tensor parallelism
barely pays for itself, which is exactly what the fraud experiments exploit.

**Why 120 s.** Callbacks arrive in ~5 s bulks, so a 30 s window holds ~6 deliveries and one
delivery is ±17 %. `sweep_b300_fp8_30s_noise_demo.log` shows the same FP8 arm scoring 1279
at batch 64 on a 30 s window against 1599 on 120 s — pure noise. Before the accounting fix
the same points read +17.8 % / +16.7 % / +40 %.

### MTP speculative decoding (H200)

MTP is a decode-time mechanism; PoC v2 is a pure 1024-token prefill, so it cannot affect the
proof — and does not:

| | PoC batches 8/16/32/64 | fingerprint vs honest | KV tokens |
|---|---|---:|---:|
| FP8 | 1248 / 1376 / 1408 / 1408 | — | 1 049 888 |
| FP8 + MTP-2 | 1248 / 1376 / 1408 / 1408 | 0.1993 median (floor: 0.2025) | 951 680 |

Serving, by contrast, gains materially:

| Scenario | FP8 out tok/s | MTP-2 out tok/s | Δ |
|---|---:|---:|---:|
| s1 long prompt, sequential | 93.5 | 112.6 | **+20 %** |
| s2 short prompt, 30 runners | 1224.2 | 1336.7 | +9 % |
| s3 very long, sequential | 79.1 | 102.1 | **+29 %** |
| s4 very long, 20 runners | 275.8 | 294.6 | +7 % |

Costs: 9.4 % of KV, and s2 TTFT doubles (0.188 → 0.432). Since MTP is undetectable in the
proof and free to enable, allowing it implies baking it into the image — otherwise the gain
accrues only to operators who happen to discover it.

### Fingerprint reproduction

```bash
python3 scripts/l2_matrix.py artifacts nonces_b200_fp8_s1.json nonces_b300_fp8_s1.json
python3 scripts/l2_matrix.py artifacts          # every same-seed pair
```

Divergence, where it exists, is spread over **all** nonces rather than concentrating on
`n % batch_size == 0`; the "first nonce in batch" artefact seen with other models does not
dominate here. Cross-seed control pairs land at ~1.40, the expected asymptote.

### Resources

| Host | weights/rank | KV tokens | bringup |
|---|---:|---:|---:|
| 4×H200 | 69.28 GiB | 1 049 888 | 225 s |
| 2×B300 | 138.16 GiB | 1 172 144 | 508 s |
| 4×B200 | 69.28 GiB | 2 046 704 | 293 s |

KV costs ≈ **164 KiB/token** at fp8 (2 × 80 layers × 8 KV heads × 128) — heavy compared with
other models here, though irrelevant for prefill PoC (batch 64 × 1024 = 65 536 tokens).
Total FP8 weights are **276 GiB**, which is why a single 275 GB B300 cannot host this model
(≈242 GiB usable at `gmu 0.90`) while a 4-bit fraud arm can.

## Findings

1. **No vLLM port needed** — 0.25.1 registry already has Hy3 and its MTP model.
2. **Bit-exact PoC is an architecture property, scoped to one machine.** Holds on B300 TP=2
   and B200 TP=4, fails on H200 TP=4, and does not transfer B200↔B300.
3. **The 0.40 gate is invalid for Hy3 on Hopper** — 4.1 % honest false positives.
4. **MTP: free for PoC, +20…29 % for serving.** A policy decision, not a detection problem.
5. **Measure throughput on ≥120 s windows** with the corrected accounting.
6. **Serving must be measured on a warm engine.** On B300 a cold-boot run reported 56 tok/s
   on s3 where the warm run gave 87.
7. **`pow/stop` before compressa-perf is mandatory**, otherwise every request returns
   `503 poc_generation_active` and the database silently ends up empty.
8. **compressa-perf 0.2.7 drops metrics**: it inserts them in a loop with `conn.commit()`
   commented out and then crashes on PDF generation (`logo.png`), so `THROUGHPUT_*` and
   `RPS` are usually lost. Recompute from `measurements`:
   `sum(n_output) / (max(end_time) - min(start_time))`.

## Files

```
artifacts/
  nonces_h200_fp8_{s1,s1_r2,s2,s3}.json        honest FP8, 4×H200 (s1_r2 = repeat of s1)
  nonces_h200_fp8_mtp2_{s1,s1_r2}.json         + MTP-2 speculative decoding
  nonces_b300_fp8_{s1,s1_r2,s2,s3}.json        honest FP8, 2×B300
  nonces_b200_fp8_{s1,s1_r2,s2,s3}.json        honest FP8, 4×B200
  sweep_b300_fp8_120s.log, sweep_b200_fp8_120s.log        valid sweeps
  sweep_h200_*_30s_BUGGY.log                              invalid timing, kept for the MTP comparison
  sweep_b300_fp8_30s_noise_demo.log                       why 30 s is not enough
  serving_*.sqlite                                        compressa-perf databases
scripts/
  patch_hy3.py  run_pow_generation.py  collect_artifacts.py  l2_matrix.py  poc_seeds.json
```

## Reproducibility checklist

- [x] Image pinned by digest; model quantisation described from `config.json`
- [x] Every script referenced above committed under `scripts/`
- [x] All L2 tables reproducible from committed artifacts via `scripts/l2_matrix.py`
- [x] 3 seeds behind every fingerprint claim; cross-seed control included
- [x] Invalid measurements labelled in place rather than dropped
- [x] Scope limit of the bit-exactness result stated explicitly
- [x] No internal-tooling links, absolute paths, or sibling-repo references
