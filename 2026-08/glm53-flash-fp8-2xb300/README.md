# GLM-5.3-Flash — 2×B300 — honest FP8 PoC baseline (2030 nonces/min, PoC caps out at batch 32)

**Date:** 2026-08-26
**Model:** `zai-org/GLM-5.3-Flash` — native FP8 (e4m3, dynamic activations), FP8 KV cache,
320B-total / 18B-active multimodal MoE, 288 routed experts × top-8 + 1 shared, 45 layers
(34 KDA linear-attention + 11 NoPE sparse-MLA), 1 MTP layer, 1M declared context.
Snapshot `3f1971b7b5f7a528c9c4ef6212c8785298a8c24a`, 306 GB, 62 shards.
**Hardware:** 2× NVIDIA B300 SXM6 AC (SXM, 275040 MiB each, 1100 W, driver 610.57.04, sm_103),
NV18 full mesh.
**Image:** `ghcr.io/kaitakuai/vllm-poc:glm53-poc-v4-ed8873884`
**Digest:** `sha256:31b42acc1d85688a20e4ef8e6de718829062097cd6f3457f83e9e4fea892f123`

## Summary

First run of GLM-5.3-Flash on the PoC v2 prefill scheme. The model works and is **27 %
faster than Hy3 on the same topology**, but the PoC path has a hard ceiling: **batch 48
and above kill the engine** with a CUDA illegal memory access inside the sparse-MLA
indexer, and no flag combination avoids it. Batch 32 is both the working maximum and the
throughput optimum, so the ceiling costs nothing today — but a node configured with batch
64 silently produces zero nonces.

| Variant | PoC throughput (nonces/min) | Δ |
|---|---:|---:|
| Hy3 FP8, 2×B300 TP=2 ([`../hy3-fp8-2xb300`](../hy3-fp8-2xb300/)) | 1599 | — |
| **GLM-5.3-Flash FP8, 2×B300 TP=2, batch 32** | **2030** | **+27 %** |

## Environment

| Parameter | Value |
|---|---|
| CUDA version | 13.0.1 |
| NVIDIA driver | 610.57.04 |
| Python version | 3.12.3 |
| PyTorch version | 2.13.0+cu130 |
| vLLM | 0.28.0.dev0+glm53.gonka.sampler1 |
| FlashInfer | 0.6.17 (cubin + jit-cache cu130) |
| compressed-tensors | 0.17.0 |
| gonka-poc plugin | 0.1.4 |
| OS / base image | Ubuntu 24.04, built on `vllm/vllm-openai:glm53-flash` |

Upstream vLLM does **not** support this model: `Glm5NextForConditionalGeneration` is absent
from the registry in `v0.25.1`, `v0.27.0`, `v0.28.0` and `main`. Support ships only in the
vendor image `vllm/vllm-openai:glm53-flash`, whose branch is not an ancestor of `main`
(759 files differ, including files `main` has deleted). See
[the vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash) — *"Use the dedicated
GLM-5.3-Flash docker until support lands in the standard vLLM image."*

## Config

```bash
# scripts/serve_fp8_tp2.sh
SNAP=$(ls -d /root/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash/snapshots/*/ | head -1)
export VLLM_ENGINE_READY_TIMEOUT_S=3600
exec gonka-vllm-serve \
  --model "$SNAP" \
  --served-model-name glm53 \
  --tensor-parallel-size 2 \
  --kv-cache-dtype fp8 \
  --max-model-len 131072 \
  --max-num-batched-tokens 65536 \
  --max-num-seqs 128 \
  --worker-extension-cls gonka_poc.worker.PoCWorkerExtension \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 --enable-auto-tool-choice \
  --host 0.0.0.0 --port 8081
```

### What changed vs the default

| Parameter | Default (image as-shipped) | This run | Why |
|---|---|---|---|
| launcher | `vllm serve` | **`gonka-vllm-serve`** | `vllm serve` loads only the `[vllm.general_plugins]` hook; the `/api/v1/pow/*` router lives in the console script |
| model argument | positional | **`--model <path>`** | the positional argument is silently ignored and `Qwen/Qwen3-0.6B` loads instead |
| model reference | repo id | **local snapshot path** | `Glm5NextProcessor.from_pretrained` does a raw `open()` and cannot resolve a repo id |
| `--worker-extension-cls` | unset | **`gonka_poc.worker.PoCWorkerExtension`** | without it PoC generation fails in the background with `'Worker' object has no attribute 'execute_poc_forward'` |
| `--max-num-batched-tokens` | 16384 | **65536** | 16384 caps the PoC at batch 16 (16 × 1024 = 16384) |
| `--max-num-seqs` | image default | 128 | tested as a hypothesis for the batch ceiling; it is **not** the cause (see below) |
| `--kv-cache-dtype` | unset | `fp8` | FP8 KV works on Blackwell only; Hopper needs BF16 KV + `--no-enable-flashinfer-autotune` |

## Validation

### Throughput sweep

120 s measurement window per batch, seed `s1`, 5 s warm-up. A 30 s window carries ±17 %
bulk-delivery noise and must not be used for comparison (see `CONTRIBUTING.md`).

| batch | nonces in 120 s | nonces/min | per card | note |
|---:|---:|---:|---:|---|
| 16 | 3765 | 1882 | 941 | |
| **32** | **4061** | **2030** | **1015** | best |
| 48 | — | — | — | engine dies, CUDA IMA |
| 64 | 512 | 256 | — | dies after ~20 s; the figure is an artefact of a dead engine, not throughput |

### The batch ceiling is a code bug, not a tuning knob

```
gonka_poc/worker/extension.py:122       execute_poc_forward
gonka_poc/poc/poc_model_runner.py:239   _create_v1...
gonka_poc/_compat/v0_28.py:203          build_attn_metadata
vllm/v1/attention/backends/mla/indexer.py:453
triton/runtime/jit.py -> RuntimeError: Triton Error [CUDA]: an illegal memory access
```

Three runs, isolating one variable at a time:

| `max-num-batched-tokens` | `max-num-seqs` | batch 48 | batch 64 |
|---:|---:|---|---|
| 65536 | default | — | IMA after 512 nonces |
| **131072** (2.7× headroom) | default | IMA after 240 nonces | 0 (context already poisoned) |
| 131072 | **128** | IMA after 528 nonces | 0 |

The `maxnbt ≥ batch × seq_len` rule that governs DeepSeek-V4 does **not** explain this:
batch 16 under `maxnbt=16384` runs fine at exact equality, while batch 64 under
`maxnbt=65536` fails at the same equality. The dependency is on the number of requests in
the PoC batch, not on any token budget.

**An IMA poisons the CUDA context**: every measurement taken afterwards in the same process
is wrong. Batch 16 measured 1665/min in a poisoned run and 1882/min cleanly — a 13 %
understatement. Restart the engine after any IMA and discard the earlier numbers.

Raising `maxnbt` is not free — KV cache shrinks with it:

| `max-num-batched-tokens` | GPU KV cache |
|---:|---:|
| 16384 | 11,406,540 tokens |
| 65536 | 10,151,526 tokens |
| 131072 | 8,303,411 tokens |

### Nonce collection

1000 nonces per seed at batch 32, three seeds from the standard set
(`scripts/poc_seeds.json`). Throughput here independently confirms the sweep.

| seed | nonces | nonces/min |
|---|---:|---:|
| s1 | 1000 | 1883 |
| s2 | 1000 | 1929 |
| s3 | 1000 | 1929 |

Scale sanity check — nonce sets from **different** seeds are computed from different inputs
and sit at the asymptote of the metric, which fixes the ceiling of the L2 scale:

| pair | median L2 |
|---|---:|
| s1 ↔ s1 (itself) | 0.0000 |
| s1 ↔ s2 | 1.4207 |
| s1 ↔ s3 | 1.4125 |

### Startup profile

| | |
|---|---|
| cold (first launch on the box) | ~17 min |
| warm (TileLang / DeepGEMM caches populated) | 5–9 min |
| weights load | 62 shards, ~1 s/shard |
| memory | 153 GiB/card weights, 254.4 GiB/card total at gmu 0.92 |

`TORCH_CUDA_ARCH_LIST` in the base image is `7.5 8.0 8.6 8.9 9.0 10.0 12.0` — it has no
`10.3a`, so B300 (sm_103) runs through the sm_100 path. **This caused no problem**: no
`no kernel image` errors, TileLang compiled the hyper-connection kernels
(`mhc_pre_big_fuse_with_norm_tilelang`, `mhc_post_tilelang`) in ~10 s, and DeepGEMM warmed
1720 configurations in ~10 s. For comparison, Hy3 NVFP4 on B200 spent 20+ minutes on 4145
DeepGEMM configurations.

The log also reports `Breakable CUDA graph enabled` and `enforce_eager=False`: as on
DeepSeek-V4, `torch.compile` is force-disabled for this architecture, so the meaningful
compilation variable is `--enforce-eager`, not `mode`.

### Inference performance

**Not measured.** The honest and fraud checkpoints together (306 GB + 183 GB) do not fit in
the 500 GB box disk, so the honest weights were deleted to make room for the fraud arm
before any serving benchmark ran. A serving comparison needs a box with ≥ 600 GB of disk.

## Files

- [`artifacts/nonces_fp8_s{1,2,3}.json`](artifacts/) — 1000 nonce fingerprints per seed
  (12-dim fp16 `vector_b64`), the reference set for every fraud comparison
- [`artifacts/collector_config_s{1,2,3}.json`](artifacts/) — collector metadata per run
- [`artifacts/sweep_16_32_64_maxnbt65536.log`](artifacts/sweep_16_32_64_maxnbt65536.log) — the headline sweep
- [`artifacts/sweep_48_64_maxnbt131072.log`](artifacts/sweep_48_64_maxnbt131072.log),
  [`artifacts/sweep_48_64_maxnbt131072_seqs128.log`](artifacts/sweep_48_64_maxnbt131072_seqs128.log) —
  the two runs that rule out `maxnbt` and `max-num-seqs` as causes of the batch ceiling
- [`artifacts/collect.log`](artifacts/collect.log) — nonce collection, all three seeds
- [`scripts/serve_fp8_tp2.sh`](scripts/serve_fp8_tp2.sh) — exact engine launch
- [`scripts/sweep.sh`](scripts/sweep.sh), [`scripts/collect.sh`](scripts/collect.sh) — drivers
- [`scripts/run_pow_generation.py`](scripts/run_pow_generation.py),
  [`scripts/collect_artifacts.py`](scripts/collect_artifacts.py),
  [`scripts/poc_seeds.json`](scripts/poc_seeds.json) — the PoC harness as executed

The engine log for this arm was overwritten by the fraud run on the same box and is lost;
the fraud engine logs are in the sibling folder.

`scripts/run_pow_generation.py` differs from the repository's other copies in exactly three
lines: this image serves the PoC router at `/api/v1/pow/*`, without the `inference/`
segment that the mlnode images use.

## Findings / recommendation

1. **GLM-5.3-Flash is 27 % faster than Hy3 for PoC on identical hardware** — 2030 vs 1599
   nonces/min on 2×B300 TP=2. Consistent with 18B active parameters against 21B.
2. **PoC is capped at batch 32.** Batch ≥ 48 kills the engine through the sparse-MLA
   indexer, and neither `max-num-batched-tokens` (up to 2.7× headroom) nor
   `max-num-seqs 128` helps. This needs a code fix. Until then, a node configured above
   batch 32 produces nothing.
3. **`max-num-batched-tokens` must be raised to at least 32768** for batch 32; the 16384
   default silently limits PoC to batch 16.
4. **Three launch requirements are undocumented and two of them fail silently** — see the
   config table. Missing `--worker-extension-cls` in particular leaves the router
   answering `200 OK` while generating zero nonces; missing `--model` loads a completely
   different model. Both should be enforced by the image or by a pre-flight check.
5. **The sm_100 fallback on B300 is a non-issue** — the missing `10.3a` in the arch list
   costs nothing measurable at startup.

## Related

- fraud arm on the same box: [`../glm53-flash-nvfp4-libertai-2xb300/README.md`](../glm53-flash-nvfp4-libertai-2xb300/README.md)
- comparable model, same topology: [`../hy3-fp8-2xb300/README.md`](../hy3-fp8-2xb300/README.md)
- campaign this methodology comes from: [`../hy3-summary/README.md`](../hy3-summary/README.md)
- vLLM recipe (model requirements, hardware matrix): https://recipes.vllm.ai/zai-org/GLM-5.3-Flash

## Reproducibility checklist

- [x] A reader with only this folder can reach the headline result by following the README
      top to bottom.
- [x] Hardware stated exactly: GPU model, count, driver version, interconnect.
- [x] Image pinned by tag + digest.
- [x] Every command copy-pasteable; the only placeholder is the box address.
- [x] Every script the steps invoke is committed under `scripts/`.
- [x] No `.claude/` links, no absolute user paths, no sibling-repo paths.
- [x] All artifacts referenced in the report exist in `artifacts/`.
- [x] Expected outputs stated: sweep prints `Result: N nonces, M/min` per batch; collection
      writes `nonces_1000.json` per seed.
- [x] Known gotchas listed with fixes (launcher, `--model`, worker extension, `maxnbt`,
      `HOST_IP`, IMA-poisoned context).
- [x] Measurements invalidated by the IMA are named as invalid, with the magnitude of the
      error.
- [x] Serving explicitly reported as not measured, with the reason.
