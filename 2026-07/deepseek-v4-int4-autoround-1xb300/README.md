# DeepSeek-V4-Flash INT4: it does run — and it is twice as slow as honest FP8

**Date:** 2026-07-25
**Models:** `Intel/DeepSeek-V4-Flash-W4A16-AutoRound` (INT4, auto-round / auto_gptq packing, group 128) vs `deepseek-ai/DeepSeek-V4-Flash` (FP8) vs `nvidia/DeepSeek-V4-Flash-NVFP4`
**Hardware:** 1× NVIDIA B300 SXM6 AC (275,040 MiB, 1100 W), driver 580.126.09, TP=1
**Base image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash:0.2.13-vllm0.25.1-overlay-k4` (`sha256:2af898fa…8cbb28a`)
**Patched image:** base + [vllm-project/vllm#45645](https://github.com/vllm-project/vllm/pull/45645) applied to the installed `vllm` package (`scripts/pr45645_vllm_only.diff`)
**PoC:** v2 plugin, seq_len 1024, k_dim 12, `--max-model-len 400000`

> V4 thresholds are **not calibrated**. L2 values are distances only — no PASS/FRAUD verdicts.

## Why this run exists, and what it overturns

Earlier reports in this series concluded that **INT4 is impossible for V4**: every GPTQ /
AutoRound attempt died in the weight loader with

```
AttributeError: 'ColumnParallelLinear' object has no attribute 'weight_scale_inv'
```

That conclusion was wrong about the *cause*. INT4 is not impossible — it is **absent from
the released vLLM**. Intel's model card points at an open pull request, and that PR turns
out to be **188 lines of pure Python across three files, no CUDA**. It applies cleanly to
the 0.25.1 tree in our image (all hunks succeed, largest offset 107 lines), so no rebuild
is needed — a `patch` and a `docker commit`.

The decisive hunk is in `flashmla.py::_o_proj`:

```python
if not hasattr(self.wo_a, "weight_scale_inv"):
    # Using BF16 reference wo_a path (same as ROCm).
```

which is precisely the attribute our failures reported missing. Intel's checkpoint keeps
`wo_a` at 16 bits (`extra_config: {"wo_a": {"bits": 16}}`) while packing everything else as
INT4 — 34,123 × `qweight` / `qzeros` / `scales`.

**With the patch the model loads, serves, and answers correctly.** Weights 142.7 GiB,
engine up in 210 s, no cold JIT at all (NVFP4 needs ~15 min), KV cache 2,700,687 tokens,
max concurrency 6.75× at 400k context, zero failed requests.

## Result — INT4 is not a threat, it is a handicap

PoC batch sweep, eager, same card, same day:

| batch | honest FP8 | NVFP4 | **INT4** |
|------:|-----------:|------:|---------:|
| 8 | 1184 | 2192 | **560** |
| 16 | 1664 | 2720 | **800** |
| **32** | **1664** | 2368 | **832** |

**INT4 delivers half the nonces of the honest model** (832 vs 1664 at batch 32) and about a
third of NVFP4's.

The reason is in the patch itself: it does not add INT4 kernels, it *routes around their
absence*. `_o_proj` falls back to the BF16 reference path, and
`_materialize_mxfp_wo_a_bf16` dequantises weights back to bf16 at load. The model is stored
in 4 bits and computed through 16 — paying an unpacking cost on every forward.

The memory saved does not compensate either: 142.7 GiB of weights instead of ~250 GiB, but
KV cache grows only from 2,661,034 to 2,700,687 tokens (+1.5 %), because
`--gpu-memory-utilization 0.90` caps the pool before the saving matters.

## L2 — INT4 separates three times better than NVFP4

1000 nonces per pair, same `block_hash` / `public_key`.

| Pair | N | median L2 | p95 | >0.4 |
|---|---:|---:|---:|---:|
| honest B300 vs honest B200 | 1000 | 0.188 | 0.370 | 3.7 % |
| honest B300 vs honest H200 | 1000 | 0.189 | 0.361 | 2.7 % |
| NVFP4 vs honest B300 | 1000 | 0.210 | 0.414 | 6.1 % |
| **INT4 vs honest B300** | 1000 | **0.296** | 0.530 | **19.0 %** |
| **INT4 vs honest B200** | 1000 | **0.297** | 0.545 | 19.7 % |
| **INT4 vs honest H200** | 1000 | **0.296** | 0.543 | 20.1 % |
| INT4 vs NVFP4 | 1000 | 0.308 | 0.543 | 22.2 % |

INT4's offset is **0.296 against every honest set regardless of machine** — stable to the
third decimal, so it is a model-level signal, not measurement noise. Compare with the
honest floor of 0.188, which is entirely machine noise.

Machine-readable: `artifacts/l2_matrix.json`.

## The fraud scale, now complete

Placing every measured V4 vector on one axis:

| What is compared against honest V4 | median L2 | >0.4 | throughput vs honest |
|---|---:|---:|---:|
| itself, repeated on the same card | 0.000 | 0 % | — |
| honest on a different card | 0.188 | ~3 % | — |
| **NVFP4** | **0.210** | 6.1 % | **+37 %** |
| **INT4 AutoRound** | **0.296** | 19.0 % | **−50 %** |
| V4-Base (checkpoint substitution) | 0.443 | 61.5 % | needs 2× the GPUs |
| a completely different model (Kimi / MiniMax / Qwen3) | ~1.40 | 100 % | — |

The pattern is the point: **the more profitable a cheat is, the harder it is to detect.**

- NVFP4 — the only vector that is *both* profitable (+37 %) and near-invisible (0.210 against
  a 0.188 honest floor).
- INT4 — three times more visible, and *unprofitable*: nobody patches vLLM in order to earn
  half as much.
- Checkpoint substitution — trivially visible and needs twice the hardware.
- Foreign models — saturate the metric at ~1.40, the asymptote for uncorrelated 12-dim
  vectors.

**Consequence for threshold work:** the defence problem is not a spectrum, it is a single
target. Everything except NVFP4 is either loud or economically pointless.

## Caveats

- **The patch is closed, not merged.** Why it was closed was not investigated. It is
  adequate as a research instrument; it is not a basis for anything shipped.
- **The patch's effect on the honest path was not measured.** The BF16 fallback is gated on
  `hasattr(wo_a, "weight_scale_inv")`, which honest FP8 satisfies, so the branch should not
  fire — but "should not" is an argument, not a measurement. The clean check is to run
  honest FP8 on the patched image and compare nonces with the unpatched run; if they are
  bit-identical, the 0.296 figure is purely quantisation. **That check has not been run**,
  so treat 0.296 as an upper bound that may contain a patch component.
- Only the eager arm is reported here. The CUDA-graph arm adds nothing for PoC on this box
  (see `../deepseek-v4-flash-1xb300-cudagraph-ab/`).

## Files

| Path | What |
|---|---|
| `artifacts/int4_eager_{nonces.json,poc_sweep.log}` | INT4 run |
| `artifacts/honest_fp8_eager_*` , `nvfp4_eager_*` | the two comparison runs, same card |
| `artifacts/l2_matrix.json` | the table above |
| `scripts/pr45645_vllm_only.diff` | the patch, test file stripped |
| `scripts/full_test_int4.sh` | the driver (uses the patched image) |
| `scripts/patch_runner.py` | the only modification to the image's runner args |
| `scripts/{run_pow_generation,collect_artifacts,compare_nonces,metrics}.py` | sweep, collection, L2, serving metrics |

## Reproduce

```bash
curl -sL https://github.com/vllm-project/vllm/pull/45645.diff -o pr.diff   # strip a/tests/*
docker run -d --name p --entrypoint sleep <base-image> infinity
docker cp scripts/pr45645_vllm_only.diff p:/tmp/
docker exec p bash -c 'cd /usr/local/lib/python3.12/dist-packages && patch -p1 --fuzz=5 < /tmp/pr45645_vllm_only.diff'
docker commit p mlnode-v4-int4:pr45645
./scripts/full_test_int4.sh Intel/DeepSeek-V4-Flash-W4A16-AutoRound int4
```

Pass `MODEL=` to `run_pow_generation.py` — it is pinned to the default checkpoint and
otherwise answers `409 params mismatch`, surfaced as a bare `502` in the sweep log.

## Reproducibility checklist

- [x] Patch committed in-tree, applied to a pinned base image referenced by digest
- [x] All three models measured on the same GPU, same day, same args
- [x] Raw sweeps and nonce sets committed
- [x] L2 regenerated from committed artifacts
- [x] Unverified assumption (patch effect on the honest path) stated explicitly
- [x] No links to `.claude/`, no absolute local paths, no host addresses
- [x] No verdicts asserted — V4 thresholds are not calibrated
