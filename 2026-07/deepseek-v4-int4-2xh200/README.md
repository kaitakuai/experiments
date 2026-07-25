# DeepSeek-V4-Flash INT4 on 2× H200: ties honest FP8 again — and never reproduces itself

**Date:** 2026-07-25
**Models:** `Intel/DeepSeek-V4-Flash-W4A16-AutoRound` (INT4, auto-round / auto_gptq, group 128) vs `deepseek-ai/DeepSeek-V4-Flash` (FP8)
**Hardware:** 2× NVIDIA H200 SXM (143,771 MiB, 700 W, NV18), driver 580.159.03, CUDA 13.0, TP=2
**Base image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash:0.2.13-vllm0.25.1-overlay-k4` (`sha256:2af898fa…8cbb28a`)
**Patch:** [vllm-project/vllm#45645](https://github.com/vllm-project/vllm/pull/45645) (`scripts/pr45645_vllm_only.diff`)
**PoC:** v2 plugin, seq_len 1024, k_dim 12, `--max-model-len 400000`

> V4 thresholds are **not calibrated**. L2 values are distances only — no PASS/FRAUD verdicts.

## Why this run exists

INT4 had been measured on Blackwell (half the honest throughput, via the patch's BF16
fallback) and on 4× H100 (dead level with honest, via a real W4A16 Marlin kernel). This is
the third configuration: Hopper again, but TP=2 instead of TP=4, to see whether the parity
survives a different card count — and it is the last empty row of the benchmark table.

## Result 1 — parity, to one nonce, for the second time

| batch | honest FP8 | INT4 eager | INT4 graphs |
|------:|-----------:|-----------:|------------:|
| **32** | **1216** | **1215** | **1216** |

Alongside the other two configurations:

| config | honest FP8 | INT4 | Δ | MoE path |
|---|---:|---:|---:|---|
| 4× H100 TP=4 | 1536 | 1535 | −0.07 % | W4A16 MARLIN |
| **2× H200 TP=2** | **1216** | **1215** | **−0.08 %** | **W4A16 MARLIN** |
| 1× B300 TP=1 | 1664 | 832 | −50 % | BF16 fallback (no Marlin on sm_100) |

**Speed is decided by whether a W4A16 Marlin kernel exists, not by architecture generation
or card count.** Where it does, INT4 ties honest FP8 exactly; where it does not, the patch
falls back to dequantise-to-BF16 and halves throughput.

CUDA graphs change nothing (1215 → 1216), matching honest FP8 on the same box, which also
scores 1216 in both modes. Four independent measurements on this hardware — two models ×
two modes — all land on the same number.

| | value |
|---|---|
| **MoE backend** | `W4A16 MARLIN` |
| **ATTN backend** | `sparse_mla` (C128A) |
| weights | 72.65 GiB/GPU (~145 GiB total) |
| GPU KV cache | 1,282,446 tokens |
| max concurrency @400k | 3.21× |

Note the memory: ~145 GiB total, the same as honest FP8's ~149 GiB. **INT4 needs two H200s,
exactly like the honest model.** On 2× H100 (80 GB each) the same 145 GiB does not leave
room for KV and the load fails — which is why that configuration needs four cards. The
requirement is set by the checkpoint's size, and Intel's INT4 checkpoint is only 2.5 %
smaller than FP8 because it keeps `wo_a`, `embed` and `gate` at 16 bits.

## Result 2 — INT4 does not reproduce itself, on any architecture

Collecting twice on the **same box**, changing only the compilation mode:

| model / card | bit-identical | median L2 |
|---|---:|---:|
| **INT4 / H100** | **0/1000** | 0.1396 |
| **INT4 / H200** | **0/1000** | 0.1402 |
| **INT4 / B300** | **0/1000** | 0.1386 |
| NVFP4 / B200 | 968/1000 | 0.0000 |
| honest FP8 / B300 | 968/1000 | 0.0000 |

**This corrects the B300 report.** That report observed the same 0/1000 on Blackwell and
attributed it to the patch's BF16 fallback. H100 and H200 run a genuine Marlin kernel — the
fallback is not used — and produce the identical result. The cause is therefore the
**INT4 quantisation itself**: unpacking 4-bit weights is sensitive to an operation order
that changes with the compilation mode. Honest FP8 and NVFP4 under the same test reproduce
96.8 % bit-for-bit.

Consequence beyond fraud: **INT4 is a poor choice for an honest node too.** Such a node
would not agree with itself across a configuration change, and would fail any check that
relies on reproducibility.

## Result 3 — L2 against honest: 0.297, the fourth independent confirmation

| Pair | median L2 | >0.4 |
|---|---:|---:|
| **INT4 H200 vs honest H200** | **0.297** | 21.6 % |
| INT4 H200 (graphs) vs honest H200 | 0.294 | — |
| INT4 H100 vs honest H100 | 0.295 | 20.8 % |
| INT4 B300 vs honest B300 | 0.296 | 19.0 % |
| INT4 H200 vs INT4 H100 | 0.145 | 1.4 % |
| INT4 H200 vs INT4 B300 | 0.144 | 1.1 % |
| honest vs honest, different GPU models | 0.188 | ~3 % |
| NVFP4 vs honest | 0.210 | 6.1 % |

INT4's offset against honest is **0.294–0.298 across three architectures, two compute paths
and two compilation modes**. Any two INT4 runs sit 0.14 apart regardless of where they ran —
the same 0.14 seen between modes on one box, which is what Result 2 is about.

Machine-readable: `artifacts/l2_matrix.json`.

## Result 4 — serving

Zero failed requests in all eight runs.

| Scenario | eager | graphs | Δ |
|---|---:|---:|---:|
| s1 long prompt, sequential | 11.4 | **83.3** | +628 % |
| s2 short prompt, concurrent | 212.6 | **1109.5** | +422 % |
| s3 very long, sequential | 9.6 | **66.0** | +587 % |
| s4 very long, max concurrency | 72.4 | **166.5** | +130 % |

The same flag that moves PoC by one nonce moves serving by up to 6×. This is the fourth
V4 configuration showing the split, and the reason is unchanged: the PoC forward at batch 32
is not waiting on kernel launches, the decode loop — paying a full launch round per token —
is.

Against honest FP8 in eager on this hardware (7.25 / 151.5 / 8.91 / 53.9 tok/s), INT4 is
8–57 % faster. That is the opposite sign from 4× H100, where INT4 lost 18 % under maximum
concurrency. With half the workers there is less cross-GPU synchronisation to pay for, so
the smaller weights help instead of hurting.

## The verdict on INT4, across all three configurations

| axis | INT4 vs honest |
|---|---|
| PoC throughput | equal on Hopper (2 configs), half on Blackwell |
| memory | equal (~145 vs ~149 GiB) |
| GPUs required | equal (2× H200, 4× H100) |
| self-reproducibility | **0/1000 vs honest 968/1000** |
| detectability | 0.297 vs NVFP4's 0.210 — **3× more visible** |

INT4 gives a cheater nothing on any axis, is harder to hide, and cannot even reproduce
itself. **NVFP4 remains the only vector that is simultaneously profitable (+37…42 % PoC) and
near-invisible (0.210 against a 0.188 honest floor).**

## Files

| Path | What |
|---|---|
| `artifacts/int4_{eager,graphs}_poc_sweep.log` | both sweeps |
| `artifacts/int4_{eager,graphs}_nonces.json` | nonce sets per mode |
| `artifacts/int4_{eager,graphs}_serving.json` | serving metrics |
| `artifacts/int4_{eager,graphs}_backends.txt` | resolved backends |
| `artifacts/honest_fp8_eager_*` | honest reference, same hardware |
| `artifacts/int4_h100_reference_nonces.json` | the 4×H100 INT4 set, for the cross-config comparison |
| `artifacts/l2_matrix.json` | the table above |
| `scripts/` | patch, box prep, run driver, sweep, collection, L2, metrics |

## Reproduce

```bash
./scripts/prep.sh 2 --patch      # deps, libnvrtc symlink, PR patch, runner args (TP=2), API
hf download Intel/DeepSeek-V4-Flash-W4A16-AutoRound
./scripts/run_model.sh Intel/DeepSeek-V4-Flash-W4A16-AutoRound h200-int4
python3 scripts/compare_nonces.py artifacts/int4_eager_nonces.json artifacts/honest_fp8_eager_nonces.json
```

Rent by `gpu_ram` (≥139 GB for H200), require `cuda_max_good >= 13.0` — a 570-series driver
kills the CUDA-13 image at `init_device` — and confirm `NV#` in `nvidia-smi topo -m`. sm_90
needs the `libnvrtc.so` symlink or FlashInfer's JIT fails to link `-lnvrtc`. Pass `MODEL=`
to `run_pow_generation.py`.

## Reproducibility checklist

- [x] Image by tag and digest; patch committed in-tree
- [x] Both modes measured; honest reference from the same hardware and image
- [x] Raw sweeps, nonce sets, serving metrics and backend records committed
- [x] L2 regenerated from committed artifacts
- [x] A wrong explanation from the B300 report identified and corrected, with the evidence
- [x] No links to `.claude/`, no absolute local paths, no host addresses
- [x] No verdicts asserted — V4 thresholds are not calibrated
