# Kimi-K2.6 INT4 — 4×B200 — b200-k5-kimi-1 image validation

**Date:** 2026-05-02
**Model:** `moonshotai/Kimi-K2.6` (compressed-tensors INT4, MLA, 384 routed × top_k=8)
**Hardware:** 4× NVIDIA B200 SXM (Vast.ai inst 36027724, 178.35 GiB HBM/GPU, NV18 mesh, sm_100)
**Image:** `ghcr.io/kaitakuai/mlnode-full:0.2.12-vllm0.20.0-b200-k5-kimi-1`

## Summary

This is a **port + memory-budget tuning** experiment, not a perf change.
The b200-k5-kimi-1 image already bakes in our B300 best-config patches
(TP=4 default, FlashInfer mxint4 MoE env, MLA seq_lens fix, eager mode).
On 4×B200 those defaults OOM at startup because B200 has only 178 GiB
HBM/GPU vs B300's 275 GiB — Kimi-K2.6 weights alone consume ~155 GiB
per worker (TP=4 sharding) and the vision-encoder profile_run wants
another 14 GiB on top.

We re-tune three knobs to fit memory **without** disabling the vision
tower or quantizing further. With the tuned config, vLLM cold-starts in
~100 s, sustains **2240 nonces/min @ batch=32** (≈ 92 % of B300
per-instance throughput), and produces nonces that **PASS** the canonical
Gonka chain validation against every existing B300/H100/B200_old baseline
(mean L2 in the 0.19-0.21 floor for cross-anything Kimi INT4).

## Symptom: B200 OOM at vision-encoder profile_run

Observed on a fresh `b200-k5-kimi-1` cold-start with the unmodified
baked-in defaults `--gpu-memory-utilization 0.85`,
`--max-num-batched-tokens 131072`. Sequence:

| Step | Result |
|---|---:|
| Model load (TP=4, INT4) | 140.8 GiB / GPU, 38 s |
| KV cache allocation (util=0.85) | OK, 18 GiB pool |
| `Encoder cache will be initialized with budget of 131072 tokens, 31 vision_chunk items max` | started |
| Vision encoder `profile_run` peak alloc | **14.32 GiB needed**, only 5.56 GiB free → OOM |

Worker stack:
```
Tried to allocate 14.32 GiB. GPU 0 has 178.35 GiB total, 5.56 GiB free.
This process has 172.78 GiB memory in use.
158.01 GiB allocated by PyTorch, 12.33 GiB reserved.
```

Followed by `EngineCore failed to start. Failed core proc(s): {}` and
`vLLM process exited prematurely`. Re-tries with `--enforce-eager`
(already on) and removing `--enable-expert-parallel` did not help.

Full evidence in [`artifacts/oom-evidence.txt`](artifacts/oom-evidence.txt).

## Root cause

Kimi-K2.6 INT4 leaves its full vision tower
(`KimiK25ForConditionalGeneration`) loaded by default. vLLM's
`profile_run` for the multimodal encoder reserves a workspace
proportional to `max-num-batched-tokens`:

```
  workspace ≈ K · max_num_batched_tokens · sizeof(activation)
   ↳ 131072 → ~14 GiB        (b200-k5-kimi-1 default)
   ↳  65536 → ~ 7 GiB        (still 0.38 GiB short on B200)
   ↳  32768 → ~ 4 GiB        (fits with ~3 GiB margin)
```

On B300 (275 GiB / GPU) this 14 GiB is invisible — there is ~120 GiB
free after model + KV. On B200 (178 GiB / GPU) the post-model headroom
is 6-8 GiB, less than what vision profile asks for.

The KV-cache calculation also hits a separate ceiling: at
`gpu-mem-util=0.75` the budget falls below the model's resident size
(`Available KV cache memory: -22.15 GiB`); at 0.85 it is barely
positive (`-4.31 GiB`); at 0.95 it has room (~17 GiB pool).

## Fix design (config-only, no image rebuild)

Three knob deltas, all live-patched in `runner.py` `_b300_forced` dict
inside the running image. No image rebuild required.

### A) `--gpu-memory-utilization`: 0.85 → 0.95

```
'--gpu-memory-utilization': '0.95'
```

Bumps PyTorch's headroom by ≈ 18 GiB / GPU so KV cache can be allocated
above the resident model footprint. Headroom for vision-encoder
workspace becomes 7 GiB free (vs negative at 0.75, 5.56 at 0.85).

### B) `--max-num-batched-tokens`: 131072 → 32768

```
'--max-num-batched-tokens': '32768'
```

Shrinks vision-encoder `profile_run` workspace from ~14 GiB to ~4 GiB.
Fits inside the 7 GiB post-model headroom with margin. Side-effect:
caps PoC v2 effective batch size at 32 on B200 (`32 × seq_len 1024 =
32768` token budget). Largest tested batch on B200 sweep is 32; B300
runs supported up to 128.

### C) `--enable-expert-parallel`: removed

Re-tested with EP both ways; EP added ~3 GiB of all-to-all overhead
per rank without helping the OOM (workspace remains 14 GiB without
vision pruning). The non-EP path also matches the chain-validator
reference layout used by older H100 / B200_old baselines.

The vision tower is **not** disabled (`--limit-mm-per-prompt` was
considered and rejected — operator preference is to keep the model
fully loaded).

### Other knobs left at the b200-k5-kimi-1 image bake

```
--tensor-parallel-size 4              # 4×B200 → 1 instance
--max-model-len 120000
--max-num-seqs 128
--logprobs-mode processed_logprobs
--compilation-config '{"mode": 0, "cudagraph_mode": "NONE"}'  # eager
--trust-remote-code
--attention-backend CUTLASS_MLA
ENV VLLM_USE_FLASHINFER_MOE_INT4=1    # Blackwell-only, sm_100, +138 % perf vs Marlin
```

## Validation

### Sanity (cold-state probe)

| Check | Result |
|---|---|
| 4× B200 SXM visible | ✓ 178.35 GiB / GPU, driver 590.48.01 |
| NVLink topology | ✓ NV18 full mesh between all 4 GPUs |
| NVLink per-link BW | ✓ 53.125 GB/s × 18 = 956 GB/s aggregate |
| PCIe link | ✓ Gen5 × 16 |
| Compute capability | ✓ sm_100 (Blackwell DC) |
| CUDA / torch / NCCL | ✓ 13.0 / 2.11.0+cu130 / NCCL 2.28.9 |
| /dev/shm / disk | ✓ 566 GiB / 605 GiB free |
| Image patches baked | ✓ TP=4, FlashInfer env, MLA fix, eager all on |

### Cold start (with tuned knobs)

| Step | Wall-clock |
|---|---:|
| Kimi-K2.6 download (HF Hub + `hf_transfer`) | 8 min (555 GB) |
| Model load to VRAM (TP=4) | 38 s |
| Per-rank ptxas JIT + DeepGEMM warmup | hidden inside loading window |
| Vision encoder `profile_run` (32k workspace) | ~10 s |
| First `inference_healthy=true` | **100 s end-to-end** |

### Phase-3 throughput sweep (sweep_kimi.py, callback aggregator)

| batch | nonces (30 s window) | nonces/min |
|---:|---:|---:|
|  8 | 864 | 1 728 |
| 16 | 1 040 | 2 080 |
| 32 | 1 120 | **2 240 ★** |

**Best: 2240 nonces/min @ batch=32** on 4×B200 single instance.

Comparison to B300 best-config (TP=4 single instance, FlashInfer eager):
`B200 / B300 ≈ 2240 / 2432 = 92 %` per-instance throughput. Per-GPU:
B200 = 560/GPU, B300 = 608/GPU → B200 is ~9 % slower per GPU, matching
the spec gap in HBM bandwidth (B300 has more).

### Canonical Gonka chain validation (mean L2 + binomtest)

Run via [`gonka-l2-validate`](../../../.claude/skills/gonka-l2-validate/SKILL.md)
skill (formulas lifted byte-for-byte from `vllm/poc/data.py` +
`vllm/poc/validation.py`).

| Pair | N | mean L2 | n_mismatch @ thr=0.4 | verdict (chain proto + calibrated p_mis=0.02) |
|---|---:|---:|---:|---|
| **B200_new ↔ B300 Marlin** | 988 | **0.1909** | 17 (1.72%) | **PASS** p=0.77 |
| B200_new ↔ H100 ref TP=16 | 1000 | 0.2048 | 16 (1.60%) | PASS p=0.85 |
| B200_new ↔ B200_old (vLLM 0.19) | 1000 | 0.1917 | 12 (1.20%) | PASS p=0.98 |

For comparison, the existing cross-pair baselines from earlier B300
work (all also PASS):

| Pair | mean L2 |
|---|---:|
| B300 FlashInfer ↔ B300 Marlin (self-arch, cross-kernel) | 0.187 |
| B300 FlashInfer ↔ B200_old | 0.189 |
| B300 Marlin ↔ B200_old | 0.193 |
| B300 FlashInfer ↔ H100 ref | 0.203 |
| B300 Marlin ↔ H100 ref | 0.205 |
| H100 ref ↔ B200_old | 0.206 |

**All mean L2 in 0.19-0.21 corridor.** This is the fundamental Kimi-K2.6
INT4 cross-anything floor: 384 routed experts × top_k=8 + discrete MoE
routing means a single-bit numeric perturbation at any layer flips the
expert set → activations diverge to the end of the network. Cross-arch
adds only ~0.02 on top of cross-kernel.

**At chain-default `dist_threshold=0.4` + calibrated `p_mismatch=0.02`
(per official PoC v2 validation report), every pair PASSES the binomial
fraud test.** The strict `0.02 / 0.001` defaults are intended for
self-validation on identical hardware and FAIL on every cross-anything
INT4 pair (this is expected).

### Sanity inference (FP-side regression probe)

`/v1/chat/completions` response to "What is the capital of France?
Answer in one short sentence.":

> *"The user is asking for the capital of France and wants the answer
> in one short sentence. ... The capital of France is Paris."*

Coherent, factual, no degradation from the FlashInfer MoE swap or the
small-batch budget.

## Migration guide

For operators on the bare `b200-k5-kimi-1` image facing the OOM at
cold start:

```bash
docker exec mlnode-308 sed -i \
  "s|'--gpu-memory-utilization': '0.85'|'--gpu-memory-utilization': '0.95'|" \
  /app/packages/api/src/api/inference/vllm/runner.py

docker exec mlnode-308 sed -i \
  "s|'--max-num-batched-tokens': '131072'|'--max-num-batched-tokens': '32768'|" \
  /app/packages/api/src/api/inference/vllm/runner.py

docker restart mlnode-308
```

Then POST `up/async` **without** `--enable-expert-parallel`:

```json
{
  "model": "moonshotai/Kimi-K2.6",
  "dtype": "auto",
  "additional_args": [
    "--trust-remote-code",
    "--attention-backend", "CUTLASS_MLA",
    "--max-num-seqs", "128",
    "--enforce-eager"
  ]
}
```

Cold start ≈ 100 s after this point. Vision tower remains loaded.

If `kaitakuai/mlnode` ships a **k6** image variant, the natural fix is
to bake these three deltas into `b200.py` `_b200_forced` so users don't
have to live-patch.

## Files

- [`artifacts/nonces_1000.json`](artifacts/nonces_1000.json) — 1120
  Kimi-K2.6 PoC v2 nonces collected at batch=32 on 4×B200, used as
  input to the canonical L2 validation
- [`artifacts/config.json`](artifacts/config.json) — collection metadata
  (GPU, vLLM version, batch_size, seq_len, k_dim, timestamp)
- [`artifacts/l2_5way.json`](artifacts/l2_5way.json) — canonical pairwise
  L2 + binomtest results across all 5 collected baselines (B200_new,
  B300_FI, B300_Marlin, H100_ref, B200_old)
- [`artifacts/l2_5way.png`](artifacts/l2_5way.png) — stacked histograms
  per pair with mean / median / 0.02 / 0.2 / 0.4 threshold lines
  (gonka-style)

## Related

- B300 perf attribution + skill design:
  [`../kimi_k26_b300_eager_flashinfer/README.md`](../kimi_k26_b300_eager_flashinfer/README.md)
- Canonical L2 skill:
  [`../../../.claude/skills/gonka-l2-validate/SKILL.md`](../../../.claude/skills/gonka-l2-validate/SKILL.md)
- FlashInfer MXINT4 MoE provenance: vLLM PR
  [#32437](https://github.com/vllm-project/vllm/pull/32437) (first shipped
  in 0.16.0, env `VLLM_USE_FLASHINFER_MOE_INT4=1`, sm_100 only)
- Watcher cold-start fix that's already in this image
  (k3 → k4 carry-over): [`../../2026-04/qwen235b-fp8-8xb300-watcher-cold-start-fix/README.md`](../../2026-04/qwen235b-fp8-8xb300-watcher-cold-start-fix/README.md)
