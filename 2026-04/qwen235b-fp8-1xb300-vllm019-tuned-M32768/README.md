# Qwen3-235B-A22B FP8 — 1×B300 — vLLM 0.19.0 — TUNED MoE for M=32768

**Date:** 2026-04-28 / 29
**Model:** `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`
**Quantization:** FP8 (block-wise w8a8, block_shape=[128,128])
**Hardware:** 1× NVIDIA B300 SXM6 AC (Blackwell Ultra, sm_103a, 275 GB HBM3e)
**vLLM:** 0.19.0 (Kaitaku PoC v2 build)
**MLNode:** 0.2.12 (image `ghcr.io/kaitakuai/mlnode-full:0.2.12-vllm0.19.0-h100-k1` + sm_103a fixes)

## Summary

This run pairs the **same baseline setup** with a `fused_moe` config tuned for the
real PoC workload (`M = batch × seq_len = 32 × 1024 = 32768`). Tuning ran
`benchmark_moe.py --batch-size 32768 --tune` for 2 h 57 min (10629 s) on the same
B300, exploring 640 Triton config candidates. The resulting config eliminated the
`Using default MoE config. Performance might be sub-optimal!` warning from vLLM
startup — but the throughput needle barely moved.

## Tuned MoE config

`E=128,N=1536,device_name=NVIDIA_B300_SXM6_AC,dtype=fp8_w8a8,block_shape=[128,128].json`:

```json
{
  "triton_version": "3.6.0",
  "32768": {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 256,
    "GROUP_SIZE_M": 16,
    "num_warps": 4,
    "num_stages": 4
  }
}
```

Saved at `artifacts/moe_configs/...` and cached at `.work/moe_configs/b300/...M32768.json`.

## Results

### Phase 3 (5×35 s sweep)

| Batch | Baseline (default MoE) | This run (tuned MoE) | Δ |
|---|---|---|---|
| 8 | 640 | **656** | **+2.5 %** |
| 16 | 800 | 800 | 0 |
| **32** | **832** | **832** | **0** ★ |
| 64 | 0 | 0 | — |
| 128 | 0 | 0 | — |

### Continuous collection (1000 nonces, batch=32, logprobs_count=0)

| | Baseline | Tuned |
|---|---|---|
| Nonces collected | 1024 | 1056 |
| Wall time (s) | 76.0 | 76.0 |
| **Throughput (n/min)** | **798** | **823** (+3 %) |

## Key observation

**MoE GEMM is not the bottleneck.** Triton autotune found a config marginally
better than vLLM's default heuristic for M=32768 (the actual shape PoC sends).
With the warning gone but no real win, the slow path is elsewhere — most likely
in:

- DeepGEMM block-quant linear layers (E8M0 path on B300 — vLLM 0.20.0 has a
  direct fix in #40552 for B200 that we should test)
- FlashInfer attention / TRTLLM prefill on sm_103a
- something in the vLLM 0.19 → 0.15 regression baseline

Confirmed via the M-logger patch (`fused_moe.py:try_get_optimal_moe_config`)
that PoC inference forwards through M values **8192, 16384, 32768** (one per
batch size 8/16/32 × seq_len 1024). CUDA-graph capture pre-warms M=1..256.

## Reproduction

Identical to the baseline experiment, plus:

```bash
# Capture real M values (one-time, after first vLLM start)
docker exec b300-bench python3 /tmp/patch_log_m.py
# Restart vLLM, run a short PoC bench, then:
docker exec b300-bench grep -aE "FUSED_MOE_M" /tmp/mlnode.log \
    | grep -oP "num_tokens=\d+" | sort -un

# Tune for the captured M (here 32768)
docker exec b300-bench /usr/bin/python3 \
    /vllm-workspace/benchmarks/kernels/benchmark_moe.py \
    --model /data/hf/Qwen3-235B-A22B-Instruct-2507-FP8 \
    --dtype fp8_w8a8 --tp-size 1 --batch-size 32768 \
    --tune --trust-remote-code \
    --save-dir /tmp/moe_configs

# Install + restart vLLM
docker exec b300-bench cp /tmp/moe_configs/*.json \
    /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs/
```

## Files

- `artifacts/nonces_1000.json` — 1056 nonces (823/min steady state)
- `artifacts/config.json` — collector config
- `artifacts/logprobs_100.json` — empty (`--logprobs-count 0`)
- `artifacts/mlnode.log` — full startup + serving log (no "default MoE config" warning)
- `artifacts/bench.log` — Phase 3 results
- `artifacts/moe_configs/E=128,N=1536,device_name=NVIDIA_B300_SXM6_AC,dtype=fp8_w8a8,block_shape=[128,128].M32768.json`

## Next steps

1. **vLLM 0.20.0 upgrade test** (highest expected ROI). Released 2026-04-27, includes
   #40552 "RMS norm + quant fusion fix on DeepGEMM UE8M0 path for B200" — directly
   in our regression suspect path. Quick to test via `pip install vllm==0.20.0`
   over the existing venv if dependencies cooperate.
2. **`--max-num-batched-tokens 32768`** in `runner.py` hardcodes — vLLM blog
   recommendation for GB300. Quick to test.
3. **Attention backend swap** — force FLASH_ATTN or TRITON_ATTN vs current
   auto-pick FLASHINFER+TRTLLM, measure delta.
4. **Profile with `nsys`** — find the actual bottleneck. Cost: setup time, but
   should give a sharper next-step direction than further blind tuning.
