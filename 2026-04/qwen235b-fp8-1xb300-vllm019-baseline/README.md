# Qwen3-235B-A22B FP8 — 1×B300 — vLLM 0.19.0 — BASELINE

**Date:** 2026-04-26
**Model:** `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`
**Quantization:** FP8 (block-wise w8a8, block_shape=[128,128])
**Hardware:** 1× NVIDIA B300 SXM6 AC (Blackwell Ultra, sm_103a, 275 GB HBM3e)
**vLLM:** 0.19.0 (Kaitaku PoC v2 build, base `kaitakuai/vllm:v0.19.0-pocv2-alpha3`)
**MLNode:** 0.2.12 (image `ghcr.io/kaitakuai/mlnode-full:0.2.12-vllm0.19.0-h100-k1` + B300 sm_103a fixes)

## Summary

This is the **baseline** for Qwen3-235B-A22B FP8 on a single B300 with vLLM 0.19. The
H100 image (`mlnode-full:0.2.12-vllm0.19.0-h100-k1`) was loaded as-is on the B300 host
and patched in-place with the same five sm_103a compatibility fixes that ship in the
existing `mlnode-overlay:*-b300-k2` (which is built on vLLM 0.15). No vLLM rebuild
was performed.

Best-batch throughput: **832 nonces/min** at `batch_size=32`. Continuous collection
of 1024 nonces yielded **798 nonces/min**. This is **~19% below the user-reported
1024 nonces/min** observed on the same hardware with vLLM 0.15 — a regression to
investigate.

Likely root cause is the missing MoE autotune config for B300 (vLLM logs
`Using default MoE config. Performance might be sub-optimal!`). DeepGEMM is also
enabled despite `VLLM_MOE_USE_DEEP_GEMM=0` being injected into the subprocess env
by the runner; the env-var path either is not honored in 0.19 or is overridden by
the vLLM platform check.

## Hardware

| Component | Value |
|---|---|
| GPU | 1× NVIDIA B300 SXM6 AC (sm_103a) |
| VRAM | 275040 MiB |
| Driver | 580.126.09 |
| Host CUDA | 13.0 |
| Image CUDA | 12.9.1 |
| Host RAM | 270 GiB |
| Disk | 400 GB virtual disk |

## Software

| Component | Version |
|---|---|
| Image base | `ghcr.io/kaitakuai/mlnode-full:0.2.12-vllm0.19.0-h100-k1` |
| OS (image) | Ubuntu (Python 3.12.13) |
| torch | 2.10.0+cu129 |
| torch arch list | sm_70, sm_75, sm_80, sm_86, sm_90, sm_100, sm_120 (no sm_103) |
| vllm | 0.19.0 |
| flashinfer | 0.6.6 (jit-cache uninstalled, JIT-compiled on first run) |
| Triton ptxas | replaced with CUDA 12.9 ptxas |

## Reproduction

```bash
# 0. host setup (B300 instance, Ubuntu 24.04, NVIDIA driver 580, docker + nvidia-runtime)
docker pull ghcr.io/kaitakuai/mlnode-full:0.2.12-vllm0.19.0-h100-k1

# model on host (~221 GB)
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
    Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
    --local-dir /root/hf/Qwen3-235B-A22B-Instruct-2507-FP8

# container with overridden entrypoint so we can patch before MLNode starts
docker run -d --name b300-bench --gpus all --ipc=host --shm-size=32g \
    -v /root/hf:/data/hf:ro \
    -p 8081:8081 -p 5001:5001 \
    --entrypoint /bin/bash \
    ghcr.io/kaitakuai/mlnode-full:0.2.12-vllm0.19.0-h100-k1 -c "sleep infinity"

# B300 sm_103a fixes (same as overlay/hw/b300/Dockerfile, applied at runtime)
docker exec b300-bench bash -c '
    cp /usr/local/cuda-12.9/bin/ptxas /usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/ptxas
    pip uninstall -y -q flashinfer-jit-cache
    for d in /usr/local/lib/python3.12/dist-packages/nvidia/*/include/; do
      for f in "$d"*.h; do
        [ -f "$f" ] && ln -sf "$f" "/usr/local/cuda/include/$(basename "$f")"
      done
    done
    rm -f /usr/local/cuda-*/compat/libcuda.so*
    ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/local/cuda-12.9/compat/libcuda.so
'

# MLNode patches
docker exec b300-bench sed -i 's/MAX_UNHEALTHY_COUNT = 3/MAX_UNHEALTHY_COUNT = 9999/' \
    /app/packages/api/src/api/watcher.py
# subprocess env:  VLLM_USE_V1=1 + VLLM_USE_FLASHINFER_MOE_FP8=0 + VLLM_MOE_USE_DEEP_GEMM=0
# (see patches.sh for python-based runner.py rewrite)
# B300 hardcodes (force gpu_mem_util=0.95, max_model_len=131072, logprobs_mode=processed_logprobs;
# default tensor_parallel_size=1, max_num_seqs=128) — same payload as overlay/hw/b300/Dockerfile

# start MLNode
docker exec -d b300-bench bash -c '
    source /app/packages/api/.venv/bin/activate
    PYTHONPATH=/app:/app/packages/api/src nohup python -m uvicorn api.app:app \
        --host 0.0.0.0 --port 8081 --app-dir /app/packages/api/src > /tmp/mlnode.log 2>&1
'

# bring vLLM up via MLNode API (additional_args minimal; rest is hardcoded)
curl -s -X POST http://127.0.0.1:8081/api/v1/inference/up/async \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "/data/hf/Qwen3-235B-A22B-Instruct-2507-FP8",
        "dtype": "auto",
        "additional_args": [
            "--served-model-name", "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
            "--trust-remote-code"
        ]
    }'

# benchmark (run_pow_generation.py with batch sizes [8, 16, 32, 64, 128] and patched
# MLNODE_URL → http://127.0.0.1:8081)
HOST_IP=127.0.0.1 python -u /tmp/run_pow_generation.py --phase 3 --skip-check

# nonce collection (1024 nonces at safe batch_size=32)
python -u /tmp/collect_artifacts.py \
    --url http://127.0.0.1:8081 \
    --model Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
    --output-dir /tmp/artifacts \
    --nonces 1000 --batch-size 32 --logprobs-count 0 \
    --gpu B300_SXM6 --vllm-version 0.19.0
```

## Startup profile

| Phase | Time |
|---|---|
| Weight load (220.35 GiB into VRAM) | 57.7 s |
| Initial profiling/warmup | 11.1 s |
| Dynamo bytecode transform | 7.5 s |
| Graph compile (range 1..32768) | 7.5 s |
| `torch.compile` total | 18.1 s |
| DeepGEMM warmup (1404 kernels) | 21 s |
| FlashInfer JIT autotune | 1.3 s |
| CUDA graph capture (PIECEWISE 35 + FULL 19) | ~10 s |
| **Cold start total (load → ready)** | **~170 s** |

VRAM after load: 229.5 GB used → 44.6 GB free.
After KV cache + CUDA graphs allocation: 271 GB used → 3 GB free.

KV cache:
- Available: **27.33 GiB**
- Pool size: **152,432 tokens**
- Concurrency at 131,072-token requests: **1.16×**

Backends picked:
- attention: `FLASHINFER` with HND KV layout, TRTLLM prefill (auto-detected)
- MoE FP8: `MoEPrepareAndFinalizeNoDPEPModular` (TRITON path)
- DeepGEMM: **enabled** (E8M0, despite `VLLM_MOE_USE_DEEP_GEMM=0` in subprocess env — possible no-op in 0.19)

⚠️ **Warning observed in startup log:**
```
WARNING [fused_moe.py:1090] Using default MoE config. Performance might be sub-optimal!
Config file not found at .../E=128,N=1536,device_name=NVIDIA_B300_SXM6_AC,dtype=fp8_w8a8,block_shape=[128,128].json
```

## Results — Phase 3 (5×35 s batch sweep)

| Batch | Nonces (30 s) | Nonces/min |
|---|---|---|
| 8 | 320 | 640 |
| 16 | 400 | 800 |
| **32** | **416** | **832** ★ |
| 64 | 0 | 0 (PoC engine stuck) |
| 128 | 0 | 0 (PoC engine stuck) |

**Best batch:** 32. **Best throughput:** 832 nonces/min.

Batch 64 / 128 returned 0 nonces — the PoC engine got stuck rather than OOM (vLLM
itself stayed alive and continued to serve `/v1/chat/completions` afterwards). This is
the documented behaviour for tight-memory configurations in the `poc-benchmark` agent
playbook.

## Results — continuous collection

| Metric | Value |
|---|---|
| Nonces collected | 1024 |
| Wall time | 76.0 s |
| Effective throughput | **798 nonces/min** |
| Batch size | 32 |
| `logprobs_count` | 0 |

Steady-state throughput is slightly below the peak benchmark sample (832 → 798) which
matches the typical 30-second-window vs. 76-second-window variance.

## Comparison

| Configuration | Throughput | Δ vs vLLM 0.15 baseline |
|---|---|---|
| **This run** — 1× B300, vLLM 0.19, mlnode-full h100→b300 patches | **832 / min** | **−19 %** |
| 1× B300, vLLM 0.15 (user prior measurement) | 1024 / min | (reference) |
| 4× H100 SXM, Qwen3-235B FP8, TRITON MoE (project memory) | 1388 / min | — (≈ 347 / GPU) |

So **B300 ≈ 2.4× single H100** at 832 / min, but we are leaving ~19 % on the table
relative to the 0.15 measurement on the same silicon.

## Key observations

1. **MoE config missing for B300** — biggest suspected cost. vLLM ships pretuned
   `E=128,N=1536,device_name=...,dtype=fp8_w8a8,block_shape=[128,128].json` for H100,
   B200, etc., but not for B300_SXM6_AC. The default heuristic config is in use,
   warned by `fused_moe.py:1090`. Generating this file via vLLM's MoE benchmark
   tuner is the first optimization step.
2. **DeepGEMM unconditionally on.** Setting `VLLM_MOE_USE_DEEP_GEMM=0` in the
   subprocess env did not prevent the DeepGEMM warmup from running. Need to confirm
   whether 0.19 still honors that variable, and whether the warmup means DeepGEMM
   is actually used at runtime (vs. just measured).
3. **TRTLLM prefill auto-selected** — same as on H100 with this image; not B300-specific.
4. **CUDA graph capture is wide** — 35 PIECEWISE + 19 FULL graphs, max capture size 256.
   This is FULL_AND_PIECEWISE mode (vLLM 0.19 default). Worth comparing to PIECEWISE-only
   on later runs.
5. **PoC engine stuck at batch ≥ 64** — same pattern documented by the
   `poc-benchmark` agent for tight-memory configs. Not a vLLM crash.
6. **TORCH_CUDA_ARCH_LIST does not include sm_103** — only `sm_100, sm_120`. Torch
   kernels are forward-compatible via PTX from sm_100; we have not yet measured
   whether a torch rebuild with `10.3+PTX` would close any gap.

## Files

- `artifacts/nonces_1000.json` — 1024 collected nonces (batch=32)
- `artifacts/config.json` — collector configuration
- `artifacts/logprobs_100.json` — empty (collected with `--logprobs-count 0`)
- `artifacts/mlnode.log` — full MLNode/vLLM startup + serving log

## Next steps (optimization phase)

1. **Tune the missing MoE config for B300** via `python -m vllm.benchmarks.kernels.benchmark_moe --tune`,
   save result into `vllm/model_executor/layers/fused_moe/configs/E=128,N=1536,device_name=NVIDIA_B300_SXM6_AC,dtype=fp8_w8a8,block_shape=[128,128].json`,
   re-run benchmark. Expect biggest single win (~10–20 %).
2. **Force TRITON over DeepGEMM** properly. Investigate the 0.19 selection logic
   in `vllm/model_executor/layers/fused_moe/` and pick the working knob (env, code patch,
   or `--quantization-param-path`).
3. **Compare attention backends.** The H100 image auto-picks FLASHINFER+TRTLLM. Try
   forcing FLASH_ATTN or TRITON_ATTN to measure the cost.
4. **Compare CUDA graph modes.** `cudagraph_mode=PIECEWISE` only vs. current
   `FULL_AND_PIECEWISE`.
5. **Repeat the 0.15 measurement on the same host** as a sanity-check that the
   1024 / min reference is reproducible.
