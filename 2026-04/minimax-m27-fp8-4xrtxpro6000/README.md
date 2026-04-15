# PoC Benchmark: MiniMax-M2.7 (FP8) on 4×RTX PRO 6000 Blackwell

**Date:** 2026-04-14
**Model:** `MiniMaxAI/MiniMax-M2.7`
**Quantization:** FP8 block-wise 128×128 (E4M3, dynamic activations; `gate`, `e_score_correction_bias`, `lm_head` left unquantized)
**Hardware:** 4×NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GiB each)
**vLLM:** 0.19.0 + kaitakuai PoC v2 overlay (`mb/feat/port-pocv2-vllm-0.19` @ `f9477d1`)

## Summary

Standard PoC v2 nonce-generation benchmark. `MiniMax-M2.7` is shipped natively as FP8 (block-wise 128×128 W8A8) with a `MiniMaxM2ForCausalLM` MoE architecture (62 layers, 48 attention heads, 8 KV heads, max position 196 608). On-disk model footprint: 215 GiB across 125 safetensors shards.

The choice of FP8 MoE backend was forced by hardware/library compatibility:

| Backend | Status on this stack | Reason |
|---|---|---|
| `FLASHINFER_TRTLLM` / `FLASHINFER_CUTLASS` | ❌ rejected | FlashInfer FP8 MoE kernels (tested 0.6.6 and 0.6.7.post3) do not support the MiniMax 128×128 block-wise layout |
| `DEEPGEMM` | ❌ rejected | `deep_gemm` v2.1.1.post3 supports SM90 (Hopper) and SM100 (datacenter Blackwell B100/B200) but **not SM120** (workstation Blackwell, RTX PRO 6000) |
| **`TRITON`** | ✅ used | Universal Triton fallback; ~30–50 % slower than vendor kernels but the only working FP8 MoE path on SM120 with this layout in vLLM 0.19.0 |

## Hardware

| Parameter | Value |
|-----------|-------|
| GPU | 4× NVIDIA RTX PRO 6000 Blackwell Server Edition (97 887 MiB each, SM 12.0) |
| NVIDIA Driver | 580.95.05 (CUDA 13.0 capable) |
| CPU | Intel(R) Xeon(R) Platinum 8470Q (208 vCPUs) |
| RAM | 1.0 TiB total, 961 GiB available |
| Local disk (model) | `/dev/md0` XFS, 350 GiB total / 136 GiB free after model copy |
| Shared disk (model source) | shared FUSE mount, 2.0 TiB |
| Provider | partner GPU provider (bare-metal node) |

## Software

| Component | Version |
|-----------|---------|
| vLLM | 0.19.0 (Python wheel from PyPI) |
| PoC v2 overlay (vLLM) | `kaitakuai/vllm` `mb/feat/port-pocv2-vllm-0.19` @ `f9477d1` |
| MLNode packages (api/common/pow/train) | `kaitakuai/rtx-pro-6000` `vendor/gonka-source/mlnode/packages` |
| torch | 2.10.0+cu128 |
| triton | 3.6.0 |
| flashinfer-python | 0.6.7.post3 |
| flashinfer-cubin | 0.6.7.post3 |
| Python | 3.12.3 (Miniconda) |
| OS | Ubuntu 22.04.5 LTS |

### Environment quirks (must be applied at image build / before `vllm serve`)

- **torch 2.10.0 duplicate `TritonTemplate` files** — the wheel ships three legacy `*.py` files alongside their successors that re-register the same kernel names. Importing `torch._inductor.lowering` then crashes with `AssertionError: duplicate template name`, breaking any vLLM inference. Remove them once at image-prep time:
  ```bash
  rm -f /root/miniconda3/lib/python3.12/site-packages/torch/_inductor/kernel/{flex_attention,flex_decoding,mm_scaled_grouped}.py
  ```
  This fix is built into our golden `vllm0.19.0-poc` image (see `gonka-deploy/image/rtx_pro_6000/stage1_normalize.sh` step 3/5).

- **`VLLM_USE_FLASHINFER_MOE_FP8` and `VLLM_USE_FLASHINFER_MOE_FP4` must NOT be set** for MiniMax-M2.7 — both backends reject the layout. The default `/etc/profile.d/vllm-poc.sh` enables them, so they have to be unset for this model.

## Reproduction

### 0. Provider-side prep (golden image)

Use the `vllm0.19.0-poc` golden image built per `gonka-deploy/image/rtx_pro_6000/stage{1,2}_*.sh`. The image already includes:
- vLLM 0.19.0 + PoC v2 overlay applied to `site-packages/vllm/`
- torch 2.10.0 duplicate `TritonTemplate` files removed
- `flashinfer-cubin` 0.6.6 (precompiled cubins, no JIT-on-first-run)
- `/etc/profile.d/vllm-poc.sh` with `VLLM_ALLOW_INSECURE_SERIALIZATION=1`

If working from a fresh stock image, replay [`stage1_normalize.sh`](../../../gonka-deploy/image/rtx_pro_6000/stage1_normalize.sh) and [`stage2_poc_patch.sh`](../../../gonka-deploy/image/rtx_pro_6000/stage2_poc_patch.sh).

### 1. Download model

`MiniMax-M2.7` is **215 GiB** in 125 safetensors shards. The most reliable recipe on a bare-metal node with slow / flaky network to HF:

```bash
# install hf_transfer for parallel chunked download (5–10× faster on good links)
uv pip install --python /root/miniconda3/bin/python3.12 hf_transfer

# write a tiny retry-loop wrapper (huggingface-cli sometimes fails HEAD calls after long sessions)
cat > /root/hf_retry.sh <<'EOF'
#!/bin/bash
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=hf_xxx               # <- your token; without it HF egress is even slower
attempt=0
while true; do
  attempt=$((attempt+1))
  echo "=== attempt $attempt $(date -Iseconds) ===" >> /root/hf_download.log
  /root/miniconda3/bin/huggingface-cli download MiniMaxAI/MiniMax-M2.7 \
      --local-dir /mnt/shared/MiniMaxAI/MiniMax-M2.7 \
      --max-workers 8 \
      --token "$HF_TOKEN" >> /root/hf_download.log 2>&1
  rc=$?
  echo "=== attempt $attempt exited rc=$rc ===" >> /root/hf_download.log
  [ $rc -eq 0 ] && break
  sleep 10
done
EOF
chmod +x /root/hf_retry.sh
nohup /root/hf_retry.sh >/dev/null 2>&1 & disown
```

Wall time observed: **~3 h 30 min** total, ~16–34 MiB/s, one HEAD-timeout retry. Save to the shared FUSE mount for reuse across follow-up runs.

### 2. Copy model to local NVMe (10× faster vLLM load)

```bash
mkdir -p /root/local-nvme/models/MiniMaxAI
nohup rsync -a --info=progress2 \
    /mnt/shared/MiniMaxAI/MiniMax-M2.7/ \
    /root/local-nvme/models/MiniMaxAI/MiniMax-M2.7/ \
    > /root/rsync.log 2>&1 & disown
```

Wall time observed: **22 min 56 s** at sustained 159 MB/s (FUSE→local XFS). Final size 215 GiB.

### 3. Install MLNode packages

```bash
# Build tarball from kaitakuai/rtx-pro-6000 submodule and ship to instance
cd gonka-deploy/image/rtx_pro_6000
tar czf /tmp/mlnode_packages.tar.gz \
    -C vendor/gonka-source/mlnode/packages api common pow train
scp /tmp/mlnode_packages.tar.gz root@<instance>:/tmp/

# On instance:
mkdir -p /app/packages && cd /app/packages && tar xzf /tmp/mlnode_packages.tar.gz

# Extra Python deps required by MLNode
uv pip install --python /root/miniconda3/bin/python3.12 \
    toml accelerate fire fastrlock h2 termcolor typer-slim setuptools-scm tenacity

# Patches (same as Kimi recipe)
sed -i 's/MAX_UNHEALTHY_COUNT = 3/MAX_UNHEALTHY_COUNT = 9999/' \
    /app/packages/api/src/api/watcher.py
sed -i 's|VLLM_PYTHON_PATH = "/usr/bin/python3.12"|VLLM_PYTHON_PATH = "/root/miniconda3/bin/python3.12"|' \
    /app/packages/api/src/api/inference/vllm/runner.py
sed -i 's/env\["VLLM_USE_V1"\] = "0"/env["VLLM_USE_V1"] = "1"/' \
    /app/packages/api/src/api/inference/vllm/runner.py
```

### 4. Re-apply PoC v2 overlay (only needed if vllm wheel was reinstalled)

If anything ran `pip install --force-reinstall vllm==0.19.0` after the golden image was built, the patches in `entrypoints/openai/api_server.py`, `sampling_params.py`, `v1/sample/sampler.py`, etc. were reverted. The freshly-added `vllm/poc/` directory survives, but `include_router(poc_router)` in `api_server.py` does not — and without it the `/api/v1/pow/*` routes 404.

```bash
# ship the vendor/vllm/vllm tree from kaitakuai/rtx-pro-6000
cd gonka-deploy/image/rtx_pro_6000/vendor/vllm
find vllm -name '*.py' -print0 | tar -cz --null -T - -f /tmp/vllm_overlay.tgz
scp /tmp/vllm_overlay.tgz root@<instance>:/tmp/

# On instance:
cd /tmp && tar xzf vllm_overlay.tgz && cd vllm
find . -name '*.py' | tar -cf - -T - | \
    tar -xf - -C /root/miniconda3/lib/python3.12/site-packages/vllm/

# Sanity check
grep -n include_router \
    /root/miniconda3/lib/python3.12/site-packages/vllm/entrypoints/openai/api_server.py
# Expect: line 251: app.include_router(poc_router)
```

### 5. Start MLNode

`VLLM_ALLOW_INSECURE_SERIALIZATION=1` is required for PoC v2 with TP>1. `SAFETENSORS_FAST_GPU=1` shaves ~5 s off the load. **Do not set `VLLM_USE_FLASHINFER_MOE_FP{4,8}`** — they will crash startup on this layout.

```bash
cd /app/packages/api/src
VLLM_ALLOW_INSECURE_SERIALIZATION=1 SAFETENSORS_FAST_GPU=1 \
PYTHONPATH="/app:/app/packages/api/src:/app/packages/pow/src:/app/packages/train/src:/app/packages/common/src" \
nohup /root/miniconda3/bin/python3.12 -m uvicorn api.app:app \
    --host 0.0.0.0 --port 8081 --app-dir /app/packages/api/src \
    > /var/log/mlnode.log 2>&1 & disown
```

### 6. Start vLLM via MLNode

```bash
curl -s -X POST http://127.0.0.1:8081/api/v1/inference/up/async \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/root/local-nvme/models/MiniMaxAI/MiniMax-M2.7",
    "dtype": "auto",
    "additional_args": [
      "--served-model-name", "MiniMaxAI/MiniMax-M2.7",
      "--tensor-parallel-size", "4",
      "--gpu-memory-utilization", "0.92",
      "--max-num-seqs", "128",
      "--max-model-len", "32768",
      "--trust-remote-code",
      "--enable-auto-tool-choice",
      "--tool-call-parser", "minimax_m2",
      "--reasoning-parser", "minimax_m2_append_think"
    ]
  }'

# Wait for {"is_running":true,...}
while true; do
  s=$(curl -s http://127.0.0.1:8081/api/v1/inference/up/status | grep -o '"is_running":[a-z]*')
  echo "$s"; [ "$s" = '"is_running":true' ] && break; sleep 10
done

# Sanity: PoC routes must answer 200
curl -s -o /dev/null -w 'pow/status %{http_code}\n' \
    http://127.0.0.1:5001/api/v1/pow/status
```

### 7. Patch and run benchmark

```bash
sed -i 's|MLNODE_URL = "http://127.0.0.1:8080"|MLNODE_URL = "http://127.0.0.1:8081"|' \
    run_pow_generation.py
sed -i 's|MODEL_NAME = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"|MODEL_NAME = "MiniMaxAI/MiniMax-M2.7"|' \
    run_pow_generation.py

export HOST_IP=127.0.0.1
python3 -u run_pow_generation.py --skip-check
```

(`--skip-check` bypasses Phase 0 because vLLM is already up via MLNode.)

## Startup profile

Measured on the second hot-cache start (first cold start is dominated by FUSE/disk reads):

| Phase | Duration |
|-------|----------|
| Model loading (from local NVMe) | 29.81 s |
| Model memory / GPU (after weights) | 56 GiB × 4 |
| Memory profile + KV-cache allocation | ~10 s |
| Warmup model | included in `init engine` below |
| `init engine (profile, create kv cache, warmup model)` | 40.59 s |
| Available KV cache | 29.19 GiB → **493 712 tokens** |
| **Total cold start** (MLNode → vLLM ready) | **~100 s** |
| First inference batch after warmup | additional cold-start; benchmark auto-detects empty 5 s warmup and restarts generation |
| Steady-state GPU memory | 91.7 GiB / 96 GiB on each GPU |

## Results

Two parallelism configurations were measured against identical PoC v2 workloads on the same instance: **TP=4** (`--tensor-parallel-size 4 --pipeline-parallel-size 1`) and **PP=4** (`--tensor-parallel-size 1 --pipeline-parallel-size 4`).

### Phase 3 — batch-size sweep (30 s measurement window per batch, 5 s warmup discarded)

| Batch Size | TP=4 nonces/min | PP=4 nonces/min |
|-----------:|----------------:|----------------:|
| 2 | 680 | 516 |
| 8 | **848 ★** | 768 |
| 16 | 800 | 768 |
| 32 | 768 | **768 ★** |
| 64 | 0 (OOM) | 768 |

- **TP=4 best:** batch=8 → **848 nonces/min** (peak), but engine OOMs at batch=64.
- **PP=4 best:** batch=32 (or any of 8/16/32/64) → **768 nonces/min** (flat plateau, no OOM through batch=64).
- TP=4 is ~10 % faster at its peak; PP=4 is more robust under high concurrency and uses less GPU memory (~83 GiB vs ~91 GiB per GPU because PP avoids attention-weight duplication).

### Phase 1 — generation + self-validation

| Configuration | Async batch=8 throughput | Sync→Sync determinism | Async→Sync replay |
|---|---|---|---|
| **TP=4** | 424 nonces / 848/min | 20/20, **0 mismatches** | 424/424, **0 mismatches**, p_value = 1.0 |
| **PP=4** | 384 nonces / 768/min | 20/20, **0 mismatches** | 384/384, **0 mismatches**, p_value = 1.0 |

### Phase 2 — fraud detection (both configurations)

| Test | Vectors | Expectation | TP=4 | PP=4 |
|---|---|---|---|---|
| Fresh honest (sync→sync) | 32 | validation passes | ✅ passed | ✅ passed |
| INT4-quantized fraud vectors | 32 | validation rejects | ✅ rejected | ✅ rejected |

## Backend trace (from vLLM startup log)

```
Using TRITON Fp8 MoE backend out of potential backends:
    [AITER, FLASHINFER_TRTLLM, FLASHINFER_CUTLASS, DEEPGEMM,
     TRITON, MARLIN, BATCHED_DEEPGEMM, BATCHED_TRITON, XPU]
Using FLASH_ATTN attention backend out of potential backends:
    [FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION]
Selected CutlassFP8ScaledMMLinearKernel for Fp8LinearMethod
PoC engine patch applied successfully to AsyncLLM (V1)
```

`AsyncLLM (V1)` here refers to vLLM's V1 engine architecture, not PoC v1. Both MLNode router (`api/inference/pow_v2_routes.py`, tag `"PoC v2"`) and vLLM overlay (`vllm/poc/routes.py` from `mb/feat/port-pocv2-vllm-0.19`) are **PoC v2**.

## Key observations

- **TRITON is the only working FP8 MoE backend on SM120 with MiniMax's 128×128 block-wise layout in vLLM 0.19.0** (applies equally to TP and PP). Both FlashInfer FP8 MoE paths reject the configuration (verified on FlashInfer 0.6.6 and 0.6.7.post3); DeepGEMM rejects the device entirely (no SM120 support yet, only SM90/SM100). This caps achievable throughput at ~30–50 % below what vendor-tuned kernels would deliver on the same silicon.
- **TP=4 vs PP=4 trade-off:**
  - TP=4 wins peak throughput by ~10 % (848 vs 768 nonces/min) but **OOMs at batch=64** because TP duplicates attention buffers across all 4 GPUs (~91 GiB used per GPU, 5 GiB headroom).
  - PP=4 is throughput-flat from batch=8 to batch=64 (768 nonces/min) and uses only ~83 GiB per GPU; **no OOM** at any tested batch. Better choice for sustained high-concurrency or anywhere the workload may spike past batch=32.
  - Both configurations preserve PoC v2 determinism end-to-end (p_value = 1.0 on Async→Sync replay).
- **Throughput plateau** at batch=8 (TP) or batch=8+ (PP) is consistent with the kimi-k25-int4-4×B200 baseline (sweet spot at 32–64 there because B200 has more KV headroom and faster INT4 kernels). For RTX PRO 6000 Blackwell + FP8, batch=8 is the practical maximum on TP, batch=32 on PP.
- Comparison with kimi-k25-int4-4×B200 (1024 nonces/min): MiniMax-M2.7 / RTX PRO 6000 Blackwell achieves **~83 % of B200's PoC throughput** on TP=4 (~75 % on PP=4) despite using a smaller workstation Blackwell — better than the raw FLOPs ratio would suggest, primarily because PoC v2 is bound by MoE expert dispatch latency, not raw matmul.
- **PoC v2 self-validation passes byte-for-byte** (`p_value = 1.0`) on both TP and PP — the TRITON FP8 MoE path is numerically deterministic across async and sync execution, which is the property PoC v2 actually requires. The TRITON throughput penalty does not affect correctness, only speed.

## Artifacts

- [`artifacts/nonces_1000.json`](artifacts/nonces_1000.json) + [`artifacts/config.json`](artifacts/config.json) — 1 056 PoC nonce vectors (`seq_len=1024`, `k_dim=12`) collected on **TP=4** with `batch_size=8`. Sustained rate **823 nonces/min** (77 s) — within 3 % of Phase-3 peak 848/min, delta is the per-batch HTTP callback overhead.
- [`artifacts/nonces_1000_pp4.json`](artifacts/nonces_1000_pp4.json) + [`artifacts/config_pp4.json`](artifacts/config_pp4.json) — 1 056 PoC nonce vectors collected on **PP=4** with `batch_size=32`. Sustained rate **773 nonces/min** (82 s) — within 1 % of Phase-3 plateau 768/min.
- [`artifacts/compressa-perf.sqlite`](artifacts/compressa-perf.sqlite) and [`artifacts/compressa-results.txt`](artifacts/compressa-results.txt) — see [`compressa-perf.md`](compressa-perf.md) for the inference benchmark (TP=4 only).

Reproduction (TP=4):

```bash
python3 -u collect_artifacts.py \
    --url http://127.0.0.1:8081 \
    --model 'MiniMaxAI/MiniMax-M2.7' \
    --output-dir /root/poc_artifacts \
    --nonces 1000 --batch-size 8 --logprobs-count 0 \
    --gpu 'RTX PRO 6000 Blackwell' \
    --vllm-version '0.19.0+poc-overlay-f9477d1'
```

Reproduction (PP=4 — bring vLLM up via MLNode with `--tensor-parallel-size 1 --pipeline-parallel-size 4`, then):

```bash
python3 -u collect_artifacts.py \
    --url http://127.0.0.1:8081 \
    --model 'MiniMaxAI/MiniMax-M2.7' \
    --output-dir /root/poc_artifacts_pp4 \
    --nonces 1000 --batch-size 32 --logprobs-count 0 \
    --gpu 'RTX PRO 6000 Blackwell' \
    --vllm-version '0.19.0+poc-overlay-f9477d1 (PP=4 TP=1)'
```

Note: `--url` points to **MLNode** on `:8081`, not vLLM's `:5001`. The script hits MLNode endpoints (`/api/v1/inference/pow/init/generate`, `/stop`); MLNode proxies to vLLM's PoC v2 routes (`/api/v1/pow/*`). It also opens a callback HTTPServer on port `9998` to receive the streamed nonce batches (different from the `9999` used by `run_pow_generation.py`, so the two can coexist).
