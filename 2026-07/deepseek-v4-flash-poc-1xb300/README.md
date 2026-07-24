# PoC Benchmark + Cross-Implementation Validation: DeepSeek-V4-Flash on 1×B300

**Date:** 2026-07-24
**Model:** `deepseek-ai/DeepSeek-V4-Flash`
**Hardware:** 1×NVIDIA B300 SXM6 (275,040 MiB)
**vLLM:** 0.25.1 (Gonka PoC-v2 fork, forward-ported from 0.20.0)

## Summary

Two results for DeepSeek-V4-Flash PoC-v2 on vLLM 0.25.1, single B300, TP=1:

1. **Throughput sweep** in the realistic serving config (compiled `mode=3` +
   `FULL_AND_PIECEWISE` CUDA graphs, `max_model_len=400000`): **1472 nonces/min** at
   batch_size 32.
2. **Cross-implementation consensus**: the V4 PoC nonces produced by three independent
   implementations of the same v2 (murmur3) scheme — the out-of-tree plugin, the earlier
   in-tree 0.23→0.25 port, and this 0.25.1 forward-port — are near-identical (median
   pairwise L2 ≈ 0.002) and pass the chain fraud test against each other (binomial
   p ≤ 2e-44). The forward-ported V4 path is consensus-safe.

DeepSeek-V4-Flash needs three PoC-specific fixes on top of the generic PoC forward
(per-group attention metadata for its heterogeneous KV block sizes, `positions` threaded
into `CommonAttentionMetadata`, and deterministic pseudo `input_ids` for its hash-routed
MoE). The vLLM code is at
[gonka-ai/vllm#65](https://github.com/gonka-ai/vllm/pull/65), branch tip
[`7aabb623b5d4c281b1e718ba50b34db9c3a9e358`](https://github.com/gonka-ai/vllm/commit/7aabb623b5d4c281b1e718ba50b34db9c3a9e358);
the V4 support commit is
[`b87895f6a670e54fc168a2f499fb7889f964036e`](https://github.com/gonka-ai/vllm/commit/b87895f6a670e54fc168a2f499fb7889f964036e).

## Hardware

| Parameter | Value |
|-----------|-------|
| GPU | 1× NVIDIA B300 SXM6 (275,040 MiB) |
| NVIDIA Driver | 580.126.09 |
| CPU | AMD EPYC 9575F 64-Core (240 vCPUs) |
| RAM | 2,007 GB |

## Software

| Component | Version |
|-----------|---------|
| vLLM | 0.25.1 (Gonka PoC-v2 fork) |
| torch | 2.11.0+cu130 |
| Python | 3.12.13 |
| OS | Ubuntu 24.04.4 LTS |
| Base image | `vllm/vllm-openai:v0.25.1` |

## Reproduction

### 1. Download model

```bash
hf download deepseek-ai/DeepSeek-V4-Flash
```

### 2. Build the PoC image

Overlay the fork's Python onto the stock vLLM image (preserves compiled `.so`):

```bash
git clone https://github.com/gonka-ai/vllm && cd vllm
git fetch origin pull/65/head && git checkout 7aabb623b5d4c281b1e718ba50b34db9c3a9e358
docker build -f Dockerfile.quick \
  --build-arg BASE_IMAGE=vllm/vllm-openai:v0.25.1 \
  -t vllm-pocv2-v4:0.25.1 .
```

### 3. Start vLLM (realistic serving config)

```bash
docker run -d --gpus '"device=0"' --shm-size=32g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint bash vllm-pocv2-v4:0.25.1 -c \
  "python3 -m vllm.entrypoints.openai.api_server \
     --model deepseek-ai/DeepSeek-V4-Flash --trust-remote-code \
     --kv-cache-dtype fp8 --gpu-memory-utilization 0.90 \
     --max-model-len 400000 --max-num-batched-tokens 32768 \
     --compilation-config '{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"]}' \
     --host 0.0.0.0 --port 8000"
```

`--kv-cache-dtype fp8` is required for V4 (FlashMLA path). PoC forward runs
`skip_compiled=True`, so the compiled/eager choice does not change PoC throughput; the
compiled config is used so the run matches the fleet's real serving deployment.

### 4. Run the batch-size sweep

`scripts/run_pow_generation.py` targets the in-tree PoC routes (`/api/v1/pow/*`) on the
vLLM port. Run it inside the container so the callback receiver is reachable:

```bash
docker exec -e HOST_IP=127.0.0.1 <container> \
  python3 /path/scripts/run_pow_generation.py --phase 3 --skip-check
```

### 5. Collect nonces

```bash
docker exec <container> python3 /path/scripts/collect_artifacts.py \
  --url http://127.0.0.1:8000 --model deepseek-ai/DeepSeek-V4-Flash \
  --output-dir /tmp/arts --nonces 1000 --batch-size 8 --logprobs-count 0
```

### 6. Cross-implementation L2

```bash
python3 scripts/l2_crossval.py \
  artifacts/nonces_reference_plugin.json artifacts/nonces_1000.json
```

## Startup profile

| Phase | Value |
|-------|-------|
| Model weights | ~149 GiB |
| GPU KV cache (fp8) | 2,604,694 tokens |
| Max concurrency @ 400k ctx | ~6.5 sequences |
| Compilation | FlashInfer MoE autotune + CUDA graph capture |

## Results — throughput sweep

Compiled `mode=3` + `FULL_AND_PIECEWISE`, `max_model_len=400000`, seq_len 1024,
5 s warmup + 30 s measure:

| Batch Size | Nonces (30 s) | Nonces/min |
|-----------:|--------------:|-----------:|
| 8 | 624 | 1248 |
| 16 | 720 | 1440 |
| **32** ★ | **736** | **1472** |

**Best: batch=32 → 1472 nonces/min.** Throughput is nearly batch-flat (+18% from b8 to
b32) — the V4 PoC forward is compute-bound. This is ~1.6× the 0.20.0-image baseline
(~928 nonces/min).

## Cross-implementation consensus

Same v2 (murmur3) pseudo-`input_ids` scheme, 1000 common nonces per pair, chain proto
fraud test (`dist_threshold=0.40`, `p_mismatch=0.10`, `p_value_threshold=0.05`):

| A | B | median L2 | mismatch >0.40 | binomial p | Verdict |
|---|---|----------:|---------------:|-----------:|:-------:|
| plugin (out-of-tree) | in-tree (0.23→0.25) | 0.0024 | 0.20% | 1.1e-42 | **PASS** |
| plugin (out-of-tree) | qd forward-port (0.25.1) | 0.0024 | 0.10% | 2.0e-44 | **PASS** |
| in-tree (0.23→0.25) | qd forward-port (0.25.1) | 0.0000 | 0.30% | 4.1e-41 | **PASS** |

All three independent implementations agree within honest bounds. The ~0.002 median L2 is
numerical noise between kernel paths, far below the 0.40 per-nonce threshold and the 10%
allowed mismatch rate.

## Nonce collection

Collected 1064 nonces at ~720 nonces/min (batch_size=8, continuous mode). The primary set
(`artifacts/nonces_1000.json`) is the 0.25.1 forward-port; the two reference sets are the
out-of-tree plugin and the earlier in-tree port.

## Artifacts

- `artifacts/config.json` — GPU, versions, startup command, KV-cache sizing
- `artifacts/nonces_1000.json` — 1000 PoC nonce vectors, 0.25.1 forward-port (seq_len=1024, k_dim=12)
- `artifacts/nonces_reference_plugin.json` — reference set, out-of-tree plugin
- `artifacts/nonces_reference_intree.json` — reference set, in-tree 0.23→0.25 port
- `artifacts/sweep.json` — batch-size throughput sweep
- `artifacts/l2_matrix.json` — cross-implementation L2 + fraud verdicts

## Key observations

- **400k context + compiled fits on a single B300** for V4: sparse-MLA compresses KV
  enough for a 2.6M-token pool (max concurrency ~6.5 at full 400k context).
- **PoC throughput is compilation-invariant** here (b16/b32 rates are bit-identical to an
  eager `max_model_len=8192` run) because the PoC forward is `skip_compiled`.
- **The forward-port is consensus-safe**: nonces match the out-of-tree reference within
  numerical noise (p ≤ 2e-44), the same verdict as the earlier in-tree port.

## Reproducibility checklist

- [x] Model download command
- [x] Image build from a pinned commit (full 40-char SHA)
- [x] Full vLLM startup command (`artifacts/config.json`)
- [x] Sweep + collection commands and scripts (`scripts/`)
- [x] Hardware + software versions
- [x] Nonce sets and cross-validation script committed
- [x] No `.claude/` paths, no sibling-repo paths; cross-repo references pinned to SHA
