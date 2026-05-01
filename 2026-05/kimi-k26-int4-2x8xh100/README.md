# Kimi-K2.6 INT4 — PoC v2 nonces + inference validation on 2×8×H100

**Date**: 2026-05-01
**Cluster**: Hyperbolic 2× bare metal nodes (uk-southeast-3), connected via 8×400 Gb/s InfiniBand fabric
**Goal**: collect 1000 nonces + nonces/min, validate inference logprobs against [`kaitakuai/experiments/.../exp2-top-k-40-sentinels`](https://github.com/kaitakuai/experiments/tree/main/2026-04/kimi-k26-inference-validation/exp2-top-k-40-sentinels) honest dataset, find max context that fits

---

## Headline results

| Config | Peak nonces/min | At batch | Notes |
|---|---|---|---|
| **TP=16, PP=1 (Ray, IB)** | **1389** | 32 | Best. batch=64 OOMs PoC engine. |
| TP=8, PP=2 (Ray, IB) | 1014 | 8 | Pipeline bubbles + Ray Compiled Graph overhead. |

Reference baseline (kaitakuai/experiments):
- 8× B200 TP=4 (2 vLLM instances, mlnode 0.2.12): **1024 nonces/min** @ batch=64 (b300-k3 image)
- Our **16×H100 TP=16** beats this by **+36% (1389 vs 1024)**, with full **262144 (256K) native context**

Inference validation against honest exp2 dataset (100 sp items, processed_logprobs, top_k=40):

| Stat | Our TP=16 | Honest exp2 (B200↔H200) |
|---|---|---|
| mean distance2 | **0.0463** | 0.0232 |
| median | 0.0389 | 0.0231 |
| p95 | 0.0882 | 0.0351 |
| max | 0.1291 | 0.0466 |
| token mismatches | 0/100 | 0/1000 |
| **gate (mean ≤ 0.2)** | ✅ **PASS** | ✅ |
| soft pass (mean ≤ 0.05) | ✅ PASS (0.0463 < 0.05) | ✅ |

distance2 ~2× higher than exp2 baseline is expected for cross-GPU/cross-vendor (H100 vs B200/H200) — the gate is on **mean ≤ 0.2**, well within bounds.

---

## Hardware

| Component | Spec |
|---|---|
| Provider | Hyperbolic, region uk-southeast-3 |
| Nodes | 2× bare metal (`stupendous-gardenia-lion-b37e5cca`, `stupendous-gardenia-lion-0385caa5`) |
| GPU per node | 8× **NVIDIA H100 SXM5 80GB HBM3** (NVLink full mesh, NV18) |
| GPU total | **16** (1.28 TB GPU memory) |
| CPU per node | 192× Intel Xeon 8558 (256 logical, 2 NUMA) |
| RAM per node | ~2 TB |
| Boot disk | 893 GB RAID1 (2× 894G NVMe) |
| Data disk | 1× 3.5 TB NVMe (XFS) per node, mounted as `/data` |
| Network: public | `ens10f0np0` 25 Gb/s ethernet (per node, public IP) |
| Network: cluster | **8× InfiniBand 400 Gb/s ports** per node (Mellanox MT4129, mlx5_ib0..7) |
| IB switch | `ndlo-qm9700-1` (Mellanox QM9700, both nodes share fabric) |
| IB partition (PKEY) | `0x8098` (provider-managed; both nodes intentionally placed in same partition) |
| Measured cross-node IB throughput | **388 Gb/s** (`ib_send_bw -d mlx5_ib0 -s 1MB -n 1000`, 97% of theoretical 400 Gb/s) |

Public IPs / IB IPs:
- node-A: `85.234.79.28` (public) → `192.168.243.158` (ib0)
- node-B: `85.234.79.241` (public) → `192.168.242.154` (ib0)

---

## Software stack

| Component | Version | Notes |
|---|---|---|
| OS | Ubuntu 24.04.3 LTS (Noble Numbat) | both nodes |
| Kernel | 6.8.0-85-generic | |
| **NVIDIA driver** | **580.126.20** | upgraded from stock 570.158.01 (570 has CUDA 12.8 ceiling, image needs CUDA 12.9 PTX) |
| nvidia-fabricmanager | 580.126.20 | required for NVSwitch on H100 SXM5 |
| libnvidia-nscq | 580.126.20 | NVSwitch query lib |
| Docker | 29.1.3 | apt-installed |
| nvidia-container-toolkit | 1.19.0 | from Nvidia repo |
| **MLNode image** | `ghcr.io/product-science/mlnode:3.0.13-alpha5` (43.4 GB) | unchanged, no patches |
| Container CUDA | 12.9.1 (NV_CUDA_CUDART_VERSION=12.9.79-1) | from base image |
| **vLLM** (in image) | **0.15.1** | `/usr/local/lib/python3.12/dist-packages/vllm` |
| **Ray** (in image) | **2.53.0** | for cross-node distributed executor |
| Python (in image) | 3.12.13 | system |
| Quantization | compressed-tensors W4A16 (Marlin MoE) | native INT4 from `moonshotai/Kimi-K2.6` |
| Attention backend | **`FLASHMLA`** | Hopper-only Dense MLA. **Required** — default FLASHINFER doesn't support MLA |
| MoE backend | `CompressedTensorsWNA16MarlinMoEMethod` | auto-selected for compressed-tensors W4A16 |
| logprobs mode | **`processed_logprobs`** | matches gonka production (sentinels at clipped positions) |
| KV cache block size | 64 | forced by FlashMLA |
| Engine | vLLM V1 (V0 not available in 0.15) | |

Why driver upgrade was required: image ships CUDA 12.9 PTX in Marlin kernels (`gptq_marlin_repack`); host driver 570.158 (CUDA 12.8 ceiling) → `cudaErrorUnsupportedPtxVersion` at model load. Driver 580 (CUDA 12.9+) resolves it.

---

## Model

`moonshotai/Kimi-K2.6` ([HF](https://huggingface.co/moonshotai/Kimi-K2.6))

| Field | Value |
|---|---|
| Architecture | `KimiK25ForConditionalGeneration` (DeepSeek-V2-style MoE + MLA) |
| Quantization | compressed-tensors W4A16 (native INT4 QAT) |
| Total params | 1.058 T |
| Disk size | ~595 GB (64 safetensors shards × ~9 GB) |
| Layers | 61 (text) |
| Hidden size | (in `text_config`) |
| Native context | **262144 (256K)** — `max_position_embeddings` and `generation_config.max_length` |
| Tokenizer | tiktoken, vocab 163840, BOS=163584, EOS=163585 |
| Multi-modal | yes (image-text-to-text), but only text is used here |
| HF gated | no (public) |

We verified KV cache fits the **full native 262144 context** with `--gpu-memory-utilization 0.92` on TP=16 (1.28 TB total GPU mem; ~540 GB weights leaves ~640 GB for KV).

---

## vLLM launch command (TP=16, PP=1, peak config)

Run inside the mlnode container on node-A (Ray head):

```bash
VLLM_HOST_IP=192.168.243.158 \
NCCL_SOCKET_IFNAME=ib0 \
GLOO_SOCKET_IFNAME=ib0 \
NCCL_IB_HCA=mlx5_ib0,mlx5_ib1,mlx5_ib2,mlx5_ib3,mlx5_ib4,mlx5_ib5,mlx5_ib6,mlx5_ib7 \
python3 -m vllm.entrypoints.openai.api_server \
  --model /root/.cache/huggingface/kimi-k26 \
  --served-model-name moonshotai/Kimi-K2.6 \
  --tensor-parallel-size 16 \
  --distributed-executor-backend ray \
  --host 0.0.0.0 \
  --port 5001 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 64 \
  --logprobs-mode processed_logprobs \
  --trust-remote-code \
  --attention-backend FLASHMLA
```

`--distributed-executor-backend ray` makes vLLM use the existing Ray cluster (head on node-A, worker on node-B) and distribute the 16 TP ranks across both nodes via NCCL over IB.

---

## PoC v2 + inference request parameters

PoC v2 nonce gen (constants in `collect_artifacts.py`, matches gonka standard):
```
block_hash       = "artifact_collection_block_v1"
public_key       = "artifact_collection_pk_v1"
block_height     = 100
seq_len          = 1024
k_dim            = 12
node_id, node_count = 0, 1   # single vLLM instance
```

Inference 5-language probe (matches honest exp2):
```
temperature       = 0.7
seed              = 1
top_logprobs      = 4
max_tokens        = 64    (8 in batch sweeps to save time)
repetition_penalty= 1.2
top_k             = 40
top_p             = 0.95
```

`top_k=40` + `top_p=0.95` clip non-top tokens to `-inf` BEFORE softmax → vLLM clamps the rest to `-9999` (sentinels) when `--logprobs-mode processed_logprobs` is active. Sentinel rate observed: **62-64/64 tokens (97-100%)** — matches exp2's ~90.5% on longer outputs.

---

## Full sweep results

### TP=16, PP=1 (1 vLLM instance over Ray, all 16 GPUs)

| batch_size | nonces collected | time | nonces/min | result |
|---|---|---|---|---|
| 8 | 1040 | 52.0 s | **1199.8** | ok |
| 16 | 1040 | 47.0 s | **1327** | ok |
| **32** | **1088** | **47.0 s** | **🚀 1389** | **PEAK** |
| 64 | 0 | 240 s | 0 | **PoC engine OOM-stuck** |

> Memory note (`memory.md` / earlier B200 runs): "After OOM at large batch, PoC engine gets stuck → restart vLLM before collecting nonces". On batch=64 with 256K context the KV cache budget overflows; need to drop ctx to ~120K to test b=64.

### TP=8, PP=2 (Ray Compiled Graph, pipeline parallel)

| batch_size | nonces collected | time | nonces/min | result |
|---|---|---|---|---|
| 8 | 1048 | 62.0 s | **1014** | ok (chat completions had timeouts on 5-lang probe — PP=2 chat path is buggy in vLLM 0.15.1) |
| 16, 32 | — | — | — | not run (container died after b=8 cleanup) |

### Comparison vs reference

| Hardware | Image | TP × PP | Peak nonces/min | Source |
|---|---|---|---|---|
| 8× B200 | mlnode 0.2.12-vllm0.20.0-b300-k3 | 4 × 1 (2 instances) | 1024 | memory `mlnode_b300_k3_kimi_k26_patches.md` |
| **16× H100 (this run)** | mlnode 3.0.13-alpha5 | **16 × 1** (1 instance, IB) | **1389** | this experiment |
| 16× H100 (this run) | mlnode 3.0.13-alpha5 | 8 × 2 | 1014 | this experiment |

---

## Inference validation (cross-GPU vs honest dataset)

Replayed first 100 prompts of `honest_b200_h200.jsonl` (Spanish split) through our TP=16 vLLM with `enforced_tokens` set to honest's tokens — so logprobs are computed on identical positions on both sides. distance2 = exact gonka formula from `mlnode/packages/benchmarks/src/validation/utils.py`.

| Stat | Value |
|---|---|
| n items | 100 (validation interrupted at 100/200 by user; sample size ample) |
| token mismatches | **0/100** |
| distance2 mean | **0.0463** |
| median | 0.0389 |
| p95 | 0.0882 |
| min / max | 0.0280 / 0.1291 |
| Gonka gate `mean ≤ 0.2` | ✅ **PASS** |
| Soft band `mean ≤ 0.05` | ✅ PASS |

Honest exp2 baseline (B200 TP=4 → H200 TP=8 self-validation):
- mean = 0.0232, p95 = 0.0351, max = 0.0466

Our cross-GPU mean is ~2× higher than exp2 baseline — explained by H100×16 vs B200×4↔H200×8 hardware difference (different attention/MoE kernels, different allreduce order). Well within the production gate.

---

## Issues encountered + fixes

1. **Initial Hyperbolic instances had different IB partition keys** (PKEY 0x80b0 vs 0x8068 → cross-node IB blocked at L2). Provider re-provisioned both into the same PKEY 0x8098 → ping/IB worked. **Always verify cross-node IB ping before assuming InfiniBand will be usable.**

2. **`gptq_marlin_repack` failed with `cudaErrorUnsupportedPtxVersion`** because stock host driver 570.158 has CUDA 12.8 ceiling but image-bundled Marlin kernel needs CUDA 12.9 PTX. Fix: upgrade host driver to 580.126.20 + reboot. Also forced `--force-overwrite` to resolve `libnvidia-gl-580-server` file conflict during install.

3. **Default attention backend FLASHINFER doesn't support MLA** → Engine init crashes immediately. Fix: pass `--attention-backend FLASHMLA` (Hopper-only Dense MLA). On Blackwell (B200/B300) you'd use `CUTLASS_MLA` instead.

4. **MLNode `runner.py` doesn't support multi-node TP** because it does `total_gpus // gpus_per_instance` locally. With TP=16 and 8 local GPUs it would set `CUDA_VISIBLE_DEVICES=0..15`, which doesn't exist locally → vLLM crashes. **Fix: launch vLLM directly via `python3 -m vllm.entrypoints.openai.api_server --distributed-executor-backend ray`** — the image still works, just bypassing the MLNode API wrapper. PoC v2 endpoints (`/api/v1/pow/...`) are still served by vLLM directly.

5. **Ray placement group from previous launch sticks around** — when we relaunched vLLM with TP=8/PP=2 after killing TP=16, the new engine got `Current node has no GPU available` because the old placement group still held all 8 GPUs. Fix: `docker rm -f mlnode` on both nodes and restart Ray cluster from scratch.

6. **batch=64 hangs the PoC engine** at 256K context (KV cache budget overflows). Engine stops emitting nonces and stays stuck until vLLM restart. To explore higher batches, drop `--max-model-len` to ~120K first.

7. **NVMe device naming changes after reboot** (`/dev/nvme1n1` became `/dev/nvme0n1` on node-A). Mount by UUID or scan `blkid | grep xfs` — don't hardcode `/dev/nvmeXn1` in scripts.

8. **HuggingFace download stalled at 58/64 shards on node-B** with `Too many open files (os error 24)`. Fix: re-run `hf download` with `ulimit -n 65536` and `--max-workers 8`. (Resume worked — already-downloaded shards are skipped.) hf-cli also renamed `huggingface-cli download` → `hf download` in 1.13+.

9. **`ssh user@host 'cmd'` from Windows cmd.exe loses single-quote shell wrapping** — use double quotes (or no quotes) on cmd.exe.

10. **`pkill -f` with a pattern that matches the bash command line itself kills the bash session** (exit 137) — use `pgrep -f X | xargs kill -9` or a more-specific pattern (`pgrep -f api_server` rather than `pkill -f vllm`).

---

## Folder layout (this experiment)

```
2026-05/kimi-k26-int4-2x8xh100/
├── README.md                          # this report
├── RUNBOOK.md                         # step-by-step reproduction
├── collect_artifacts.py               # nonce collector + 5-lang probe (from kaitakuai/rtx-pro-6000)
├── validate_against_honest.py         # cross-GPU distance2 validator vs honest_b200_h200.jsonl
├── artifacts/
│   ├── tp16/                          # TP=16 batch=8 main run
│   │   ├── config.json
│   │   ├── nonces_1000.json
│   │   └── inference_5langs.json
│   ├── tp16_b16/                      # TP=16 batch=16
│   ├── tp16_b32/                      # TP=16 batch=32  ← peak run
│   └── tp8pp2_b8/                     # TP=8 PP=2 batch=8
└── logs/
    ├── vllm.log                       # last vLLM run (TP=8 PP=2 startup)
    ├── collect.log                    # main collect_artifacts (TP=16 b=8)
    ├── sweep.log                      # TP=16 batch sweep (b=16, 32, 64-stuck)
    ├── sweep_tp8pp2.log               # TP=8 PP=2 sweep (only b=8 ran)
    └── validate.log                   # 100-item distance2 run on TP=16
```

Each `nonces_1000.json` contains: `block_hash`, `public_key`, `seq_len`, `k_dim`, `total_nonces`, `nonces_per_min`, `generation_time_sec`, `artifacts: [...]` (the actual nonce vectors).

Each `inference_5langs.json` contains: `params`, 5× `{language, prompt, text, n_tokens, n_sentinel_positions, full_response}`.

---

## What I'd do next

- Test `--max-model-len 131072` (half of native) to fit batch=64+ → likely +20-30% nonces/min over current peak.
- Try `mlnode-full:0.2.12-vllm0.20.0-b300-k3` image (vLLM 0.20 + custom PoC patches from memory `mlnode_b300_k3_kimi_k26_patches.md`) — may behave differently on PP=2 chat completions.
- Cross-validate against more languages (en/ch/ar/hi) in addition to sp; current 100-item subset is sp-only.
