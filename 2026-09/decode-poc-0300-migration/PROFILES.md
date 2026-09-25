# Serving profiles for decode-PoC on vLLM 0.30.0

Engine: vLLM `v0.30.0` (`ced6857af`) plus the residual of [gonka-ai/vllm#114](https://github.com/gonka-ai/vllm/pull/114);
plugin `gonka-poc` 0.2.0, [gonka-ai/gonka-vllm-plugins#19](https://github.com/gonka-ai/gonka-vllm-plugins/pull/19).
Date: 2026-09-25. Status column: **measured** = booted and measured on 0.30 in this campaign (`README.md`);
**carried over** = the frozen 0.25.1 / 0.28.1 profile, not re-measured on 0.30.

## Rules that hold for every cell

- `max_cudagraph_capture_size` = `--max-num-seqs` (N). A capture grid below N halves PoC and chat.
- `--max-model-len` from the chain: MiniMax 180000, DeepSeek 400000, GLM 400000 (GLM's engine default 1,048,576 also fits).
- `--kv-cache-dtype fp8` everywhere except GLM on B300 (below); `--no-enable-prefix-caching`.
- `--logprobs-mode processed_logprobs --worker-extension-cls gonka_poc.worker.PoCWorkerExtension --trust-remote-code`.
- PoC requests with `batch_size = 0` (engine-managed admission); `POC_BATCH_SIZE_DEFAULT` only sets the fallback.
- FlashInfer autotune: on for MiniMax and DeepSeek (default), **off for GLM** (`--no-enable-flashinfer-autotune`, illegal memory access on Hopper).
- Runtime: the PyPI wheel works for MiniMax and DeepSeek NVFP4 on Blackwell. **GLM and DeepSeek FP8 on Hopper run from the official image** `vllm/vllm-openai:v0.30.0` (residual overlaid, plugin installed): the PyPI FlashInfer 0.6.18.post1 wheel fails vLLM's sm90 sparse-MLA check (`ckv_scale_arr` positional).
- 0.30 needs more memory than 0.25.1 for MiniMax: the frozen `gpu-memory-utilization` fails (autotuner on B300, validation on H100). One step down is the rule; DeepSeek and GLM are not memory-bound.
- The plugin is installed with `pip install --no-deps`; add `scipy` to the image.

## MiniMax-M2.7 (`--attention-backend FLASHINFER`, fp8 KV, `--max-model-len 180000`, `--served-model-name MiniMaxAI/MiniMax-M2.7`)

| cards | TP | gmu | N = capture | MNBT | MoE backend | status | PoC nonces/s | chat requests/s at c | R |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| 1×B300 | 1 | **0.95** (0.97 fails in the autotuner) | 608 | 32768 | auto = flashinfer_trtllm | measured | 26.8 | 29.0 at 608 | 0.92 |
| 4×H100 | 4 | **0.90** (0.94 runs out of memory at a 250-nonce validation) | 752 / capture 768 | 32768 | triton | measured | 35.3 | 36.8 at 768 | 0.96 |
| 2×H200 | 2 | 0.95 → try 0.92 | 640 | 8192 | triton | carried over (0.28.1: 27.7 / 31.0 at 640) | | | |
| 2×B200 | 2 | 0.95 → try 0.93 | 1536 | 32768 | auto = flashinfer_trtllm | carried over (0.25.1: 57.1 / 62.9 at 1536) | | | |
| 4×A100 | 4 | 0.92 | 704 / capture 608 | 8192 | auto | carried over; not in the chain | | | |

Open on 0.30: gmu 0.92 on H100 (the 0.28.1 point) was not tried; the H200 and B200 gmu is a guess by analogy, verify before pinning.

Boot (B300):

```
vllm serve /models/MiniMax-M2.7 --served-model-name MiniMaxAI/MiniMax-M2.7 --host 127.0.0.1 --port 8000 \
  --trust-remote-code --tensor-parallel-size 1 --attention-backend FLASHINFER --max-model-len 180000 --kv-cache-dtype fp8 \
  --no-enable-prefix-caching --max-num-seqs 608 --gpu-memory-utilization 0.95 --max-num-batched-tokens 32768 \
  --logprobs-mode processed_logprobs --worker-extension-cls gonka_poc.worker.PoCWorkerExtension \
  --compilation-config '{"max_cudagraph_capture_size":608}'
```

H100: `--tensor-parallel-size 4 --gpu-memory-utilization 0.90 --max-num-seqs 752 --moe-backend triton --compilation-config '{"max_cudagraph_capture_size":768}'`.

## DeepSeek-V4-Flash-0731 (fp8 KV, `--max-model-len 400000`, `--tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --enable-auto-tool-choice`, `--served-model-name deepseek-ai/DeepSeek-V4-Flash-0731`)

| cards | checkpoint | TP | gmu | N = capture | MNBT | MoE backend | status | PoC nonces/s | chat requests/s at c | R |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| 1×B300 | NVFP4 | 1 | 0.90 | 2048 | 32768 | auto = flashinfer_trtllm NvFp4 | measured | 47.1 | 43.7 at 1024 | 1.08 |
| 4×H100 | FP8 | 4 | 0.90 | 640 | 16384 | auto = marlin | measured (image) | 21.4 | 23.8 at 768 | 0.90 |
| 2×H200 | FP8 | 2 | 0.90 | 768 | 32768 | auto = marlin | carried over (0.28.1: 19.8–20.8 / 21.9) | | | |
| 2×B200 | FP8 or NVFP4 | 2 | 0.90 | 2048 | 32768 | auto | carried over (0.25.1: 46.2 / 41.3 at 1024; NVFP4 47.3 / 43.2) | | | |

Open on 0.30: H100 at N 1024 (the best 0.28.1 point, 24.9 nonces/s) was not measured at gmu 0.90; the KV pool on 0.30 is about half of the 0.25.1 one at the same flags, so check capacity before raising N. DeepSeek warm-up must equal the run size (TileLang JIT inside the run halves the first points).

Boot (H100, inside the image):

```
vllm serve /models/DeepSeek-V4-Flash-0731 --served-model-name deepseek-ai/DeepSeek-V4-Flash-0731 --host 127.0.0.1 --port 8000 \
  --trust-remote-code --enable-auto-tool-choice --tensor-parallel-size 4 --gpu-memory-utilization 0.90 --max-model-len 400000 \
  --max-num-batched-tokens 16384 --kv-cache-dtype fp8 --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 \
  --no-enable-prefix-caching --max-num-seqs 640 --logprobs-mode processed_logprobs --worker-extension-cls gonka_poc.worker.PoCWorkerExtension \
  --compilation-config '{"max_cudagraph_capture_size":640}'
```

B300 NVFP4: model dir `DeepSeek-V4-Flash-0731-NVFP4`, `--tensor-parallel-size 1 --max-num-seqs 2048 --max-num-batched-tokens 32768 --compilation-config '{"max_cudagraph_capture_size":2048}'`.

## GLM-5.3-Flash (`--block-size 2304`, `--tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice`, `--no-enable-flashinfer-autotune --no-disable-hybrid-kv-cache-manager`, `--served-model-name zai-org/GLM-5.3-Flash`)

**Mandatory on 0.30 (decided with [@vbgd0](https://github.com/vbgd0), 2026-09-25):**
`--kda-prefill-backend triton --attention-config '{"sparse_mla_force_mqa": true}'`.
They keep the 0.28.1 prefill kernels so the 15 September artifacts stay valid; without them a 0.30 validator sees
0.28.1 artifacts at the cross-hardware level (`README.md`, GLM section). Every node runs the same setting.
Cost: PoC within boot noise; chat −8% on H100, +5% on B300.

| cards | TP | KV | gmu | N = capture | MNBT | extra | status | PoC nonces/s | chat requests/s at c | R |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| 8×H100 | 8 | fp8 | 0.95 | 256 (512: chat +7%, PoC same, R 0.85) | 8192 | `--disable-custom-all-reduce` | measured (image) | 17.5 | 18.7 at 256 | 0.93 |
| 2×B300 | 2 | bf16 (`auto`; fp8 equal on 0.28.1, not re-measured) | 0.90 | 256 (perf profile of the freeze: gmu 0.95, N 512, MNBT 16384 → 20.5) | 32768 (65536 → `CUDA_ERROR_INVALID_VALUE` in the trtllm FMHA) | | measured | 17.2–17.4 | 17.3 at 256 | 0.99 |
| 4×H200 | 4 | fp8 | 0.90 | 256 | engine default | | carried over (0.28.1: 14.2 / 15.1 at 256) | | | |
| 4×B200 | 4 | fp8 | 0.90 | 1024 | 32768 | `--disable-custom-all-reduce` | carried over (0.28.1: 35.0 / 35.7 at 1024) | | | |

`POC_BATCH_SIZE_DEFAULT`: 16 on H100, 32 on B300 (fallback only; the chain sends `batch_size = 0`).

Boot (8×H100, inside the image):

```
vllm serve /models/GLM-5.3-Flash --served-model-name zai-org/GLM-5.3-Flash --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 8 --gpu-memory-utilization 0.95 --kv-cache-dtype fp8 --block-size 2304 --max-num-seqs 256 \
  --max-num-batched-tokens 8192 --logprobs-mode processed_logprobs --worker-extension-cls gonka_poc.worker.PoCWorkerExtension \
  --tool-call-parser glm47 --reasoning-parser glm45 --trust-remote-code --enable-auto-tool-choice --no-enable-flashinfer-autotune \
  --no-enable-prefix-caching --no-disable-hybrid-kv-cache-manager --disable-custom-all-reduce \
  --compilation-config '{"max_cudagraph_capture_size":256}' \
  --kda-prefill-backend triton --attention-config '{"sparse_mla_force_mqa": true}'
```

2×B300: `--tensor-parallel-size 2 --gpu-memory-utilization 0.90 --kv-cache-dtype auto --max-num-batched-tokens 32768`, no `--disable-custom-all-reduce`.

## Sources

`README.md` in this folder (0.30 measurements, 2026-09-24); the `hw.json` and FREEZE files of the 9 September devkit
(0.25.1) and of the 15 September GLM devkit (0.28.1); the 0.28.1 fleet measurements of 2026-09-15 … 09-16 (H100, B300, H200).
