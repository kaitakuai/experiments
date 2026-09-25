# decode-PoC on vLLM 0.30.0: migration check for MiniMax-M2.7, DeepSeek-V4-Flash and GLM-5.3-Flash

- **Dates:** 2026-09-23 … 2026-09-25
- **Models:** `MiniMaxAI/MiniMax-M2.7` @ `d494266a`; `deepseek-ai/DeepSeek-V4-Flash-0731` @ `7872f01b` and `MJPansa/DeepSeek-V4-Flash-0731-NVFP4` @ `64d64cd8`; `zai-org/GLM-5.3-Flash` @ `eb9eb208`
- **Hardware:** 1×B300 SXM6 (Nebius), 8×H100 80GB (Verda; 4×H100 for MiniMax and DeepSeek, 8×H100 for GLM), 2×B300 SXM6 (Vast)
- **Stack:** vLLM `v0.30.0` (PyPI wheel, or the image `vllm/vllm-openai:v0.30.0` on Hopper) plus the 35-file residual `kaitakuai/vllm` @ `c303828d0` and the plugin `gonka-poc` 0.2.0, `kaitakuai/gonka-vllm-plugins` @ `fef179f`, branches `poc-as-chat-vllm-0.30.0-dev`

The decode-PoC stack moved from vLLM 0.25.1 (MiniMax, DeepSeek; [2026-09/decode-poc-0251-freeze](../decode-poc-0251-freeze))
and 0.28.1 (GLM; [2026-09/decode-poc-0281-glm-freeze](../decode-poc-0281-glm-freeze)) to one base, vLLM 0.30.0,
which carries GLM-5.3-Flash upstream ([vllm-project/vllm#53906](https://github.com/vllm-project/vllm/pull/53906)).
This folder is the check of that move: does the 0.30 stack boot the three models on the frozen profiles, does a 0.30
validator accept the reference corpora of the frozen points, and what do PoC and chat throughput do.

Plan agreed with [@vbgd0](https://github.com/vbgd0) on 2026-09-23: the release of decode-PoC targets 0.30.0 if all three
models come up; Kaitaku boots the models on B300 and H200-class hardware, validates against the 0.25.1 / 0.28.1
reference artifacts on the same pairs and measures performance on the known settings; the residual and the plugin go
to him as PRs ([gonka-ai/vllm#114](https://github.com/gonka-ai/vllm/pull/114),
[gonka-ai/gonka-vllm-plugins#19](https://github.com/gonka-ai/gonka-vllm-plugins/pull/19)) and he builds the testnet.
Bit-identical output across versions was not expected and is not required: the chain statistic decides.

---

## Contents

- [Code](#code) — residual, plugin, checkpoints, kits
- [Launch settings](#launch-settings) — the seven measured cells; full recipes in `PROFILES.md`
- [Result in one table](#result-in-one-table)
- [Throughput](#throughput) — PoC and chat against the frozen points
- [Validation](#validation) — a 0.30 validator on the 0.25.1 / 0.28.1 reference corpora and on its own
- [GLM-5.3-Flash: two prefill kernels](#glm-53-flash-two-prefill-kernels) — the cause of the GLM shift and the two flags that remove it
- [Round, gates, replay](#round-gates-replay)
- [Profile changes on 0.30](#profile-changes-on-030)
- [Reproduce](#reproduce)
- [Data not in this repository](#data-not-in-this-repository)

---

## Code

| item | value |
| --- | --- |
| engine | [kaitakuai/vllm · poc-as-chat-vllm-0.30.0-dev](https://github.com/kaitakuai/vllm/tree/poc-as-chat-vllm-0.30.0-dev) @ c303828d0: stock `v0.30.0` plus 35 files, overlaid on the wheel or the image |
| plugin | [kaitakuai/gonka-vllm-plugins · poc-as-chat-vllm-0.30.0-dev](https://github.com/kaitakuai/gonka-vllm-plugins/tree/poc-as-chat-vllm-0.30.0-dev) @ fef179f, `gonka-poc` 0.2.0 |
| upstream | [gonka-ai/vllm#114](https://github.com/gonka-ai/vllm/pull/114) (residual, base `release/v0.30-decode-int`) · [gonka-ai/gonka-vllm-plugins#19](https://github.com/gonka-ai/gonka-vllm-plugins/pull/19) (one plugin for the three models, base `decode/vlm030`) |
| runtime, Blackwell | the PyPI wheel `vllm==0.30.0` in a venv (MiniMax, DeepSeek NVFP4, GLM on the Vast image) |
| runtime, Hopper | the image `vllm/vllm-openai:v0.30.0` with the residual overlaid and the plugin installed: the PyPI FlashInfer 0.6.18.post1 fails vLLM's sm90 sparse-MLA check (`ckv_scale_arr` positional) and GLM and DeepSeek FP8 do not boot from the wheel on H100 |
| MiniMax | MiniMaxAI/MiniMax-M2.7 · d494266a (FP8 weights) |
| DeepSeek, FP8 | deepseek-ai/DeepSeek-V4-Flash-0731 · 7872f01b |
| DeepSeek, NVFP4 | MJPansa/DeepSeek-V4-Flash-0731-NVFP4 · 64d64cd8 |
| GLM-5.3-Flash | zai-org/GLM-5.3-Flash · eb9eb208 (FP8 weights); safetensors-header and config fingerprints equal the 0.28.1 freeze provenance |
| reference corpora | the honest corpora of the frozen points: MiniMax B300 and H100 (0.25.1, 10 block hashes × 250 nonces), DeepSeek NVFP4 B300 and FP8 H100 (0.25.1, 10 × 250), GLM B300 and H100 (0.28.1, 10 × 250) |
| kits | the generation and validation kits of the 9 September and 15 September devkits, unchanged |

## Launch settings

The seven measured cells. Every cell is the frozen profile of its card, with the changes noted in
[Profile changes on 0.30](#profile-changes-on-030). Boot lines, per-model flags and the rules behind them: `PROFILES.md`.

| cards | model | runtime | TP | KV | MoE | gpu-memory-utilization | N = max-num-seqs = capture size | max-num-batched-tokens |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| 1×B300 | MiniMax | wheel | 1 | fp8 | `flashinfer_trtllm` | 0.95 (frozen 0.97) | 608 | 32768 |
| 4×H100 | MiniMax | wheel | 4 | fp8 | `triton` | 0.90 (frozen 0.94) | 752 / capture 768 | 32768 |
| 1×B300 | DeepSeek NVFP4 | wheel | 1 | fp8 | `flashinfer_trtllm` NvFp4 | 0.90 | 2048 | 32768 |
| 4×H100 | DeepSeek FP8 | image | 4 | fp8 | `marlin` Mxfp4 | 0.90 | 640 | 16384 |
| 8×H100 | GLM | image | 8 | fp8 | `triton` | 0.95 | 256 | 8192 |
| 2×B300 | GLM | image | 2 | bf16 | `flashinfer_trtllm` | 0.90 | 256 | 32768 |

N is passed twice, as `--max-num-seqs N` and `max_cudagraph_capture_size N`; PoC requests carry `batch_size 0`, so the
engine schedules the batch. The validation cells were taken at the profile of the reference corpora (MiniMax B300 at
gpu-memory-utilization 0.97, which boots for validation and fails for the throughput run; DeepSeek H100 at 0.85 and
N 1024 as the reference boot). GLM on both cards was measured at N 256, the N of the reference corpora.

## Result in one table

| model × card | boots | 0.30 validator on the reference corpora | 0.30 validator on its own corpora | round / gate / replay | PoC, nonces/s, 0.30 vs frozen | chat, requests/s, 0.30 vs frozen |
| --- | --- | --- | --- | --- | --- | --- |
| MiniMax 1×B300 | yes | 6.3% points: the "same card, other boot" level | 0.06–0.31% | OK / 503→200 / 24 of 24 | 26.8 vs 29.9 (gmu 0.95 vs 0.97) | 29.0 vs 31.6 at c608 |
| MiniMax 4×H100 | yes (gmu 0.94 runs out of memory at validation; 0.90) | 6.6% points: the same-boot level of H100 | 6.0–7.0% | OK / 503→200 / 24 of 24 | 35.3 vs 43.2 (gmu 0.90 vs 0.94) | 36.8 vs 42.2 at c768 |
| DeepSeek NVFP4 1×B300 | yes | 10.1% points, 0.29% of points above τ 0.02: the numerics moved | 0.76%, 0.019% above τ 0.02 (as frozen) | OK / 503→200 / 24 of 24 | 47.1 vs 42.4 (+11%) | 43.7 vs 37.2 at c1024 (+18%) |
| DeepSeek FP8 4×H100 | yes (image) | 9.0% points, 0.25% above τ 0.02 | 7.6%, 0.17% above τ 0.02 (frozen 0.03%) | OK / 503→200 / 24 of 24 | 21.4 vs 22.5 (−5%) | 23.8 vs 22.7 at c768 (+5%) |
| GLM 8×H100 | yes (image only) | 12.7% points with the 0.30 defaults; **9.9% with the two prefill flags**, equal to a 0.28.1 validator on this box | 6.1–6.6% | OK / 503→200 / 24 of 24 | 17.9 vs 15.75 (+14%); 17.5 with the flags | 20.5 vs 16.3 at c256 (+25%); 18.7 with the flags |
| GLM 2×B300 | yes | 14.5% points with the defaults; **6.5% with the two prefill flags**, the same-card band of the freeze | 5.0–6.3% | OK / 503→200 / 24 of 24 | 16.4–17.7 vs 16.3 at N 256 | 16.4 (defaults), 17.3 (flags) at c256 |

Read across: the 0.30 stack boots all three models on the frozen profiles; MiniMax validates the 0.25.1 references at
the same-card level; DeepSeek's numerics moved between 0.25.1 and 0.30 but the distance stays far below the fraud arm
of the freeze; GLM needs two flags to keep the 0.28.1 references valid, and with them the 0.30 validator reproduces
the 0.28.1 one. Not bit-identical anywhere, which was expected: the FlashInfer kernels of the 0.30 wheel differ.

## Throughput

**nonces/min per 8 GPUs = nonces/s × 60 × 8 ÷ GPUs in the configuration**; **R = nonces/s ÷ requests/s**, the units
being comparable (a nonce is 256 prompt tokens plus 256 decode steps, a chat request 256 prompt plus 256 generated).
PoC runs are 3,000 nonces at once (1,000 for GLM) after a warm-up, ramp-up and drain included; chat is `vllm bench serve`
with `ignore_eos`, three waves at the frozen concurrency. Frozen figures are the 0.25.1 and 0.28.1 rows of the freeze
folders at the same profile.

| cards · model | PoC, nonces/s | nonces/min per 8 GPUs | chat, requests/s | at concurrency | tpot p50, ms | R | frozen PoC / chat | change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1×B300 · MiniMax, gmu 0.95 | 26.82 | 12,874 | 29.03 | 608 | 65 | 0.92 | 29.93 / 31.64 (gmu 0.97) | −10% / −8% |
| 4×H100 · MiniMax, gmu 0.90 | 35.33 | 4,240 | 36.75 | 768 | 61 | 0.96 | 43.23 / 42.22 (gmu 0.94) | −18% / −13% |
| 1×B300 · DeepSeek NVFP4 | 47.07 | 22,594 | 43.74 | 1024 | 67 | 1.08 | 42.4 / 37.2 | +11% / +18% |
| 4×H100 · DeepSeek FP8 | 21.42 | 2,570 | 23.82 | 768 | 88 | 0.90 | 22.5 / 22.7 | −5% / +5% |
| 8×H100 · GLM, 0.30 defaults | 17.92 | 1,075 | 20.45 | 256 | 45 | 0.88 | 15.75 / 16.34 (N 256) | +14% / +25% |
| 8×H100 · GLM, `marlin` MoE | 17.79 | 1,067 | 19.94 | 256 | 45 | 0.89 | | same as `triton` |
| 8×H100 · GLM, two prefill flags | 17.46 | 1,048 | 18.74 | 256 | 46 | 0.93 | | −3% / −8% against the defaults |
| 2×B300 · GLM, 0.30 defaults, two boots | 16.44 / 17.66 | 3,946 / 4,238 | 16.39 | 256 | 48 | 1.08 | 16.34 / — (N 256; chat frozen at N 1024 only) | PoC within boot noise |
| 2×B300 · GLM, two prefill flags, two boots | 17.38 / 17.19 | 4,171 / 4,126 | 17.29 | 256 | 46 | 0.99 | | chat +5% against the defaults |

The MiniMax losses are the KV capacity at the lower `gpu-memory-utilization` (B300: 535 nonces in the KV cache against
623 frozen; H100: 605 against about 750), not a per-row regression; a like-for-like point needs 0.92 on H100 with the
validation memory re-checked. DeepSeek on B300 gains on both axes; on H100 at the frozen N 640 it loses 5% on PoC,
and the H100 KV pool on 0.30 is about half of the 0.25.1 one at the same flags (DeepSeek B300: 1,118,619 tokens against
2,318,630), so N cannot be raised without checking capacity. GLM gains on Hopper from the two new prefill kernels
(+14% PoC, +25% chat with the defaults); pinning the 0.28.1 kernels costs 3% PoC and 8% chat on H100 and nothing on
B300, where chat is 5% faster with the pinned kernels.

## Validation

A validator on the 0.30 stack re-validates the reference corpora of the frozen points (teacher-forced prefill of the
chain, the same rule as the chain), and its own corpora generated on the same boot. Points %: the share of compared
points whose reflection index differs, mean over hashes [range]. The frozen rows are the published cells of the freeze
folders; the fraud rows are quoted from them for scale.

### MiniMax-M2.7, partition 250

| validator → prover corpora | points % at τ = 0 | nonces flagged at τ 0.02 / 0.025 |
| --- | ---: | ---: |
| 0.30 B300 → 0.25.1 B300 references, 10 hashes | 6.30 [5.73–6.80] | 6.6% / 1.5% |
| 0.30 B300 → own corpora, same boot, 3 hashes | 0.16 [0.06–0.31] | 0 / 0 |
| frozen: B300 same boot | 0.11 [0.00–0.45] | 0 / 0 |
| frozen: B300 same card, other boot and plugin build | 6.31 [5.67–6.90] | 7% / 2% |
| frozen: B300 validating H200 / H100 / A100 corpora | 7.32 / 8.03 / 8.11 | 14–26% / 5–14% |
| frozen: fraud QuantTrio on B300 | 12.10 | 63% / 41% |
| 0.30 H100 → 0.25.1 H100 references, 10 hashes | 6.57 [5.82–7.19] | 11.0% / 5.7% |
| 0.30 H100 → own corpora, same boot, 3 hashes | 6.40 [6.03–7.01] | 3.6–11.2% / 0.8–4.0% |
| frozen: H100 same boot | 6.57 [5.64–7.10] | 11% / 6% |
| frozen: H100 validating B300 / H200 / A100 corpora | 7.92 / 7.44 / 7.80 | 17–23% / 7–13% |

On B300 the 0.30 stack is as deterministic within a boot as the 0.25.1 line and sits on the "same card, other boot and
build" row against the references, below every cross-hardware honest cell. On H100 (TP 4) it sits on the same-boot
level of that card. Equivalent at the chain statistic on both cards.

### DeepSeek-V4-Flash, point level

Rates are per cent of points at τ = 0 / 0.01 / 0.02 / 0.05 / 0.1, the grid the DeepSeek kit prints.

| validator → prover corpora | τ = 0 | 0.01 | 0.02 | 0.05 | 0.1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.30 NVFP4 B300 → 0.25.1 NVFP4 B300 references, 10 hashes | 10.11 [10.00–10.32] | 1.99 | 0.294 | 0.0005 | 0 |
| 0.30 NVFP4 B300 → own corpora, same boot, 3 hashes | 0.76 [0.38–1.26] | 0.115 | 0.019 | 0.001 | 0 |
| frozen: NVFP4 B300 same boot / other boot | 1.49 / 1.99 | — | 0.018 / 0.021 | 0 | 0 |
| frozen: NVFP4 B300 ↔ FP8 (cross-quantisation) | 8.08 | 1.03 | 0.087 | 0 | 0 |
| 0.30 FP8 H100 → 0.25.1 FP8 H100 references, 10 hashes | 9.04 [8.87–9.26] | 1.54 | 0.247 | 0.0139 | 0.0002 |
| 0.30 FP8 H100 → own corpora, same boot, 3 hashes | 7.57 [7.50–7.61] | 0.92 | 0.166 | 0.0166 | 0 |
| frozen: FP8 H100 same boot | 6.74 | 0.46 | 0.027 | 0 | 0 |
| frozen: fraud REAP-145B, any validator | 42.0 | 26.9 | 16.6 | 3.38 | 0.149 |

Within a boot the NVFP4 stack on B300 is as deterministic as before (0.019% of points above τ 0.02 against 0.018%
frozen); the FP8 stack on H100 is noisier than the 0.25.1 line even within a boot (0.166% against 0.027%). Against the
0.25.1 references both cards show a systematic shift: 9–10% of points on every hash, 0.25–0.29% above τ 0.02, three
times the frozen cross-quantisation level, and 56× below the REAP arm at τ 0.02 and about 7,000× at τ 0.05. The
DeepSeek numerics moved between 0.25.1 and 0.30 (candidates: the RoPE change on the sparse SWA layers of
[vllm-project/vllm#54815](https://github.com/vllm-project/vllm/pull/54815) and the kernel migrations); a 0.25.1 prover
under a 0.30 validator lands above the honest cross-quantisation cell and far from fraud. Validation passes; the
0.25.1 goldens are not a same-boot reference for a 0.30 fleet.

### GLM-5.3-Flash, partition 200, points % at τ = 0

The 0.28.1 references are the honest B300 and H100 corpora of the 15 September freeze. Every GLM cell is a validation
cell (no GLM nonce reproduces on another boot without a mismatching point).

| validator → prover corpora | H100 | B300 |
| --- | ---: | ---: |
| 0.30 defaults → 0.28.1 references, 10 hashes | 12.68 [9.77–18.25] | 14.53 [9.93–19.02] |
| 0.30 defaults → own corpora, same boot, 3 hashes | 6.29 [6.10–6.58] | 5.70 [5.02–6.31] |
| **0.30 with `--kda-prefill-backend triton` and `sparse_mla_force_mqa` → 0.28.1 references** | **9.88 [9.09–10.79]** | **6.53 [6.13–7.00]** |
| 0.28.1 stack on the same box → 0.28.1 references | 9.87 [9.06–10.52] | — |
| frozen: same card, other boot | 7.06 [6.22–8.23] | 5.08 [4.72–5.57] |
| frozen: other card, honest | 11.2–12.5 | 11.2–13.6 |
| frozen: fraud W4A16 | 20.8 | 21.0 |

With the 0.30 defaults a validator sees the 0.28.1 references at the cross-hardware level of the freeze, with three
hashes on H100 (h04, h08, h09) and five on B300 above it. With the two flags the 0.30 validator reproduces the
0.28.1 validator on every hash: on H100 9.88% against 9.87% for the 0.28.1 stack booted on the same box, on B300
6.53% inside the same-card band. Within a boot, 0.30 is at the usual GLM same-card level on both cards.

The H100 box itself adds about 3 pp at the same version: the 0.28.1 stack validates its own references at 9.9% here
against 7.1% for another boot on the freeze's H100 node. This node initialises the FlashInfer allreduce + RMSNorm fusion
workspace (`backend=mnnvl`), the freeze node did not; the `pass_config` flag does not remove that workspace, so it was
not isolated further.

## GLM-5.3-Flash: two prefill kernels

[@vbgd0](https://github.com/vbgd0) asked for the cause of the GLM shift (2026-09-24). Ruled out first, by inspection:
the checkpoint (fingerprints equal the freeze provenance; `eb9eb208` is still the latest revision), the attention
backend (`FLASHINFER_MLA_SPARSE_SM90`, fp8 KV, on both), the sparse-MLA planner dtype patch (identical hunk), the KDA
decode kernel (0.30 changed only its addressing), the MoE expert kernel (`marlin` = `triton`, table below) and MoE
routing (the runner computes the gate itself on both bases).

Between the base of the 0.28.1 branch (`385dce36b`) and `v0.30.0` two prefill paths of this model changed:

1. **KDA prefill (34 of 45 layers).** 0.30 adds FlashKDA (`vllm-project/FlashKDA`, fused CUDA, sm90a / sm100a / sm120a)
   and selects it by default for bf16, head_dim 128 and a bounded gate, all true for GLM-5.3-Flash. 0.28.1 ran the
   Triton `chunk_kda_with_fused_gate` path, which is unchanged. Switch: `--kda-prefill-backend triton`.
2. **Sparse-MLA prefill (11 layers).** In 0.28.1 no dense-MHA prefill backend supported this model's MLA dimensions
   (256 / 0 / 256); the boot log says `No MLA prefill backend supports this model; sparse MLA will use the top-k MQA path
   only`, so prefill rows went through the same fp8-KV absorbed-MQA kernel as decode. 0.30 added `MLADimensions(256, 0, 256)`
   to the FlashAttention prefill backend (`Using FLASH_ATTN MLA prefill backend`), so prefill rows run dense bf16 MHA over
   `kv_b_proj` outputs, without the fp8 rounding of the cache. Switch: `--attention-config '{"sparse_mla_force_mqa": true}'`.

Validation is a teacher-forced prefill of the chain, so the validator side moved on both kernels; generation moved
through the prompt prefill and the KV entries it writes. Bisect on the 8×H100 box, one container boot per cell, the ten
0.28.1 H100 references re-validated each time (points % at τ = 0, mean [range]):

| validator | prover corpora | points % |
| --- | --- | ---: |
| 0.28.1 stack, this box | 0.28.1 references | 9.87 [9.06–10.52] |
| 0.30 defaults (FlashKDA + dense-MHA prefill) | 0.28.1 references | 12.68 [9.77–18.25] |
| 0.30, `--moe-backend marlin` | 0.28.1 references | 11.87 [8.35–18.15] |
| 0.30, `pass_config.fuse_allreduce_rms=false` | 0.28.1 references | 12.58 [9.92–18.54] |
| 0.30, `--kda-prefill-backend triton` only | 0.28.1 references | 13.03 [9.15–19.54] |
| 0.30, `sparse_mla_force_mqa` only | 0.28.1 references | 13.33 [9.68–19.36] |
| **0.30, both flags** | 0.28.1 references | **9.88 [9.09–10.79]** |
| 0.28.1 stack, this box | 0.30-default corpora (h01–h03) | 11.11 [11.01–11.20] |
| 0.28.1 stack, this box | 0.30-default corpora (h04, h08, h09) | 16.76 [16.19–17.73] |
| 0.30, both flags | 0.30-default corpora (h01–h03) | 11.23 [11.05–11.44] |
| 0.30, `--kda-prefill-backend triton` | 0.30-default corpora (h01–h03) | 10.62 [10.34–10.83] |
| 0.30, `sparse_mla_force_mqa` | 0.30-default corpora (h01–h03) | 15.76 [10.75–19.01] |
| 0.30, `marlin` | 0.30-default corpora (h01–h03) | 9.56 [9.20–10.12] |
| 0.30, `marlin`, same boot | own corpora (h04, h08, h09) | 7.04 [6.47–7.51] |

Reading:

- With both flags the 0.30 validator reproduces the 0.28.1 validator on every hash; the outlier hashes return to 10%.
  The two kernel changes are the whole cause: nothing else in the diff (mHC TileLang rewrite, DeepGEMM pin, FlashInfer
  0.6.18 → 0.6.18.post1, torch nightly) is visible in this statistic.
- Each kernel alone is a cross-hardware-sized shift against corpora made with the other kernel, so a fleet must run
  the same prefill kernels on both sides; mixed provers and validators sit at the freeze's cross-hardware level, and the
  mixed pair 0.28.1 ↔ 0.30 has its own outlier hashes (h04, h08, h09), as every hardware pair in the freeze had.
- The three outlier hashes are ordinary on 0.30 itself (same-boot 6.5–7.5%), so the gap is specific to the
  0.28.1 → 0.30 change of the GLM numerics and depends on the input.
- One tilelang-related cell is invalid: on CUDA 0.30 has no mHC fallback (`ImportError: tilelang is required for mhc`).
- The same finding transfers to 2×B300 unchanged, with the same two switches (14.53% → 6.53%).

**Decision (Kaitaku with [@vbgd0](https://github.com/vbgd0), 2026-09-25): pin both flags in the GLM profile and keep the
15 September artifacts.** Recorded in the GLM launch profile (`PROFILES.md`, every GLM boot) and in the plugin fork's
decision log. The alternative, keeping the 0.30 defaults for their faster prefill and regenerating the GLM artifacts on
0.30 across the fleet, was not taken. In both cases every node must run the same setting.

## Round, gates, replay

Every cell passed the same three checks as the freeze points:

- **Gates** (`r1_gates.json` in `artifacts/provenance/`): 8 artifacts of 257 steps, k in 0..15, no NaN steps;
  self-validation of 64 nonces at the card's level (MiniMax B300 0.0% at TP 1, H100 6.15% at TP 4; DeepSeek NVFP4 B300
  1.71%, FP8 H100 8.4%; GLM H100 5.9%, B300 7.0%). Free-generation repeats are bit-identical only at TP 1 with the
  autotuner off (MiniMax B300); at TP > 1 or with the autotuner on the chains fork at the first differing step, which the
  kit records and does not require. The kit's verdict `FAIL` on GLM B300 is a bookkeeping artifact: the synthesized
  provenance lacked `tp`, so the TP 1 bit-exact rule was applied to a TP 2 boot.
- **Mining round** through `/api/v1/pow/init/generate`: `GENERATING`, chat answers 503 during the round and 200 after
  `/stop`, on every cell. The first chat request after PoC work exceeded 60 s on both boxes (compile of the chat path,
  one-off).
- **Replay equivalence** (24 originals replayed on the same server, `scripts/b300/run_replay.py`): 24 of 24 PASS on
  every cell; similarity 1.0 on MiniMax H100, minimum 0.9815 on MiniMax B300 (one EOS-terminated replay), 0.969–0.975
  on DeepSeek and GLM, median 1.0 or 0.985.

## Profile changes on 0.30

- **MiniMax needs more memory.** The frozen `gpu-memory-utilization` fails: 0.97 on B300 dies in the FlashInfer
  autotuner (`trtllm_fp8_block_scale_moe` asks 3.4 GB with 1.7 GB free) and then at graph capture; 0.94 on 4×H100 runs
  a 250-nonce teacher-forced validation out of memory (382 MiB requested, 316 MiB free) and degrades into the known
  `prev_k` race. One step down: 0.95 on B300, 0.90 on H100. DeepSeek and GLM are not memory-bound.
- **GLM and DeepSeek FP8 on Hopper run from the image**, not the wheel (Code).
- **GLM on 0.30 pins the 0.28.1 prefill kernels**: `--kda-prefill-backend triton --attention-config '{"sparse_mla_force_mqa": true}'`
  on every node.
- **First boot on Nebius B300** compiles the trtllm fused-MoE kernels for sm_103a (about 25 minutes) and needs
  `python3.12-dev` on the image; GLM on 2×B300 with `max-num-batched-tokens` 65536 fails in the trtllm FMHA
  (`CUDA_ERROR_INVALID_VALUE`), 32768 boots.
- **DeepSeek warm-up must equal the run size**: TileLang JIT inside the run halves the first points.
- The plugin installs with `pip install --no-deps`; `scipy` is added to the image.

Carried-over profiles (2×H200, 2×B200, 4×B200, 4×A100) were not re-measured on 0.30; their rows in `PROFILES.md` are the
frozen settings with the memory step noted as a guess to verify before pinning.

---

## Reproduce

Points % and the same-card nonce shares of every cell are rebuilt from `artifacts/` without a GPU:

```bash
python3 scripts/common/tau_cells.py artifacts/validations/minimax/validator_b300_030/prover_b300_0251
```

`artifacts/validations/<model>/validator_<card>_<stack>/prover_<card>_<stack>/` holds the kit's validation summary
(`r2_validate_summary_*.json`: points and mismatching points per hash) and, for the self, both-flags and 0.28.1-stack
cells, the `npz_*` counters (mismatching points of each nonce on the kit's τ grid, plus the compared-point count).
DeepSeek cells are the `xq_val_*` summaries of the DeepSeek kit (per-hash rates and the point-level τ grid).
`artifacts/provenance/` holds the boot records (KV capacity in tokens, N, MoE backend, GPU, checkpoint fingerprint) of
every generation, validation and throughput boot, the gate outputs and the corpus manifests; `artifacts/perf/<card>/`
the PoC runs (`perf_poc_*.json`) and chat sweeps (`r_chat_sweep_*.json`, `vllm bench serve` outputs);
`artifacts/stage/` the plan logs of the runs. `scripts/` are the box scripts as run: environment set-up
(`h100/mkvenv.sh`, `h100/setup030.sh`, `b300/prep_vast.sh`), boots, the staged runs (`common/stage_mm.sh`,
`h100/stage_ds.sh`), the GLM bisect (`h100/glm_bisect.sh`, `h100/glm_cellC_then_old.sh`, `h100/glm_cellD.sh`) and
the replay check.

## Data not in this repository

Corpora with per-step vectors (`corp_honest_*_gen.json`, 44 MB each), the per-nonce validation records
(`val_honest_*_v250.json`), the replay originals and responses, and the server and boot logs of every cell are too
large for the tree: 6.7 GB local. They go into the [poc-devkit-0300-2026-09-25 on Google Drive](https://drive.google.com/drive/folders/18Kbd796D42Bf4zd4daDBMNBAnFTAbUeD) (6.7 GB, sha256 manifest) together
with the residual and plugin tarballs used on the boxes, the reference corpora and the launch profiles. The npz counters
of the mixed-kernel GLM cells (0.30 defaults or one flag against the 0.28.1 references) are kept in the bundle, not
in this tree.
