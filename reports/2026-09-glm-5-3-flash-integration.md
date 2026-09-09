# GLM-5.3-Flash integration — vLLM 0.28 image, PoC on Hopper, release inputs and proposal (2026-08-26 … 2026-09-08)

## Summary

The [kaitaku.ai](https://github.com/kaitakuai) team (Mykola [@baychak](https://github.com/baychak), Pavlo [@clanster](https://github.com/clanster)) brought [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) to a releasable state at the request of [@gmorgachev](https://github.com/gmorgachev) (2026-08-26) and [@vbgd0](https://github.com/vbgd0) (2026-08-27, 08-31, 09-03). The work covers:

- an MLNode image on vLLM 0.28 built from the `vllm/vllm-openai:glm53-flash` branch — the model (`Glm5NextForConditionalGeneration`) is absent from every vLLM release, so no 0.25.x build can load it;
- PoC on Hopper: the failure with FlashInfer 0.6.17 traced to the FP8 KV-cache path, fixed by FlashInfer 0.6.18 (released 2026-08-29); batch ceilings established per GPU generation;
- a hardware profile on four topologies (2×B300, 4×B200, 4×H200, 8×H100) with honest, NVFP4 and REAP50 arms, and a cross-hardware inference-validation run (2×B300 → 4×H200, 4000 generations) — 11 experiment directories;
- the PoC sampler residual ported to the GLM base ([gonka-ai/vllm#105](https://github.com/gonka-ai/vllm/pull/105), [#106](https://github.com/gonka-ai/vllm/pull/106)), vLLM 0.28 compatibility in the plugin ([gonka-ai/gonka-vllm-plugins#9](https://github.com/gonka-ai/gonka-vllm-plugins/pull/9)), MLNode on the new residual ([gonka-ai/gonka#1724](https://github.com/gonka-ai/gonka/pull/1724));
- governance proposal [#101](https://gonka.gg/network/proposals/101) — add GLM-5.3-Flash, remove Kimi-K2.6 and GLM-5.2-FP8 from `poc_params.models`.

Release inputs and the consensus parameters were independently recomputed by [@vbgd0](https://github.com/vbgd0) from the committed artifacts in [gonka-ai/gonka#1734](https://github.com/gonka-ai/gonka/pull/1734). Proposal #101 was submitted on 2026-09-08 (voting until 2026-09-10); activation epoch 394.

---

## Image bring-up on vLLM 0.28 (2026-08-26 … 2026-08-31)

vLLM support for GLM-5.3-Flash exists only on the `glm53-flash` branch of the official image ([vllm-project/vllm#53906](https://github.com/vllm-project/vllm/pull/53906)). Three images were built as FlashInfer moved from 0.6.17 to 0.6.18.

| Image | vLLM | FlashInfer | Purpose |
| --- | --- | --- | --- |
| `ghcr.io/kaitakuai/vllm-poc:glm53-poc-v4-ed8873884` | `0.28.0.dev0+glm53.gonka.sampler1` | 0.6.17 | first PoC run on Blackwell (2×B300) |
| `ghcr.io/kaitakuai/mlnode-b300-glm-5-3-flash:0.2.14-vllm0.28-glm53-test-k3` | `0.28.0.dev0+glm53.gonka.sampler1` | 0.6.18 | Blackwell profiles (2×B300, 4×B200) |
| `ghcr.io/kaitakuai/mlnode-h100-glm-5-3-flash:0.2.14-vllm0.28-glm53-test-k3` | `0.28.0.dev0+glm53.gonka.sampler1` | 0.6.18 | Hopper profiles (4×H200, 8×H100) |

**Conclusion.** The residual + `gonka-poc` plugin stack ports to a non-release vLLM base without engine changes; the model-specific work was the KV-cache / FlashInfer combination and the sparse-MLA indexer patch described below.

### PoC on Hopper: FP8 KV cache and FlashInfer 0.6.18 (2026-08-27 … 2026-08-31)

On Hopper with FlashInfer 0.6.17, PoC generation produced up to ~100 nonces and then died with a CUDA illegal memory access. Blackwell was unaffected up to batch 32.

| GPU | KV cache | FlashInfer | Attention backend | Result |
| --- | --- | --- | --- | --- |
| 2×B300, 4×B200 | fp8 | 0.6.17 / 0.6.18 | `FLASHINFER_MLA_SPARSE` | PoC works, batch up to 32 |
| 4×H200, 8×H100 | fp8 | 0.6.17 | `FLASHINFER_MLA_SPARSE_SM90` | illegal memory access after ~100 nonces |
| 4×H200 | fp8 | 0.6.18 | `FLASHINFER_MLA_SPARSE_SM90` | **PoC works, batch 16, 1439 nonces/min** ([2026-09/glm53-flash-fp8-4xh200](https://github.com/kaitakuai/experiments/tree/main/2026-09/glm53-flash-fp8-4xh200)) |
| 8×H100 | fp8 | 0.6.18 | `FLASHINFER_MLA_SPARSE_SM90` | PoC works, batch 16 (`--gpu-memory-utilization 0.95` required) |

`--max-num-batched-tokens` must be at least `PoC batch × 1024`; below that the run yields zero nonces rather than degrading, and on 4×H200 a batch of 24 with the shipped 16384 budget raises `CUDA_ERROR_ILLEGAL_ADDRESS` (XID 31) on all four GPUs. Batch 48 collapses on every arm regardless of budget (documented in [gonka-ai/gonka#1734](https://github.com/gonka-ai/gonka/pull/1734)).

**Conclusion.** Hopper needs FlashInfer 0.6.18 and an FP8 KV cache; the working batch limits are 16 on Hopper and 32 on Blackwell, and the node-config files in #1734 carry exactly these values.

---

## Hardware profile and separability (2026-08-26 … 2026-09-02)

Honest runs on four topologies, two fraud arms (NVFP4 quantisation, REAP50 expert pruning), three shared seeds, 1000 nonces per seed. Per [gonka-ai/gonka#1734](https://github.com/gonka-ai/gonka/pull/1734), every aggregate was recomputed by the core team from these artifacts.

| GPU (TP) | Image | PoC batch | Nonces/min | Directory |
| --- | --- | ---: | ---: | --- |
| 2×B300 (2) | `glm53-poc-v4-ed8873884` | 32 | 2030 | [2026-08/glm53-flash-fp8-2xb300](https://github.com/kaitakuai/experiments/tree/main/2026-08/glm53-flash-fp8-2xb300), re-analysed in [2026-09/glm53-flash-fp8-2xb300](https://github.com/kaitakuai/experiments/tree/main/2026-09/glm53-flash-fp8-2xb300) |
| 4×B200 (4) | `…glm53-test-k3` | 32 | **2727** ★ | [2026-09/glm53-flash-fp8-4xb200](https://github.com/kaitakuai/experiments/tree/main/2026-09/glm53-flash-fp8-4xb200) |
| 4×H200 (4) | `…glm53-test-k3` | 16 | 1439 | [2026-09/glm53-flash-fp8-4xh200](https://github.com/kaitakuai/experiments/tree/main/2026-09/glm53-flash-fp8-4xh200) |
| 8×H100 (8) | `…glm53-test-k3` | 16 | 1775 | [2026-09/glm53-flash-fp8-8xh100](https://github.com/kaitakuai/experiments/tree/main/2026-09/glm53-flash-fp8-8xh100) |

Fraud arms: [2026-08/glm53-flash-nvfp4-libertai-2xb300](https://github.com/kaitakuai/experiments/tree/main/2026-08/glm53-flash-nvfp4-libertai-2xb300), [2026-09/glm53-flash-nvfp4-libertai-4xb200](https://github.com/kaitakuai/experiments/tree/main/2026-09/glm53-flash-nvfp4-libertai-4xb200), [2026-09/glm53-flash-reap50-patrickbdevaney-4xh200](https://github.com/kaitakuai/experiments/tree/main/2026-09/glm53-flash-reap50-patrickbdevaney-4xh200). The matrix is consolidated in [2026-09/glm53-flash-cross-hardware-summary](https://github.com/kaitakuai/experiments/tree/main/2026-09/glm53-flash-cross-hardware-summary).

Two facts from the matrix shaped the release inputs (both stated in #1734): honest runs of the same seed on the same box are bit-identical except at `index % 16 == 0` — the first sequence of each collection batch — on 4×H200 and 4×B200 (63 of 1000 nonces), but only 1 of 1000 on 8×H100 at TP=8; and honest pairs across GPU generations sit at a measurable floor while both fraud arms sit well above it. The golden reference artifact in #1734 is therefore baked from the 8×H100 TP=8 arm.

**Conclusion.** Separability holds across three GPU generations for both fraud arms; at the gate proposed in #1734 (`dist_threshold` 0.44) the worst honest cross-generation pair is 11.7 % past the gate and the mildest fraud arm 31.7 %.

### Inference validation (2026-09-02)

1000 multilingual prompts generated on 2×B300 and replayed on 4×H200 for the honest model and the NVFP4 build, in both logprobs modes — 4000 generations, 4000 replays, 0 length mismatches ([2026-09/glm-5-3-flash-inference-validation](https://github.com/kaitakuai/experiments/tree/main/2026-09/glm-5-3-flash-inference-validation)).

| Mode | Honest mean `distance2` | NVFP4 mean | Ratio | Best F1 |
| --- | ---: | ---: | ---: | ---: |
| processed | 0.024969 | 0.054328 | 2.18× | 0.918 |
| raw | 0.029738 | 0.076863 | 2.58× | 0.998 |

**Conclusion.** With `ValidationParams.logprobs_mode = processed_logprobs` unchanged, a `validation_threshold` of 0.951 rejects 0.5 % of honest answers and catches 64.2 % of NVFP4 substitutions per replay (per #1734).

---

## Upstream code and release branch (2026-09-05 … 2026-09-08)

Because the base is a branch, not a release, a dedicated release line was agreed: S0 = the `glm53-flash` vLLM image, S1 = residual, S2 = residual + plugin + FlashInfer 0.6.18. [@vbgd0](https://github.com/vbgd0) created [`release/v0.28.0-glm53`](https://github.com/gonka-ai/vllm/tree/release/v0.28.0-glm53) on 2026-09-07.

| PR | Title | Author | Status |
| --- | --- | --- | --- |
| [gonka-ai/vllm#104](https://github.com/gonka-ai/vllm/pull/104) | chore(base): upstream vLLM at the GLM-5.3-Flash residual base (933876c38) | [@baychak](https://github.com/baychak) | merged 2026-09-07 (base sync) |
| [gonka-ai/vllm#105](https://github.com/gonka-ai/vllm/pull/105) | feat(poc): port the PoC sampler residual to the GLM-5.3-Flash base | [@baychak](https://github.com/baychak) | merged 2026-09-07 |
| [gonka-ai/vllm#106](https://github.com/gonka-ai/vllm/pull/106) | fix(poc): fold the GLM-5.3-Flash Stage-4 fixes into the residual | [@baychak](https://github.com/baychak), four commits by [@clanster](https://github.com/clanster) | merged 2026-09-08 |
| [gonka-ai/gonka-vllm-plugins#9](https://github.com/gonka-ai/gonka-vllm-plugins/pull/9) | feat(compat): support vLLM 0.28 for the GLM-5.3-Flash base | [@baychak](https://github.com/baychak) | merged 2026-09-08 |
| [gonka-ai/gonka#1724](https://github.com/gonka-ai/gonka/pull/1724) | feat(mlnode): GLM-5.3-Flash on the vLLM 0.28 residual | [@baychak](https://github.com/baychak) | open |
| [gonka-ai/gonka#1734](https://github.com/gonka-ai/gonka/pull/1734) | GLM-5.3-Flash: release inputs and proposed consensus parameters | [@vbgd0](https://github.com/vbgd0) (core) | open; node-config files for B300, B200, H200, 8×H200, H100 |

[@vbgd0](https://github.com/vbgd0) rebuilt and re-ran the stack on 2026-09-08: nonces match. Tracking issue: [gonka-ai/gonka#1691](https://github.com/gonka-ai/gonka/issues/1691).

**Conclusion.** Everything needed to build the release from source is upstream; no Kaitaku image is part of the handover.

### Governance entry (chain side)

Proposal [#101](https://gonka.gg/network/proposals/101) (submitted 2026-09-08, voting until 2026-09-10), built on the parameter set of proposal #100 so that devshard v4.1 is carried unchanged:

| Parameter | Value |
| --- | --- |
| `poc_params.models` | add `zai-org/GLM-5.3-Flash` (`seq_len` 1024, `dist_threshold` 0.44, `p_mismatch` 0.10, `p_value_threshold` 0.05, `weight_scale_factor` 0.62, `penalty_start_epoch` 394); remove `moonshotai/Kimi-K2.6` and `zai-org/GLM-5.2-FP8` |
| `MsgRegisterModel` | `zai-org/GLM-5.3-Flash` @ `04c4e9e95c5da8862dced7e5056455116f83a7e0`, `validation_threshold` 0.951, `--max-model-len 400000`, `--kv-cache-dtype fp8`, GLM tool-call and reasoning parsers |

`weight_scale_factor` 0.62 was calibrated by [@vbgd0](https://github.com/vbgd0) so that a B200 host switching to GLM-5.3-Flash gains about 7 % weight; for other GPUs the optimal model does not change. MiniMax-M2.7 and DeepSeek-V4-Flash-0731 parameters are untouched.

**Conclusion.** The model and its parameters were proposed by Kaitaku, independently validated by the core team, and submitted jointly; activation is scheduled for epoch 394 (2026-09-15).

---

## Participants

[kaitakuai](https://github.com/kaitakuai) ([@baychak](https://github.com/baychak), [@clanster](https://github.com/clanster)).
Gonka core team ([@gmorgachev](https://github.com/gmorgachev), [@vbgd0](https://github.com/vbgd0), [@mtvnastya](https://github.com/mtvnastya), [@tcharchian](https://github.com/tcharchian)).

| Participant | GitHub | Role | Contribution |
| --- | --- | --- | --- |
| Pavlo | [@clanster](https://github.com/clanster) | kaitakuai | Image bring-up on the `glm53-flash` branch, Hopper PoC fix (FP8 KV + FlashInfer 0.6.18, indexer patch), all 11 experiment directories, inference validation, four Stage-4 fixes in #106 |
| Mykola | [@baychak](https://github.com/baychak) | kaitakuai | Image chain (S0–S2), Hopper diagnosis on 4×H200, residual port #105/#106, plugin compat #9, MLNode #1724, release-branch plan, proposal #101 |
| Gleb Morgachev | [@gmorgachev](https://github.com/gmorgachev) | Gonka core team | Request, review of the proposal (registry args, devshard v4.1 ordering) |
| Vladislav Bogdanov | [@vbgd0](https://github.com/vbgd0) | Gonka core team | Release branch `release/v0.28.0-glm53`, independent re-run and verification, #1734, coefficient calibration |
| Anastasia Matveeva | [@mtvnastya](https://github.com/mtvnastya) | Gonka core team | Proposal timing and text, coefficient table, removal of Kimi-K2.6 / GLM-5.2-FP8 |
| Tania Charchian | [@tcharchian](https://github.com/tcharchian) | Gonka core team | Issue #1691 |
