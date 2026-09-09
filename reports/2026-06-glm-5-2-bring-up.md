# GLM-5.2 bring-up on vLLM 0.23.0 — PoC, context and images (2026-06-22 … 2026-06-30)

## Summary

The [kaitaku.ai](https://github.com/kaitakuai) team (Mykola [@baychak](https://github.com/baychak), Pavlo [@clanster](https://github.com/clanster)) brought `zai-org/GLM-5.2-FP8` up on the Gonka PoC v2 stack at the request of the core team (2026-06-22; release context 450k tokens and an inference-validation threshold requested on 2026-06-24). The work covers:

- Migration of the image chain to vLLM 0.23.0 with the PoC code moved into the `gonka-poc` pip plugin — the first plugin-based profiles (`mlnode-b300-minimax-m2-7:0.2.13-vllm0.23.0-k1`, `mlnode-b300-kimi-k2-6:0.2.13-vllm0.23.0-k1`), including a Starlette 1.3 middleware fix in the plugin
- Nine experiment directories under [`2026-06/glm-5.2-*`](https://github.com/kaitakuai/experiments/tree/main/2026-06): PoC and inference on 8×B200, 4×B300 and 8×H200; eager vs CUDA graphs; DeepGEMM; full 1M vs 450k context; cross-hardware L2; an AWQ-INT4 arm; inference validation in both logprobs modes
- The enforced-tokens ingestion fix [kaitakuai/vllm#13](https://github.com/kaitakuai/vllm/pull/13) that repaired inference validation on plugin images
- Deployable images for B300, B200 and H200 (`mlnode-foundry` #70–#75), offered for announcement on 2026-06-30

GLM-5.2 entered governance through proposal #79 «Add Kimi K2.6 and GLM 5.2 model» (submitted 2026-06-26, PASSED). The model never bootstrapped on the network; proposal #101 (2026-09-08) removes `zai-org/GLM-5.2-FP8` from `poc_params.models`.

---

## Image chain on vLLM 0.23.0 with the `gonka-poc` plugin (2026-06-21 … 2026-06-24)

The 0.23.0 migration was done so that everything vLLM lets a plugin own left the fork; what remains in-tree is the `residual/vllm-poc` branch of [kaitakuai/vllm](https://github.com/kaitakuai/vllm/tree/residual/vllm-poc). The dedicated GLM image bakes the GLM profile into `runner.py`; earlier runs used the Kimi image with a runner patch ([`glm-5.2-poc-backend-sweep/scripts/runner_patch_glm.py`](https://github.com/kaitakuai/experiments/tree/main/2026-06/glm-5.2-poc-backend-sweep)).

| Resource | Link |
|----------|------|
| PoC plugin | `gonka-poc` (pip; pinned by digest in each profile) |
| Residual | [`kaitakuai/vllm:residual/vllm-poc`](https://github.com/kaitakuai/vllm/tree/residual/vllm-poc) |
| GLM profile | [mlnode-foundry#70](https://github.com/kaitakuai/mlnode-foundry/pull/70) `b200-glm-5-2`, [#72](https://github.com/kaitakuai/mlnode-foundry/pull/72) inference compiled by default, PoC eager |
| Validation fix | [kaitakuai/vllm#13](https://github.com/kaitakuai/vllm/pull/13) `enforced_tokens` request ingestion → [mlnode-foundry#73](https://github.com/kaitakuai/mlnode-foundry/pull/73) |
| B300 / H200 | [mlnode-foundry#74](https://github.com/kaitakuai/mlnode-foundry/pull/74), [#75](https://github.com/kaitakuai/mlnode-foundry/pull/75) (`kv-cache-dtype fp8` + `mnbt 16384` on H200 after Pavlo's test) |

**Conclusion.** The residual carried the sampler-side enforced-token engine but not the HTTP→`SamplingParams` bridge, so validators' `enforced_tokens` were silently dropped; #13 closed that and the inference-validation runs below confirm 0 token mismatches on the fixed image.

---

## PoC and inference per GPU (2026-06-22 … 2026-06-24)

`vllm bench serve`, 200 requests, 1024→256 tokens, concurrency 32. Hardware rented on Vast.ai.

| GPU (TP) | Image | Backend | PoC eager | PoC CUDA graphs | Inference, CUDA graphs | Notes |
|----------|-------|---------|----------:|----------------:|-----------------------:|-------|
| [8×B200 (8)](https://github.com/kaitakuai/experiments/tree/main/2026-06/glm-5.2-fp8-8xb200) | Kimi image + runner patch | triton | 1016 | 768 (−24 %) | 817 tok/s, TPOT 25 ms (eager: 157 tok/s, 170 ms) | DeepGEMM crashed in the linear kernel, FlashInfer-CUTLASS MoE hung |
| [8×B200 (8)](https://github.com/kaitakuai/experiments/tree/main/2026-06/glm-5.2-deepgemm-8xb200) | `mlnode-b200-glm-5-2:0.2.13-vllm0.23.0-k1` | DeepGEMM | **1517** ★ | 1054 | 583 tok/s, TPOT 25 ms (eager: 196 tok/s, 77 ms) | same host measured 928 with the previous plugin — the gain is the plugin, not the kernel |
| [8×B300 (2 × TP=4)](https://github.com/kaitakuai/experiments/tree/main/2026-06/glm-5.2-deepgemm-4xb300) | same | DeepGEMM | 896 per engine, **1792 per box** | — | 556 tok/s, TPOT 42 ms | TP=8 on the same box: 1078 |
| [8×H200 (8)](https://github.com/kaitakuai/experiments/tree/main/2026-06/glm-5.2-fp8-8xh200) | Kimi image + runner patch | triton | 576 | 512 | 561 tok/s at 450k context, TPOT 42 ms | `fp8_e4m3` KV rejected by FLASHMLA_SPARSE; `fp8_ds_mla` needed for long context |

**Conclusion.** Eager for PoC, CUDA graphs for inference — the policy baked into the profiles by #72. On B300 the 753B FP8 weights fit on four cards, so an 8-GPU box runs two TP=4 engines and gives 66 % more nonces than one TP=8 engine.

---

## Context: 1M vs 450k (2026-06-24)

The release context had to fit 8×H200 (about 400k tokens, per the core team).

| GPU (TP) | KV dtype | KV cache | Outcome |
|----------|----------|---------:|---------|
| 4×B300 (4) | `fp8_e4m3` | 1 124 928 tokens | full 1 048 576 context fits; PoC 864 vs 896 nonces/min at 32k — no measurable cost |
| 8×H200 (8) | bf16 | 553 668 tokens | 553k max |
| 8×H200 (8) | `fp8_ds_mla` | 711 167 tokens | serving stable up to 450k; a 600k bench returned prefill-only |

**Conclusion.** 450k was taken as the release context (Gleb, 2026-06-24); the images were later built with `--max-model-len 400000` to match the governance entry.

---

## Cross-hardware L2 and validation arms (2026-06-23 … 2026-06-25)

Canonical L2 with `k_dim 12`, gate `dist_threshold 0.4 / p_mismatch 0.02`, 1000 nonces per pair.

| Pair | mean L2 | past gate | Verdict |
|------|--------:|----------:|---------|
| FP8 H200 ↔ FP8 B200 | 0.188 | 0.7 % | PASS |
| FP8 B300 ↔ FP8 B200 | 0.193 | 0.8 % | PASS |
| FP8 B300 ↔ FP8 H200 | 0.189 | 0.9 % | PASS |

An AWQ-INT4 arm ([`glm-5.2-awq4bit-8xb200`](https://github.com/kaitakuai/experiments/tree/main/2026-06/glm-5.2-awq4bit-8xb200), TP=4) was measured against the B200, H200 and B300 honest sets. Inference validation ([`glm-5.2-inference-validation`](https://github.com/kaitakuai/experiments/tree/main/2026-06/glm-5.2-inference-validation)) replayed 1000 prompts in five languages between 8×B200 and 8×H200 in both `processed_logprobs` and `raw_logprobs` modes; a paired KV-dtype ablation (bf16 vs `fp8_ds_mla`, 200 prompts) moved the honest mean by 0.3 %, so KV dtype is not a factor in the cross-hardware spread.

**Conclusion.** Honest PoC vectors are interchangeable across B200, H200 and B300; the AWQ arm is separated by the PoC gate on every reference. Validation-side numbers were shared with the core team on 2026-06-25.

---

## Chain entry and what did not happen

- Proposal #79 «Add Kimi K2.6 and GLM 5.2 model» (submitted 2026-06-26) passed; the registry entry carries `--max-model-len 400000`, `validation_threshold 0.75`.
- Images for B300, B200 and H200 were ready on 2026-06-30 and offered for announcement; no announcement was requested.
- The model did not bootstrap on the network; proposal #101 (2026-09-08) removes it from `poc_params.models` so hosts stop paying the missing-delegation penalty.

---

## Participants

[kaitakuai](https://github.com/kaitakuai) ([@baychak](https://github.com/baychak), [@clanster](https://github.com/clanster)).
Gonka core team ([@gmorgachev](https://github.com/gmorgachev), [@tcharchian](https://github.com/tcharchian), [@mtvnastya](https://github.com/mtvnastya)).

| Participant | GitHub | Role | Contribution |
|-------------|--------|------|--------------|
| Pavlo | [@clanster](https://github.com/clanster) | kaitakuai | All nine experiment directories: PoC sweeps on B200 / B300 / H200, context investigation, cross-hardware L2, AWQ arm, inference validation, H200 profile parameters |
| Mykola | [@baychak](https://github.com/baychak) | kaitakuai | vLLM 0.23.0 migration and `gonka-poc` plugin packaging, kaitakuai/vllm#13, mlnode-foundry #70–#75, B300 / B200 / H200 images |
| Gleb Morgachev | [@gmorgachev](https://github.com/gmorgachev) | Gonka core team | Request, `gm/port-pocv2-vllm-0.23.0` branch, 450k context decision, inference-threshold requirement |
| Tania Charchian | [@tcharchian](https://github.com/tcharchian) | Gonka core team | Timeline tracking (2026-06-24) |
| Anastasia Matveeva | [@mtvnastya](https://github.com/mtvnastya) | Gonka core team | Proposal #79 |
