# DeepSeek-V4-Flash integration — PoC, vLLM 0.25.1 port and governance entry (2026-07-17 … 2026-08-27)

## Summary

The [kaitaku.ai](https://github.com/kaitakuai) team (Mykola [@baychak](https://github.com/baychak), Pavlo [@clanster](https://github.com/clanster)) brought `deepseek-ai/DeepSeek-V4-Flash` to a governance-approved PoC model at the request of the Gonka core team (2026-07-17 … 07-31). The work covers:

- PoC support for the DeepSeek-V4 model family (per-group KV metadata, positions, pseudo ids)
- First model shipped on the residual + `gonka-poc` plugin stack for vLLM 0.25.1 (the port itself is covered in the [plugin report](https://github.com/kaitakuai/experiments/blob/main/reports/2026-07-gonka-poc-plugin-and-residual.md))
- Two experiment campaigns: 12 directories on the original checkpoint, 10 on `DeepSeek-V4-Flash-0731` — four GPU topologies, fraud arms, inference validation, seed stability, DSpark
- Replay hooks for the V2 model runner, needed for DSpark ([gonka-ai/vllm#92](https://github.com/gonka-ai/vllm/pull/92)), and follow-up fixes
- Release candidates and verification of the core team's rc1 / rc3 images on Kaitaku hardware
- Governance: proposal #94 (model added), proposals #97 / #98 (`weight_scale_factor` correction after the nonce/min measurement fix by Vlad)

DeepSeek-V4-Flash-0731 was added to `poc_params.models` by proposal #94 (PASSED, submitted 2026-08-10; `penalty_start_epoch` 360). Proposal #98 (PASSED, 2026-08-27) raised `weight_scale_factor` from 0.214 to 0.246.

---

## PoC on DeepSeek-V4 and the vLLM 0.25.1 port (2026-07-17 … 2026-07-31)

### PoC fix for the DeepSeek-V4 family (2026-07-18 … 2026-07-20)

Earlier runs of DeepSeek-V4-Flash produced no PoC. Pavlo extended the PoC runner in the `gonka-poc` plugin so that the V4 code path is covered without model-specific customisation.

| Resource | Link |
|----------|------|
| Plugin change | `kaitakuai/gonka-poc#14` (commit `3a119d8`, now in `main` of [gonka-ai/gonka-vllm-plugins](https://github.com/gonka-ai/gonka-vllm-plugins)) |
| First working image | `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash:0.2.13-vllm0.25.1-overlay-k4` |
| Cross-implementation check | [`2026-07/deepseek-v4-flash-poc-1xb300`](https://github.com/kaitakuai/experiments/tree/main/2026-07/deepseek-v4-flash-poc-1xb300) — plugin, in-tree fork and `qd` port agree (median L2 0.0024 / 0.0000, p = 1.000) |

**Conclusion.** PoC generation and validation on DeepSeek-V4 agree with the in-tree fork and the `qd` port within the cross-validation gate; the fix is generic, not per model.

### Port to vLLM 0.25.1 (2026-07-21 … 2026-07-31)

Two variants were submitted against `release/v0.25.1`: a full in-tree port with DeepSeek-V4 support ([gonka-ai/vllm#65](https://github.com/gonka-ai/vllm/pull/65), Pavlo, 45 files) and a thin residual plus the out-of-tree `gonka-poc` plugin ([#66](https://github.com/gonka-ai/vllm/pull/66) → [#78](https://github.com/gonka-ai/vllm/pull/78), Mykola). The core team chose the residual; DeepSeek-V4 support lives in the plugin (`kaitakuai/gonka-poc#14`, Pavlo) and shipped with plugin `v0.1.1`. The port, the fix stack and the plugin transfer are described in the [plugin report](https://github.com/kaitakuai/experiments/blob/main/reports/2026-07-gonka-poc-plugin-and-residual.md) and are not repeated here.

**Conclusion.** DeepSeek-V4 was the first model to ship on the residual + plugin stack; the model-specific code is confined to the plugin.

---

## Experiments (2026-07-24 … 2026-08-11)

### Original checkpoint — PoC throughput and validation (2026-07-24 … 2026-07-27)

Honest arm `deepseek-ai/DeepSeek-V4-Flash`, image `overlay-k4`.

| GPU (TP) | vLLM | Best batch | Nonces/min | Notes |
|----------|------|-----------:|-----------:|-------|
| [1×B300 (1)](https://github.com/kaitakuai/experiments/tree/main/2026-07/deepseek-v4-flash-poc-1xb300) | 0.25.1 | 32 | 1472 | ~1.6× the 0.20.0 image |
| [4×H100 (4)](https://github.com/kaitakuai/experiments/tree/main/2026-07/deepseek-v4-flash-4xh100) | 0.25.1 | 32 | 1536 | eager = CUDA graphs for PoC |
| [2×H200 (2)](https://github.com/kaitakuai/experiments/tree/main/2026-07/deepseek-v4-flash-2xh200) | 0.25.1 | 32 | 1216 | |
| [2×B200 (2)](https://github.com/kaitakuai/experiments/tree/main/2026-07/deepseek-v4-flash-2xb200) | 0.25.1 | 32 | **2304** ★ | +71 % with CUDA graphs |

Also in the campaign: a CUDA-graph A/B on 1×B300 ([`deepseek-v4-flash-1xb300-cudagraph-ab`](https://github.com/kaitakuai/experiments/tree/main/2026-07/deepseek-v4-flash-1xb300-cudagraph-ab)), five-seed stability with the batch-boundary artifact quantified ([`deepseek-v4-seed-stability-1xb300`](https://github.com/kaitakuai/experiments/tree/main/2026-07/deepseek-v4-seed-stability-1xb300)), `l2_crossval.py` aligned to the chain's arithmetic, and inference validation executor B300 → validator 2×H200 ([`deepseek-v4-flash-inference-validation`](https://github.com/kaitakuai/experiments/tree/main/2026-07/deepseek-v4-flash-inference-validation), 0/6000 length mismatches). Fraud arms (NVFP4, W4A16-AutoRound, INT4) were measured on each topology; results are in the experiment directories.

**Conclusion.** PoC throughput and cross-implementation agreement were established on four topologies; the honest cross-validation floor and the batch-boundary artifact are documented for the threshold work.

### DeepSeek-V4-Flash-0731 rerun and DSpark (2026-07-31 … 2026-08-11)

Rerun on the new checkpoint. Ten directories under [`2026-08/deepseek-v4-flash-0731-*`](https://github.com/kaitakuai/experiments/tree/main/2026-08).

| Item | Result |
|------|--------|
| PoC nonces/min, 1×B300 / 2×B200 / 2×H200 | 1728 / 2304 / 1215 — unchanged vs the original checkpoint |
| DSpark (speculative decoding) | up to 3.39× decode on 4×H100 with output preserved; bit-identical PoC on 2×B200 |
| Inference validation, B300 → 2×H200 | 0/6000 mismatches; stale-checkpoint arm separates cleanly |
| Threshold normalization ([`…-threshold-normalization`](https://github.com/kaitakuai/experiments/tree/main/2026-08/deepseek-v4-flash-0731-threshold-normalization), Mykola) | `R = D / f_honest(N)`: TP at FP 5 % from 34.7 % to 68.8 %; 64.0 % when applied to the July data |
| DSpark loader bug ([`…-nvfp4-dspark-1xb300`](https://github.com/kaitakuai/experiments/tree/main/2026-08/deepseek-v4-flash-0731-nvfp4-dspark-1xb300)) | `Fp8Config.from_config` ignored `ignore`; fixed in [kaitakuai/vllm#20](https://github.com/kaitakuai/vllm/pull/20) |
| Throughput accounting fix | [kaitakuai/experiments#7](https://github.com/kaitakuai/experiments/pull/7) by [@vbgd0](https://github.com/vbgd0): post-boundary callbacks excluded (0–40 % overstatement) |

**Conclusion.** The 0731 checkpoint changed nothing for PoC throughput; DSpark works for both PoC and serving once the replay path and the loader are fixed.

---

## Engine fixes and release candidates (2026-07-31 … 2026-08-13)

### Replay hooks for the V2 model runner (2026-07-31 … 2026-08-03)

DSpark only runs on the V2 model runner, where the replay validation hooks were missing.

| PR | Title | Author | Outcome |
|----|-------|--------|---------|
| [gonka-ai/vllm#92](https://github.com/gonka-ai/vllm/pull/92) | feat(poc): port replay hooks to the V2 model runner | [@baychak](https://github.com/baychak) | merged 2026-08-03 |
| [kaitakuai/vllm#18](https://github.com/kaitakuai/vllm/pull/18) | fix(poc): make V2 replay work with per-request logprobs mode and speculation | [@clanster](https://github.com/clanster) | folded into #92 |
| [gonka-ai/vllm#96](https://github.com/gonka-ai/vllm/pull/96) | fix(sched): skip requests absent from req_id_to_index | [@clanster](https://github.com/clanster) via [@baychak](https://github.com/baychak) | open |
| [gonka-ai/vllm#97](https://github.com/gonka-ai/vllm/pull/97) | fix(deepseek_v4): keep DSpark draft experts off the NVFP4 path | [@clanster](https://github.com/clanster) via [@baychak](https://github.com/baychak) | open |
| [gonka-ai/gonka#1560](https://github.com/gonka-ai/gonka/pull/1560) | fix(mlnode): link libnvrtc.so where the linker looks | [@baychak](https://github.com/baychak) | merged 2026-08-07 |

### Release candidates (2026-07-23 … 2026-08-13)

| Stage | Date | Content |
|-------|------|---------|
| Kaitaku candidates | 2026-07-23 … 08-01 | first stack built entirely from upstream sources (`release/v0.25.1` after #78, plugin v0.1.1, mlnode 0.2.14), then the V2 replay hooks |
| Core team `mlnode:3.0.14-post2-vllm0.25.1-rc1` … rc3 | 2026-08-03 … 08-07 | verified on Kaitaku B300; four defects found in review and fixed before rc3 |
| Rebuild on `mlnode:3.0.16` | 2026-08-13 | nonces/min identical to the previous build |

**Conclusion.** Every release candidate, including the core team's rc1 / rc3, went through the same PoC checks on B300 before shipping.

---

## Governance entry (2026-08-09 … 2026-08-27)

| Proposal | Content | Status |
|----------|---------|--------|
| [#94](https://gonka.gg/network/proposals/94) | Add `deepseek-ai/DeepSeek-V4-Flash-0731` (`seq_len` 1024, `dist_threshold` 0.41, `p_mismatch` 0.10, `weight_scale_factor` 0.214); parameters verified by the core team; submitted by Kaitaku | PASSED |
| [gonka-ai/gonka#1640](https://github.com/gonka-ai/gonka/pull/1640) | fix(benchmarks): rate PoC nonces against the interval that produced them — Vlad's measurement fix ([experiments#7](https://github.com/kaitakuai/experiments/pull/7)) ported to the benchmark script; the original nonce/min was overstated by about 13 % | merged 2026-08-25 |
| [#97](https://gonka.gg/network/proposals/97) / [#98](https://gonka.gg/network/proposals/98) | `weight_scale_factor` 0.214 → 0.246 (recomputed with the core team); #97 expired without quorum, #98 resubmitted unchanged | #98 PASSED 2026-08-27 |

**Conclusion.** The coefficient correction traces to the measurement fix by [@vbgd0](https://github.com/vbgd0); the proposal text credits it.

---

## Participants

[kaitakuai](https://github.com/kaitakuai) ([@baychak](https://github.com/baychak), [@clanster](https://github.com/clanster)).
Gonka core team ([@gmorgachev](https://github.com/gmorgachev), [@vbgd0](https://github.com/vbgd0), [@mtvnastya](https://github.com/mtvnastya), [@tcharchian](https://github.com/tcharchian)).

| Participant | GitHub | Role | Contribution |
|-------------|--------|------|--------------|
| Pavlo | [@clanster](https://github.com/clanster) | kaitakuai | PoC fix for DeepSeek-V4, in-tree port #65, both experiment campaigns, DSpark, [kaitakuai/vllm#18](https://github.com/kaitakuai/vllm/pull/18) / [#19](https://github.com/kaitakuai/vllm/pull/19) / [#20](https://github.com/kaitakuai/vllm/pull/20), rc review, proposal #94 submission |
| Mykola | [@baychak](https://github.com/baychak) | kaitakuai | #92, threshold normalization, release-candidate images, [#1560](https://github.com/gonka-ai/gonka/pull/1560), [#1640](https://github.com/gonka-ai/gonka/pull/1640), proposals #97 / #98 |
| Gleb Morgachev | [@gmorgachev](https://github.com/gmorgachev) | Gonka core team | Scope, `release/v0.25.1` branch, review of experiment results |
| Vladislav Bogdanov | [@vbgd0](https://github.com/vbgd0) | Gonka core team | Review and merge of #92, rc1 / rc3 images, 0731 rerun request, throughput accounting fix ([experiments#7](https://github.com/kaitakuai/experiments/pull/7)), coefficient recomputation |
| Anastasia Matveeva | [@mtvnastya](https://github.com/mtvnastya) | Gonka core team | Proposal #94 draft, coefficient recomputation for #97 / #98 |
| Tania Charchian | [@tcharchian](https://github.com/tcharchian) | Gonka core team | Issue tracking (#1408), status follow-up |

