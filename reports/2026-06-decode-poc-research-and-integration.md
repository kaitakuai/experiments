# decode-PoC — research, cross-hardware validation and integration into the vLLM 0.25.1 plugin stack (2026-06-09 … 2026-09-08, work in progress)

## Summary

The [kaitaku.ai](https://github.com/kaitakuai) team (Mykola [@baychak](https://github.com/baychak), Pavlo [@clanster](https://github.com/clanster)) worked on the decode-based PoC scheme authored by Ilia ([@axeltec-gonka](https://github.com/axeltec-gonka), [gonka-ai/gonka#1135](https://github.com/gonka-ai/gonka/issues/1135)) as a contributor: experiments on rented hardware, research findings adopted into the scheme, and the integration of the scheme into the residual + `gonka-poc` plugin stack on vLLM 0.25.1. The work covers:

- June separability and Goodhart experiments on decode-PoC ([2026-06/decode-poc-separability-compiled-vs-eager](https://github.com/kaitakuai/experiments/tree/main/2026-06/decode-poc-separability-compiled-vs-eager), [2026-06/decode-poc-goodhart-deepgemm-vs-flashinfer](https://github.com/kaitakuai/experiments/tree/main/2026-06/decode-poc-goodhart-deepgemm-vs-flashinfer)), closed by the 2026-07-03 write-up on compiled vs eager × Householder reflections.
- Cross-hardware validation of the scheme with Ilia's tooling: B300 prover against H200 / A100 / H100 validators, MiniMax-M2.7, Qwen3-235B and Kimi-K2.6 data.
- Research contributions taken into the scheme or its calibration: per-nonce reflection seed, precomputed codebook, the vector channel `poc_vector_artifacts` with margin-gate calibration, artifact size measurements.
- The full A100 / H100 / H200 / B300 cross-validation matrix (4 seeds) and unified calibration requested by the core team; performance and fairness measurements (PoC vs inference on five deployment configurations, nonces/min on vLLM 0.25.1).
- Expert-seeding-window sweep with Ilia that reduced the PoC/inference ratio spread across GPU types from 1.7 to 1.19, plus two bugs found and fixed in the scheme.
- Migration of decode-PoC from the in-tree vLLM 0.20 branch to the plugin architecture on vLLM 0.25.1 with artifacts unchanged from the 0.20 branch: [gonka-ai/vllm#100](https://github.com/gonka-ai/vllm/pull/100) and [gonka-ai/gonka-vllm-plugins#8](https://github.com/gonka-ai/gonka-vllm-plugins/pull/8), tested by [@vbgd0](https://github.com/vbgd0) on 2026-09-02.
- Extension of the scheme to DeepSeek-V4-Flash and the `poc-as-chat` scheduling variant that [@vbgd0](https://github.com/vbgd0) took as the integration base on 2026-09-08.

This is an interim report: the track is not finished. Integration into the network release continues on the `poc-as-chat` base; the open items are listed in the Status section below and tracked in [gonka-ai/gonka#1688](https://github.com/gonka-ai/gonka/issues/1688), [#1689](https://github.com/gonka-ai/gonka/issues/1689) and [#1690](https://github.com/gonka-ai/gonka/issues/1690).

---

## Separability and Goodhart experiments (2026-06-09 … 2026-07-04)

Ilia asked on 2026-06-09 for parallel experiments on different hardware and models. The June bundles used MiniMax-M2.7 FP8 as the honest arm and an AWQ arm as fraud, prover 1×B300 TP=1, validator 4×H100 TP=4, vLLM 0.20.0.

| Experiment | Question | Outcome |
| --- | --- | --- |
| [decode-poc-separability-compiled-vs-eager](https://github.com/kaitakuai/experiments/tree/main/2026-06/decode-poc-separability-compiled-vs-eager) | Does cross-hardware separability survive compiled vs eager decode? | Compiled decode keeps the honest/AWQ gap; eager decode loses it (N=64 per cell) |
| [decode-poc-goodhart-deepgemm-vs-flashinfer](https://github.com/kaitakuai/experiments/tree/main/2026-06/decode-poc-goodhart-deepgemm-vs-flashinfer) | Does decode-PoC reward the DeepGEMM backend that hurts serving? | With compiled decode PoC score and serving move together (−8.7 % / −49.3 %), where prefill PoC had rewarded DeepGEMM with +40 % |
| 2026-07-03 write-up (chat MLNode on Gonka) | compiled/eager × Householder reflections on/off, MiniMax, B300 → A100 | Compiled generation + Householder reflections separates best; a per-nonce reflection seed instead of one seed per block hash removes the axis collapse on hardware or quantisation change |

An earlier observation (2026-06-10) that the compiled prefill PoC already validates across GPU generations on vLLM 0.15 / 0.19 / 0.20 was shared with Ilia as a reason not to keep the eager path.

**Conclusion.** The experiments supported the decode-based direction; the per-nonce reflection seed became part of the scheme.

---

## Cross-hardware validation with Ilia's tooling (2026-06-24 … 2026-07-10)

Ilia's branch `poc-v0.20-decode-poc-cg` with `run_scope.sh` and the report renderer was used as-is from 2026-07-08 so that results stay comparable between hardware.

| Date | Prover → validator | Models | Delivered |
| --- | --- | --- | --- |
| 2026-06-24 … 06-25 | 1×B300 TP=1 → 4×H200 TP=4, 4×A100 TP=4 | MiniMax-M2.7 FP8 vs AWQ; Qwen3-235B | k-id vs continuous L2 comparison, `decode-poc-handoff.tgz` with `tools/analyze.py`; precomputed codebook proposed |
| 2026-07-09 … 07-10 | 1×B300, 4×A100, H100 | MiniMax-M2.7 | Full run through Ilia's tooling, archived results |
| 2026-07-18, 07-22 | as above | Kimi-K2.6 | Two data deliveries; Kimi later dropped from the decode track by the core team (2026-08-27) |

**Conclusion.** The precomputed codebook was reused in the scheme (2026-06-24); the MoE-related noise seen in these runs was attributed to expert routing, which led to seeded routing.

---

## Research contributions (2026-06-27 … 2026-07-23)

| Contribution | Where it went |
| --- | --- |
| Per-nonce seed for Householder reflections; averaging PoC vectors over seeds with k-id used only for chaining | Ilia tested both (2026-07-13); the per-nonce seed is in the scheme, the averaging was superseded by his margin-gate |
| Dominant-axis whitening of PoC vectors | Not adopted — Ilia's seeded routing removed the noise it targeted |
| Vector channel `poc_vector_artifacts`: prover ships compact pre-discretisation vector slices, validator computes a continuous `vector_score` — [axeltec-software/vllm#4](https://github.com/axeltec-software/vllm/pull/4) on `poc/margin-gate`; margin-gate calibration on B300+TRITON vs 4×A100+FlashAttention, three campaigns | Reviewed by Ilia; the PR was closed once the artifact format was fixed by the core team on 2026-07-23 (12 coordinates × 256 decode steps, [@gmorgachev](https://github.com/gmorgachev)) |
| Off-chain artifact size per nonce for prefill / discrete k-id / vector / combined variants on B300 | Input to the artifact-format decision |
| Bench client fix: chat client silently capped in-flight requests at 100 (httpx pool) — [axeltec-software/vllm#5](https://github.com/axeltec-software/vllm/pull/5) | Merged by Ilia 2026-08-07 |

**Conclusion.** Of the four proposals, the per-nonce seed and the precomputed codebook are in the scheme; the vector channel informed the artifact-format decision; whitening was not needed.

---

## Full cross-validation matrix and unified calibration (2026-07-27 … 2026-07-31)

After a weak-separability case was found on one prover/validator pair, [@mtvnastya](https://github.com/mtvnastya) and [@gmorgachev](https://github.com/gmorgachev) asked for the complete picture and the raw artifacts.

| Item | Content |
| --- | --- |
| Generation | Honest FP8 and AWQ arms on A100, H100, H200, B300; 4 Householder seeds each |
| Validation | All prover/validator pairs — 76 cells |
| Calibration | Unified per-model calibration parameters (margin-gate τ, mismatch rate) with the recalculation script `tools/wave_unified3.py` in the archive |
| Delivery | Raw artifacts on Google Drive — [`decode-poc-hw-matrix-2026-07-29.tgz`](https://drive.google.com/file/d/1GymtaGbV9NAL238ZVjOWlpsm6KnnImag/view) (full matrix, 2026-07-29) and [`decode-poc-hardcase-handoff-20260727-r2.tgz`](https://drive.google.com/file/d/1tYR7Lr24mGAWDmt-FRqnwE5SNjUDLQR7/view) (the weak pair, 2026-07-28); summary for the core team; answers to [@mtvnastya](https://github.com/mtvnastya) on honest-validator false positives (2026-07-31) |

**Conclusion.** A single set of calibration parameters per model works for the Hopper/Blackwell fleet; the details are consensus-security material and stay with the core team.

---

## Performance and fairness (2026-07-29 … 2026-08-19)

| Measurement | Configurations | Result |
| --- | --- | --- |
| PoC vs pure inference throughput | 4×A100, 4×H100, 2×H200, 1×B300 (+2×B300 on 2026-08-01) | Full batch sweep for PoC and concurrency sweep for chat; PoC scales with the card roughly like inference; the same 1×B300 on two hosts differed by 53 % vs 83 % of peak |
| Expert-seeding-window sweep with Ilia (16 / 32 / 64 / 128 / 256) | several deployment configurations | Best point 256; PoC/inference ratio spread across GPU types 1.7 → 1.19; two bugs found and fixed — one in the inference data-collection tool, one in the algorithm (Ilia, 2026-08-09) |
| nonces/min on vLLM 0.25.1, MiniMax-M2.7, for weight redistribution (requested by the core team 2026-08-19) | 4×A100 1145 (batch 600) · 4×H100 2453 (600) · 2×H200 1890 (584) · 1×B300 1822 (536) | Preliminary numbers |
| Correlation with nominal HBM bandwidth | same four | Pearson 0.962 for PoC, 0.897 for chat |

Raw data for these measurements was not committed to `kaitakuai/experiments`; the July runs are inside [`decode-poc-hw-matrix-2026-07-29.tgz`](https://drive.google.com/file/d/1GymtaGbV9NAL238ZVjOWlpsm6KnnImag/view) and the vLLM 0.25.1 runs in the [`poc-devkit-2026-08-26`](https://drive.google.com/drive/folders/1mI1MHu5xs-EfBO3R6Zh-cjOxngyKvoUp) bundle on Google Drive.

**Conclusion.** With the expert-seeding window at 256 the PoC/inference ratio is close to uniform across GPU types; the core team and the scheme author accepted these results as the base for the release (2026-08-06).

---

## Documentation (2026-08-10, 2026-08-27)

| Date | Item |
| --- | --- |
| 2026-08-10 | Description of the current decode-PoC version for the core team; Ilia later published `POC_DECODE.md` |
| 2026-08-25 … 08-27 | Canonical bundle: reproducible fixed point, raw data and reference sets — [`poc-devkit-2026-08-26`](https://drive.google.com/drive/folders/1mI1MHu5xs-EfBO3R6Zh-cjOxngyKvoUp) on Google Drive; the description was rewritten in English |

**Conclusion.** The reproducible fixed point and its description are in the bundle; documentation is kept in English.

---

## Migration to the vLLM 0.25.1 plugin stack (2026-08-16 … 2026-09-02)

Plan agreed with the core team on 2026-08-17: migration by 16.08, thresholds for Kimi and DeepSeek by 24.08, image polish by 31.08. Item 3 was dropped on 2026-08-26: the hand-over unit is code, not images.

| Step | Date | Resource |
| --- | --- | --- |
| decode scheme ported from the in-tree 0.20 branch into the plugin, engine untouched, artifacts unchanged from 0.20 | 2026-08-15 … 08-17 | [gonka-ai/gonka-vllm-plugins#6](https://github.com/gonka-ai/gonka-vllm-plugins/pull/6) (draft) |
| Mixed part (inference in parallel with PoC) restored with Ilia after it had been dropped in the port; retest on all configurations | 2026-08-19 | branch `axeltec/mixed-poc` |
| Regression 0.20 → 0.25 on MiniMax located; fix by Ilia in the MoE gate (fixed, uniform expert distribution); performance gap 8–9 % | 2026-08-21 … 08-24 | [axeltec-software/gonka-vllm-plugins#1](https://github.com/axeltec-software/gonka-vllm-plugins/pull/1), [#2](https://github.com/axeltec-software/gonka-vllm-plugins/pull/2), [#3](https://github.com/axeltec-software/gonka-vllm-plugins/pull/3) (prefill stalls removed, nsys on 1×B300) |
| Upstream PRs on the integration branches created by [@vbgd0](https://github.com/vbgd0) | 2026-08-26 | [gonka-ai/vllm#100](https://github.com/gonka-ai/vllm/pull/100) (engine seams, `release/v0.25.1-decode-int`), [gonka-ai/gonka-vllm-plugins#8](https://github.com/gonka-ai/gonka-vllm-plugins/pull/8) (mixed decode-PoC — PoC and chat share one batch, `decode-poc-int`); fallback logic duplicating the plugin removed |
| Review of the prefill/decode compatibility changes and of the network transition sequence | 2026-08-27 … 08-31 | [kaitakuai/gonka-vllm-plugins#1](https://github.com/kaitakuai/gonka-vllm-plugins/pull/1), [#2](https://github.com/kaitakuai/gonka-vllm-plugins/pull/2) (author [@vbgd0](https://github.com/vbgd0)), [#3](https://github.com/kaitakuai/gonka-vllm-plugins/pull/3) (author Ilia) |

**Conclusion.** All core-team workloads pass on #100 + #8 (2026-09-02); the new and the old PoC are implemented separately.

---

## DeepSeek-V4-Flash and the `poc-as-chat` base (2026-08-28 … 2026-09-08)

| Step | Date | Resource |
| --- | --- | --- |
| Extension of decode-PoC to DeepSeek-V4-Flash: one blocker and a model-agnostic fix; separability shown on H100 (TP=4) and B300 | 2026-08-28 … 09-03 | [gonka-ai/gonka#1690](https://github.com/gonka-ai/gonka/issues/1690) |
| `rolling-poc-admission` branch on vLLM 0.28 (six fixes, PoC/inference ratio 1.07) — set aside: the integration stays on vLLM 0.25.1 and outside vLLM internals | 2026-09-02 … 09-03 | branch `axeltec/rolling-poc-admission` |
| `poc-as-chat`: PoC rows scheduled like chat requests, admission layer removed; fixes the decode-PoC hangs on Hopper with DeepSeek, keeps separability and throughput | 2026-09-06 … 09-08 | [kaitakuai/vllm#22](https://github.com/kaitakuai/vllm/pull/22), [kaitakuai/gonka-vllm-plugins#4](https://github.com/kaitakuai/gonka-vllm-plugins/pull/4) |
| The core team took `poc-as-chat-vllm-0.25.1-dev` as the integration base and merged its own updates into it | 2026-09-08 | [kaitakuai/gonka-vllm-plugins#5](https://github.com/kaitakuai/gonka-vllm-plugins/pull/5), [kaitakuai/vllm#23](https://github.com/kaitakuai/vllm/pull/23) |
| In progress: DeepSeek + MiniMax reruns on the merged branches and coefficient re-selection | from 2026-09-08 | [gonka-ai/gonka#1688](https://github.com/gonka-ai/gonka/issues/1688), [#1689](https://github.com/gonka-ai/gonka/issues/1689), [#1690](https://github.com/gonka-ai/gonka/issues/1690) |

**Conclusion.** The integration base for the network release is `poc-as-chat` on vLLM 0.25.1; reference artifacts and the coefficient pass are the open items.

---

## Status (2026-09-08)

| State | Item |
| --- | --- |
| Done | Research findings adopted into the scheme (per-nonce reflection seed, precomputed codebook, expert-seeding window 256); cross-hardware matrix and unified calibration delivered to the core team; decode-PoC ported to the vLLM 0.25.1 plugin stack ([gonka-ai/vllm#100](https://github.com/gonka-ai/vllm/pull/100), [gonka-ai/gonka-vllm-plugins#8](https://github.com/gonka-ai/gonka-vllm-plugins/pull/8)) and tested by [@vbgd0](https://github.com/vbgd0); `poc-as-chat` base accepted for integration |
| In progress | DeepSeek-V4-Flash and MiniMax-M2.7 reruns on the merged `poc-as-chat` branches; coefficient re-selection; reference artifacts for the release — [#1688](https://github.com/gonka-ai/gonka/issues/1688), [#1689](https://github.com/gonka-ai/gonka/issues/1689), [#1690](https://github.com/gonka-ai/gonka/issues/1690) |
| Next | First full MLNode image on decode-PoC for MiniMax (testing), then DeepSeek with coefficients; merge of #100 / #8 after the reruns; network transition sequence agreed with the core team on 2026-08-29 … 08-31 (two-mode image → vote → activation) |

**Conclusion.** The research and the migration are delivered; the release integration is the remaining work.

---

## Participants

[kaitakuai](https://github.com/kaitakuai) ([@baychak](https://github.com/baychak), [@clanster](https://github.com/clanster)).
Gonka core team ([@gmorgachev](https://github.com/gmorgachev), [@vbgd0](https://github.com/vbgd0), [@mtvnastya](https://github.com/mtvnastya), [@tcharchian](https://github.com/tcharchian)); independent contributor [@axeltec-gonka](https://github.com/axeltec-gonka).

| Participant | GitHub | Role | Contribution |
| --- | --- | --- | --- |
| Mykola | [@baychak](https://github.com/baychak) | kaitakuai | Experiments and write-ups, cross-hardware validation, vector channel and margin-gate calibration, 76-cell matrix, performance and fairness measurements, expert-window sweep, documentation, migration to the 0.25.1 plugin stack (#6, #100, #8), DeepSeek extension, `poc-as-chat` (#22, #4) |
| Pavlo | [@clanster](https://github.com/clanster) | kaitakuai | Data collection on rented hardware, reruns for the coefficient pass |
| Ilia Slavutin | [@axeltec-gonka](https://github.com/axeltec-gonka) | independent | Author of the decode-PoC scheme (#1135); seeded routing, margin-gate, MoE gate fix; joint experiments and review |
| Gleb Morgachev | [@gmorgachev](https://github.com/gmorgachev) | Gonka core team | Scope, methodology requirements, artifact format decision |
| Vladislav Bogdanov | [@vbgd0](https://github.com/vbgd0) | Gonka core team | Integration branches, review and testing of #100 / #8, prefill/decode compatibility (#1, #2, #5, #23), nonces/min requirements |
| Anastasia Matveeva | [@mtvnastya](https://github.com/mtvnastya) | Gonka core team | Separability questions, model lineup |
| Tania Charchian | [@tcharchian](https://github.com/tcharchian) | Gonka core team | Issues #1688–#1690 |
