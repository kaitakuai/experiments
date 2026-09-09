# PoC as a vLLM plugin — residual + `gonka-poc` architecture and its adoption (2026-06-16 … 2026-09-08)

## Summary

The [kaitaku.ai](https://github.com/kaitakuai) team (Mykola [@baychak](https://github.com/baychak), Pavlo [@clanster](https://github.com/clanster)) replaced the in-tree PoC port that had to be redone on every vLLM release with a two-part layout: a thin residual patch inside vLLM and an out-of-tree, pip-installable plugin `gonka-poc`. The work covers:

- the plugin itself — 30 commits from scaffold to hand-over in the Kaitaku repository, transferred to [gonka-ai/gonka-vllm-plugins](https://github.com/gonka-ai/gonka-vllm-plugins) on 2026-07-28;
- the residual port to vLLM 0.25.1 — [gonka-ai/vllm#78](https://github.com/gonka-ai/vllm/pull/78) (no behaviour change) and the follow-up fixes [#79](https://github.com/gonka-ai/vllm/pull/79), [#80](https://github.com/gonka-ai/vllm/pull/80), [#82](https://github.com/gonka-ai/vllm/pull/82);
- the review round with the core team — [gonka-vllm-plugins#1](https://github.com/gonka-ai/gonka-vllm-plugins/pull/1), [#2](https://github.com/gonka-ai/gonka-vllm-plugins/pull/2), [#3](https://github.com/gonka-ai/gonka-vllm-plugins/pull/3) and release `v0.1.1`;
- three model and version additions carried by the plugin without touching the residual: DeepSeek-V4 support, decode-PoC ([#6](https://github.com/gonka-ai/gonka-vllm-plugins/pull/6), [#8](https://github.com/gonka-ai/gonka-vllm-plugins/pull/8)) and vLLM 0.28 for GLM-5.3-Flash ([#9](https://github.com/gonka-ai/gonka-vllm-plugins/pull/9)).

The MLNode release images `3.0.14-post2-vllm0.25.1-rc1` and `3.0.16` are built on `release/v0.25.1` + `gonka-poc v0.1.1`; on 2026-07-29 the core team verified that, for the models already on the network, the resulting release behaves as the previous one.

---

## Why a plugin (2026-06-16 … 2026-07-24)

Until July every vLLM version bump meant re-porting the PoC code inside the vLLM tree. For vLLM 0.25.1 both approaches were built and submitted side by side so the core team could compare them.

| Approach | PR | Files | Lines | Verdict |
| --- | --- | ---: | ---: | --- |
| In-tree port ("fat-fork") | [gonka-ai/vllm#65](https://github.com/gonka-ai/vllm/pull/65) — port Gonka PoC-v2 + DeepSeek-V4 support to v0.25.1 | 45 | +5893 | closed 2026-07-26 in favour of the residual |
| Residual + out-of-tree plugin | [gonka-ai/vllm#66](https://github.com/gonka-ai/vllm/pull/66) — thin residual for v0.25.1 (alternative to #65) | 20 | +1124 | accepted as the direction; re-split into [#78](https://github.com/gonka-ai/vllm/pull/78) |

The residual keeps only what vLLM offers no extension point for (sampler hooks, enforced-token replay ingestion, the shm ring size); everything else — PoC generation, validation, gating middleware, engine `_compat` shims — lives in `gonka-poc` and is installed with `pip`. The image pipeline is staged: S0 upstream vLLM base → S1 residual → S2 residual + plugin (the artifact handed to the core team) → S3/S4 Kaitaku per-GPU profiles and overlays.

The core team accepted the split on 2026-07-24 as the way to reduce the operational cost of vLLM and MLNode upgrades.

**Conclusion.** The residual + plugin split was chosen over the in-tree port; the residual for 0.25.1 is 887 lines and has stayed unchanged since 2026-07-29.

---

## The plugin before the transfer (2026-06-16 … 2026-07-24)

The plugin was started in `kaitakuai/gonka-poc` for vLLM 0.23 and grew through four architecture reviews before hand-over. The old repository's issue and PR threads (27) are archived in [`experiments/archive/gonka-poc-issues`](https://github.com/kaitakuai/experiments/tree/main/archive/gonka-poc-issues); its full commit history is now `main` of `gonka-ai/gonka-vllm-plugins`.

| Date | PR (old repo) | Author | What |
| --- | --- | --- | --- |
| 2026-06-16 | scaffold, `#2`–`#7` | [@baychak](https://github.com/baychak) | initial scaffold for vLLM 0.23.x, contract tests, `_compat` dispatch, gating middleware, two rounds of must-fix items |
| 2026-06-17 | `#8`–`#11` | [@baychak](https://github.com/baychak) | code and doc defects, process gates in CI (grep-lint, real-wheel smoke, exception path) |
| 2026-06-22 | `#12` | [@baychak](https://github.com/baychak) | reset `middleware_stack` before `add_middleware` (Starlette 1.3.x) |
| 2026-06-24 | `#13` | [@baychak](https://github.com/baychak) | unlock MoE workspace around the PoC forward (vLLM 0.23 DeepGEMM) |
| 2026-07-20 | `#17` | [@baychak](https://github.com/baychak) | vLLM 0.25.x support — `_compat/v0_25` + dual-version dispatch |
| 2026-07-20 | `#14` | [@clanster](https://github.com/clanster) | DeepSeek-V4 support — per-group metadata, positions, pseudo ids |
| 2026-07-20 | `#18` | [@baychak](https://github.com/baychak) | validation on leased KV blocks — inference keeps running (the logic of [@qdanik](https://github.com/qdanik)'s `qd/combine-poc-and-inference` branch, folded in at [@gmorgachev](https://github.com/gmorgachev)'s request of 2026-07-19) |
| 2026-07-21 … 07-24 | `#19`–`#26` | [@baychak](https://github.com/baychak) | pre-transfer cleanup, deployment-agnostic docs, contributor guide, one contract suite per vLLM minor |

**Conclusion.** By 2026-07-24 the plugin supported vLLM 0.23 and 0.25 through one dispatch layer, carried DeepSeek-V4 and the PoC-during-inference logic, and was documented for hand-over (tag `v0.1.0a0`).

---

## Transfer and review round (2026-07-24 … 2026-07-31)

On 2026-07-24 Kaitaku proposed a GitHub transfer of the repository to the `gonka-ai` namespace; [@vbgd0](https://github.com/vbgd0) completed it on 2026-07-28 as `gonka-ai/gonka-vllm-plugins` and reviewed the plugin with four notes on 2026-07-27: abort in-flight requests by internal id, RNG under `poc_stronger_rng`, no validation on scratch-capable engines, naming.

| PR | Title | Merged | What |
| --- | --- | --- | --- |
| [gonka-vllm-plugins#1](https://github.com/gonka-ai/gonka-vllm-plugins/pull/1) | fix(compat): abort in-flight requests by internal id, not external | 2026-07-29 | review note 1 |
| [gonka-vllm-plugins#2](https://github.com/gonka-ai/gonka-vllm-plugins/pull/2) | chore: drop vLLM 0.23 support, target 0.25.x only | 2026-07-29 | −547 lines; 0.25.x is the only supported minor |
| [gonka-vllm-plugins#3](https://github.com/gonka-ai/gonka-vllm-plugins/pull/3) | docs: say what consensus actually requires — derivation, not bits | 2026-07-29 | wording fixed after [@gmorgachev](https://github.com/gmorgachev)'s correction that bit-identical output was never required |
| — | release `v0.1.1` (by [@vbgd0](https://github.com/vbgd0)) | 2026-07-29 | first release from the `gonka-ai` namespace |

The residual side landed in the same days. [gonka-ai/vllm#78](https://github.com/gonka-ai/vllm/pull/78) (port only, +887) was merged on 2026-07-29 with a table of ten follow-up PRs, each showing one change; [@vbgd0](https://github.com/vbgd0) merged three and closed seven.

| PR | Title | Outcome |
| --- | --- | --- |
| [gonka-ai/vllm#79](https://github.com/gonka-ai/vllm/pull/79) | stop a disabled grammar from leaking another request's mask | merged 2026-07-30 |
| [gonka-ai/vllm#80](https://github.com/gonka-ai/vllm/pull/80) | range-check and bound replay ids before the engine sees them | merged 2026-07-30 |
| [gonka-ai/vllm#82](https://github.com/gonka-ai/vllm/pull/82) | make the shm ring size a knob instead of a hardcoded 64 | merged 2026-07-30 |
| [#81](https://github.com/gonka-ai/vllm/pull/81), [#83](https://github.com/gonka-ai/vllm/pull/83)–[#88](https://github.com/gonka-ai/vllm/pull/88) | V2 replay guard, replay exhaustion, logprobs-mode override, per-step syncs, token-id format, logging, `/v1/completions` replay fields | closed at reviewer's verdict; [#81](https://github.com/gonka-ai/vllm/pull/81) superseded by [#92](https://github.com/gonka-ai/vllm/pull/92) |

**Conclusion.** Plugin `v0.1.1` and residual `release/v0.25.1` after #78/#79/#80/#82 became the stack the release images are built from.

---

## Maintenance after the transfer (2026-08-12 … 2026-08-13)

| Date | PR | Author | What |
| --- | --- | --- | --- |
| 2026-08-12 | [gonka-vllm-plugins#4](https://github.com/gonka-ai/gonka-vllm-plugins/pull/4) — route plugin logs through vLLM's handler | [@baychak](https://github.com/baychak) | `nonces per min` had disappeared from the release logs ([@gmorgachev](https://github.com/gmorgachev), 2026-08-12); merged 2026-08-13 |
| 2026-08-13 | [gonka-vllm-plugins#5](https://github.com/gonka-ai/gonka-vllm-plugins/pull/5) — preserve captured MoE workspace | [@vbgd0](https://github.com/vbgd0) | the MoE-workspace unlock added in `#13` caused a warm-up regression on MiniMax (longer warm-up, CUDA graph capture); found by [@vbgd0](https://github.com/vbgd0) and removed (−201 lines) |

**Conclusion.** One regression traced to the plugin was caught before the DeepSeek bootstrap and reverted; the residual was not involved.

---

## What the plugin carried afterwards (2026-07-20 … 2026-09-08)

Each later addition changed the plugin only.

| Addition | Where | Residual change |
| --- | --- | --- |
| DeepSeek-V4-Flash PoC (per-group KV metadata, positions, pseudo ids) | `#14` by [@clanster](https://github.com/clanster), in `main` since the transfer; shipped in the 3.0.16 images and proposal 94 | none |
| decode-PoC ported from the 0.20 in-tree branch | [gonka-vllm-plugins#6](https://github.com/gonka-ai/gonka-vllm-plugins/pull/6) (draft, 2026-08-17) → [#8](https://github.com/gonka-ai/gonka-vllm-plugins/pull/8) mixed decode-PoC (2026-08-26, open; tested by [@vbgd0](https://github.com/vbgd0) 2026-09-02) | engine seams only — [gonka-ai/vllm#100](https://github.com/gonka-ai/vllm/pull/100) (+407, open) |
| vLLM 0.28 for GLM-5.3-Flash | [gonka-vllm-plugins#9](https://github.com/gonka-ai/gonka-vllm-plugins/pull/9) — `_compat/v0_28.py`, +517, merged 2026-09-08 | residual re-based onto the GLM branch as [gonka-ai/vllm#105](https://github.com/gonka-ai/vllm/pull/105); PoC logic untouched |

Tags on the `gonka-ai` repository: `v0.1.0a0`, `v0.1.1`, `v0.1.2`, `v0.1.3`, `v0.1.4` (the last one is what the GLM-5.3-Flash images pin).

**Conclusion.** Three additions — a new model, a new PoC scheme and a new vLLM minor — went in through the plugin's `_compat` and worker layers without re-porting the residual.

---

## Participants

[kaitakuai](https://github.com/kaitakuai) ([@baychak](https://github.com/baychak), [@clanster](https://github.com/clanster)).
Gonka core team ([@vbgd0](https://github.com/vbgd0), [@gmorgachev](https://github.com/gmorgachev)).

| Participant | GitHub | Role | Contribution |
| --- | --- | --- | --- |
| Mykola | [@baychak](https://github.com/baychak) | kaitakuai | architecture (residual + plugin), plugin from scaffold to `v0.1.0a0`, residual port #78 and follow-ups, `_compat` layers for 0.23/0.25/0.28, transfer, review-round PRs #1–#4, #9 |
| Pavlo | [@clanster](https://github.com/clanster) | kaitakuai | in-tree port #65 for comparison, DeepSeek-V4 support in the plugin (`#14`), testing of every stage on B300 |
| Vladislav Bogdanov | [@vbgd0](https://github.com/vbgd0) | Gonka core team | review notes of 2026-07-27, repository transfer, merges of #78–#82 and plugin PRs, release `v0.1.1`, regression fix #5, release images |
| Gleb Morgachev | [@gmorgachev](https://github.com/gmorgachev) | Gonka core team | transfer decision, correction on consensus requirements (derivation, not bits), logging request behind #4, request to fold in the `qd/combine-poc-and-inference` branch |
| Daniil Yankouski | [@qdanik](https://github.com/qdanik) | | author of `qd/combine-poc-and-inference` (PoC validation without interrupting inference), whose logic is `#18` in the plugin |
