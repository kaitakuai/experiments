# MiniMax-M2.7 integration and post-deploy stabilization (2026-05-05 … 2026-06-03)

## Summary

The [kaitaku.ai](https://github.com/kaitakuai) team (Mykola [@baychak](https://github.com/baychak), Pavlo [@clanster](https://github.com/clanster)) handled MiniMax-M2.7 onboarding into the v0.2.13 governance upgrade and the post-deploy bug-fixing that followed across the v0.2.13 cycle. The work covers:

- PoC v2 port from vLLM 0.19.0 to upstream 0.20.0 — base image for the MiniMax integration ([gonka-ai/vllm#37](https://github.com/gonka-ai/vllm/pull/37), image `ghcr.io/kaitakuai/vllm:0.20.0-pocv2`)
- Cross-hardware MiniMax-M2.7 FP8 validation on B200 / H100 / A100 / H200 (vLLM 0.19 / 0.20 grid)
- Final validation pass under the micro-update deadline; refresh of 2026-04 H200 artifacts after a stale `config.json` was caught
- Review of [#1226](https://github.com/gonka-ai/gonka/pull/1226) (per-model dispatch + tool-message shape + reasoning_split)
- Post-deploy Kimi-K2.6 bug-fixes that shipped in v0.2.13: `thinking_token_budget` resolver, corner-case clamp, reasoning-burn quarantine exemption
- Post-deploy cross-model benchmarks requested by Gleb - Qwen3-235B on 4×H100 SXM at 240k context

MiniMax-M2.7 was activated on mainnet via the v0.2.13 upgrade (commit `178086202`, merged 2026-05-26; `minimaxStartEpoch=278`).

---

## MiniMax-M2.7 — version testing

### PoC v2 port to vLLM 0.20.0 (2026-05-05)

| Resource | Link |
|----------|------|
| PR | [gonka-ai/vllm#37](https://github.com/gonka-ai/vllm/pull/37) |
| Branch | [`kaitakuai/vllm:mb/feat/port-pocv2-vllm-0.20`](https://github.com/kaitakuai/vllm/tree/mb/feat/port-pocv2-vllm-0.20) |
| Base image | `ghcr.io/kaitakuai/vllm:0.20.0-pocv2` |

The port is the foundation for all subsequent MiniMax images and the MiniMax governance entry.

### MiniMax-M2.7 FP8 cross-hardware grid (2026-05-07 … 2026-05-12)

Tested across the available hardware tiers; data lives in [`kaitakuai/experiments/2026-05/minimax-m27-fp8-2xh200/artifacts/`](https://github.com/kaitakuai/experiments/tree/main/2026-05/minimax-m27-fp8-2xh200) and a cross-hardware [Google sheet](https://docs.google.com/spreadsheets/d/1C-XnWDElh5fS9DmtdY04pEIzL60Lfmh8MnWSsCdiX64/).

| GPU | vLLM | Δ vs vLLM 0.15.1 + PR #36 |
|-----|------|--------------------------:|
| 2×H200 | 0.20.0 | +12 % |
| 4×H100 SXM | 0.20.0 | **+40 %** ★ |
| 4×A100 | 0.20.0 | +4 % |
| 2×B200 | 0.20.0 | (baseline established) |

A stale `config.json` (vLLM 0.19, which does not support MiniMax) in the 2026-04 H200 directory was caught by Gleb on 2026-05-12; Pavlo rebuilt the artifacts under the correct 0.20.0 path.

### Final validation under the micro-update deadline (2026-05-12 … 2026-05-13)

Pavlo re-ran the H200 PoC vectors after Gleb's catch, and ran every filter in `devshard/cmd/devshardctl/request_filters.go` against vLLM 0.20 — no engine crashes, no validation drift. This pass blocked the MiniMax release until the stale artifact was replaced.

### PR #1226 review (2026-05-23)

Mykola reviewed [gonka-ai/gonka#1226](https://github.com/gonka-ai/gonka/pull/1226) — Danya's per-model dispatch + tool-message shape + reasoning_split. Spot-checked tool-message shape and reasoning-split behavior on the MiniMax route.

### MiniMax governance entry (chain side)

Spec landed in [`inference-chain/app/upgrades/v0_2_13/upgrades.go:minimaxGovernanceModel()`](https://github.com/gonka-ai/gonka/blob/main/inference-chain/app/upgrades/v0_2_13/upgrades.go):

```
ModelArgs: ["--enable-auto-tool-choice", "--kv-cache-dtype", "fp8",
            "--tool-call-parser", "minimax_m2",
            "--reasoning-parser", "minimax_m2_append_think"]
VRam: 320          ThroughputPerNonce: 5000        minimaxStartEpoch: 278
HfCommit: d494266a4affc0d2995ba1fa35c8481cbd84294b
```

---

## Post-deploy Kimi bug-fixes (shipped in v0.2.13)

David repeatedly flagged Kimi-K2.6 behavior issues in the run-up to the v0.2.13 release; the cluster below resolves them.

| PR | Title | Author | Notes |
|----|-------|--------|-------|
| [#1202](https://github.com/gonka-ai/gonka/pull/1202) | default `thinking_token_budget` for Kimi-K2.6 reasoning split | [@clanster](https://github.com/clanster) | Whitelisted `thinking_token_budget`; default = `max_tokens / 2` per Moonshot recommendation |
| [#1204](https://github.com/gonka-ai/gonka/pull/1204) | hotfix test for #1202 | [@clanster](https://github.com/clanster) | CI repair |
| [#1212](https://github.com/gonka-ai/gonka/pull/1212) | strip `thinking_token_budget` for non-Kimi models | [@baychak](https://github.com/baychak) | Prevents `thinking_token_budget` leaking onto MiniMax/Qwen routes |
| [#1213](https://github.com/gonka-ai/gonka/pull/1213) | fix gateway tests broken by per-model access gate | [@baychak](https://github.com/baychak) | 5/6 failing tests on `main` repaired |
| [#1227](https://github.com/gonka-ai/gonka/pull/1227) | clamp Kimi-K2.6 `max_tokens >= 16` to keep probe content non-empty | [@baychak](https://github.com/baychak) | corner case where `thinking_token_budget=0` + `max_tokens<16` returned empty content; gateway was penalizing the node |
| [#1233](https://github.com/gonka-ai/gonka/pull/1233) (review) | exempt Kimi-K2.6 reasoning-burn empties from host quarantine | [qdanik](https://github.com/qdanik) (core) | reviewed by [@baychak](https://github.com/baychak) |

---

## Post-deploy cross-model benchmarks (requested by Gleb)

### Qwen3-235B FP8 on 4×H100 SXM with vLLM 0.20.0 at 240k context (2026-05-29 … 2026-05-30)

Artifacts at [`kaitakuai/experiments/2026-05/qwen3-235b-vllm020-240k`](https://github.com/kaitakuai/experiments/blob/main/2026-05/qwen3-235b-vllm020-240k/README.md).

| Config | Best batch | Outcome |
|--------|-----------:|---------|
| `--max-num-batched-tokens 32768 --gpu-memory-utilization 0.96`, `--max-model-len 240000` | 32 | full 240k context fits and PoC runs cleanly on vLLM 0.20.0 |

### `kv_scratch` cleanup (2026-05-27)

Gleb on 2026-05-28: the existing `kv_scratch` Stage2 workaround (from Tamaz) is superseded by his upstream commit `052648bf`. Mykola verified equivalence and rebased the `mlnode-foundry` Stage2 base onto the upstream commit, removing the carried patch.

---

## Participants

[kaitakuai](https://github.com/kaitakuai) ([@baychak](https://github.com/baychak), [@clanster](https://github.com/clanster)).
Gonka core team ([@gmorgachev](https://github.com/gmorgachev), [GLiberman](https://github.com/GLiberman).
[qdanik](https://github.com/qdanik).

| Participant | GitHub | Role | Contribution |
|-------------|--------|------|--------------|
| Pavlo | [@clanster](https://github.com/clanster) | kaitakuai | MiniMax FP8 cross-hardware validation (B200 / H100 / A100 / H200), final validation pass, request-filter validation on vLLM 0.20, thinking_token_budget cluster PRs #1202 / #1204 |
| Mykola | [@baychak](https://github.com/baychak) | kaitakuai | PoC v2 port to vLLM 0.20.0 (PR #37 + `vllm:0.20.0-pocv2` image), thinking_token_budget cluster PRs #1212 / #1213 / #1227, PR #1226 review, PR #1233 review, `kv_scratch` cleanup |
| Gleb | [@gmorgachev](https://github.com/gmorgachev) | Gonka core team | MiniMax governance entry, micro-update deadline, post-deploy benchmark request (Qwen3 240k), `kv_scratch` cleanup direction |
| David | [GLiberman](https://github.com/GLiberman) | Gonka core team | Kimi `thinking_token_budget` and corner-case requirements |
| Danya | [qdanik](https://github.com/qdanik) |  | PR #1226 (per-model dispatch / tool-message shape / reasoning_split), PR #1233 (Kimi reasoning-burn quarantine fix) |

