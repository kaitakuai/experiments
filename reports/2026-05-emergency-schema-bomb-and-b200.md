# Emergency response — schema-bomb defense and B200 image stabilization (2026-05-11 … 2026-05-31)

## Summary

The [kaitaku.ai](https://github.com/kaitakuai) team (Mykola [@baychak](https://github.com/baychak), Pavlo [@clanster](https://github.com/clanster)) handled the network-critical incident that hit Kimi-K2.6 hosts during epoch 266 and the partner-side B200 image crash cluster that surfaced in parallel. The work covers:

- Prompt-of-death investigation flagged by Hyperfusion (matrix test of failure modes on Qwen and Kimi)
- Gateway defense cascade (8 merged PRs + 2 reviews) in `gonka-ai/gonka` to filter requests that crash vLLM
- B200 Kimi image stabilization: reproduced engine death under sustained load, shipped fixed image, validated 72-hour soak
- Stall request mass-update — direct DM outreach to 5 affected node operators

All defense PRs shipped in `devshard-0.2.13` (commit `178086202`, merged 2026-05-26).

---

## Prompt-of-death investigation (2026-05-11 … 2026-05-13)

Hyperfusion reported a "prompt of death" for Qwen that hung mlnodes. David Liberman temporarily disabled Transfer Agent completion. The pair ran a matrix test of known failure modes:

| Vector | Models tested | Outcome |
|--------|---------------|---------|
| `n=1638400` | Qwen, Kimi | rejected upstream |
| `structured_outputs` payloads | Qwen, Kimi | reproducible engine hang on 0.15.1; works on 0.19+ |
| `presence_penalty` / `frequency_penalty` extreme values | Qwen, Kimi | clamp recommended |
| `messages[].content` with `image_url` | Qwen, Kimi | rejected at validator |
| `thinking: {"type": "enabled"}` hang | Kimi | reproduced on 0.15, fixed on 0.20 |

**Conclusion.** Findings filed in the `dl/devshards-gateway-to-main` branch and used as the basis for the defense PR cascade below.

---

## Epoch 266 — Kimi voting-power collapse (2026-05-16 22:01 → 05-17 05:42)

In epoch 266 many nodes serving Kimi-K2.6 failed to commit nonces; Kimi VP came in at 412k against the 488k threshold, so 132 k Kimi-delegations were discarded by the chain and the guardian tiebreaker did not resolve (preserved-flag overlap).

Log analysis across gonka-1 / gonka-3:

| Host | Symptom | Hypothesis |
|------|---------|------------|
| gonka-1 | vLLM stuck in init, `/health` returns 503 | engine deadlock after malformed request |
| gonka-3 | failed before PoC start | upstream timeout chain |
| gonka-2 | not serving Kimi | not affected |

**Root cause.** A request carrying a deeply-nested `response_format.json_schema` (200+ levels of nesting) reproducibly killed `/health` without producing a stack trace on mlnode-021 (B300) — silent engine death. The defense cascade below was built on top of this finding.

---

## Gateway defense cascade — PRs in `gonka-ai/gonka` (2026-05-14 … 2026-05-19)

All PRs ship in `devshard-0.2.13`. Author column shows the kaitaku.ai contributor; reviewers from the core team are listed in Participants.

| PR | Title | Author | Notes |
|----|-------|--------|-------|
| [#1170](https://github.com/gonka-ai/gonka/pull/1170) | strip `min_tokens` when `stop_token_ids` present (vLLM crash mitigation) | [@clanster](https://github.com/clanster) | upstream issue [vllm-project/vllm#42363](https://github.com/vllm-project/vllm/issues/42363) — still open on 0.20.2 |
| [#1171](https://github.com/gonka-ai/gonka/pull/1171) | strip `prompt_logprobs` to mitigate vLLM OOM crash | [@clanster](https://github.com/clanster) | upstream [vllm-project/vllm#41031](https://github.com/vllm-project/vllm/issues/41031) |
| [#1172](https://github.com/gonka-ai/gonka/pull/1172) | clamp `temperature` ∈ [0, 2], force-validation logprobs | [@clanster](https://github.com/clanster) | open-ended temperature hung vLLM at `temperature=999999` |
| [#1174](https://github.com/gonka-ai/gonka/pull/1174) | implement request filters for chat messages and parameters | [qdanik](https://github.com/qdanik) (core) | **full review + on-stand repro** on mlnode-021 by [@baychak](https://github.com/baychak) — confirmed schema-bomb hypothesis |
| [#1177](https://github.com/gonka-ai/gonka/pull/1177) | add OpenAI observability fields (`user` / `metadata` / `parallel_tool_calls` / `stream_options`) on top of #1180 | [@baychak](https://github.com/baychak) | safer whitelist + extension on top of #1180; rebase under #1180 |
| [#1184](https://github.com/gonka-ai/gonka/pull/1184) | special-token literal sanitizers for tools and messages content | [@baychak](https://github.com/baychak) | defense-in-depth against prompt-injection via `<\|im_end\|>` / `<\|im_user\|>` etc. — CVE `GHSA-hpv8-x276-m59f` |
| [#1187](https://github.com/gonka-ai/gonka/pull/1187) | bump `tools.parameters` MaxDepth 5 → 16 | [@baychak](https://github.com/baychak) | #1180 default depth ≤5 broke real agent payloads (OpenClaw / Cursor / Claude Code / MCP, depth ≤12 observed) |
| [#1195](https://github.com/gonka-ai/gonka/pull/1195) | bump `tools.parameters` MaxNodes 128 → 256 | [@baychak](https://github.com/baychak) | second-pass relaxation for legit agent schemas after #1187 |

**Stress test (2026-05-18, [@clanster](https://github.com/clanster)).** 109 hand-crafted requests against the new whitelist: numeric edge cases, regex backtracking, `max_tokens` up to `--max-model-len`, `messages=2048`, schema flood, Kimi parser quirks. **0 engine crashes, 0 `/health` degradations.**

---

## B200 Kimi image stabilization (2026-05-18 … 2026-05-22)

A partner reported eight `mlnode-full:0.2.12-vllm0.20.0-b200-k5-kimi-1` containers (4×B200 each) restarting 0–2× per day under production load.

**Reproduction.** Repeated token `Xl` + 16 parallel requests reliably killed the vLLM engine within 5–60 minutes. Root cause: `enforce_eager=true` combined with B200 KV-cache pressure (upstream [vllm-project/vllm#40926](https://github.com/vllm-project/vllm/issues/40926)).

**Fix.** New image [`ghcr.io/kaitakuai/mlnode-b200-kimi-k2-6:0.2.13-vllm0.20.0-q.int4-k2`](https://github.com/kaitakuai/experiments/tree/main/2026-05/kimi_k26_int4_4xb200_q-int4-k2) — removed `--enforce-eager`, enabled `cudagraph_mode: FULL_AND_PIECEWISE`.

| Image | Engine deaths | Container-hours | Rate |
|-------|--------------:|----------------:|-----:|
| Old `b200-k5-kimi-1` | 9 | 96 | 2.25 / day |
| **New `0.2.13-...q.int4-k2`** | **0** | **72** | **0 / day** ★ |

Announced in the DevOps channel on 2026-05-23.

---

## Participants

[kaitakuai](https://github.com/kaitakuai) ([@baychak](https://github.com/baychak), [@clanster](https://github.com/clanster)).
Gonka core team ([@gmorgachev](https://github.com/gmorgachev), [libermans](https://github.com/libermans).
[qdanik](https://github.com/qdanik)).

| Participant | GitHub | Role | Contribution |
|-------------|--------|------|--------------|
| Pavlo | [@clanster](https://github.com/clanster) | kaitakuai | Defense PRs #1170 / #1171 / #1172, stress-test 109 requests on the new whitelist, log analysis for epoch 266 |
| Mykola | [@baychak](https://github.com/baychak) | kaitakuai | PR #1174 on-stand repro (mlnode-021 B300) confirming the schema-bomb hypothesis, defense PRs #1177 / #1184 / #1187 / #1195, B200 image reproduction + fixed image + 72-hour soak, stall mass-update outreach |
| Gleb | [@gmorgachev](https://github.com/gmorgachev) | Gonka core team | Epoch 266 coordination, review of defense PRs |
| David | [libermans](https://github.com/libermans) | Gonka core team | Initial incident flagging (epoch 266 voting-power collapse, stalled-request reports), defense PR review |
| Danya | [qdanik](https://github.com/qdanik) |  | PR #1174 (request filters base implementation), PR #1180 |
