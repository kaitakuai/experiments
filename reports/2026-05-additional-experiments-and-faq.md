# Additional benchmarks, network operations, and community documentation (2026-04-30 … 2026-05-30)

## Summary

Additional work in the v0.2.12 → v0.2.13 cycle that did not fit the prior bounty's "MiniMax-and-bug-fixes" framing. The [kaitaku.ai](https://github.com/kaitakuai) team (Mykola [@baychak](https://github.com/baychak), Pavlo [@clanster](https://github.com/clanster)) handled:

- Kimi-K2.6 INT4 on a multi-node 2×8×H100 cluster with InfiniBand — at the request of Gleb
- Discovery and PR-formalization of the `VLLM_USE_FLASHINFER_MOE_INT4=1` burst on B300 / B200
- Community FAQ — 17 inference questions for [@tcharchian](https://github.com/tcharchian)

---

## PoC throughput — Kimi-K2.6 INT4 on multi-node 2×8×H100 cluster

Deployed Kimi-K2.6 INT4 on a Hyperbolic uk-southeast-3 cluster of two 8×H100 80GB nodes connected by InfiniBand 8×400 Gb/s. vLLM 0.15.1 ran as a single instance with `TP=16` over Ray distributed executor on NCCL/IB. Full native context of 262 144 (256 K) tokens fit in KV cache at `--gpu-memory-utilization 0.92`.

| Config | Best batch | Nonces/min |
|--------|-----------:|-----------:|
| [2×8×H100 InfiniBand, TP=16](https://github.com/kaitakuai/experiments/tree/main/2026-05/kimi-k26-int4-2x8xh100) | 32 | **1389** ★ |
| 2×8×H100, TP=8 + PP=2 (control) | 32 | 1014 |

**Conclusion.** TP=16 over IB is the better topology for Kimi-K2.6 INT4 on this hardware tier — TP=8 + PP=2 leaves ~27 % on the table due to pipeline bubbles.

---

## `VLLM_USE_FLASHINFER_MOE_INT4=1` burst on B300 / B200

Discovered on 2026-05-02 while iterating on Blackwell configurations. The flag is only available in vLLM 0.16+ on `sm_100+`. Required flag combination:

```
VLLM_USE_FLASHINFER_MOE_INT4=1 \
  --attention-backend CUTLASS_MLA \
  --max-num-seqs 128 \
  --enable-expert-parallel \
  --enforce-eager
```

| GPU (TP) | Without flag | With flag | Δ |
|----------|-------------:|----------:|--:|
| [8×B300](https://github.com/kaitakuai/experiments/tree/main/2026-05/kimi_k26_b300_eager_flashinfer) | 2048 | **5120** ★ | +150 % |
| 8×B200 | (baseline) | **3660** | — |

Validation across the new config: mean L2 = 0.2034, stat-test = 0.4 — within tolerance.

---

## Community FAQ — 17 inference questions (2026-05-22 … 2026-05-30)

Six review iterations (v1 → v6) of `handoff-improved-answers-ru-v6.md`. For each of 17 community questions:

| Question category | Items |
|-------------------|-------|
| Token Limits | Q1, Q2, Q3, Q4, Q5, Q6 |
| Speed and Performance | Q7, Q8 |
| Tool Use and Agent Compatibility | Q9, Q10, Q11 |
| Web Search and Fetching | Q12, Q13, Q14 |
| Caching and Infrastructure | Q15, Q16 |
| Multimodality | Q17 |

Each answer was verified live against the `gonka-api.org` broker; cross-checked against the `dl/devshards-gateway-to-main` branch and chat-archived findings. Three rounds of multi-agent verification workflow caught internal-info leaks (epoch references, partner names) before publication.

---

## Participants

[kaitakuai](https://github.com/kaitakuai) ([@baychak](https://github.com/baychak), [@clanster](https://github.com/clanster)).
Gonka core team ([@gmorgachev](https://github.com/gmorgachev), [libermans](https://github.com/libermans), [tcharchian](https://github.com/tcharchian)).

| Participant | GitHub | Role | Contribution |
|-------------|--------|------|--------------|
| Pavlo | [@clanster](https://github.com/clanster) | kaitakuai | Kimi-K2.6 INT4 on 2×8×H100 InfiniBand cluster (1389 nonces/min), `VLLM_USE_FLASHINFER_MOE_INT4=1` discovery and validation on B300 / B200 |
| Mykola | [@baychak](https://github.com/baychak) | kaitakuai | mlnode investigations and hotfixes, community FAQ — 17 questions / 6 review iterations |
| Gleb | [@gmorgachev](https://github.com/gmorgachev) | Gonka core team | Initial requests (2×8×H100 cluster, network002 heads-up), PR-formalization suggestion for the MOE INT4 flag |
| David | [libermans](https://github.com/libermans) | Gonka core team | mlnode `gonka1d694r…` flagging and joint debug |
| Tania | [tcharchian](https://github.com/tcharchian) | Gonka core team | Community FAQ request, TEE working-session initiation |
