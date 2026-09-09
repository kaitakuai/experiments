# ML-node observability — metrics exporter, DAPI federation and community Grafana (2026-07-06 … 2026-09-08)

## Summary

The [kaitaku.ai](https://github.com/kaitakuai) team (Mykola [@baychak](https://github.com/baychak), Pavlo [@clanster](https://github.com/clanster)) built the ML-node monitoring path that ships in the vLLM 0.25.1 MLNode release, and hosts the community Grafana that reads it. The work covers:

- A design agreed with Gonka core and [@a-kuprin](https://github.com/a-kuprin) on 2026-07-06 … 07-08: metrics on by default, `GONKA_METRICS=off` to opt out, an allowlist of vLLM families plus GPU and host metrics, no placement labels, no sidecars.
- The ML-node metrics exporter `GET /api/v1/metrics` (schema v1) and the DAPI federation endpoint `GET /v1/mlnodes/metrics` — [gonka-ai/gonka#1469](https://github.com/gonka-ai/gonka/pull/1469) merged into `upgrade-v0.2.14` on 2026-07-19; the exporter and the scheduler-heartbeat liveness fix merged to `main` on 2026-08-17 inside [gonka-ai/gonka#1536](https://github.com/gonka-ai/gonka/pull/1536).
- Grafana dashboards — network-node view, ML-node drill-down and whole-network overview with PoC/cPoC markers — hosted on Kaitaku nodes since July.
- The public deployment repository [kaitakuai/gonka-grafana](https://github.com/kaitakuai/gonka-grafana) (Grafana + VictoriaMetrics, `docker-compose up`) and the read-only community instance at [monitoring.kaitaku.ai](https://monitoring.kaitaku.ai), live since 2026-09-04.

The MLNode side reached the network with the vLLM 0.25.1 MLNode image (August 2026 update). Tracking issue: [gonka-ai/gonka#1692](https://github.com/gonka-ai/gonka/issues/1692).

---

## Design (2026-07-06 … 2026-07-08)

The agenda came from [@a-kuprin](https://github.com/a-kuprin) and [@gmorgachev](https://github.com/gmorgachev): expose ML-node load, carry it to the participant's node, and make it readable next to inference traffic. Mykola wrote the architecture document and the work plan; Artur (Hyperfusion) advised on vLLM Prometheus families and DCGM; [@qdanik](https://github.com/qdanik) had started an earlier observability stage and supplied an agent-generated DAPI prototype ([kAIPraxisBot/gonka@2bc354bf](https://github.com/kAIPraxisBot/gonka/commit/2bc354bf)) that the federation code was built on.

| Decision | Choice |
|---|---|
| Source of ML-node metrics | vLLM `/metrics` + NVML + `/proc`; no privileged containers, no DCGM sidecar |
| What is exported | allowlist of 17 vLLM families, `mlnode_gpu_*`, `mlnode_host_*`, `mlnode_up`, `gonka_metrics_schema_info` |
| Privacy | no hostname, IP, provider or placement labels; `model_name` normalised |
| Default | on; `GONKA_METRICS=off` returns 404 |
| Retention on the node | none beyond the in-memory scrape cache |

**Conclusion.** The path is "as tiny as possible": read what vLLM and NVML already expose, federate it through DAPI, keep the schema versioned.

---

## Implementation — fork, review, upstream (2026-07-10 … 2026-08-17)

Pavlo implemented both halves in the Kaitaku fork; Mykola reviewed them there before they went upstream.

| PR | Title | Author | Status |
|---|---|---|---|
| [kaitakuai/gonka#14](https://github.com/kaitakuai/gonka/pull/14) | feat(mlnode): metrics exporter — GET /api/v1/metrics, schema v1 | [@clanster](https://github.com/clanster) | 13 review rounds, approved 2026-07-16 |
| [kaitakuai/gonka#15](https://github.com/kaitakuai/gonka/pull/15) | feat(dapi): public mlnode metrics federation — /v1/mlnodes/metrics (Phase 2) | [@clanster](https://github.com/clanster) | 8 review rounds, approved 2026-07-16 |
| [gonka-ai/gonka#1469](https://github.com/gonka-ai/gonka/pull/1469) | feat(dapi): public mlnode metrics federation — /v1/mlnodes/metrics | [@clanster](https://github.com/clanster) | +850/−2; reviewed and merged by [@a-kuprin](https://github.com/a-kuprin) 2026-07-19 into `upgrade-v0.2.14` |
| [gonka-ai/gonka#1421](https://github.com/gonka-ai/gonka/pull/1421) | fix(mlnode): scheduler-heartbeat liveness for vLLM runner + damped proxy health probe | [@baychak](https://github.com/baychak) | superseded by #1536 |
| [gonka-ai/gonka#1501](https://github.com/gonka-ai/gonka/pull/1501) | feat(mlnode): metrics exporter — GET /api/v1/metrics, schema v1 | [@baychak](https://github.com/baychak) | superseded by #1536 |
| [gonka-ai/gonka#1536](https://github.com/gonka-ai/gonka/pull/1536) | mlnode vLLM 0.25.1: exporter, heartbeat, and release profiles | [@vbgd0](https://github.com/vbgd0) | merged to `main` 2026-08-17; "Supersedes #1421 and #1501" |

Review of #1469 moved the rate limit from the echo handler into a separate nginx zone and kept PoC and devshard paths exempt. #1536 carries the Kaitaku commits with authorship intact (`645235fc` exporter — clanster; `3ee92e5c` heartbeat liveness, `ac25403a` exporter guide and ADR — baychak) and credits the team in its Acknowledgements.

**Conclusion.** Both halves are upstream: federation since v0.2.14, exporter and heartbeat since the vLLM 0.25.1 MLNode release.

### Follow-ups requested by DAPI (2026-08-11 … 2026-08-13)

After the merge [@a-kuprin](https://github.com/a-kuprin) fixed the endpoint mount and added host-ping observability in [gonka-ai/gonka#1580](https://github.com/gonka-ai/gonka/pull/1580), which specifies an MLNode `GET /api/v1/clock`.

| PR | Title | Status |
|---|---|---|
| [gonka-ai/gonka#1586](https://github.com/gonka-ai/gonka/pull/1586) | feat(mlnode): mount GET /api/v1/clock for dapi's ping job | approved by [@a-kuprin](https://github.com/a-kuprin) 2026-08-13, not merged |
| [gonka-ai/gonka#1590](https://github.com/gonka-ai/gonka/pull/1590) | fix(mlnode): accept DAPI's JSON POSTs sent without Content-Type | open, no review |

**Conclusion.** Two small MLNode changes are ready; both need retargeting to `main` before merge.

---

## Grafana and hosting (2026-07-14 … 2026-08-17)

Pavlo built the dashboards against the federation endpoint of the Kaitaku nodes and has hosted Grafana since 2026-07-16.

| Dashboard | Content |
|---|---|
| `gonka-network` | one participant's network node: its ML nodes, engine load, queue, KV usage, GPU temperature/power/clocks, host CPU and memory |
| `gonka-node` | drill-down into one ML node |
| `gonka-overview` | whole network via epoch auto-discovery; PoC/cPoC markers added on request of [@gmorgachev](https://github.com/gmorgachev) (2026-08-17); token counts include validation traffic and say so |

The hosted instance additionally sends Telegram notifications on node state; alerting is not part of the public repository.

**Conclusion.** Core team used the overview from 2026-08-17; the token-count caveat became a banner in the public repository.

---

## Community release (2026-08-27 … 2026-09-08)

On 2026-08-27 [@gmorgachev](https://github.com/gmorgachev) asked for a repository any operator can bring up with `docker-compose up`, and for Kaitaku to keep hosting the community instance; the collector config for one's own MLNode is to land in the main repository later.

| Item | Result |
|---|---|
| [kaitakuai/gonka-grafana](https://github.com/kaitakuai/gonka-grafana) | Grafana + VictoriaMetrics stack (2026-08-28): the three dashboards, datasource provisioning, `scrape.yml`, epoch auto-discovery and PoC-phase annotation scripts; compose project `gonka-monitoring`, arm64 note, disclaimer banner (2026-09-04) |
| [monitoring.kaitaku.ai](https://monitoring.kaitaku.ai) | read-only community instance, live 2026-09-04 |
| [gonka-ai/gonka#1692](https://github.com/gonka-ai/gonka/issues/1692) | tracking issue opened by [@tcharchian](https://github.com/tcharchian); reported ready 2026-09-08 |

**Conclusion.** The stack is reproducible from the public repository; Kaitaku maintains and hosts the community instance.

---

## Participants

- [kaitakuai](https://github.com/kaitakuai) ([@baychak](https://github.com/baychak), [@clanster](https://github.com/clanster)).
- Gonka core team ([@gmorgachev](https://github.com/gmorgachev), [@vbgd0](https://github.com/vbgd0), [@tcharchian](https://github.com/tcharchian)); independent contributors [@a-kuprin](https://github.com/a-kuprin), [@qdanik](https://github.com/qdanik), Artur (Hyperfusion).

| Participant | GitHub | Role | Contribution |
|---|---|---|---|
| Pavlo | [@clanster](https://github.com/clanster) | kaitakuai | metrics exporter, DAPI federation (#14, #15, #1469), all Grafana dashboards, `gonka-grafana` repository, hosting |
| Mykola | [@baychak](https://github.com/baychak) | kaitakuai | architecture document and plan, review of #14/#15, upstream packaging (#1501, #1531), heartbeat liveness (#1421), `/api/v1/clock` (#1586), #1590 |
| Aleksandr Kuprin | [@a-kuprin](https://github.com/a-kuprin) | independent | agenda, review and merge of #1469, endpoint mount fix and `/clock` spec (#1580), review of #1586 |
| Daniil Yankouski | [@qdanik](https://github.com/qdanik) | independent | earlier observability stage; DAPI prototype the federation was built on |
| Artur | — | Hyperfusion | vLLM metrics and DCGM consultation |
| Gleb Morgachev | [@gmorgachev](https://github.com/gmorgachev) | Gonka core team | scope, metric list, overview markers, community-release format |
| Vladislav Bogdanov | [@vbgd0](https://github.com/vbgd0) | Gonka core team | hardening and merge of exporter and heartbeat in #1536 |
| Tania Charchian | [@tcharchian](https://github.com/tcharchian) | Gonka core team | tracking issue #1692 |
