# decode-PoC — GLM-5.3-Flash on vLLM 0.28 and the migration to vLLM 0.30.0 (2026-09-10 … 2026-09-25, work in progress)

## Summary

The [kaitaku.ai](https://github.com/kaitakuai) team (Mykola [@baychak](https://github.com/baychak), Pavlo [@clanster](https://github.com/clanster)) continued the decode-PoC track after the [frozen point on vLLM 0.25.1](https://github.com/kaitakuai/experiments/blob/main/reports/2026-06-decode-poc-research-and-integration.md) at the request of [@vbgd0](https://github.com/vbgd0). The work covers:

- decode-PoC on GLM-5.3-Flash, the model that took Kimi's place on the decode track: plugin support, the sampler residual on the vendor vLLM 0.28.1 base, and the measured state on four topologies (2×B300, 4×H200, 8×H100, 4×B200) — [2026-09/decode-poc-0281-glm-freeze](https://github.com/kaitakuai/experiments/tree/main/2026-09/decode-poc-0281-glm-freeze);
- MLNode fixes for the GLM release ([gonka-ai/gonka#1751](https://github.com/gonka-ai/gonka/pull/1751)) and the cross-hardware check of the first-nonce fix [gonka-ai/gonka-vllm-plugins#10](https://github.com/gonka-ai/gonka-vllm-plugins/pull/10) by [@vbgd0](https://github.com/vbgd0);
- upstream PRs on the GLM line: [gonka-ai/vllm#112](https://github.com/gonka-ai/vllm/pull/112) (base moved to the vendor tree, merged), [gonka-ai/vllm#113](https://github.com/gonka-ai/vllm/pull/113) (residual, open), [gonka-ai/gonka-vllm-plugins#12](https://github.com/gonka-ai/gonka-vllm-plugins/pull/12) (merged 2026-09-21 together with [#8](https://github.com/gonka-ai/gonka-vllm-plugins/pull/8));
- MiniMax-M2.7 and DeepSeek-V4-Flash measured on the 0.28.1 base, which fixed the release plan of 2026-09-16 (DeepSeek and MiniMax on 0.25.1, GLM on 0.28);
- the migration to vLLM 0.30.0 after the core team moved the release to that version on 2026-09-23: residual and plugin ported, seven cells measured on B300 and H100, the old reference artifacts validated by the 0.30 stack, one validator regression on GLM located and closed by two launch flags — [2026-09/decode-poc-0300-migration](https://github.com/kaitakuai/experiments/tree/main/2026-09/decode-poc-0300-migration); PRs [gonka-ai/vllm#114](https://github.com/gonka-ai/vllm/pull/114) and [gonka-ai/gonka-vllm-plugins#19](https://github.com/gonka-ai/gonka-vllm-plugins/pull/19).

This is an interim report: the release on 0.30.0 is being assembled by the core team. Open items are in the Status section.

---

## decode-PoC on GLM-5.3-Flash, vLLM 0.28 (2026-09-11 … 2026-09-16)

On 2026-09-11 [@vbgd0](https://github.com/vbgd0) asked for decode-PoC on GLM-5.3-Flash with the smallest possible change, targeting `release/v0.28-decode-int`, with nonces/min and validation vectors as the deliverable. The model had entered the network the same week ([proposal #101](https://gonka.gg/network/proposals/101)).

| Step | Date | Resource |
| --- | --- | --- |
| MLNode 3.0.17 does not start on the Ubuntu 24.04 base (`useradd: UID 1000 is not unique`); fix: stop creating `appuser` in the entrypoints | 2026-09-10 … 09-11 | [gonka-ai/gonka#1751](https://github.com/gonka-ai/gonka/pull/1751), merged by [@vbgd0](https://github.com/vbgd0) |
| Plugin support for the GLM architecture (`Glm5NextForConditionalGeneration`: KDA layers, sparse MLA, 42 MoE routers seeded); the "simplest overlay" turned out not to be simple — the model differs from DeepSeek and MiniMax more than those two differ from each other | 2026-09-11 … 09-14 | branch `poc-as-chat-vllm-0.28.0-glm-dev` |
| Residual moved from the July upstream ancestor to the vendor tree of the official image (`385dce36b`, 0.28.1rc1); bf16 KV works on H200 from that base | 2026-09-14 | branch `poc-as-chat-vllm-0.28.1-glm-dev` |
| Launch settings per card: N as `--max-num-seqs` = `max_cudagraph_capture_size`, MoE kernel and KV type per generation; 4×H200 decode-PoC from 6.4 to 14.2 nonces/s | 2026-09-13 … 09-15 | `PROFILES.md` in the devkit |
| H200 and Blackwell hosts voting each other invalid on GLM ([@gmorgachev](https://github.com/gmorgachev), 2026-09-14): the first nonce of a batch was not validated; fix by [@vbgd0](https://github.com/vbgd0), re-checked B300 against H200 on five seeds | 2026-09-14 | [gonka-ai/gonka-vllm-plugins#10](https://github.com/gonka-ai/gonka-vllm-plugins/pull/10), MLNode 3.1.0 |
| Frozen point: one code revision, four configurations, throughput, R, the prover × validator matrix, data on Drive (3.4 GB) | 2026-09-13 … 09-16 | [2026-09/decode-poc-0281-glm-freeze](https://github.com/kaitakuai/experiments/tree/main/2026-09/decode-poc-0281-glm-freeze) ([experiments#13](https://github.com/kaitakuai/experiments/pull/13)) |

Throughput on the frozen point (PoC first, then chat, one boot per row):

| cards · GLM-5.3-Flash | PoC, nonces/s | nonces/min per 8 GPUs | chat, requests/s | R |
| --- | ---: | ---: | ---: | ---: |
| 2×B300 | 21.04 | 5,050 | 21.48 | 0.980 |
| 4×H200 | 14.18 | 1,702 | 15.11 | 0.938 |
| 8×H100 | 15.35 | 921 | 18.14 | 0.846 |
| 4×B200 | 35.02 | 4,202 | 35.67 | 0.982 |

**Conclusion.** decode-PoC runs on GLM-5.3-Flash on all four topologies of the network; the measured state and the launch settings are published, the raw data is on Drive.

---

## Upstream PRs and the release-base decision (2026-09-15 … 2026-09-21)

| Step | Date | Resource |
| --- | --- | --- |
| Base of `release/v0.28-decode-int` moved to the vendor tree `385dce36b`, so the residual diff is 21 files instead of the whole branch | 2026-09-16 | [gonka-ai/vllm#112](https://github.com/gonka-ai/vllm/pull/112), merged by [@vbgd0](https://github.com/vbgd0) |
| Residual for GLM on that base; an inference-validation change through vLLM trace replay was reverted at the core team's request — inference validation is to be reworked separately | 2026-09-16 … 09-17 | [gonka-ai/vllm#113](https://github.com/gonka-ai/vllm/pull/113), open |
| Plugin: decode-PoC on vLLM 0.28 for GLM, PoC and chat in one batch | 2026-09-16 → merged 2026-09-21 | [gonka-ai/gonka-vllm-plugins#12](https://github.com/gonka-ai/gonka-vllm-plugins/pull/12) |
| MiniMax-M2.7 and DeepSeek-V4-Flash on the 0.28.1 GLM base (4×H100, 1×B300, 2×H200), same launch settings as the 0.25.1 freeze: MiniMax PoC within −2 … −9 %, DeepSeek PoC +10 … +15 % and chat +22 … +23 % | 2026-09-15 … 09-16 | Pavlo; data in the Drive devkit |
| Release plan agreed with [@vbgd0](https://github.com/vbgd0): DeepSeek and MiniMax stay on vLLM 0.25.1, GLM ships on its own 0.28 base; later versions after the release | 2026-09-16 | — |
| Teacher-forced validation data added to both devkits for the core team's rewrite of the decode statistic ([gonka-ai/gonka-vllm-plugins#16](https://github.com/gonka-ai/gonka-vllm-plugins/pull/16), [#17](https://github.com/gonka-ai/gonka-vllm-plugins/pull/17) by [@vbgd0](https://github.com/vbgd0)) | 2026-09-21 | Drive devkits |

**Conclusion.** The GLM line is upstream: base and plugin merged, residual under review. The 0.28.1 numbers for the other two models did not justify moving them before the release.

---

## Migration to vLLM 0.30.0 (2026-09-23 … 2026-09-25)

vLLM 0.30.0 (tagged 2026-09-21) includes GLM-5.3-Flash officially. On 2026-09-23 [@vbgd0](https://github.com/vbgd0) set the plan: if all three models start on 0.30.0, the decode-PoC release goes out on it; step 0 — the three models on B300 and H200, then a PR to him and a testnet built by him; step 1 — validation against the 0.25.1 / 0.28.1 reference artifacts on the same pairs, bit-identity not expected; step 2 — throughput on the known settings.

| Step | Date | Resource |
| --- | --- | --- |
| Residual ported to 0.30.0 (35 files) and one plugin for the three models (0.2.0) | 2026-09-23 … 09-24 | branches `poc-as-chat-vllm-0.30.0-dev` in [kaitakuai/vllm](https://github.com/kaitakuai/vllm) and [kaitakuai/gonka-vllm-plugins](https://github.com/kaitakuai/gonka-vllm-plugins) |
| Seven cells measured: MiniMax on 1×B300 and 4×H100, DeepSeek NVFP4 on 1×B300 and FP8 on 4×H100, GLM on 8×H100 and 2×B300; each cell: boot, validation of the old references by the 0.30 validator, same-boot self-validation, a mining round through `/init/generate`, replay equivalence 24/24 | 2026-09-24 | [2026-09/decode-poc-0300-migration](https://github.com/kaitakuai/experiments/tree/main/2026-09/decode-poc-0300-migration) |
| GLM validator regression on 0.30 located in two hours at [@vbgd0](https://github.com/vbgd0)'s request: two new prefill kernels in 0.30 (FlashKDA for the KDA layers, dense MHA prefill in sparse MLA) move the validator away from the 0.28.1 references; with `--kda-prefill-backend triton` and `sparse_mla_force_mqa` the 0.30 validator matches the 0.28.1 one; confirmed on 2×B300 the same evening | 2026-09-24 | same directory, `PROFILES.md` |
| Decision with [@vbgd0](https://github.com/vbgd0): GLM on 0.30 with the two prefill flags | 2026-09-25 | — |
| Upstream PRs on the branches created by [@vbgd0](https://github.com/vbgd0): residual on `release/v0.30-decode-int`, plugin on `decode/vlm030`; updated launch parameters for 0.30.0 handed over | 2026-09-25 | [gonka-ai/vllm#114](https://github.com/gonka-ai/vllm/pull/114), [gonka-ai/gonka-vllm-plugins#19](https://github.com/gonka-ai/gonka-vllm-plugins/pull/19), both open |

Throughput on 0.30.0 against the frozen points (PoC nonces/s, chat requests/s):

| model × cards | PoC 0.30 vs freeze | chat 0.30 vs freeze | note |
| --- | ---: | ---: | --- |
| MiniMax-M2.7 1×B300 | 26.8 vs 29.9 | 29.0 vs 31.6 | gpu-memory-utilization 0.97 → 0.95 on 0.30 |
| MiniMax-M2.7 4×H100 | 35.3 vs 43.2 | 36.8 vs 42.2 | 0.94 → 0.90: validation ran out of memory at 0.94 |
| DeepSeek-V4-Flash NVFP4 1×B300 | 47.1 vs 42.4 | 43.7 vs 37.2 | |
| DeepSeek-V4-Flash FP8 4×H100 | 21.4 vs 22.5 | 23.8 vs 22.7 | |
| GLM-5.3-Flash 8×H100 | 17.9 vs 15.75 | 20.5 vs 16.3 | default 0.30 kernels; with the two flags PoC 17.5, chat 18.7 |
| GLM-5.3-Flash 2×B300 | 17.4 vs 16.3 | 17.3 | with the two flags |

Validation: the 0.30 stack is as deterministic as before within a boot and validates the 0.25.1 MiniMax references at the "same card, other boot" level of the freeze; the DeepSeek references sit above that band and are to be re-recorded on 0.30; GLM matches the 0.28.1 references with the two flags. On Hopper GLM runs only from the official image: the PyPI FlashInfer wheel fails vLLM's sparse-MLA check on sm90.

**Conclusion.** All three models run decode-PoC on vLLM 0.30.0 with known launch settings; the code is in two upstream PRs and the testnet build is with the core team.

---

## Status (2026-09-25)

| State | Item |
| --- | --- |
| Done | decode-PoC on GLM-5.3-Flash on four topologies with published measured state; [gonka-ai/vllm#112](https://github.com/gonka-ai/vllm/pull/112) and [gonka-ai/gonka-vllm-plugins#12](https://github.com/gonka-ai/gonka-vllm-plugins/pull/12) merged; [gonka-ai/gonka#1751](https://github.com/gonka-ai/gonka/pull/1751) merged; residual and plugin on 0.30.0 with seven measured cells and the GLM prefill fix |
| In progress | Review of [gonka-ai/vllm#114](https://github.com/gonka-ai/vllm/pull/114) and [gonka-ai/gonka-vllm-plugins#19](https://github.com/gonka-ai/gonka-vllm-plugins/pull/19); testnet build by the core team; [gonka-ai/vllm#113](https://github.com/gonka-ai/vllm/pull/113) pending the decision on the 0.28 line |
| Next | H200 and B200 cells on 0.30.0; reference artifacts re-recorded on 0.30.0 with the core team's decode statistic ([#16](https://github.com/gonka-ai/gonka-vllm-plugins/pull/16), [#17](https://github.com/gonka-ai/gonka-vllm-plugins/pull/17)); coefficients for the release |

**Conclusion.** The decode-PoC code base for the release is on vLLM 0.30.0 and upstream; the remaining work is review, testnet and reference artifacts.

---

## Participants

| Name | GitHub | Affiliation | Role in this period |
| --- | --- | --- | --- |
| Mykola Baichak | [@baychak](https://github.com/baychak) | kaitaku.ai | GLM decode-PoC, residual and plugin ports, 0.30 campaign, upstream PRs |
| Pavlo | [@clanster](https://github.com/clanster) | kaitaku.ai | MLNode images and 3.1.0 checks, MiniMax and DeepSeek on the 0.28.1 base, host operations |
| Vladislav | [@vbgd0](https://github.com/vbgd0) | Gonka core team | Requests and release plan, review and merges, first-nonce fix, decode statistic rewrite |
| Gleb Morgachev | [@gmorgachev](https://github.com/gmorgachev) | Gonka core team | Scope; reported the H200 vs Blackwell validation failure on GLM |
