# Hy3 evaluation — PoC baselines, quantised checkpoints and inference validation (2026-08-18 … 2026-08-20)

## Summary

The [kaitaku.ai](https://github.com/kaitakuai) team (Mykola [@baychak](https://github.com/baychak), Pavlo [@clanster](https://github.com/clanster)) evaluated [`tencent/Hy3`](https://huggingface.co/tencent/Hy3) as a candidate network model at the request of the core team (2026-08-18 … 08-19: H100 vs B300 PoC performance on the current 3.0.16 release, inputs for a possible proposal). The work covers:

- honest FP8 PoC baselines on four topologies — 2×B300, 4×B200, 4×H200, 8×H100 — all on the existing MLNode 3.0.16 image, no vLLM port needed (`HYV3ForCausalLM` and `HYV3MTPModel` ship in vLLM 0.25.1);
- a re-measurement of the 4×H200 baseline after the throughput accounting fix in [kaitakuai/experiments#7](https://github.com/kaitakuai/experiments/pull/7) (Vlad) invalidated the first run;
- three quantised checkpoints measured at their minimal topologies, with cross-hardware L2 across five hosts;
- MTP speculative decoding measured for both PoC and serving;
- cross-hardware inference validation 2×B300 → 4×H200, which exposed that speculation on the validator silently breaks replay validation — fixed in [kaitakuai/vllm#21](https://github.com/kaitakuai/vllm/pull/21).

The core team reproduced the results on 2026-08-20. No governance proposal followed.

---

## Honest PoC baselines — four topologies (2026-08-19)

Image `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash-0731:3.0.16-overlay-k5` (vLLM 0.25.1, mlnode 3.0.16) on every host. All throughput figures use the corrected script from [kaitakuai/experiments#7](https://github.com/kaitakuai/experiments/pull/7) and a 120 s window.

| GPU (TP) | vLLM | Nonces/min | Per card | Notes |
|---|---|-----:|-----:|---|
| [4×B200 (4)](https://github.com/kaitakuai/experiments/tree/main/2026-08/hy3-fp8-4xb200) | 0.25.1 | **1888** | 472 | bit-identical on repeat |
| [2×B300 (2)](https://github.com/kaitakuai/experiments/tree/main/2026-08/hy3-fp8-2xb300) | 0.25.1 | 1599 | **800** ★ | bit-identical on repeat |
| [8×H100 (8)](https://github.com/kaitakuai/experiments/tree/main/2026-08/hy3-fp8-8xh100) | 0.25.1 | 1344 | 168 | not bit-identical on repeat |
| [4×H200 (4)](https://github.com/kaitakuai/experiments/tree/main/2026-08/hy3-fp8-4xh200) | 0.25.1 | 1248 | 312 | re-measured; first run invalid (see below) |

The first 4×H200 run (1408 nonces/min) was taken with the pre-fix accounting script and a 30 s window; it overstated throughput by 12.8 % and is kept in the folder marked invalid. The redo with the fixed script and a 120 s window gives 1248. Serving figures were not affected by the bug.

**Conclusion.** B300 gives the best per-card PoC rate (800 nonces/min per card); Hopper is 2.5–4.8× lower per card. On Blackwell the fingerprint is bit-identical on repeat, on Hopper it is not — the property is architectural and does not transfer between machines.

### Cross-hardware honest floor

| Comparison | n | Median L2 |
|---|---:|---:|
| honest FP8 ↔ honest FP8, different cards and repeats (5 pairs) | 5000 | 0.201 |

Every non-identical honest pair lands in 0.1977–0.2043 across chip, TP width, host, datacentre and driver version. **Conclusion.** The honest floor is a fleet-wide constant of the same order as measured for DeepSeek-V4 and GLM-5.3-Flash; a single network-wide gate is defensible.

---

## Quantised checkpoints (2026-08-19)

Three publicly available quantised builds were measured at their minimal topologies and cross-validated against the honest arms: [`RedHatAI/Hy3-NVFP4-FP8`](https://github.com/kaitakuai/experiments/tree/main/2026-08/hy3-nvfp4-redhatai-4xb200) (llm-compressor), [`cyankiwi/Hy3-AWQ-INT4`](https://github.com/kaitakuai/experiments/tree/main/2026-08/hy3-int4-cyankiwi-4xh200) (Marlin; also [8×H100](https://github.com/kaitakuai/experiments/tree/main/2026-08/hy3-int4-cyankiwi-8xh100), [4×H100](https://github.com/kaitakuai/experiments/tree/main/2026-08/hy3-int4-cyankiwi-4xh100), [2×H200](https://github.com/kaitakuai/experiments/tree/main/2026-08/hy3-int4-cyankiwi-2xh200)), [`r0b0tlab/Hy3-295B-NVFP4`](https://github.com/kaitakuai/experiments/tree/main/2026-08/hy3-nvfp4-r0b0tlab-2xb300) (ModelOpt; also [1×B300](https://github.com/kaitakuai/experiments/tree/main/2026-08/hy3-nvfp4-r0b0tlab-1xb300), [4×B200](https://github.com/kaitakuai/experiments/tree/main/2026-08/hy3-nvfp4-r0b0tlab-4xb200)).

| Arm | Hosts | Result |
|---|---|---|
| honest FP8 | 2×B300, 4×B200, 4×H200, 8×H100 | one floor, 0.20 |
| three quantised builds | 3 machines, 4 topologies, 2 drivers | each separates from the honest floor on every measured pair |

**Conclusion.** All three quantised builds are separable from honest on the measured pairs, and the fingerprint of a checkpoint reproduces across machines to within 0.26 %. The per-arm distances, the fraud-side throughput economics and the serving-quality comparison are in the [campaign summary](https://github.com/kaitakuai/experiments/tree/main/2026-08/hy3-summary).

---

## MTP speculative decoding (2026-08-19)

Measured on 4×H200 with `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`; no draft checkpoint is needed, the MTP layer reuses the target weights.

| Configuration | PoC | Serving (sequential) | Serving (under load) | KV cache |
|---|---|---:|---:|---:|
| FP8 | baseline | — | — | — |
| FP8 + MTP-2 | identical to the nonce; L2 0.1993 vs floor 0.2025 | +20…29 % | +7…9 % | −9.4 % |

**Conclusion.** MTP cannot influence a prefill-only proof and does not; it is a serving gain only. Whether to enable it in the release images is a policy decision.

---

## Inference validation — cross-hardware 2×B300 → 4×H200 (2026-08-20)

1000 multilingual prompts × two logprobs modes, generated on B300 and replayed on 4×H200, honest and both quantised arms; 6000 replays, 0 length mismatches ([hy3-inference-validation](https://github.com/kaitakuai/experiments/tree/main/2026-08/hy3-inference-validation)).

| Configuration | distance2 | Length mismatches |
|---|---:|---:|
| no speculation (floor, processed) | 0.0185 | 0 |
| executor speculates, validator does not | 0.0180 | 0 |
| validator speculates, stock image | 0.2072 | 82/100 |
| validator speculates, with the fix | 0.0229 | **0** |

Cause: Hy3 + MTP runs on the V1 model runner, whose `RejectionSampler` carries no enforced-token hook, and an accepted draft books two emitted tokens while the reply carries one, so the replay index runs ahead of the output — with no error anywhere. **Fix.** Keep replaying requests out of speculation at scheduling time: [kaitakuai/vllm#21](https://github.com/kaitakuai/vllm/pull/21) (Pavlo, 2026-08-20); other requests in the batch keep speculating. The same fix was later carried into the GLM-5.3-Flash residual via [gonka-ai/vllm#106](https://github.com/gonka-ai/vllm/pull/106).

**Conclusion.** Speculation on the executor is invisible to validation; on the validator it must be suppressed for replay requests.

---

## Participants

[kaitakuai](https://github.com/kaitakuai) ([@baychak](https://github.com/baychak), [@clanster](https://github.com/clanster)).
Gonka core team ([@gmorgachev](https://github.com/gmorgachev), [@vbgd0](https://github.com/vbgd0)).

| Participant | GitHub | Role | Contribution |
|---|---|---|---|
| Pavlo | [@clanster](https://github.com/clanster) | kaitakuai | All PoC and inference-validation experiments on five hosts, campaign summary, [kaitakuai/vllm#21](https://github.com/kaitakuai/vllm/pull/21) |
| Mykola | [@baychak](https://github.com/baychak) | kaitakuai | Coordination, image, carry-over of #21 into [gonka-ai/vllm#106](https://github.com/gonka-ai/vllm/pull/106) |
| Gleb Morgachev | [@gmorgachev](https://github.com/gmorgachev) | Gonka core team | Request (H100 vs B300 exploration) |
| Vladislav Bogdanov | [@vbgd0](https://github.com/vbgd0) | Gonka core team | Request on 3.0.16, throughput accounting fix [kaitakuai/experiments#7](https://github.com/kaitakuai/experiments/pull/7), independent reproduction |

