# NVFP4 on 1×B300: DSpark was never incompatible — a loader bug hid it, and the fix removes the fraud's only visible tell

**Date:** 2026-08-10
**Candidate:** `MJPansa/DeepSeek-V4-Flash-0731-NVFP4` @ `64d64cd89bc63a66aa46506da89d7821f7491c62` (164 GB on disk, 48 shards)
**Honest counterpart:** `deepseek-ai/DeepSeek-V4-Flash-0731` @ `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` (156 GB)
**Hardware:** 1× NVIDIA B300 SXM6 AC 275 GB (1100 W), TP=1, driver 580.126.09, CUDA 13, 60 CPU cores
**Image:** `ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash-0731:3.0.14-post2-vllm0.25.1-rc3-overlay-k4`
**Digest:** `sha256:de43e29193bc4cda1b3aa47f3ce24601851b95026b5d84031858a3b091e052b9` (amd64 manifest `sha256:c0a1804737b447f186b79d7d8edd29022ecff364a4e2997c0c43b2e8d8b07d43`)
**vLLM:** 0.25.1

> V4 thresholds are **not calibrated**. Distances quoted from sibling reports are distances only.

## This corrects two earlier reports

`../deepseek-v4-flash-0731-nvfp4-1xb300` (2026-08-01) and
`../deepseek-v4-flash-0731-nvfp4-2xb200` (2026-08-02) both concluded that NVFP4 **cannot**
speculate, and built a detection argument on it: an honest node speculates, a quantised one
cannot, so decode timing betrays the fraud even when its PoC vectors look clean.

That conclusion was wrong. The acceptance collapse those reports measured is real and
reproduces exactly, but its cause is a vLLM loader bug, not a property of the quantisation.
With a one-branch fix, NVFP4 speculates as well as the honest checkpoint.

**The consequence runs the wrong way for the network.** The behavioural tell we thought we had
does not exist once the bug is fixed, and the trade-off we described — "+63 % PoC bought at the
price of no speculative decoding" — is not a trade-off at all. NVFP4 wins on PoC *and* keeps
inference.

## Summary

- **Reproduced the collapse on the stock image:** acceptance 1.15–1.24 tokens per chunk across
  four scenarios, against 3.6–6.0 for the honest checkpoint. With 7 draft tokens, essentially
  none are ever accepted.
- **Root cause: NVFP4 kernels reading MXFP4 weights in the DSpark draft.** The conversion leaves
  the draft (`mtp.*`) experts in the source format on purpose and says so with an `ignore` list
  the FP8 loader never reads; the global `moe_quant_algo: NVFP4` then reaches the draft layers,
  which vLLM builds at `layers.{num_hidden_layers + i}`.
- **The fix restores acceptance to 3.5–6.05**, i.e. honest-checkpoint level, and raises serving
  throughput 2.7–5.6× with zero failed requests.
- **The official NVIDIA checkpoint has the identical layout**, so `nvidia/DeepSeek-V4-Flash-NVFP4`
  is affected by the same bug. This is not specific to the community port.
- **PoC is unaffected either way: 2816 nonces/min at batch 32, +61 % over honest.** The draft
  takes no part in the PoC forward.
- **Validation replay (`enforced_tokens`) is unaffected**: 64/64 tokens reproduced in all three
  builds, and greedy output is byte-identical across them.
- **The k4 image needs no changes.** It started on NVFP4 with an empty `additional_args`; the
  entire baked profile applied unmodified.

## Result 1 — the collapse, and that it is the loader

Acceptance in tokens per streamed chunk. `s1` 20k prompt sequential, `s2` 2k prompt at
concurrency 30, `s3` 45k prompt long decode, `s4` 45k prompt at concurrency 20.

| build | s1 | s2 | s3 | s4 |
|---|---:|---:|---:|---:|
| stock k4 | 1.205 | 1.149 | 1.203 | 1.244 |
| stock k4, repeat | 1.205 | 1.145 | 1.192 | — |
| **+ fix (layer-index form)** | 5.000 | 3.636 | 5.171 | 5.534 |
| **+ fix (`quantized_layers` form)** | 3.750 | 3.485 | **6.053** | 5.341 |
| honest 0731, same image | 4.225 | 3.588 | 6.038 | 5.557 |
| honest 0731, July reference | 5.37 | 3.64 | 6.04 | 5.69 |

The attribution is a **round trip on one machine**: patch applied → acceptance rises, file
restored to stock → acceptance falls back to 1.2, patch reapplied → rises again. Same box, same
weights, same engine, same seeds; only `quant_config.py` differed. A single before/after pair
would not have been enough, because the earlier claim had stood on exactly one.

s1 varies between the two fixed builds (5.00 vs 3.75) — it is a five-request scenario dominated
by TTFT and it is the noisiest cell in this instrument. s3, the long-decode scenario, is the
stable one, and there the fix lands on 6.053 against the honest 6.038.

Serving throughput, same runs, zero failed requests anywhere:

| build | s1 | s2 | s3 | s4 |
|---|---:|---:|---:|---:|
| stock k4 | 76.0 | 697.6 | 105.0 | 935.0 |
| + fix (layer-index) | 423.7 | 3149.5 | 454.4 | 2521.0 |
| + fix (`quantized_layers`) | 330.9 | 2685.9 | 525.6 | 2337.7 |

**These are not comparable to the honest figures in the sibling reports**, which were measured
on a different box (30 cores against 60 here). Acceptance is the model-level quantity and is
comparable; tokens per second is not. The honest-vs-NVFP4 serving comparison on identical
hardware is still missing — see the open questions below.

## Result 2 — root cause

Both NVFP4 conversions of V4 convert only the target experts and preserve the draft:

```json
"moe_quant_algo": "NVFP4",
"ignore": ["*.attn.*", "*.ffn.shared_experts.*", "head", "mtp.*"],
"quantized_layers": { "layers.0.ffn.experts": {...}, ...43 entries, none for the draft }
```

The community port's own receipt states it: *"0731 DSpark/MTP tensors are preserved; only
layers.\*.ffn.experts routed projections are converted."* Comparing tensor indices confirms it —
4705 `mtp.*` tensors in both checkpoints, none gaining the `weight_scale` / `weight_scale_2` /
`input_scale` companions that the converted experts gained.

Two things then combine:

1. `Fp8Config.from_config` reads `ignored_layers` and `modules_to_not_convert`. It does **not**
   read `ignore`, so the exclusion list is dropped silently and `ignored_layers` is empty.
2. `moe_quant_algo` is global, so `get_quant_method` hands `ModelOptNvFp4FusedMoE` to every
   `RoutedExperts` — including the three draft layers.

Instrumenting the loader (`scripts/instrument_prefixes.py`) shows what it is actually asked
about: 46 `RoutedExperts`, named `model.layers.{i}.ffn.experts` — 43 target layers plus 43/44/45
for the draft. Note the `model.` prefix, which the config keys do not carry; a naive membership
test against `quantized_layers` matches nothing and would silently disable NVFP4 everywhere.

So the draft's experts are read with NVFP4 semantics over MXFP4 bytes. The draft emits garbage,
every speculative token is rejected, and only the free bonus token survives — acceptance 1.2.
The target model is untouched, which is why output quality, PoC vectors and the nonce distances
in the earlier reports all looked healthy. The defect was visible only through speculation.

The fix consults `quantized_layers`, the checkpoint's own per-layer map, rather than applying
`moe_quant_algo` globally: `scripts/quant_config_draft_moe.patch`, submitted as
[kaitakuai/vllm#20](https://github.com/kaitakuai/vllm/pull/20). A layer-index heuristic also
works today but would break a future conversion that does quantise the draft; the map form
degrades correctly in every case, and keeps the old behaviour when a checkpoint ships no map.

## Result 3 — PoC

Nonces/min, `run_pow_generation.py --phase 3`, two sweeps back to back:

| batch | NVFP4 sweep 1 | NVFP4 sweep 2 | honest, same image | gain |
|---:|---:|---:|---:|---:|
| 8 | 2320 | 2544 | 1648 | +54 % |
| 16 | 2720 | 2720 | 1696 | +60 % |
| 32 | **2816** | 2752 | 1728 | **+61 %** |

2816 at batch 32 reproduces the 2816 measured on 2026-08-01 on a different card of the same
model, and the +61 % matches the +63 % reported then. The draft layers do not participate in
the PoC forward, so these figures hold for the stock and fixed builds alike; they were taken on
the fixed build.

## Result 4 — validation replay is not disturbed

The fix moves a quantisation method, not the sampler, but replay under speculation is the path
[kaitakuai/vllm#18](https://github.com/kaitakuai/vllm/pull/18) had to repair, so it was worth
confirming rather than assuming. `scripts/enforced_test.py` generates a greedy completion with
top-logprobs, then replays that token sequence through `enforced_tokens`:

| build | tokens | reproduced | logprobs |
|---|---:|---:|---|
| stock k4 | 64 | 64 | present |
| + fix (layer-index) | 64 | 64 | present |
| + fix (`quantized_layers`) | 64 | 64 | present |

The generated text was byte-identical in all three, which is the stronger statement: the fix
does not change what the model produces, only what the draft proposes.

## Result 5 — memory

| | KV cache | note |
|---|---:|---|
| NVFP4 | 1,956,349 tokens | |
| honest 0731, same image and card model | 2,211,880 tokens | **−11.6 %** for NVFP4 |

Worth stating plainly because it is counter-intuitive: **NVFP4 is not a memory win here, it is a
memory loss.** 0731's experts are already FP4 (MXFP4); the conversion swaps the scale layout and
adds per-block scale tensors, so the checkpoint grows from 156 GB to 164 GB. On a 275 GB card
this is irrelevant — a single 400k-token request needs about 15 GiB of KV and the headroom is
enormous. On 80 GB cards it is not irrelevant: the working 4×H100 profile (`gmu 0.85`,
`maxnbt 16384`) was the only one of four that both started and survived load, with roughly
4 GiB of headroom above the 400k requirement. NVFP4 would consume about half of that. Untested.

## Environment

Engine args, taken from the log and identical to the image's baked profile because the API
request carried `additional_args: []`:

`--tensor-parallel-size 1 --gpu-memory-utilization 0.90 --max-model-len 400000
--max-num-batched-tokens 32768 --kv-cache-dtype fp8 --logprobs-mode processed_logprobs
--worker-extension-cls gonka_poc.worker.PoCWorkerExtension --tokenizer-mode deepseek_v4
--tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --trust-remote-code
--enable-auto-tool-choice --speculative-config '{"method":"dspark","num_speculative_tokens":7,
"draft_sample_method":"greedy"}'`

Reported by the engine: `quantization=deepseek_v4_fp8`, `expert_dtype resolved to 'fp4'`,
`Using V2 Model Runner`, `DSpark draft model loaded: 96 params` — the same draft parameter count
as on the honest checkpoint, which is the first hint that loading succeeds and inference through
the draft is what fails. Bring-up took 570 s cold and 135 s on a warm kernel cache.

**The k4 image required no fixes.** `gcc -lnvrtc` links, `VLLM_USE_V2_MODEL_RUNNER` is absent
from the environment so vLLM selects the V2 runner itself, and the venv carries the API
dependencies. The two defects that needed manual repair on k9/k10 are closed.

Two delivery notes that cost hours and are worth recording:

- **ghcr throttles layer downloads per IP.** After a few tens of GB, `docker pull` sits on
  `Pulling fs layer` with no `Downloading` lines and no disk growth, while HuggingFace keeps
  serving the same host at 27–35 MB/s and ghcr *manifests* answer in 0.2 s. It is not a hung
  daemon. `scripts/fetch_image_oci.sh` fetches blobs into an OCI layout and hands it to
  `docker load`; it mints a fresh token per blob (they expire in ~5 min), resumes with explicit
  `Range` offsets (`curl -C -` does not work — the blob URL redirects to the CDN and the Range
  header is lost), runs one stream at a time, and backs off when a round gains almost nothing.
- **`conversion-receipt.json` in the NVFP4 repo does not describe the published files.** Its
  byte counts and sha256 values are from before upload and differ from every shard by 128 B to
  337 KB. Verify against the HF API instead; `scripts/fetch_weights.sh` does.

## What is missing from this folder, and why

The box was lost to a host-key rotation between the last measurement and artifact retrieval, as
happened on 2026-08-07 with the turnkey B300 host. Consequently:

- **No raw `serving.json`, no engine logs, no sweep logs.** Every number in this report is
  transcribed from the run output rather than regenerated from a committed log.
  `artifacts/summary.json` carries a `_provenance` field saying so. The sibling honest B300
  report has the same limitation for its DSpark arm.
- **No nonce sets.** The fraud-detection question — whether NVFP4 remains statistically
  invisible — is not re-opened here; it was measured in the 2026-08-01 report with three seeds
  and is unaffected by this fix, since the draft plays no part in nonce generation.
- **TTFT and TPOT** were not recorded per arm.

What this does *not* weaken: the round-trip attribution (three engine restarts with the file
swapped), the PoC figures (two sweeps, matching an independent earlier measurement), and the
replay checks — all of which were read directly from the run output as it happened.

## Files

| path | what |
|---|---|
| `artifacts/summary.json` | every table above, with an explicit provenance note |
| `scripts/quant_config_draft_moe.patch` | the fix, as submitted to kaitakuai/vllm#20 |
| `scripts/fetch_image_oci.sh` | pull the image when ghcr throttles `docker pull` |
| `scripts/fetch_weights.sh` | fetch and verify the NVFP4 checkpoint |
| `scripts/nvfp4_setup.sh` | container + API, with the no-fixes diagnostics |
| `scripts/engine_up.sh` | bring the engine up on the baked profile and wait |
| `scripts/measure_arm.sh` | one arm: engine facts, replay, serving, two PoC sweeps |
| `scripts/enforced_test.py` | validation replay probe (`gen` then `replay`) |
| `scripts/instrument_prefixes.py` | logs the prefixes the quant config is asked about |
| `scripts/serving_bench.py` | serving load generator (counts tokens via `usage`, not SSE chunks) |
| `scripts/run_pow_generation.py`, `scripts/collect_artifacts.py` | PoC tooling, committed as patched |
| `scripts/l2_crossval.py`, `scripts/poc_seeds.json` | analysis and the fixed seed set |

The PoC scripts are committed **as patched**: `run_pow_generation.py` needs
`API_PREFIX = "/api/v1/inference"` (the bare `/api/v1/pow/*` family is the legacy PoW v1
service), `MLNODE_URL` on port 8081, `HOST_IP=127.0.0.1`, and its hardcoded `MODEL_NAME` edited
for the checkpoint under test. `collect_artifacts.py` needs its seed arguments promoted to
module globals.

## Reproduce

```bash
bash scripts/fetch_weights.sh                       # 164 GB, verified against the HF API
IMG=ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash-0731:3.0.14-post2-vllm0.25.1-rc3-overlay-k4
docker pull "$IMG" || IMG="$IMG" bash scripts/fetch_image_oci.sh   # if ghcr throttles
IMAGE="$IMG" bash scripts/nvfp4_setup.sh

docker cp scripts nvfp4a:/root/scripts
docker exec nvfp4a bash -lc 'cd /app && source /app/packages/api/.venv/bin/activate && \
  TAG=stock bash /root/scripts/measure_arm.sh'

# apply the fix and repeat; then restore quant_config.py and repeat again
docker exec nvfp4a bash -lc 'cd /usr/local/lib/python3.12/dist-packages/vllm && \
  cp models/deepseek_v4/quant_config.py /root/quant_config.py.bak && \
  patch -p1 < /root/scripts/quant_config_draft_moe.patch'
docker exec nvfp4a bash -lc 'cd /app && source /app/packages/api/.venv/bin/activate && \
  TAG=fixed bash /root/scripts/measure_arm.sh'
```

## Open questions

- **Honest vs NVFP4 serving on identical hardware.** The honest numbers here come from another
  box; the comparison this report can defend is acceptance, not throughput.
- **Other topologies.** Only 1×B300 TP=1 was measured. H100 at TP=4 is the one with real risk,
  since NVFP4 costs roughly half its remaining headroom.
- **`--max-num-seqs`** is still baked nowhere and still untested as the lever against OOM under
  concurrency on 80 GB cards.
- **`Fp8Config` still ignores `ignore`.** Harmless today only because the other excluded modules
  are not MoE. The next checkpoint with a different layout will fail the same silent way.
- **The fraud question is now harder.** With speculation working, NVFP4 has no known behavioural
  tell, and the earlier reports' nonce distances put it inside the honest noise band.

## Reproducibility checklist

- [x] Image pinned by digest; model pinned by revision
- [x] The three builds differ by exactly one file, and that file is committed as a patch
- [x] Attribution is a round trip (stock → fixed → stock → fixed), not a single before/after
- [x] Acceptance reported directly, so "speculation works" is observed rather than inferred
- [x] Two PoC sweeps, cross-checked against an independent earlier measurement of the same card
- [x] Validation replay checked in every build, not assumed unaffected
- [x] The official NVIDIA checkpoint's config inspected, so the blast radius is stated from evidence
- [ ] **Tables regenerated from committed logs** — not possible; the host was lost, numbers are transcribed and flagged as such
- [x] Unretrieved artifacts named explicitly, with what they do and do not weaken
- [x] Corrections to the two earlier reports stated up front
- [x] No links to `.claude/`, no absolute local paths, no host addresses
