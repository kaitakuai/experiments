# decode-PoC — compiled/eager separability matrix (reproducibility bundle)

Measures the **cross-validator separability** of a chained decode Proof-of-Compute
seal under every combination of *compiled* vs *eager* execution for the **prefill**
step and the **decode** steps. Goal: determine whether running the seal in a
compiled runtime (fused kernels / inductor) preserves the ability of an honest
validator on **different hardware** to confirm an honest prover while rejecting a
**quantization-downgrade forgery** — and whether prefill and decode contribute
that separability differently.

This bundle contains the experiment harness, a frozen codebook, and the measured
outputs. The decode-PoC seal module itself is **not** redistributed here — it is
Gonka's code (issue #1135); obtain it from the upstream Gonka vLLM fork.

## What is measured

Two independent axes produce the signal:

1. **Cross-hardware** — the *prover* generates a reference decode trajectory on
   one GPU type (single-GPU, FP8, `TP=1`); the *validator* re-derives each step
   **teacher-forced** on a different GPU type (4-GPU, `TP=4`). Per-step k-id
   mismatch fraction = how often the validator's quantized hidden-state index
   differs from the prover's. Teacher-forcing each step against the reference
   avoids divergence cascades, so the per-step profile is comparable across steps.

2. **Honest vs fraud** — the validator runs the same teacher-forced profile with
   (a) the honest full-precision model and (b) an AWQ-quantized stand-in for a
   compute-downgrade forgery. `separability = mean(fraud profile) − mean(honest profile)`.
   Larger gap = the seal better distinguishes real compute from a cheaper fake.

## Hardware roles (provider-agnostic)

| Role      | GPUs | Parallelism | Model dtype                 |
| --------- | ---- | ----------- | --------------------------- |
| Prover    | 1    | `TP=1`      | FP8 (honest)                |
| Validator | 4    | `TP=4`      | FP8 (honest) + AWQ (fraud)  |

Any two distinct GPU generations work; the point is hardware heterogeneity
between prover and validator. `TP=1` vs `TP=4` also exercises MoE expert-routing
differences, which dominate the absolute mismatch level on MoE models.

## Container

The seal runs inside an OCI image we refer to as `<MLNODE_IMAGE>`. Rebuild recipe:

- Base: **vLLM 0.20.0** (CUDA build matching the target GPUs).
- Overlay the decode-PoC seal module — **not included in this bundle**; obtain it
  from the upstream Gonka vLLM fork (decode-PoC, issue #1135) — into the installed
  vLLM at `<site-packages>/vllm/poc/`, then apply the compiled-mode patch below.
- No other modification is required; the seal is selected by the env vars below.

## The compiled-mode patch (our change under test)

Our patch to the upstream `poc_model_runner.py` enables compiled execution of the seal. Stock
decode-PoC hard-pins eager (`skip_compiled=True`). The patch makes that follow
the server's `enforce_eager` flag, and adds two fixes/controls:

- **dummy `input_ids` under compilation** — the seal feeds `inputs_embeds` with
  `input_ids=None`; a compiled MoE `forward` dereferences `input_ids.size()` and
  crashes on `None`. Under compilation we pass `torch.zeros(...)` of the right
  shape (prefill `[batch·seq]`, decode `[batch]`). Embeddings are *not* dropped.
- **per-component eager flags** — `GONKA_POC_PREFILL_EAGER=1` forces eager for the
  prefill (step 0); `GONKA_POC_DECODE_EAGER=1` forces eager for the decode steps.
  Absent ⇒ that component follows the compiled server. This is what lets one
  server config isolate prefill vs decode.

Server config used: `--compilation-config '{"custom_ops": ["all"]}'` (inductor
fused kernels). Full CUDA-graph capture is left at vLLM default.

## Frozen codebook (mandatory)

The seal quantizes each hidden state to the nearest point of a fixed sphere
codebook (`SPHERE_POINTS=16`, `SPHERE_DIM=256`). Prover and validator **must load
the identical codebook file**, else every comparison is a false mismatch.
`code/codebook.pt` is that frozen tensor. Point `GONKA_POC_SPHERE_CODEBOOK` at it
on **both** sides. Verify the sha256 matches on prover and validator before
trusting any number.

```
sha256(codebook.pt) = ba6c9a8f125c27638d44efccfe6a2017ef51ad1a6633304b99594697b69f5044
```

## Run

Prover (single GPU). Produces `ref_cc.json`, `ref_ee.json`, `ref_ec.json` in
`/hf/poc-tests` — three free-run reference trajectories, one per config:

```
bash code/matrix-gen.sh         # configs: cc (compiled both), ee (eager both), ec (eager prefill+compiled decode)
```

Copy the three `ref_*.json` **and** `codebook.pt` to the validator's
`/hf/poc-tests`. Validator (4 GPUs), honest then fraud:

```
bash code/matrix-prof.sh MiniMaxAI/MiniMax-M2.7      honest   # -> prof_honest_{cc,ee,ec}.json
bash code/matrix-prof.sh QuantTrio/MiniMax-M2.7-AWQ  fraud    # -> prof_fraud_{cc,ee,ec}.json
```

Collect the six `prof_*.json` into one directory and analyze:

```
python3 code/analyze_matrix.py <dir-with-prof_*.json>
```

### Driver phases used

- `genref <out.json>` — free-run reference trajectory (`N_NONCES` × `1+STEPS` k-ids).
- `profile <ref.json>` — teacher-forced per-step mismatch vs the reference,
  written to `profile.json` (`1+STEPS` element array; element 0 = prefill).

Defaults: `N_NONCES=8`, `STEPS=64`, `SEQ_LEN=256`, model `MiniMaxAI/MiniMax-M2.7`.

## The 7 cases (slices of 3 configs)

Each `profile.json` is a per-step array: element 0 = prefill, elements 1.. = decode.
The 7 requested cases are slices of the three configs:

| # | Case                              | Config | Slice                |
| - | --------------------------------- | ------ | -------------------- |
| 1 | compiled prefill + compiled decode | `cc`  | full (all steps)     |
| 2 | compiled prefill only              | `cc`  | step 0               |
| 3 | compiled decode only               | `cc`  | steps 1..            |
| 4 | eager prefill + eager decode       | `ee`  | full (all steps)     |
| 5 | eager prefill only                 | `ee`  | step 0               |
| 6 | eager decode only                  | `ee`  | steps 1..            |
| 7 | eager prefill + compiled decode    | `ec`  | full (all steps)     |

`analyze_matrix.py` prints honest / fraud / gap for every case.

## Synthetic constants

`block_hash = "deadbeef"×8` and `public_key = "cafebabe"×8` in `code/driver.py`
are **synthetic test fixtures**, not real chain values or keys. They only seed the
deterministic per-step salt so the experiment is self-contained.

## Results

- `results/FINDINGS.md` — the 7-case table and interpretation.
- `results/SAME-HW-FINDINGS.md` — same-hardware compile-coupling (prover side).
- `results/matrix-output.txt`, `results/samehw-output.txt` — raw analyzer output.
- `results/profiles/prof_{honest,fraud}_{cc,ee,ec}.json` — the per-step profiles.
- `results/refs/ref_{cc,ee,ec}.json` — the prover reference trajectories.

## Attribution

The decode-PoC seal module is part of the Gonka project (issue #1135) and is
**not redistributed in this bundle** — obtain it from the upstream Gonka vLLM
fork. The compiled-mode patch evaluated here (see "The compiled-mode patch"
above) is our contribution, applied on top of the upstream `poc_model_runner.py`.
The harness (`driver.py`, `matrix-gen.sh`, `matrix-prof.sh`, `analyze_matrix.py`,
`analyze_samehw.py`), the frozen `codebook.pt`, and everything under `results/`
are original to this evaluation.
