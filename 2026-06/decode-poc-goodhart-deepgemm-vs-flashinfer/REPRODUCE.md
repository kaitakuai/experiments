# decode-PoC vs the Goodhart effect — DeepGEMM vs FlashInfer MoE backend

Tests whether a chained-decode Proof-of-Compute metric stays aligned with real
serving throughput when the MoE compute backend is swapped. The known failure of
the prefill-only PoC is a Goodhart symptom: switching the FP8 MoE backend to
DeepGEMM **raised** the prefill-PoC throughput while **lowering** real inference
throughput — the metric rewarded a choice that hurts the product. This experiment
asks: does the **decode**-based PoC remove that divergence?

The container image is referred to by the placeholder
`<MLNODE_IMAGE>` (rebuild recipe below).

## Design

One GPU, one model (MiniMax-M2.7, FP8). Compilation is held fixed
(`custom_ops`); the **only** variable is the MoE backend. For each backend we
measure two throughputs in the same units (decode forward-passes per second):

- **P — decode-PoC throughput.** Batched `genref` of `N_NONCES` chained-decode
  trajectories × `STEPS` steps. `P = N_NONCES · STEPS / elapsed`. With `STEPS`
  large the sequential decode steps dominate wall-clock, so P reflects decode work.
- **S — serving throughput.** Concurrent `/v1/completions` against the same server,
  `ignore_eos` + fixed `max_tokens` (pure decode), aggregate output tok/s.

**Verdict:** Goodhart is *overcome* iff the backend with higher P also has higher
S (`argmax_backend P == argmax_backend S`). It *persists* if they prefer different
backends (the prefill-PoC failure mode).

## Backends (vLLM env toggles)

| Backend    | Env                                                       |
| ---------- | -------------------------------------------------------- |
| FlashInfer | `VLLM_USE_FLASHINFER_MOE_FP8=1 VLLM_USE_DEEP_GEMM=0`      |
| DeepGEMM   | `VLLM_USE_FLASHINFER_MOE_FP8=0 VLLM_USE_DEEP_GEMM=1`      |

The server log line `Using <X> Fp8 MoE backend out of potential backends: [...]`
confirms which backend actually engaged (recorded per run as `backend-evidence`).

## Run

The seal runs inside a container built from `<MLNODE_IMAGE>` (base: vLLM 0.20.0 +
the decode-PoC module — **not included; obtain from the upstream Gonka vLLM fork,
issue #1135** — overlaid at `<site-packages>/vllm/poc/`, with our compiled-mode
patch applied).
`code/goodhart.sh` orchestrates from the host against a long-lived container named
`dpoc` that mounts the seal module at `/patch` and a data dir at `/hf`:

```
# one-time: docker run -d --name dpoc --gpus all --network host \
#   -v <data>:/hf -v <seal-module>:/patch --entrypoint bash <MLNODE_IMAGE> -c "sleep infinity"
# place code/codebook.pt at /hf/poc-tests/codebook_b300.pt
bash code/goodhart.sh           # runs flashinfer then deepgemm; writes /hf/goodhart.log
python3 code/goodhart-analyze.py /hf/goodhart.log
```

Parameters (in `goodhart.sh`): `N_NONCES=48`, `STEPS=128`, `seq_len=256`;
serving `in≈140 / out=256 / concurrency=64 / N=256`; `compile=custom_ops`.

Between backends the orchestrator frees the GPU by killing the host-visible CUDA
compute PIDs (`nvidia-smi --query-compute-apps`), because the engine's worker
subprocess holds the memory and does not match a `vllm.entrypoints` name filter.

## Frozen codebook

`code/codebook.pt`
(sha256 `ba6c9a8f125c27638d44efccfe6a2017ef51ad1a6633304b99594697b69f5044`) is the
fixed sphere codebook the seal quantizes against; point
`GONKA_POC_SPHERE_CODEBOOK` at it. Not strictly required for a throughput
measurement, but kept so the run matches the separability experiments bit-for-bit.

## Repeat measurement (error bars)

The reported eager/compiled error bars come from `code/goodhart-rep-eager.sh` and
`code/goodhart-rep-compiled.sh`, which take 3 genref + 3 serving samples per
backend on one server (no restart between samples). Mean ± std are in `FINDINGS.md`.

## Results

- `results/FINDINGS.md` — the verdict (compiled vs eager, error-barred).
- `results/goodhart.log`, `results/goodhart-eager.log` — single-run raw logs.
- `results/goodhart-compiled-rep.log`, `results/goodhart-eager-rep.log` — the
  3-repeat raw logs behind the error bars.
- `results/backend-evidence.txt` — confirmation each backend actually engaged.
- `results/gh_*_ref_*.json` — the genref trajectories produced per backend.

## Attribution

The decode-PoC seal module is part of the Gonka project (issue #1135) and is
**not redistributed in this bundle** — obtain it from the upstream Gonka vLLM
fork. The compiled-mode patch evaluated here is our contribution, applied on top
of the upstream `poc_model_runner.py`. The harness (`driver.py`, `goodhart.sh`,
`serv-bench.py`, `goodhart-analyze.py`, the `goodhart-rep-*.sh` scripts), the
frozen `codebook.pt`, and everything under `results/` are original to this
evaluation.
