# Experiment Plan: Qwen3-235B-A22B-Instruct-2507-FP8 PoC v2 on 2×B200, mlnode alpha5

**Status:** PLAN (not yet executed) — approval pending
**Author:** Mykola (@baychak) / Claude (Cowork session)
**Drafted:** 2026-04-20
**Target execution:** 2026-04-20 / 2026-04-21
**Model (single target):** `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` (shorthand "Qwen3-235B-A22B-FP8"), snapshot `e156cb4efae43fbee1a1ab073f946a1377e6b969` — same model + snapshot used in every prior Qwen run in `experiments/2026-04/` and `experiments/reports/2026-04-kimi-qwen-experiments.md`.

## Assumptions & hard constraints

- **PoC inference MUST run in eager mode (`--enforce-eager`).** This is a hard network-level requirement, not a performance choice. Compiled mode (CUDA graphs / `torch.compile` optimisations) changes the numerical path for FP8 MoE; the hash of computed nonces then diverges from other Gonka workers, breaking cross-worker validity of the submitted proofs. Any benchmark number obtained in compiled mode is **not a valid PoC target** — it is reference-only, useful only to understand headroom.
- All baselines and projections below are interpreted through this constraint: valid comparisons are eager-vs-eager only.
- The target `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` is fixed; snapshot hash `e156cb4efae43fbee1a1ab073f946a1377e6b969` must match across all runs for cross-configuration comparison to be meaningful.

## Baseline / Expected result

All numbers below are `nonces/min` on the **same** model / snapshot / PoC v2 workload (`seq_len=1024`, `k_dim=12`, 30 s measurement per batch). Only hardware, image, and mode vary.

> **PoC must remain in eager mode** (see §Assumptions above). Compiled-mode numbers in the table below are **reference-only** — they cannot be targets for a PoC benchmark because their numerical path diverges from the rest of the Gonka worker network. Valid comparisons are **eager-vs-eager only**.

### Repo history (✅ = PoC-valid eager mode; ⚠️ = compiled, reference only)

| Config | TP | Image / vLLM | Mode | Best batch | Nonces/min | Source |
|---|---:|---|---|---:|---:|---|
| 4×A100 SXM4 80 GB | 4 | mlnode `3.0.13-alpha3` / vLLM 0.15.1 | ⚠️ compiled (reference) | 8 / 16 | 480 | [`experiments/2026-04/qwen235b-fp8-4xa100`](../qwen235b-fp8-4xa100/) |
| 4×H100 SXM 80 GB | 4 | mlnode `3.0.13-alpha3` / vLLM 0.15.1 | ⚠️ compiled (reference) | 16 | 928 | [`experiments/2026-04/qwen235b-fp8-4xh100-alpha3`](../qwen235b-fp8-4xh100-alpha3/) |
| 4×H100 SXM 80 GB | 4 | `vllm/vllm-openai:v0.19.0` + gonka-source | ⚠️ compiled (reference) | 16 | 960 | [`experiments/2026-04/qwen235b-fp8-4xh100-vllm019`](../qwen235b-fp8-4xh100-vllm019/) |
| **2×B200 180 GB** | 2 | mlnode `3.0.13-alpha3` / vLLM 0.15.1 | ✅ **eager**, `FLASHINFER` | 16–64 (plateau) | **1536** ← **number to beat** | [`experiments/2026-04/qwen235b-fp8-2xb200`](../qwen235b-fp8-2xb200/) |
| 2×B200 180 GB | 2 | mlnode `3.0.13-alpha3` / vLLM 0.15.1 | ⚠️ compiled + `FLASHINFER_TRTLLM` | 64 | 1920 (**not a valid PoC target** — compiled path diverges from other workers) | `gonka-deploy/logs/b200-qwen235b-benchmark.md` |
| 2×B300 SXM6 AC 275 GB | 2 | `kaitakuai/mlnode:v0.15.1-b300` | ✅ enforce-eager | 64 | 1664 | `.work/b300-qwen235b-benchmark.md` Exp A |
| 2×B300 SXM6 AC 275 GB | 2 × TP=1 | `kaitakuai/mlnode:v0.15.1-b300` | ✅ enforce-eager | 16 | **2048** ★ | `.work/b300-qwen235b-benchmark.md` Exp C |
| 8×H100 SXM (2 × TP=4) | 4+4 | mlnode `3.0.13-alpha5` / vLLM 0.19 + Gleb env | mode not confirmed — see §Open questions | 16 (`POC_BATCH_SIZE_DEFAULT=16`) | ~2600 (1295 per 4×H100) | Gleb, 2026-04-20 — `/sessions/…/.auto-memory/project_poc_gleb_h100_config.md` |

### Per-GPU view (anchors for projection, eager only)

| Config | Mode | Per-GPU nonces/min |
|---|---|---:|
| 2×B200 alpha3 | ✅ eager | **768** ← anchor for 2×B200 alpha5 eager |
| 2×B300 TP=2 | ✅ eager | 832 |
| 2×B300 2×TP=1 | ✅ eager | 1024 |
| 4×H100 alpha3 | ⚠️ compiled (reference) | 232 |
| 4×H100 alpha5 + Gleb env | mode unconfirmed | 324 |

### Expected result on 2×B200 alpha5 eager (this experiment)

Anchored on **1536 n/min eager baseline** (2×B200 alpha3 eager, `FLASHINFER`):

- **Floor (must-hit):** alpha5 eager ≥ alpha3 eager → **≥ 1536 n/min**. No regression.
- **Stretch band:** +10–15 % from improved alpha5 kernels / FLASHINFER_TRTLLM + FLASHINFER_MOE_FP8 in eager → **~1690–1766 n/min**.
- **Optimistic ceiling:** B300 eager reached 832 per-GPU (TP=2) on a slightly larger Blackwell. If 2×B200 alpha5 eager approaches that per-GPU level → ~1664 n/min (matches B300 TP=2). Unlikely but not impossible.

Success zones for **Run A (TP=2 eager)**:

- **< 1536** → regression vs alpha3 eager → config problem, investigate before Run B;
- **1536 – 1690** → parity / small win, acceptable but underwhelming;
- **≥ 1766** → clean uplift from alpha5 image + new FlashInfer flags in eager → goal achieved;
- **≥ 1920** → notable: alpha5 eager would match the old **compiled** 2×B200 number, meaning kernels alone recovered what compilation used to buy — very strong positive result (but still only valid if measured in eager).

For **Run B (2 × TP=1 eager)**, reference delta from B300 (TP=2 → 2×TP=1) is +23 % (1664 → 2048). Projecting over Run A:

| Run A (eager) | Run B expected (eager) |
|---:|---:|
| 1536 | ~1890 |
| 1700 | ~2090 |
| 1900 | ~2340 |

> 1920 (compiled) is **not a valid target** — PoC must remain eager. It stays in the history table for context only.

## 1. Objective

Measure **PoC v2 nonce generation throughput** (`nonces/min`) for `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` on 2×B200 under **mlnode `3.0.13-alpha5`**, applying the environment and vLLM flags Gleb uses on 8×H100 to reach ~1295 nonces/min per 4×H100 group. The goal is to check whether the same settings transfer to Blackwell (sm_120) and to find a new Blackwell baseline for the FP8 Qwen PoC path.

Hypotheses:

- **H1.** Porting Gleb's env + vLLM args (eager, with Blackwell `FLASHINFER_TRTLLM` + `FLASHINFER_MOE_FP8` overrides) to 2×B200/alpha5 under TP=2 **eager** lifts the previously measured 2×B200 eager result (1536 n/min, alpha3, `FLASHINFER`) by at least 10–15 %, putting us in the 1690–1766 n/min range.
- **H2.** Running two independent TP=1 vLLM instances on 2×B200 in eager (the pattern that gave +23 % on 2×B300 eager) can beat TP=2 eager on the same box because there is no tensor-parallel collective overhead.
- **H3.** The batch-size plateau that appeared at 1536 n/min between batch 16 and batch 64 on 2×B200/alpha3 eager was a compute/kernel ceiling, not a script artifact — the new config should either lift it or move it, not hide it. (Note: 2×B200 alpha3 compiled reached 1920 at batch 64, showing headroom existed for kernels — the question is whether alpha5 eager can tap it without leaving eager mode.)

Explicit **non-goals** for this run (tracked for future experiments):

- Inference throughput (compressa-perf / TTFT/TPOT) — separate run, not mixed with PoC sweep.
- NVFP4 or INT4 quantizations — FP8 only here.
- Comparison with 4×B200 or 8×B200 scaling — only the 2×B200 slice.

## 2. Instance specification

### Offer

| Parameter | Value |
|-----------|-------|
| Provider | Vast.ai |
| Offer ID | **35002904** |
| Host ID | 57669 (same host family as prior PoC `34537523`) |
| Location | Alabama, US |
| Cost | **$7.601 / hr** |
| Network | 5098 Mbps down / 4032 Mbps up (verified) |
| Reliability | 0.994 |
| Rented flag | false at draft time — **verify still available at launch** |

### Hardware (per offer metadata)

| Parameter | Value |
|-----------|-------|
| GPU | 2× NVIDIA B200 (sm_120) |
| VRAM | 183,359 MiB (~179 GiB) per GPU — 358 GiB total |
| Expected driver | 590.48.01 (per prior run on same host family) |
| Expected CPU / RAM | 64 vCPU / ~2.3 TB (per prior run on same host family) |
| Disk allocation (request) | **400 GB** (matches `qwen235b-fp8-4xh100-alpha3` create command; leaves room after ~221 GB model) |

### Software

| Component | Source | Tag / version |
|-----------|--------|---------------|
| Container image | `ghcr.io/product-science/mlnode` | **`3.0.13-alpha5`** (public, package version 797429344) |
| vLLM | baked into image | expected 0.19-line per Gleb's use; to be confirmed at runtime |
| MLNode | baked into image | 3.0.13 alpha5 runtime |
| Bench script | `rtx-pro-6000/tests/run_pow_generation.py` (repo) | HEAD |
| Nonce collector | `rtx-pro-6000/tests/collect_artifacts.py` (repo) | HEAD |
| Model | `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` | snapshot `e156cb4efae43fbee1a1ab073f946a1377e6b969` (~221 GB, 24 safetensors) |

### Port / auth plan

- SSH keys attached to the Vast account: `id=678360` (user's `gonka`), `id=772805` (session key `claude-session-20260420T190927Z-vast-only`, generated this session, private at `kaitakuai/.work/keys/vast_session`).
- MLNode API: **port 8080** inside container (same as prior alpha3 run; vanilla `alpha5` image does not run Jupyter on 8080 so no kill needed). Vast direct-SSH mapping, not Vast's `8080` proxy — pass `--direct` to `vastai create`.
- vLLM: port 5001 (MLNode proxy setup).

## 3. Configuration (vLLM + mlnode)

Two runs in one instance, back-to-back on the same cold cache:

### Run A — TP=2 (primary, reproduces Gleb's config adapted for B200)

**Shell env (applied at MLNode startup and vLLM spawn):**

```bash
# From Gleb's 8×H100 config
export VLLM_ATTENTION_BACKEND=FLASHINFER_TRTLLM   # TRTLLM sub-backend on Blackwell; per parallel inventory
                                                  # (`gonka-deploy/logs/b200-qwen235b-benchmark.md`) this +
                                                  # VLLM_USE_FLASHINFER_MOE_FP8 were the boost flags for the
                                                  # 2×B200 baseline. Both are kernel-selection flags and
                                                  # operate inside eager mode (no CUDA graph involvement).
                                                  # Gleb's H100 env used plain `FLASHINFER`; TRTLLM is the
                                                  # Blackwell override — confirm vs Gleb's alpha5 env (Q9).
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export POC_RPC_TIMEOUT_MS=300000
export POC_BATCH_SIZE_DEFAULT=16

# Blackwell-specific FP8 MoE fast path (missing from Gleb's H100 env)
export VLLM_USE_FLASHINFER_MOE_FP8=1              # FlashInfer FP8 MoE kernel on sm_120 — runtime kernel
                                                  # selection, independent of eager/compiled mode.
```

**MLNode patches (same as alpha3):**

```bash
sed -i 's/MAX_UNHEALTHY_COUNT = 3/MAX_UNHEALTHY_COUNT = 9999/' \
  /app/packages/api/src/api/watcher.py
sed -i '/await start_vllm_proxy()/a\    setup_vllm_proxy([5001])' \
  /app/packages/api/src/api/app.py
```

**MLNode `additional_args` for vLLM:**

```json
[
  "--served-model-name", "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
  "--tensor-parallel-size", "2",
  "--enforce-eager",
  "--max-num-batched-tokens", "131072",
  "--gpu-memory-utilization", "0.92",
  "--max-num-seqs", "128",
  "--max-model-len", "240000",
  "--enable-expert-parallel",
  "--disable-custom-all-reduce",
  "--trust-remote-code"
]
```

**Deliberate deviations from Gleb's 4×H100 config:**

| Flag | Gleb (4×H100) | Here (2×B200) | Why |
|------|---|---|---|
| `--tensor-parallel-size` | 4 | 2 | Only 2 GPUs on the box |
| `--enforce-eager` | not set in memory (unconfirmed — see §Open questions) | **explicitly set — REQUIRED FOR PoC** | Compiled path diverges from other workers' numerical path → nonce hashes drift → proofs become cross-worker-invalid. Hard network-level constraint, see §Assumptions. |
| `--max-num-batched-tokens` | 65536 | **131072** | Parallel inventory (`gonka-deploy/logs/b200-qwen235b-benchmark.md`) used ≥ 131072 on 2×B200. H100 80 GB can't afford the larger scheduler buffer; B200 179 GB can. Independent of eager/compiled. |
| `--compilation-config` | n/a | **dropped** | Would only matter in compiled mode. Irrelevant under `--enforce-eager`. (Originally added to dodge Blackwell `rms_norm_kernel` compile error — not triggered in eager.) |
| `--num-gpu-blocks-override` | 15000 | **removed** | 15000 is sized for 80 GB H100; B200 has ~48–50 GiB free for KV at 0.92 util → ~30k+ blocks available. Let vLLM auto-size, then override only if we hit a ceiling. |

### Run B — 2×TP=1 (secondary, tests B300 pattern on B200)

Only if Run A succeeds and budget remains.

Start two vLLM backends, one per GPU, via two MLNode starts or a patched `setup_vllm_proxy([5001, 5002])` plus the PoC runner pointing at both. Reproduces the config that gave +23 % on 2×B300 (C in `.work/b300-qwen235b-benchmark.md`).

Per-GPU launch (example GPU 0):

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --dtype auto --port 5001 --host 0.0.0.0 \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 128 \
  --max-model-len 131072 \
  --max-num-batched-tokens 131072 \
  --enable-expert-parallel \
  --disable-custom-all-reduce \
  --trust-remote-code
```

The same with `CUDA_VISIBLE_DEVICES=1` and `--port 5002`. `--enforce-eager` is mandatory here too (PoC constraint). Env vars identical to Run A (`VLLM_ATTENTION_BACKEND=FLASHINFER_TRTLLM`, `VLLM_USE_FLASHINFER_MOE_FP8=1`, the rest).

## 4. Benchmark methodology

### Workload (PoC v2)

- Script: `rtx-pro-6000/tests/run_pow_generation.py`, invoked as `python3 -u run_pow_generation.py --phase 3 --skip-check`.
- Fixed parameters: `seq_len=1024`, `k_dim=12` (PoC v2 default).
- Patched before launch: `BATCH_SIZES_TO_TEST = [8, 16, 32, 64, 128]` (standard sweep, identical to all other 2026-04 reports for cross-comparability).
- `if not start_vllm_if_needed():` → `if False:` (skip in-script launch; vLLM is already up via MLNode).
- Per batch size: **5 s warmup + 30 s measured window** (script default; consistent with all prior reports).

### Repeats and variance

- **Each batch size: 2 independent 30 s windows**, back-to-back. Median of the two goes into the headline table, both into the raw table. Rationale: prior reports run once; at 30 s window there's easily ±5 % jitter and we have one new config to validate. Adds ~3 min per run, cheap vs instance cost.
- If the first pass shows any batch at exactly 0 nonces (OOM or engine stuck), flag it and skip the repeat for that batch (re-running OOM twice is waste).

### Metrics collected per batch

- Nonces in 30 s window (primary).
- Nonces / min (derived, primary headline).
- GPU mem usage snapshot via `nvidia-smi --query-gpu=memory.used --format=csv` at mid-window (sampled once, from second SSH).
- vLLM engine P50 / P99 TPOT from the log if exposed (secondary; no-op if absent).

### Cross-config comparison

- Run A (eager) is our primary result vs the **eager** alpha3 baseline of 1536 n/min. 1920 (compiled alpha3) is tracked only as reference — not a valid PoC target.
- Run B is contrasted directly against Run A on the same instance, same model snapshot, same env vars — only topology (2×TP=1 vs TP=2) differs; both are eager.

## 5. Data collection & reporting

### Artifacts on the instance

- `/tmp/mlnode.log` — MLNode + vLLM stdout.
- `/tmp/poc_runA.log`, `/tmp/poc_runB.log` — PoC runner stdout.
- `/tmp/nvidia-smi.csv` — periodic `nvidia-smi` snapshots sampled every 10 s during each run (via `ts` one-liner).
- `/tmp/artifacts/nonces_1000.json` — collected via `collect_artifacts.py` after Run A at the best safe batch size, for byte-level PoC v2 validation.

### Pulled back to repo

Into `kaitakuai/experiments/2026-04/qwen235b-fp8-2xb200-alpha5/`:

```
README.md                        # executed report (replaces this PLAN)
artifacts/
  nonces_1000.json               # PoC v2 self-validation payload
  runA_poc.log                   # full poc stdout
  runB_poc.log                   # (if Run B executed)
  mlnode_startup.log             # grepped startup lines (DeepGEMM / FLASHINFER / KV cache / torch.compile)
  nvidia-smi.csv                 # GPU mem / util timeseries
  instance_create.json           # Vast API response
```

### Report format

Match the structure of `qwen235b-fp8-4xh100-vllm019/README.md`: Infrastructure → Hardware → Software → Model download → Patches applied → vLLM startup → Startup profile → Benchmark parameters → Run command → **Results** (per-batch table, 2 runs each, median highlighted) → **Comparison across history** — eager-only headline table (2×B200 alpha3 eager / alpha5 TP=2 / alpha5 2×TP=1 / 2×B300 TP=2 / 2×B300 2×TP=1, incl. Gleb 8×H100 alpha5 eager if confirmed) plus a separate "Reference (compiled, not PoC-valid)" table containing the 1920 n/min 2×B200 alpha3-compiled number for context → Startup / backend notes → Key observations → `vastai destroy` footer.

## 6. Success criteria / stop rules

> **Valid comparisons are eager-vs-eager only.** 1920 n/min (compiled) is **not** a valid target — PoC must remain eager (see §Assumptions).

### Primary success — headline number

- **Run A (eager) best ≥ 1536 n/min — this is the number to beat.** 1536 is the current best known 2×B200 **eager** result (mlnode alpha3, `FLASHINFER`, [`experiments/2026-04/qwen235b-fp8-2xb200`](../qwen235b-fp8-2xb200/)). Anything below means alpha5 + FLASHINFER_TRTLLM + FLASHINFER_MOE_FP8 on B200 regressed vs alpha3 eager.

### Must-hit (any miss → investigate before continuing)

- Run A completes the batch sweep with ≥ 3 of the 5 batch sizes producing > 0 nonces.
- Run A best ≥ **1536 nonces/min** (primary criterion above).
- `collect_artifacts.py` returns a valid `nonces_1000.json` (PoC v2 self-validation `p_value ≈ 1.0`, async↔sync determinism check passes on ≥ 95 % of collected nonces).

### Stretch (nice-to-have, drive whether we push further)

- Run A best ≥ **1690 n/min** (+10 % over 1536) — clean alpha5-kernel uplift in eager.
- Run A best ≥ **1766 n/min** (+15 % over 1536) — high end of expected eager-vs-eager improvement from the new FlashInfer flags; goal achieved.
- Run A best ≥ **1920 n/min** — alpha5 eager matches the old **compiled** 2×B200 number purely from kernel improvements. Very strong positive result. (Still eager, still PoC-valid — just an aspirational bar.)
- Run B best > Run A best — 2×TP=1 pattern from B300 holds on B200.

### Stop early rules

- Spend ≥ $22 with no Run A successful result → destroy instance, write post-mortem, do not proceed to Run B.
- Run A shows < 1200 n/min at all batches → config is broken, investigate (likely FLASHINFER_TRTLLM failing on sm_120 or flag conflict); skip Run B.
- Hard ceiling: $30 total → `vastai destroy` regardless of state, commit whatever was produced.
- **If at any point we discover compile-mode is silently enabled (CUDA graph log line, torch.compile trace):** abort the run, fix `--enforce-eager`, re-measure. Numbers collected without eager are discarded.

## 7. Execution steps

Checklist; each numbered step is meant to be one logical command block, with output inspected before moving on.

1. **Pre-flight (from sandbox):**
   - Verify offer 35002904 is still rentable and still $7.60/hr via `GET /bundles/?q={...id:35002904,rented:false}`.
   - Verify session pubkey `id=772805` still attached.
   - Verify `rtx-pro-6000/tests/run_pow_generation.py` and `rtx-pro-6000/tests/collect_artifacts.py` exist in repo.

2. **Create instance:**
   ```bash
   vastai create instance 35002904 \
     --image ghcr.io/product-science/mlnode:3.0.13-alpha5 \
     --disk 400 --ssh --direct
   ```
   Capture returned instance ID; save Vast's API response to `artifacts/instance_create.json`.

3. **Wait for status `running` + SSH open.** Vast ready-check via `GET /instances/<id>/`.

4. **Probe environment (one combined SSH roundtrip):** nvidia-smi GPU model/VRAM/driver, `nproc`, `free -h`, `df -h /`, `df -h /dev/shm`, `python3 -c 'import vllm, torch; print(vllm.__version__, torch.__version__)'`. Save to `artifacts/env.txt`.

5. **Download model to `/dev/shm/hf`:**
   ```bash
   mkdir -p /dev/shm/hf
   HF_HOME=/dev/shm/hf nohup python3 -c '
   from huggingface_hub import snapshot_download
   snapshot_download("Qwen/Qwen3-235B-A22B-Instruct-2507-FP8", max_workers=16)
   ' > /tmp/download.log 2>&1 &
   ```
   Poll `tail -3 /tmp/download.log` + `du -sh /dev/shm/hf/` until "Done". Expected 4–8 min at ~5 Gbps.

6. **Apply MLNode patches** (watcher MAX_UNHEALTHY, proxy port 5001) — see §3 Run A. Restart MLNode uvicorn.

7. **Export Gleb's env vars** (persist to `/etc/profile.d/poc.sh` so spawned vLLM inherits them).

8. **Run A — start vLLM (TP=2) via MLNode API:** POST config from §3 to `/api/v1/inference/up/async`. Poll `/api/v1/inference/up/status` every 10 s until `ready` or `failed`. Save `mlnode.log` tail (startup profile, DeepGEMM, FLASHINFER lines) to `artifacts/mlnode_startup.log`.

9. **Run A — batch sweep:** upload `run_pow_generation.py`, apply the two sed patches (BATCH_SIZES, skip auto-start), run twice back-to-back: `python3 -u run_pow_generation.py --phase 3 --skip-check`. Parse and tabulate results. In parallel (second SSH), sample `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv` every 10 s into `/tmp/nvidia-smi.csv`.

10. **Run A — collect nonces:** upload `collect_artifacts.py`, run at the best safe batch size (likely 32 or 64; never ≥ OOM threshold) for 1000 nonces. Copy `/tmp/artifacts/nonces_1000.json` back to repo.

11. **Decision gate:** if Run A hit the "must-hit" criteria AND time/budget remain, proceed to Run B. Otherwise skip to teardown.

12. **Run B — bring up 2× TP=1:** shut down Run A vLLM (`/api/v1/inference/down`), patch MLNode proxy to `setup_vllm_proxy([5001, 5002])`, start two vLLM processes with `CUDA_VISIBLE_DEVICES=0/1` and ports 5001/5002. Repeat the batch sweep (both runs) for Run B.

13. **Teardown:** grep interesting lines out of `mlnode.log`; `scp` all artifacts back; `vastai destroy instance <id>`; verify via `GET /instances/<id>/` that it's gone.

14. **Report generation (in repo):** replace this PLAN file with the executed report using the same folder (`qwen235b-fp8-2xb200-alpha5/`) and the house structure from §5. Update `experiments/reports/2026-04-kimi-qwen-experiments.md` with the new row in the Qwen table.

## 8. Risks & mitigations

| # | Risk | Likelihood | Impact | Detection | Mitigation |
|---|------|------------|--------|-----------|------------|
| 1 | **Compiled mode silently enabled** (forgot `--enforce-eager`, MLNode override, or vLLM 0.19 default changed) | Medium | **Critical — invalidates result** | Check `mlnode.log` for `torch.compile`, `CUDA graph capturing`, `piecewise_compile_level`; `nvidia-smi` shows `cudaGraphLaunchKernel` activity | Verify `--enforce-eager` on command line; grep log for `Compilation mode: EAGER` or equivalent before kicking off the sweep; abort + re-measure if missing |
| 2 | `VLLM_ATTENTION_BACKEND=FLASHINFER_TRTLLM` not supported on sm_120 (B200) under vLLM in alpha5 | Low | High | vLLM startup error / fallback to default backend logged | Fall back to plain `FLASHINFER` (Gleb's H100 value); if that also fails, unset entirely; record as negative result in report |
| 3 | `VLLM_USE_FLASHINFER_MOE_FP8=1` rejected by alpha5's vLLM version | Low | Medium | Startup warning "unknown env var" or MoE kernel error | Unset and rerun; TRITON FP8 MoE fallback works on Blackwell (proven by MiniMax-M2.7 run) |
| 4 | `--enable-expert-parallel` + TP=2 is under-tested on FP8 MoE on B200 | Medium | Medium | Crash or silently wrong output | If startup fails, drop the flag and rerun; PoC nonce self-validation catches silent corruption |
| 5 | `--disable-custom-all-reduce` interacts badly with FLASHINFER_TRTLLM allreduce that vLLM 0.19 auto-selects on B200 | Medium | Medium | Startup warning → hang on first collective | If hang >60 s at collective init, remove the flag |
| 6 | OOM at large batches, same as alpha3 run (batch≥128 crashed) | High | Low | PoC reports 0 nonces, vLLM engine stuck | Expected; record, move on, restart vLLM if engine gets stuck; optionally add `--num-gpu-blocks-override` lower |
| 7 | `/dev/shm` smaller than 250 GB on this offer (HF cache won't fit) | Low | High | `df -h /dev/shm` < 250 GB in step 4 | Use HF cache at `/root/.cache/huggingface` on the overlay disk (400 GB requested — enough). Change `HF_HOME` accordingly |
| 8 | Model download stalls on Vast network | Low | Medium | `du -sh /dev/shm/hf/` doesn't grow for 60 s | Resume download (HF snapshot_download is idempotent); second retry with `max_workers=32` |
| 9 | vLLM 0.19 in alpha5 auto-rejects `--max-model-len 240000` because KV cache profiling differs on B200 | Low | Low | Startup error "KV cache needed > available" | B200 has 4× the VRAM → not expected, but fallback: lower `--max-model-len` to 200000 |
| 10 | DeepGEMM cold warmup is >10 min on B200 | Medium | Low (time only) | Startup phase log | Plan for it; the instance stays hot between Run A and Run B so warmup only pays once |
| 11 | Offer is rented out between plan approval and create call | Medium | Low | API returns "not available" | Two fallbacks pre-selected: `33945636` (1×B200 Oregon; drops to TP=1 only — single config) and `33945630` (2×B200 Oregon $7.88). Re-query offers at T-0. |
| 12 | PoC runner's `MODEL_NAME = ...` points at a different hub path than the one we cached | Low | Medium | Runner errors trying to re-download | sed the `MODEL_NAME` line in `run_pow_generation.py` to exactly `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` before executing |

## 9. Budget & timebox

| Metric | Soft cap | Hard cap | Rationale |
|--------|---------:|---------:|-----------|
| Wall clock | 2 h | 3 h | Setup (~20 min) + 1 cold start (~8 min) + Run A sweep ×2 (~10 min) + nonces (~5 min) + possible Run B with warm cache (~20 min) + teardown (~5 min) ≈ 70 min. Soft-cap adds 30 min buffer for debugging. |
| Spend | $15.20 | $22.80 | At $7.601/hr. Hard cap matches 3 h. Credit balance at draft time: $20.96 — **budget exceeds available credit; user must top up ≥ $5 before launch** (see open question **Q1**, tracked in chat with Claude — not duplicated in this document). |

Spend tracker pattern during run: every 30 min print `elapsed=HH:MM, spent=$X.XX`.

## 10. Deliverables

At end of the experiment the following must exist and be committed:

- `experiments/2026-04/qwen235b-fp8-2xb200-alpha5/README.md` — final report replacing this PLAN, with all 10 sections populated from actuals, results tables for Run A (and Run B if executed), and a comparison table vs alpha3 / compiled baseline / Gleb's 4×H100 number.
- `experiments/2026-04/qwen235b-fp8-2xb200-alpha5/artifacts/` — logs, nonces, nvidia-smi timeseries, Vast API create response.
- Updated row (or two rows) in `experiments/reports/2026-04-kimi-qwen-experiments.md` under the Qwen table.
- Memory update: if any non-obvious finding emerges (config flag conflict, sm_120 FlashInfer quirk, revised throughput baseline for 2×B200), append a `project_*.md` entry in `/sessions/.../.auto-memory/` and link from `MEMORY.md`.

Instance receipt: `vastai destroy instance <id>` + API-level confirmation recorded in the report.
