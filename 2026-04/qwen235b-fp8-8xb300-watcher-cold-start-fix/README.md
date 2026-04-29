# Qwen3-235B-A22B FP8 — 8×B300 — k3 → k4 watcher cold-start fix

**Date:** 2026-04-29 → 2026-04-30
**Model:** `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`
**Hardware:** 8× NVIDIA B300 SXM6 AC (Blackwell Ultra, sm_103a)
**Old image:** `ghcr.io/kaitakuai/mlnode-full:0.2.12-vllm0.20.0-b300-k3`
**New image:** `ghcr.io/kaitakuai/mlnode-full:0.2.12-vllm0.20.0-b300-k4`

## Summary

This is an **operability fix**, not a perf change. Per-card peak stays at
1280 nonces/min @ batch=64 on 1×B300 (10240 nonces/min per 8-GPU host).
What changed is that mlnode now survives the long cold start required
for a multi-instance vLLM deployment with `--compilation-config '{"mode":1}'`
on a fresh 8×B300 host, and stops letting the Gonka network node's
default `--tensor-parallel-size 2` win over the throughput-optimal TP=1
layout.

## Symptom: 8×B300 watcher self-kill loop

Observed 2026-04-29 on `weak-seed-flourishes-fin-03` (95.133.252.112) when
the operator brought up a fresh 8×B300 deployment with the b300-k3 image
and 4× TP=2 vLLM instances configured by the Gonka network node:

| Metric | Value |
|---|---:|
| Host load avg (1m / 5m / 15m) | 158 / 1154 / **1520** |
| Container `RestartCount` over ~5 min | **6** |
| All 8×B300 GPUs (idle, between cycles) | 0 MiB used |
| mlnode `/api/v1/state` between restarts | `STOPPED` |

Full evidence in [`artifacts/8xb300-crash-evidence.txt`](artifacts/8xb300-crash-evidence.txt).

## Root cause

Three layers stacked into a kill chain, all internal to mlnode:

```
1. runner.WAIT_FOR_SERVER_TIMEOUT = 1200 (20 min)
       polls /health on each spawned vLLM port
       ┃ raises after 20 min if /health is not 200
       ▼
2. InferenceManager._exception (sticky)
       set on runner.start() raise
       is_healthy() returns False forever after this point
       ┃
       ▼
3. watcher.MAX_UNHEALTHY_COUNT = 3, interval = 2s
       3 consecutive False reports → os._exit(1)
       ┃
       ▼
   docker compose restart: always — cycle repeats from step 1
```

Cold-start time for 4× parallel TP=2 vLLM instances on a fresh 8×B300
host with `--compilation-config '{"mode":1}'`:

```
  4× weight load (220 GiB into VRAM)         ~45 s each, parallel
+ 4× DeepGEMM warmup (2428 kernels)          ~60 s each, parallel
+ 4× FlashInfer TRTLLM JIT (sm_103a)         13 s each
+ 4× per-shape mode=1 torch.compile          dominant cost — minutes
+ 4× CPU contention from parallel compiles   load-avg blow-up
─────────────────────────────────────────────────────────────────
  ≈ 22 min total wall-clock before all 4 ports respond on /health
```

20-min runner ceiling < 22-min observed cold start → runner times out
→ kill chain fires → load avg climbs as concurrent compiles stack up
across docker restarts.

## Fix design (k4)

Two **env-driven** patches at image build time. Defaults preserve
upstream vanilla behavior; b300 image's ENV block flips both on.

### A) `runner.WAIT_FOR_SERVER_TIMEOUT` becomes env-readable

```python
# tools/fragments/hw-patches/runner-py-patches/cold-start-tolerance.py
RUNNER_OLD = "WAIT_FOR_SERVER_TIMEOUT = 1200"
RUNNER_NEW = 'WAIT_FOR_SERVER_TIMEOUT = int(os.environ.get("VLLM_RUNNER_TIMEOUT", "1200"))'
```

```dockerfile
# tools/fragments/hw-patches/_shared/cold-start-tolerance.dockerfile
ENV VLLM_RUNNER_TIMEOUT=3600   # 60 min, covers 22-min cold start with 38-min headroom
```

### B) `watcher.watch_managers` gets a session-aware first-healthy grace

```python
GRACE_FIRST_HEALTHY = os.environ.get("WATCHER_GRACE_FIRST_HEALTHY", "0") == "1"

async def watch_managers(...):
    unhealthy_counts = {m: 0 for m in managers}
    ever_healthy     = {m: False for m in managers}      # SESSION-aware:
    prev_in_session  = {m: False for m in managers}      # only flips True if
                                                         # in active session
    while True:
        await asyncio.sleep(interval)
        for m in managers:
            in_session = m.get_state().name != "STOPPED"

            # Re-arm on STOPPED transition: every up/async cycle gets a
            # fresh grace window.
            if prev_in_session[m] and not in_session and ever_healthy[m]:
                logger.info("returned to STOPPED — resetting cold-start grace")
                ever_healthy[m] = False
            prev_in_session[m] = in_session

            if not m.is_healthy():
                if GRACE_FIRST_HEALTHY and not ever_healthy[m]:
                    logger.info("not yet healthy (cold-start grace; kill inactive)")
                    continue                              # ← do NOT count
                unhealthy_counts[m] += 1
                ...
            else:
                # Trivial STOPPED-shortcut healthy does not consume the grace.
                if in_session and not ever_healthy[m]:
                    ever_healthy[m] = True
                    logger.info("reached healthy in active session — kill threshold active")
                ...
```

```dockerfile
ENV WATCHER_GRACE_FIRST_HEALTHY=1
```

The two changes compose: even if the runner timeout fires (e.g. cold
start somehow exceeds 60 min), the session-aware grace stops watcher
from killing mlnode on the unhealthy report from the resulting sticky
`_exception`.

### C) `tensor-parallel-size=1` moved from default to forced

```python
# tools/fragments/hw-patches/runner-py-patches/b300.py — INJECTION_LINES
_b300_defaults = {
    '--max-num-seqs': '128',                 # only set if absent
}
_b300_forced = {
    '--tensor-parallel-size': '1',           # ← moved here from defaults
    '--gpu-memory-utilization': '0.95',
    '--max-model-len': '120000',
    '--max-num-batched-tokens': '65536',
    '--logprobs-mode': 'processed_logprobs',
    '--compilation-config': '{"mode": 1}',
}
```

The Gonka network node passes `--tensor-parallel-size 2` in its
`up/async` additional_args based on its model-topology config, which
the previous k3 patcher let through. k4 overwrites with TP=1 — the
throughput-optimal layout on 8×B300 (8 × 1280 = 10240 nonces/min
beats ~9200 measured for TP=2 × 4).

## Validation

Run on the 1×B300 test rig (95.133.252.191) with the published GHCR
image `mlnode-full:0.2.12-vllm0.20.0-b300-k4@sha256:8bf4abcae...3dab868`:

| Step | Result |
|---|---|
| Pull image from GHCR (clean) → tar → rsync to instance → docker load | OK |
| `docker run` with operator-style override `--tensor-parallel-size 2` in additional_args | actual cmdline shows `--tensor-parallel-size 1` ✓ |
| `docker exec env` shows `VLLM_RUNNER_TIMEOUT=3600`, `WATCHER_GRACE_FIRST_HEALTHY=1` | OK |
| mlnode startup log (in STOPPED state) — should NOT log "reached healthy state" | clean — 0 false positives |
| up/async → cold start | 600 s to READY, mlnode survived |
| Watcher events during cold start | exactly 1× "reached healthy state in active session", 0× unhealthy strikes, 0× critical/kill |
| Phase-3 sweep batch=64 | **1280 nonces/min** ★ (matches k3 baseline — perf preserved) |

Full bench output: [`artifacts/k4-1xb300-perftest.log`](artifacts/k4-1xb300-perftest.log).

## Migration guide

For operators currently pinned to b300-k3, just bump the image tag in
your `compose.yml`:

```diff
 services:
   mlnode-308:
-    image: ghcr.io/kaitakuai/mlnode-full:0.2.12-vllm0.20.0-b300-k3
+    image: ghcr.io/kaitakuai/mlnode-full:0.2.12-vllm0.20.0-b300-k4
     ...
```

Then `docker compose pull && docker compose up -d`. Remove
`--tensor-parallel-size 2` from `additional_args` if it was hand-set
(k4 forces TP=1 anyway, but cleaning it up avoids confusion).

The k3 GHCR images are NOT deleted — `-kN` is an immutable-tag policy,
so existing pinned deployments keep working. The k3 dashboard entries
are removed from <https://registry.kaitaku.ai/> so new operators do
not pick the deprecated tag.

## Files

- [`artifacts/8xb300-crash-evidence.txt`](artifacts/8xb300-crash-evidence.txt) —
  load avg, RestartCount, mlnode kill-chain log, cmdlines from
  95.133.252.112 during the crash episode
- [`artifacts/k4-1xb300-perftest.log`](artifacts/k4-1xb300-perftest.log) —
  cold start poll + Phase-3 sweep on the published k4 image, validating
  perf parity with k3 and successful watcher behavior

## Related

- Image source: [`mlnode/tools/fragments/hw-patches/runner-py-patches/cold-start-tolerance.py`](https://github.com/kaitakuai/mlnode/blob/main/tools/fragments/hw-patches/runner-py-patches/cold-start-tolerance.py)
- Image source: [`mlnode/tools/fragments/hw-patches/_shared/cold-start-tolerance.dockerfile`](https://github.com/kaitakuai/mlnode/blob/main/tools/fragments/hw-patches/_shared/cold-start-tolerance.dockerfile)
- Image source: [`mlnode/tools/fragments/hw-patches/runner-py-patches/b300.py`](https://github.com/kaitakuai/mlnode/blob/main/tools/fragments/hw-patches/runner-py-patches/b300.py)
- Underlying perf experiment (k3 → k4 has no perf delta): [`qwen235b-fp8-1xb300-OVERVIEW.md`](../qwen235b-fp8-1xb300-OVERVIEW.md)
