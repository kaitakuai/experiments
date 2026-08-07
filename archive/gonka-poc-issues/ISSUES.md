# kaitakuai/gonka-poc — archived issues and PRs

Repository deleted 2026-08-07 after the plugin moved to gonka-ai/gonka-vllm-plugins. Kept for the review history the plugin commits reference (e.g. the abort fix, PR #27).

## #1 [PR/closed] docs: link sampler-residual fork (poc-sampler-residual-v0.23) + 0.23.0+gonka.sampler1 wheel

*baychak, 2026-06-16*

## Summary

Wires the sampler-residual fork branch on \`kaitakuai/vllm\` into gonka-poc's docs so operators know to install the residual wheel before the plugin.

## Background

ADR-0013 explicitly classifies 6 commits as "fork-stays until Layer 3" (sampler-stack invasion + structured-output graceful degradation — see §Options-considered "sampler-stack ... leaves the fork last"). After the 0.23.0 port lands these stay in fork; the plugin installs cleanly on stock \`vllm==0.23.0\` for everything else.

Companion work on \`kaitakuai/vllm\`:
- Branch: \`poc-sampler-residual-v0.23\` (HEAD \`595cd929\`)
- 6 sampler cherry-picks + version bump (\`0.23.0+gonka.sampler1\` via \`setup.py:get_vllm_version\` + setuptools_scm override path) + Dockerfile.quick overlay + build CI workflow
- Image: \`ghcr.io/kaitakuai/vllm-sampler-residual:0.23.0-gonka.sampler1\` (mutable) + \`...-<sha9>\` (immutable)
- CI build: https://github.com/kaitakuai/vllm/actions/runs/27617896411 (in_progress at PR open)

## Changes

- \`MIGRATION_FROM_FORK.md\` §3 expanded — fork-branch metadata, PEP-440 wheel tag, 6 commits table, operator install order
- \`README.md\` Quick start — swap stock \`vllm/vllm-openai:v0.23.0-cu129\` for the residual overlay image (flagged as placeholder until the image actually publishes)
- New "Why the sampler-residual fork?" section pointing at ADR-0013

## Known TODOs (flagged for follow-up, NOT in this PR)

- 6th SHA (structured-output graceful degradation, \`4aa865752\`) — agent noted it as "SHA pending" in the table; replace with \`4aa865752\` literal in a follow-up.
- Verify ADR-0013 wording matches the quoted phrase ("sampler-stack ... leaves the fork last") — agent didn't open the ADR file.
- The overlay image name is placeholder until first successful build lands.

## Test plan

- [x] Repo builds locally: \`python -m py_compile $(find . -name '*.py')\` clean
- [x] Contract tests still pass (no functional change)
- [ ] After merge: verify \`MIGRATION_FROM_FORK.md\` renders with correct commit-SHA links

---

## #2 [PR/closed] fix(plugin): architecture refactor — close 14 must-fix items from arch review

*baychak, 2026-06-16*

## Summary
Closes the critical and major findings from the 2026-06-16 deep arch review:
- Wires PoCWorkerExtension.execute_poc_forward via collective_rpc (no more AsyncLLM monkey-patch).
- Routes poc_model_runner.py private vllm.v1.* touches through _compat/v0_23.py.
- Wires PoCGate.activate/deactivate from routes.py; removes parallel _poc_generation_active flag.
- Deletes engine_patch.py + manager.py; makes poc/__init__.py inert.
- Adds contract test for SamplingParams residual bridge (logprobs_mode + enforced_token_ids).

## Why
Pre-refactor state: `pip install vllm==0.23.0 && pip install gonka-poc && gonka-vllm-serve ...` did not produce PoC artifacts. First PoC call would ImportError on `vllm.poc.*` (fork-only paths), middleware never activated, worker extension was NotImplementedError stub.

Post-refactor: collective_rpc path is real, gating activates, worker extension implements PoC forward through shim.

## Test plan
- [x] py_compile clean across src/ and tests/
- [x] No engine_patch references remain in src/ (only doc comment + legacy test, called out below)
- [x] No vllm.poc.* imports remain in src/
- [ ] tests/contract pass against installed vllm==0.23.0 wheel (CI exercises this)
- [ ] After merge: GPU smoke test on B200 / B300 — gonka-vllm-serve + chain orchestrator produces nonces matching baseline

## Risks (called out in TASK_SCHEMA)
- **kwarg shape for collective_rpc execute_poc_forward** — needs Mykola/Gleb review: I used keyword-only args (block_hash, public_key, nonces, seq_len, hidden_size, k_dim, poc_stronger_rng) and pass them via `kwargs=` in collective_rpc; vllm 0.23 collective_rpc accepts both `args=` and `kwargs=` but the worker-side signature uses `*, ...` so positional pass-through is impossible — this is intentional but should be verified against the real engine_client signature on hardware.
- **decode_vector / aggregation semantics** — I chose to KEEP vectors as base64 strings in the per-rank result and NOT decode_vector at the API layer; rationale: the API response itself ships base64 (look at the existing /generate endpoint return shape with `encoding={dtype:f16}`), so decoding then re-encoding would be wasted work. Verify this matches what callbacks.py and validation.py expect; if validation needs the FP32 tensor in-process they call decode_vector themselves.
- **hidden_size resolution moved from API-layer** (engine_patch.py used to compute it on AsyncLLM) to worker-side (extension.py reads `self.vllm_config.model_config.get_hidden_size()`) — this should be more correct (worker owns the live config) but is a behavior change worth flagging.
- **PP non-last rank handling** — `execute_poc_forward` in poc_model_runner returns None for non-last PP ranks; the new extension wraps that as `{artifacts: [], rank: N}`. The aggregation in `_execute_poc_forward_rpc` dedupes by nonce so a buggy double-populate doesn't corrupt output, but if the underlying forward changes semantics to also return artifacts from non-last ranks this needs revisit.
- **tests/gonka/test_chat_priority_gating.py still imports vllm.poc.engine_patch** — that test was for the legacy patch path and is now broken; out of scope for this refactor but should be rewritten or deleted in a follow-up.
- **build_attn_metadata_per_layer helper** takes `slot_mapping` as a separate kwarg and mirrors it per layer in the returned dict — this matches the existing inline behavior verbatim but the shape contract should be confirmed once the forward runs on real GPU hardware.

---

## #3 [PR/closed] Rewrite gonka-poc README.md + MIGRATION_FROM_FORK.md so the operator install story is honest

*baychak, 2026-06-16*

## Summary
Rewrite `README.md` and `MIGRATION_FROM_FORK.md` in `gonka-poc` so the operator install story is honest and complete:
- README: current install steps (`pip install vllm==0.23.0 && pip install gonka-poc`), what works after install, what does NOT yet work pre-arch-refactor, and where to find the smoke-test plan.
- MIGRATION_FROM_FORK.md: enumerate fork-only paths (`vllm.poc.*`) and where they have been re-homed (or marked as future work) in the plugin.

## Why
The pre-rewrite README read like marketing copy ("drop-in replacement for the fork") that was not true; operators following it would hit ImportError on first PoC call. The arch refactor PR (#TBD) lands the actual mechanics — this PR keeps the prose honest in the meantime.

## Test plan
- [x] README renders correctly
- [x] MIGRATION doc enumerates every fork-only path touched by the plugin
- [ ] Confirm with Mykola/Gleb that the install story matches expectations

---

## #4 [PR/closed] chore(plugin): cleanup + entrypoint polish from arch review

*baychak, 2026-06-16*

## Summary
Dead-code, logging, and entrypoint polish items from the 2026-06-16 arch review on the `gonka-poc` plugin, batched on `mb/chore/cleanup-and-polish`:
- Argument plumbing tightened on the entrypoint script
- Lazy `_compat` imports to avoid pulling private vllm.v1.* at module-load time
- Dead code paths removed (unused helpers, stale TODOs)
- Logging level + format normalized

## Why
Pure code-quality cleanup surfaced during the deep arch review. No behavior change intended.

## Test plan
- [x] py_compile clean
- [ ] tests/contract collect-only passes (CI)
- [ ] After merge: same nonce output as pre-cleanup baseline on B200/B300 smoke

---

## #5 [PR/closed] chore(tests): remove legacy test_chat_priority_gating.py + flag TODO

*baychak, 2026-06-16*

## Summary
Removes the last legacy reference to the deleted \`vllm.poc.engine_patch\` module.

## Why
The arch-refactor (gonka-poc#2) replaced the AsyncLLM.poc_request monkey-patch with the \`collective_rpc\` + \`PoCGate\` design (see ADR-0013, ADR-0014). The legacy unit test was importing the deleted \`vllm.poc.engine_patch\` module and patching \`vllm.poc.routes._poc_generation_active\` (now also removed in favor of the PoCGate single-source-of-truth).

The test cannot be salvaged — its premises were the old architecture. Deleted cleanly + documented the replacement plan in \`tests/gonka/README.md\`.

## After this PR
- \`grep -rn 'vllm.poc' src tests\` → zero matches
- Plugin tree is fully migrated off the legacy fork-only paths
- Replacement test (Starlette test client → PoCGatingMiddleware) is on the GPU smoke-validation PR backlog

## Test plan
- [x] py_compile clean
- [x] Replacement TODO documented in tests/gonka/README.md
- [x] No CI impact (existing contract tests unchanged)

---

## #6 [PR/closed] test(unit): PoCGatingMiddleware Starlette unit tests

*baychak, 2026-06-16*

## Summary
Adds a pure-unit test suite for `gonka_poc.entrypoint.gating.PoCGatingMiddleware` using Starlette's TestClient. Replaces the deleted `tests/gonka/test_chat_priority_gating.py` whose premises (AsyncLLM.poc_request + module-global `_poc_generation_active`) were removed by the arch-refactor (gonka-poc#2).

## What's covered
- Default state is inactive (no block)
- Activate → /v1/chat/completions returns 503 with Retry-After
- Activate → /v1/completions returns 503
- PoC routes (/api/v1/pow/*) are NEVER blocked even when active
- Deactivate restores 200
- Custom blocked_prefixes are honored
- 503 body shape (JSON + Retry-After header)

## Why this matters
The contract tests pin the upstream vllm surface but not our own ASGI behavior. Without this, a regression in `PoCGate.activate()` or the middleware's pattern-match logic would only be caught on hardware — which is exactly what the arch-refactor was designed to avoid.

## CI
Updates `.github/workflows/contract-tests.yml` to run `pytest tests/contract tests/unit` (single-line change). Unit tests do NOT need vllm imported, so the existing matrix and install steps suffice.

## Test plan
- [x] py_compile clean
- [x] Module imports cleanly in headless Python
- [ ] CI runs all unit tests green against vllm 0.23.0

---

## #7 [PR/closed] fix(plugin): close 7 must-fix items from v2 arch re-review (compat dispatch, abort, lifespan, kwargs, gating, qwen3)

*baychak, 2026-06-16*

## Summary
Closes 7 of 8 must-fix items from the 2026-06-16 v2 arch re-review (the 8th — ADR-0014 — was merged separately as mlnode-foundry#61).

## What lands
- **fix(_compat)** — `current()` now returns the resolved module (was binding the lru_cache function → AttributeError on first PoC forward). Compat dispatch was the **CRITICAL** runtime block from the re-review.
- **fix(routes)** — wire `await abort_all_requests(engine_client)` after `gate.activate()` in `/api/v1/pow/init/generate`. The ADR + README + live tests promised abort-in-flight; this delivers.
- **fix(entrypoint)** — `cli_env_setup()` now called before `uvloop.run` (prevents fork-multiproc crash on TP>1/PP>1). Replace `@app.on_event("startup")` warning (silently suppressed under FastAPI lifespan) with a one-shot middleware check. Add 5 missing `serve_http` kwargs (ssl_ciphers, h11_max_*, log_config, access_log, enable_ssl_refresh). Change `--gonka-poc-block-prefixes` to `nargs="+"` so a bare flag fails loudly instead of silently disabling the gate.
- **chore(plugin)** — drop orphan `qwen3_moe_config.py` (empty defaults tuple, no callers). Fix README + `__init__.py` docstring that lied about Qwen3MoE defaults being ported.
- **test** — add `tests/unit/test_compat_dispatch_smoke.py` so future compat-binding regressions break CI in 0.5 seconds instead of on hardware.
- **test** — extend contract suite to pin the new `current() → module` contract.

## Pre-existing — NOT in this PR
- GPU smoke validation on B200/B300 still pending — runtime semantics of `collective_rpc` artifact aggregation + worker-side `hidden_size` resolution must be confirmed on hardware.
- Should-fix list (logger f-strings, duplicate env constants, orphan dataclasses in data.py, contract-test gaps on attn_groups/collective_rpc/set_forward_context) is tracked as follow-up.

## Test plan
- [x] py_compile clean across src/ and tests/
- [x] grep confirms zero `@app.on_event` uses, zero qwen3_moe refs
- [ ] CI green against vllm==0.23.0

---

## #8 [PR/closed] fix(plugin): v3 code defects (gate-leak, kv_caches orphan, Retry-After mismatch)

*baychak, 2026-06-17*

## Summary
Closes 3 v3 findings:
- **MAJOR (NEW-2)** -- gate-leak on abort_all_requests/_current exception. Wraps activate->abort->spawn in try/except deactivate.
- **MAJOR** -- get_kv_cache_pool orphan: two sites bypass the compat shim with raw getattr. Both now route through compat.get_kv_cache_pool.
- **Should-fix** -- Retry-After header (1s) vs retry_after_ms body (100ms) -- 10x orchestrator confusion. Single source of truth via class constant.

## Test plan
- [x] py_compile clean
- [ ] Exception-path tests added by PR-D will exercise the gate-leak fix end-to-end

---

## #9 [PR/closed] docs(plugin): v3 doc defects (Quick start install path, auto-injection lie, ADR stubs, tests/gonka README)

*baychak, 2026-06-17*

## Summary
Closes 2 v3 must-fixes + 1 should-fix:
- **CRITICAL (#2)** -- README Quick start used `pip install gonka-poc` which 404s (not on PyPI). Switched to `pip install git+https://...`.
- **MAJOR (#4)** -- README falsely claimed gonka-vllm-serve auto-injects --worker-extension-cls. Removed lie + added truthful operator instructions.
- **MAJOR (#5 partial)** -- gonka-poc references ADR-0013/ADR-0014 from 4 sites but docs/adr/ did not exist. Added local stubs pointing at mlnode-foundry canonical sources.
- tests/gonka/README rewritten -- stale fork-style invocations + deleted test references removed.

---

## #10 [PR/closed] ci+test: 3 process-gates to break the whack-a-mole cycle (grep-lint, real-wheel smoke, exception-path)

*baychak, 2026-06-17*

## Summary
Closes the v3 architectural recommendation: install 3 process-gates so v4 cannot find the same defect classes v2 and v3 found.

1. **Grep-lint workflow** -- fails CI on any private `from vllm.v1.*` outside `_compat/`, and any `ADR-NNNN` reference whose corresponding file does not exist under `docs/adr/`. Both classes regressed across v2 and v3.

2. **Real-wheel smoke** -- extends contract-tests.yml with a `smoke-help` job that `pip install vllm==0.23.0`, `pip install -e .`, then `gonka-vllm-serve --help` end-to-end. Catches the "import path wrong, contract test enshrines wrong path" pattern v3 surfaced on cli_env_setup.

3. **Exception-path tests** -- new tests/unit/test_exception_paths.py with three failure-mode tests for init_generate. Enforces the v2 lesson that wired fixes need exception-path tests, not just happy-path tests.

## Why this matters
v1->v2->v3 found 14->8->5 must-fixes, all in the same defect categories (orphans, ADR refs without files, "structurally right wired wrong"). Architecture is stable; failure is operational hygiene. These 3 gates eliminate the recurrence vector.

## Dependencies
- PR-B (gate-leak try/except) must land first OR alongside this PR -- otherwise the exception-path tests fail by design.
- PR-C (ADR stubs) must land first OR alongside this PR -- otherwise grep-lint reports the live ADR-0013/0014 references.

---

## #11 [PR/closed] fix(plugin): close 5 v4 must-fixes (coverage boundaries + callback_task lifecycle)

*baychak, 2026-06-17*

## Summary
Closes 5 v4 must-fixes in one coordinated PR. All in two architectural neighborhoods:

### Neighborhood A — gates self-coverage

- **CRITICAL** — `scipy` missing from test deps → `tests/unit/test_exception_paths.py` silently collection-skipped in CI → **Gate-3 (exception-path) had zero coverage**. Added scipy/aiohttp to runtime deps + test extras. (`fix(deps): ...`)
- **MAJOR** — `scipy + aiohttp` missing from runtime deps. Production install on the README quick-start path was crashing on first /generate (scipy) or first callback (aiohttp). Same commit.
- **MAJOR** — `plugin.register()` had zero test coverage. Smoke-help did not enumerate `vllm.general_plugins` entry-points. Added probes to import `gonka_poc.poc.*` AND enumerate the entry-point + invoke register(). (`ci(smoke-help): ...`)
- **MAJOR** — `EngineClient.collective_rpc / get_supported_tasks / model_config` unpinned by contract suite — highest blast-radius gap (every PoC forward call). Added `test_engine_client_runtime_surface`. (`test(contract): ...`)

### Neighborhood B — resource cleanup adjacent to gate

- **MAJOR** — `callback_task` orphaned on `init_generate` exception path. Spawned BEFORE the try-block; except branch deactivated the gate but left a zombie aiohttp loop. Hoisted spawn INTO the try-block + extended exception handler to cancel the task. Also closed the store-but-not-read leak in `_cancel_poc_tasks`. Added a Gate-3 test variant with `body.url` set. (`fix(routes): ...`)

## Why this matters

v4 review concluded the 3 process-gates DID break the v1/v2/v3 "structurally right wired wrong" whack-a-mole cycle (no v4 finding was a gate-latch regression). The remaining defects all clustered in two neighborhoods: gates-self-coverage and resource-cleanup-adjacent. This PR addresses both classes.

v4 architectural observation: "first iteration of gates surfaces same-class defects; second iteration surfaces boundary-class defects; third iteration should surface zero or near-zero." This PR drives the third iteration.

## Test plan
- [x] py_compile clean across src/ and tests/
- [x] YAML lint clean on contract-tests.yml
- [ ] CI green: contract + smoke-help (now widened) + grep-lint
- [ ] v5 review verifies the 5 must-fixes closed AND finds 0-2 minor (per v4 convergence prediction)

---

## #12 [PR/closed] fix: reset middleware_stack before add_middleware (Starlette 1.3.x / vLLM 0.23)

*baychak, 2026-06-22*

**Blocker found in B300 GPU acceptance.** Starlette 1.3.1 (came with vLLM 0.23) finalizes the middleware stack eagerly, so `app.add_middleware()` after `build_app` raises `RuntimeError: Cannot add middleware after an application has started`. vLLM loads the full model (~214 GB, autotune) and *then* crashes — costly.

Fix: `app.middleware_stack = None` before `add_middleware` in the two spots that add `PoCGatingMiddleware` after `build_app`:
- `api_router.py` `build_gonka_app` (the `gonka-vllm-serve` composed path)
- `plugin.py` `_wrapped_build_app` (the bare `vllm serve` warning-carrier path)

Starlette then rebuilds the stack (incorporating the middleware) on first dispatch — safe: both run before `serve_http`, and resetting a `None` stack is a no-op.

**Validated** against the real Starlette 1.3.1 in the b300 image: reproduced the exact RuntimeError, confirmed the reset fixes it, confirmed `build_gonka_app` installs `PoCGatingMiddleware`. Added a CPU-only regression test (`test_build_gonka_app_survives_finalized_middleware_stack`); full `test_gating.py` green (12 passed).

---

## #13 [PR/closed] fix(worker): unlock MoE workspace around PoC forward (vLLM 0.23 DeepGEMM)

*baychak, 2026-06-24*

vLLM 0.23 added a lockable MoE `WorkspaceManager` (`vllm/v1/worker/workspace.py`): it sizes the MoE scratch from inference shapes during warmup and **locks** it (`gpu_model_runner.lock_workspace()`) before the PoC forward runs. The PoC forward drives the MoE with a much larger fixed shape, so modular-kernel backends (DeepGEMM, triton) raise `AssertionError: Workspace is locked but allocation ... requires N MB` — DeepGEMM PoC crashes (the +49% throughput lever is unreachable on 0.23).

Fix: wrap `_execute_poc_forward` in `unlocked_moe_workspace()` — `unlock_workspace()` → forward → `lock_workspace()` (finally). **Grows-once-then-stays**: the PoC forward grows the workspace to its high-water-mark, then it stays; inference-shaped traffic remains on the locked zero-allocation fast path.

Why NOT the global `if False and self._locked` bypass: that re-arms the realloc+`torch.accelerator.empty_cache()` path for ALL traffic, and PoC batch size oscillates → recurring caching-allocator flushes in the hot loop → the ~10-15% DeepGEMM slowdown reported from B300 acceptance. This wrap confines growth to the PoC forward.

Version-/manager-guarded: no-op on vLLM <0.23 (ImportError) and non-MoE models (no workspace manager). CPU-only regression test (4 cases). Follows vLLM's own elastic-EP unlock→work→lock pattern.

---

## #14 [PR/closed] feat(poc): DeepSeek-V4 support — per-group metadata, positions, pseudo ids

*clanster, 2026-07-18*

## What / Why

PoC v2 cannot run on `DeepseekV4ForCausalLM` (deepseek-ai/DeepSeek-V4-Flash): the forward either hard-fails on missing metadata or — much worse — silently corrupts GPU memory. Three root causes, three fixes:

1. **Per-attention-group metadata layout** (the critical one, not V4-specific). V4 registers KV cache groups with *different block sizes*: sparse MLA / indexer at `cache_config.block_size` (256) and the SWA compressor at **8** (`compressor.py`, compress_ratio=128). The runner built one `slot_mapping`/`block_table` for the main group and handed it to every group's metadata builder. For the compressor pool this addresses ~32× past its allocation:
   - **sm_90 (H100)**: PoC forward crashes — `CUDA illegal memory access` (`deep_gemm_fp8_o_proj`, async) / `CUBLAS_STATUS_EXECUTION_FAILED` (`compressor_kv_score`);
   - **sm_100 (B300)**: no crash, but OOB writes silently corrupt neighbouring engine memory — NaN serving outputs after heavy PoC batches (bs≥64), wedged PoC engine, and cross-config nonce L2 blown up from ~0.2 to ~1.2 (each config reads its own garbage).
   The layout is now built per group from each group's `kv_cache_spec.block_size` (preferring the builder's spec, which reflects `kernel_block_size` splits), and the slot algebra collapses to `seq_idx*padded_len + t`, so the mapping vectorizes to two `arange`s. **Single-group models (Kimi, Qwen, …) are unaffected — the loop reproduces the previous layout exactly.** Any future model with heterogeneous KV groups (sliding-window hybrids etc.) needs this too.

2. **`positions` threaded into `CommonAttentionMetadata`.** V4's C128A sparse-MLA builder asserts `cm.positions is not None`; grep over vLLM 0.23 shows no other backend reads the field. The tensor is shared with the model forward (built once).

3. **Deterministic pseudo `input_ids` for hash-MoE routing.** V4's first `num_hash_layers` layers route experts via `tid2eid[input_ids]` and raise on `None`; PoC has no real tokens. Ids derive from the same `(block_hash, public_key, nonce)` seed scheme as the input embeddings (`_input_ids` suffix) through the existing murmur3 pipeline — pure integer arithmetic, stable across torch versions (a consensus requirement; deliberately *not* `torch.Generator`), vectorized on device, `int32` as the routing kernels expect, gated by `hf_config.model_type == "deepseek_v4"`.

   ⚠️ The derivation scheme is a network-level convention (prover and validator must agree) — please treat the exact formula as a proposal to finalize.

## Validation

- 1×B300 TP1 and 2×B300 TP2: repeat runs at `batch_size=1` are **bit-identical** (16/16, 8/8).
- 4×H100 TP4 / 8×H100 TP8: PoC runs end-to-end (previously 100%-reproducible crash); ~1163 and ~1389 nonces/min respectively; sustained 928 nonces/min on 1×B300 (bs=16, seq_len=1024).
- Canonical cross-validation (1000 nonces, `decode_vector`→L2→binomtest): all cross-topology/cross-hardware pairs land at mean L2 ≈ 0.20 (Kimi cross-hw level); B300-TP1 ↔ H100-TP4 **passes** the chain rule (thr=0.4, p_mis=0.02). Remaining marginal pairs are dominated by timing-dependent nonce sub-batching at bs>1 (batch-variant kernels) — separate issue worth its own fix: deterministic chunking in the generate queue.
- New CPU-only unit tests pin the layout math (vectorized == naive walk, per-group block-table shapes) and the ids scheme (deterministic, ranged, per-nonce distinct): 10 passed.
- Serving unaffected (greedy logprobs bit-identical before/after on the same engine; multilingual smoke OK).

## Notes for reviewers

- Requires `--kv-cache-dtype fp8` for the V4 FlashMLA path (`DeepseekV4 FlashMLA fp8 layout only supports fp8 kv-cache`) — runner profile concern, not part of this diff.
- On sm_90 hosts the image also needs a `libnvrtc.so` symlink for FlashInfer sm_90 JIT (`ld: cannot find -lnvrtc`) — image build concern.
- The contract test `test_common_attention_metadata_fields` may want `positions` added to its pinned field list.
- Earlier drafts carried a 

…(truncated)

---

## #15 [issue/open] Mirror pseudo-input-ids derivation in the in-band PoC line before DeepSeek-V4 activation

*clanster, 2026-07-19*

The pseudo input-ids derivation for token-id-routed architectures is now frozen as a versioned convention: `docs/pseudo-input-ids-convention.md` (v1: sha256 seed with `_input_ids` suffix → murmur3_32 over token positions → mod vocab, int32), with reference vectors enforced by `test_pseudo_ids_reference_vectors`.

Before DeepSeek-V4 is activated on the network, the independent in-band PoC implementation must reproduce the reference vectors byte-for-byte, plus a cross-implementation L2 check on a shared nonce set (hardware program item B6).

Context: PR #14 review, ask 3. Measured: ids influence PoC vectors above run noise (7/8 nonces, ratios 1.06–2.25×), so derivation mismatch between implementations = validation divergence.

---

## #16 [issue/open] Deterministic nonce sub-batching for batch_size > 1 (timing-dependent chunk composition)

*clanster, 2026-07-19*

At `batch_size > 1` the generate queue's nonce sub-batching is timing-dependent: identical requests split into different chunk compositions run-to-run. Kernels are batch-variant, so this is the dominant residual nondeterminism source for PoC vectors:

- bs=1: repeat runs bit-identical (B300 TP1 16/16, TP2 8/8)
- bs=4: only the first chunk matched between two identical runs
- bs=16: 1/16 bit-identical, mean L2 ≈ 0.07 same-box (DeepSeek-V4; the same mechanism is the known ~8.5e-3 same-box floor on Kimi — V4's DSA top-k / MoE routing amplify it 5-40×)

Proposed fix: deterministic chunking in `generate_queue` — strict request-order nonces, exact `batch_size` chunks, no dependence on enqueue timing; plus protocol-pinning the batch size for generation AND validation. Alternative (costlier): pin bs=1 in chain validation (~4× throughput cost measured).

Hardware program item B2 covers the empirical dispersion study (bs=16/32, many repeats). Context: PR #14 validation notes (marginal cross-pairs attributed to this), review ask 3.

---

## #17 [PR/closed] feat(compat): vLLM 0.25.x support — _compat/v0_25 + dual-version dispatch

*baychak, 2026-07-20*

## What / why
Step 5 of the 0.25.1 migration plan: one gonka-poc package serving both the 0.23 fleet and the new 0.25.1 chain via the `_compat` dispatch (built for exactly this).

- **`_compat/v0_25.py`** — mirror of v0_23 with 3 code changes: canonical `CommonAttentionMetadata` import from `vllm.v1.attention.backend` (declaration site MOVED there in 0.25.1; `backends/utils` is now the re-export — paths swapped roles), optional `positions` kwarg (None-safe; read by V4 C128A/compressor), refreshed line pins.
- `_DISPATCH` += `(0, 25)`; `setup_server(reuse_port=False)` with TypeError fallback (the only hard signature break in the package); pyproject `>=0.23.0,!=0.24.*,<0.26`.
- Contract tests: new `test_v0_25_api_surface.py` (canonical-path pin + **new `positions` required-field pin**) + module-level version gates in both files; CI matrix extended to `0.25.1`.

## ⚠ Coordination with PR #14
When #14 lands, its `v0_23.py` changes (per-group `build_attn_metadata_per_group`, positions threading) must be **mirrored into `v0_25.py`** — the `positions` kwarg here is already forward-compatible, the per-group builder is the remaining piece. Happy to do that follow-up post-merge.

## Tests
- Local (0.23 venv): contract+unit **47 passed / 2 skipped** (v0_25 file gate-skips; 0.23 dispatch unchanged). `tests/gonka` failures are pre-existing live-suite issues outside CI scope.
- CI matrix now runs both 0.23.0 and **0.25.1** — the 0.25 leg is the real gate for this PR.
- S1 0.25.1 residual image already built: `vllm-sampler-residual:0.25.1-gonka.sampler1` (contract 14 passed) — next step after merge is the S2 `vllm-poc:0.25.1` build with this package.

AI-assisted; reviewed line-by-line by @baychak.

---

## #18 [PR/closed] feat(poc): validation on leased KV blocks — inference keeps running

*baychak, 2026-07-20*

## What / why

Port of gonka-ai/vllm `qd/combine-poc-and-inference` (Даня's 4 commits: 052648bf4, 590616ab0, 32fda8c4f, cbe5380fc) into the plugin architecture, multi-group-safe. Full design + rationale: **ADR-0015**.

Also fixes a live defect this port surfaced: the plugin's `/generate` validation path neither gated nor aborted inference while overwriting KV blocks 0..N — silent corruption of live inference. Now: borrowed-lease validation where bit-safe, abort-protected legacy path everywhere else.

Key design points:
- ONE shared lease covers all kv-cache groups (block-id namespace is pool-global); EngineCore computes the lease size per group (frontend scalar `block_size` undercounts 8× on DeepSeek-V4); worker expands pool ids per group by the kernel-split ratio (`_borrowed_layout`, fail-loud guards).
- EngineCore methods injected class-level at plugin-register time (general plugins load in the engine-core process) — no vllm fork changes; reachable over the UTILITY RPC.
- `poc_reservation` CM: FIFO lock, reserve→forwards→return (return retried ×3), abort escalation, per-chunk re-abort on the legacy path (donor behaviour), truthful `reset_prefix_cache` handling.
- **Consensus safety**: the legacy (lease-None) bit-path — including the KV-scratch embeds reuse and its `poc_stronger_rng` quirk — is byte-for-byte the deployed behaviour on ALL configs. Borrowing is enabled only where the scratch can never fire (probe `execute_poc_borrow_compat`; fp8/packed-KV models — GLM, DeepSeek-V4). Physical block ids enter only address translation, never attention math.
- `GET /api/v1/pow/versions` reports `poc_validation_inference` from an actual probe; DP>1 refused.

## Review process

Three-lens adversarial review (silent KV corruption / concurrency / consensus determinism) + verify pass: 13 confirmed findings, all closed in the third commit (incl. one critical consensus catch: the scratch path had to be RESTORED bit-exact for the legacy path). Separate KISS/DRY/bloat review: 11 items applied.

## Risk

Mining bit-path untouched on all configs. Borrowed path is new but gated by probe to scratch-free configs; hardware A/B (two leases → bit-compare) queued in the V4 experiment programme before production trust.

## Rollout / Rollback

Merge → S2 `vllm-poc:0.25.1` rebuild → mlnode-base k3 → V4 leaf k3 (full-stack test image for GPU validation). Rollback: revert; engines without the injected methods degrade automatically to the abort-based path.

## Validation

grep-lint ✅ ruff ✅ contract+unit 96 passed / 2 skipped (0.23 local; CI matrix runs 0.23.0 + 0.25.1).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

## #19 [PR/closed] chore: quality pass — logging, dead code, env consolidation, self-doc

*baychak, 2026-07-21*

12 judge-verified items from a 4-lens stack quality review (KISS/DRY/water/self-documentation). Behavior-preserving: consensus math untouched (gpu_random changes are comments + dead-function deletion only; timing constants value-preserving renames). Highlight: /generate INFO logs no longer dump full nonce lists + base64 validation vectors.

Suite: 94 passed / 2 skipped, grep-lint OK. Diff: 19 files, +194/−441.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

## #20 [PR/closed] chore(repo): pre-transfer cleanup — dead code, stale docs, junk comments

*baychak, 2026-07-21*

## What / why

Prepares the repo for the planned transfer to the gonka-ai org: a new maintainer should read present-tense truth, not fork-migration history. Junk removal only — zero behavior changes.

- **Dead code**: `execute_poc_ping` / `execute_poc_describe_kv` (zero callers, verified incl. string-form `collective_rpc` names), 3 unused callback properties, unused pytest `markers` block, `py.typed` package-data no-op.
- **Orphaned test**: `tests/gonka/test_grammar_graceful_degradation.py` — fork-era suite for behavior that lives in the residual fork, currently failing, never run by CI.
- **Junk comments**: fork-provenance/rename-history talk, stale version claims (0.23-only → 0.23.x/0.25.x), drifted repo-internal `:NN` line pins (bare file paths kept). Upstream-vLLM pin-docstrings in `_compat` untouched (policy).
- **Docs**: removed kaitakuai-internal references that 404 for a gonka-ai reader (`feedback_*.md`, ml-runtime, mlnode-foundry links — local ADRs now authoritative with plain-text provenance), fixed the single local-path leak in MIGRATION_FROM_FORK.md, workflows README now matches the real CI matrix and documents smoke-help + grep-lint.

Branch/PR hygiene done separately: 5 merged/superseded remote branches deleted, stale PR #1 closed.

## Risk

Low. Dead-code removal grep-verified; `ruff` clean; `pytest tests/unit tests/contract`: **93 passed, 2 expected skips**; `tools/grep_lint.py` exit 0.

## Rollout

Squash-merge; no image rebuild needed (no runtime behavior change).

## Rollback

Revert the squash commit.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

## #21 [PR/closed] chore: 4-lens polish — doc contract fixes, dead params, consensus-math dedup

*baychak, 2026-07-21*

## What / why

Second polish pass before the gonka-ai transfer: a 4-lens review (docs-junk / KISS / DRY / self-documentation) with adversarial verification — 27 of 30 findings survived skeptic review, all applied here.

**Doc contract fixes (the substantive part):**
- README callback contract was wrong: it described only the mining path (cadence-batched POSTs to `{url}/generated`); the queued `/generate` path posts **once at completion**, to `{url}/generated` or — previously undocumented anywhere — `{url}/validated` with the verdict payload. An integrator building the receiver from README would silently lose fraud verdicts.
- `GET /api/v1/pow/versions` (the feature-detection handshake gating borrow-based validation) added to the endpoint docs.
- `register()`'s third duty (EngineCore KV borrow/return injection, ADR-0015) documented in README + package docstring.
- Fork-retirement wording aligned with ADR-0014's DEFERRED-INDEFINITELY status; status banner refreshed (2026-07-21).

**KISS:** dead `attempt`/`hidden_size` params, dead `_task` attr, dead status writes in `clear_all`, empty `TYPE_CHECKING` block.

**DRY:** consensus math is now shared with its tests instead of mirrored — `gpu_random.derive_pseudo_input_ids` + `_inplace_layout` are **pure moves** with bit-identity pinned by the frozen reference-vector test; `wire_encoding()` and `install_gating_middleware()` single-sourced; `PLUGIN_LOADED` fixtures dedup in test_gating.

**Self-doc:** version-neutral module headers (0.23-only claims were false), corrected abort-mechanism claim in `gating.py`, NonceIterator's frozen network-wide nonce-partition contract stated, CallbackSender/CallbackQueue ownership lines, `poc_deployed` integrator pin documented.

## Risk

Low. Consensus-path changes are move-only; `test_pseudo_ids_reference_vectors` (frozen vectors) passes, pinning bit-identity. `pytest tests/unit tests/contract`: **93 passed, 2 expected skips — stable across 5 consecutive runs**; ruff clean; grep-lint exit 0.

## Rollout

Squash-merge; no image rebuild required.

## Rollback

Revert the squash commit.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

## #22 [PR/closed] docs: make the plugin deployment-agnostic in its own docs

*baychak, 2026-07-21*

## What / why

Design principle: **gonka-poc must be agnostic to mlnode-foundry** (and to any image pipeline). The code already is — zero references in `src/`, and the dependency points the right way (the pipeline pip-installs the plugin; the plugin knows only vLLM). This PR removes the last doc-level coupling:

- README status banner: dropped the image-pipeline mention; added the explicit scope statement (the package depends only on a vLLM install).
- README pointers section: deployment defaults belong to whatever pipeline ships images, never to the plugin — no repo named.
- MIGRATION_FROM_FORK.md: framed as a historical record; foundry references identified as the port-time destination with an explicit no-build/runtime-dependency statement.

ADR provenance one-liners ("originated as mlnode-foundry ADR-XXXX") are kept — provenance is history, not coupling.

## Risk
None — docs only.
## Rollout
Squash-merge.
## Rollback
Revert.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

## #23 [PR/closed] chore(transfer): close hand-over blockers before the gonka-ai transfer

*baychak, 2026-07-24*

Pre-transfer pass (Phase 1 of the hand-over plan). Content-only, no behaviour change.

## Blockers closed

| # | Fix |
|---|-----|
| 1 | `pyproject.toml`: added `[project.urls]` — the wheel carried **no** project URLs at all |
| 3 | `test_v0_25_api_surface.py`: skip message claimed a 0.25 residual wheel "none exists yet" — **it does** (`gonka-ai/vllm` release/v0.25.1 + PR #66). Reworded to state the requirement, not an owner |
| 3b | `test_v0_23_api_surface.py`: same de-owner-ing |
| 4 | ADR-0014: Layer-3 deferral was justified by *"Kaitaku does not have the bandwidth or acceptance channel"* — an internal circumstance presented as normative status. Now "DEFERRED — no owner assigned", and the next owner is invited to re-open it |
| 5 | `MIGRATION_FROM_FORK.md`: `(#9)`, `(#10)`, `PR #30` autolinked to **unrelated PRs in this repo** (#9/#10 exist here and mean something else) → qualified as `kaitakuai/vllm#N` |
| В-1 | ADR-0013/0014: `(kaitakuai internal)` was factually wrong (mlnode-foundry is public) and hid the canonical source → replaced with public links |

## Deliberately NOT in this PR

Install URLs in `README.md` / `tests/gonka/README.md` still point at `kaitakuai/gonka-poc`. Changing them now would document a path that does not exist yet — they move together with `[project.urls]` in a follow-up right after the transfer completes.

## Validation

- `tools/grep_lint.py` passes (it also checks ADR-reference liveness, which this PR touches)
- `pyproject.toml` parses; `[project.urls]` present
- `tests/unit` untouched; local collection errors are missing torch/fastapi in the sandbox, identical on clean HEAD
- CI (`contract-tests` 0.23 + 0.25, `grep-lint`) will gate the rest

---

## #24 [PR/closed] docs: add contributor guide, decision log and toolchain pin

*baychak, 2026-07-24*

Fills the gaps a new owner would otherwise have to reconstruct from commit archaeology.

- **CONTRIBUTING.md** — the two rules that keep this a plugin rather than a fork (private vLLM APIs only in `_compat/`; a failing contract test means the code moved, not the test), what makes the PoC path consensus-critical, and which helpers must not be "deduplicated" without a bit-compat run. This was folklore living in workflow comments.
- **docs/decision-log.md** — the decisions that outlive their PRs, linking to the ADRs for rationale.
- **.mise.toml** — pins the Python version the CI matrix uses.
- **.gitignore** — ignore `.work/` so the local task-context directory cannot be committed by accident.

Part of the pre-transfer pass. `tools/grep_lint.py` passes (it validates the ADR references the new decision log adds).

---

## #25 [PR/closed] docs: remove duplicated statements that had started to disagree

*baychak, 2026-07-24*

An audit before hand-over found the same facts restated in 3-5 places, with copies already diverging. For an incoming reader that is worse than verbosity: they can land on the stale copy.

## Contradictions fixed (the actual reason for this PR)

| Where | Problem |
|---|---|
| README + MIGRATION | still told the incoming owner to *"revisit under gonka-ai ownership"* — i.e. the new owner reads about themselves in the third person. Status now lives in ADR-0014 only |
| README | dated status banner (`Status (2026-07-21)`) already contradicted the tag that exists; PyPI note stated twice |
| MIGRATION | three-column table with two-cell rows — rendered misaligned on GitHub |
| decision log | "pre-transfer documentation pass" entry described the very PR that added it (breaks the file own rule) **and** claimed a cleanup that was unfinished |
| ADR-0014 | pointed at a section of `tests/gonka/README.md` that this PR removes |
| contract-tests.yml | comments cited "v3 review", "v4 must-fix #3 (MAJOR)", "fix-A" — an internal review process the next owner cannot see |

## Deduplication

- `MIGRATION` section listing work for a separate deployment repo this package does not depend on: 36 lines to 6
- ADR-0013/0014 "why this file exists" + `Provenance` (repeated the header verbatim): -40 lines
- contract-test rule now stated once in CONTRIBUTING; `workflows/README.md` links to it
- `tests/gonka/README.md` install instructions link to CONTRIBUTING instead of restating them

Net **-120 lines**.

## Not touched

Everything marked CONSENSUS-CRITICAL / "do not unify" / "bit-compat" stays verbatim, including the note that the deployed scratch path reproduces a known RNG quirk on purpose. Those comments are the only guard against a well-meaning refactor breaking consensus.

Verified: `tools/grep_lint.py` passes (it checks ADR reference liveness), no dangling cross-references remain.

---

## #26 [PR/closed] test(contract): one suite for every vllm minor instead of a copy each

*baychak, 2026-07-24*

`test_v0_25_api_surface.py` was a **copy** of the 0.23 file with identifiers renamed and the prose left untouched. In a file that only runs on 0.25 it still said *"In v0.23 the signature is…"*, *"Confirmed … in v0.23"*, and told a failing assertion to re-check a patch on `poc-sampler-residual-v0.23`. 38 618 vs 38 535 bytes; same 20 tests in both.

## What the split actually bought

Comparing the two files with comments and docstrings stripped, **exactly one assertion differed**:

```python
# 0.23
importlib.import_module("vllm.v1.attention.backends.utils")
# 0.25
importlib.import_module("vllm.v1.attention.backend")
```

`CommonAttentionMetadata` was promoted to the backend package root in 0.25. The other 13 diffs were wording.

So the per-minor split bought one import path and cost a guaranteed divergence on every future minor: whoever adds 0.26 copies 800 lines again and forgets the prose again.

## What replaces it

One `test_api_surface.py` + a `PINS` table mapping each supported minor to the symbols that actually moved:

```python
PINS = {
    "0.23": {"attention_metadata_module": "vllm.v1.attention.backends.utils"},
    "0.25": {"attention_metadata_module": "vllm.v1.attention.backend"},
}
```

Adding a minor is one table entry. An unsupported minor skips the module with a message saying what to do, instead of failing 20 assertions that all mean "nobody vetted this minor".

## Nothing was weakened

All 20 tests intact, no assertion relaxed or dropped. Verified by loading the module against fake vllm 0.23.0 / 0.25.1 / 0.26.0: the first two select the correct import path, the third skips cleanly.

The real check is CI — it runs the matrix against both real wheels.

**Net -778 lines.**

---

## #27 [PR/closed] fix(compat): abort in-flight requests by internal id, not external

*baychak, 2026-07-28*

Found by @Vladikshokoladick during plugin review — thank you, this one was worse than it looked from the outside.

## What was wrong

`abort_all_requests` enumerates ids from `output_processor.request_states`. Those keys are **internal** request ids. It then called `abort(rid)` without `internal=True`, so vLLM resolved each id through the external→internal map, found no entry, and returned having aborted nothing.

The failure was silent in both directions: `abort()` raises nothing on an unknown id, so the `aborted` counter still incremented and the log still reported the requests as aborted. PoC then began its forward pass while inference requests were still decoding on the same GPU — with the log claiming the GPU had been cleared.

## The fix

`internal=True`, plus a signature probe: the `EngineClient` ABC (`vllm/engine/protocol.py:102`) declares `abort()` without the parameter, so a client that is not `AsyncLLM` would otherwise get a `TypeError` on every id. Such a client now degrades with one warning instead.

## Test

`tests/unit/test_abort_uses_internal_ids.py` stands in for the engine and reproduces vLLM's id resolution, including the silent no-op path — CPU-only, no vllm import.

Verified it fails on the old code and passes on the new:

```
=== without the fix ===
FAILED test_in_flight_requests_are_really_aborted
1 failed, 1 passed

=== with the fix ===
2 passed
```

The two collection errors elsewhere in `tests/unit` are a missing `fastapi` in my local environment, not related to this change.

---

