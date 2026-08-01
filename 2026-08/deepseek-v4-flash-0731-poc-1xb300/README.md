# PoC Benchmark: DeepSeek-V4-Flash-0731 on 1×B300 (rerun of the -Flash campaign)

**Date:** 2026-07-31 (scaffold; runs pending)
**Model:** `deepseek-ai/DeepSeek-V4-Flash-0731` @ `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
**Hardware:** 1×NVIDIA B300 SXM6
**Stack:** `mlnode-b300-deepseek-v4-flash:0.2.14-vllm0.25.1-overlay-k10`
`@sha256:a6213dac621c1634a82940533190c9a5149b6535a5690c69ca6d3919c74c8138` — the
release-candidate image: vllm `release/v0.25.1` @04a165c0 + `gonka-vllm-plugins`
v0.1.1 + mlnode from the gonka-ai release branch `vllm-0.25.1-upgrade` @1b07e5c6. No rebuild needed:
the checkpoint keeps the architecture (`DeepseekV4ForCausalLM`, 43 layers, same
`compress_ratios` prefix), and the new `dspark_*` config fields drive the speculative
decoder only, which the PoC path does not use.

Requested by Vlad (gonka-ai) on 2026-07-31: rerun nonce/min and threshold
calibration for the refreshed checkpoint, mirroring
`../../2026-07/deepseek-v4-flash-poc-1xb300`.

## Plan

1. **Throughput sweep** — nonces/min at batch sizes {8, 16, 32, 64}, serving config
   as in the -Flash run (CUDA graphs on). Comparison column against the 1472 n/min
   baseline of the previous checkpoint.
2. **Threshold calibration** — honest-vs-honest L2 distributions (same hardware,
   two runs; then cross-run vs the OLD checkpoint's artifacts to measure how far
   the weight refresh moves vectors): per-nonce L2 against `DEFAULT_DIST_THRESHOLD`,
   binomial fraud test at the production settings (`p_mismatch=0.001`,
   `fraud threshold p<0.01`).
3. **Old-vs-new separability** — the refreshed weights must land FAR outside the
   honest tolerance against the old checkpoint (else a node could serve the old
   model undetected). Report min/median L2 and the fraud-test verdict both ways.
4. Multi-`(block_hash, public_key)` pairs — per Vlad's 2026-07-27 note, at least
   3 pairs to guard against a lucky seed.

## Differences from the -Flash campaign

- Model + pinned revision (above); `MODEL_REVISION` is now passed explicitly.
- Stage: plugin image (k9), NOT the in-tree fork — numbers comparable with the
  H100/H200/B200 plugin tables, not with the old fork-based B300 run.
- Statistical test direction follows the production convention (H0 = honest),
  after the 2026-07-27 alignment fix.

## Status

- [x] Scaffold, scripts pinned to the new revision
- [ ] B300 host allocated
- [ ] Throughput sweep
- [ ] Threshold calibration + separability
- [ ] Report
