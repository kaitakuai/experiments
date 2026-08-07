**baychak** (2026-07-24):
Context for the "hardware program item B6" reference, so this is actionable without the internal plan.

**B6 — cross-implementation check of the pseudo-input-ids derivation.** Planned budget: 1xB300 + 4xH100, ~4 GPU-hours. Blocks DeepSeek-V4 activation on the network.

Runs only after the derivation formula here is frozen (this issue) together with its reference vectors. Then two things are measured:
- one more cross-hardware pair on this implementation, and
- an L2 comparison against the independent in-band PoC implementation.

The second one is the actual gate: both implementations must reproduce the same vectors from the same `(block_hash, public_key, nonce)` seed, otherwise prover and validator disagree once V4 is live.

**baychak** (2026-07-24):
Context for the "hardware program item B2" reference, so this is actionable without the internal plan.

**B2 — nonce sub-batching determinism at bs>1.** Planned budget: 1xB300, 2-3 GPU-hours. Gates chain-level validation.

Scope:
- re-run repeat-variance at bs=16 and bs=32 on a single box, many repeats, to size the effect rather than infer it from the cross-pairs in PR #14;
- decide between the two fixes: pin the batch size for chain validation, or make the chunking in `generate_queue` deterministic.

Known state going in: bs=1 is the only strictly bit-identical repeat mode (16/16 and 8/8 in PR #14). Above bs=1, runs should currently be compared via L2, not byte-for-byte.

