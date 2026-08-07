# Archived: kaitakuai/gonka-poc issue history

The plugin repository moved to `gonka-ai/gonka-vllm-plugins` on 2026-07-28
and the old `kaitakuai/gonka-poc` was deleted on 2026-08-07. Issues do not
travel with a GitHub transfer, so the 27 threads are captured here.

Why it is worth keeping: plugin commits reference these numbers — the
abort-by-internal-id fix cites PR #27, and the design discussions behind
`_compat/`, KV-borrowing (#18) and the V4 PoC path live in these threads.

Two issues were still OPEN at deletion and were never re-filed upstream:

- **#16** — Deterministic nonce sub-batching for `batch_size > 1`
  (timing-dependent ordering)
- **#15** — Mirror the pseudo-input-ids derivation in the in-band PoC line

Their full text and comments are in `ISSUES.md` / `open-issues-comments.md`.
If either becomes relevant again, re-file upstream rather than reviving the
repository.
