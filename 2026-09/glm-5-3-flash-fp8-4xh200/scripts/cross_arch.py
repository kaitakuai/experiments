#!/usr/bin/env python3
"""Compare this run against the published 2×B300 set (2026-08), seed by seed.

Usage:  python3 scripts/cross_arch.py > artifacts/cross_arch.json

Both sides are honest arms, so anything this finds is the *floor* for cross-hardware
comparison — the noise a validator sees between two nodes that are both behaving.

Caveat baked into the numbers: the B300 set was collected on the previous image
(FlashInfer 0.6.17) and this one on `...glm53-test-k3` (FlashInfer 0.6.18). Architecture and
build are therefore confounded in this comparison; see README, "What this does not settle".
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize import THRESHOLD, batch_split, l2_map, load, stats  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B300 = os.path.join(HERE, "..", "..", "2026-08", "glm53-flash-fp8-2xb300", "artifacts")


def main():
    out = {"threshold": THRESHOLD, "reference": "2026-08/glm53-flash-fp8-2xb300", "pairs": {}}
    for seed in ("s1", "s2", "s3"):
        ref = os.path.join(B300, "nonces_fp8_%s.json" % seed)
        mine = os.path.join(HERE, "artifacts", "nonces_honest_%s.json" % seed)
        if not os.path.exists(ref):
            out["pairs"]["b300_vs_h200_%s" % seed] = {"error": "reference set not found: %s" % ref}
            continue
        _, a = load(ref)
        _, b = load(mine)
        d = l2_map(a, b)
        out["pairs"]["b300_vs_h200_%s" % seed] = {**stats(d), "by_batch_position": batch_split(d)}
    json.dump(out, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
