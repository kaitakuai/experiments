#!/usr/bin/env python3
"""Compare this run against the 4×H200 set, seed by seed.

Usage:  python3 scripts/cross_arch.py > artifacts/cross_arch.json

Both sides are honest arms, so anything this finds is the *floor* for cross-hardware
comparison — the noise a validator sees between two nodes that are both behaving.

Unlike the earlier B300↔H200 comparison, nothing is confounded here: same image
(`...glm53-test-k3`), same FlashInfer 0.6.18, same TP=4, same seeds. The only difference is the
GPU generation, so whatever this reports is the architectural floor.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize import THRESHOLD, batch_split, l2_map, load, stats  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(HERE, "..", "glm53-flash-fp8-4xh200", "artifacts")


def main():
    out = {"threshold": THRESHOLD, "reference": "2026-09/glm53-flash-fp8-4xh200", "pairs": {}}
    for seed in ("s1", "s2", "s3"):
        ref = os.path.join(REF, "nonces_honest_%s.json" % seed)
        mine = os.path.join(HERE, "artifacts", "nonces_honest_%s.json" % seed)
        if not os.path.exists(ref):
            out["pairs"]["h200_vs_b200_%s" % seed] = {"error": "reference set not found: %s" % ref}
            continue
        _, a = load(ref)
        _, b = load(mine)
        d = l2_map(a, b)
        out["pairs"]["h200_vs_b200_%s" % seed] = {**stats(d), "by_batch_position": batch_split(d)}
    json.dump(out, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
