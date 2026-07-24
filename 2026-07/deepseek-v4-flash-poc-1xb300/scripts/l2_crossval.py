#!/usr/bin/env python3
"""Cross-implementation L2 + binomial fraud test for two PoC-v2 nonce sets.

Each input is a nonces file as produced by collect_artifacts.py:
    {"block_hash": ..., "seq_len": 1024, "k_dim": 12,
     "artifacts": [{"nonce": int, "vector_b64": "<32-char base64 = 12 fp16 LE>"}, ...]}

Usage:
    python3 l2_crossval.py A.json B.json [--dist-threshold 0.40] [--p-mismatch 0.10] [--p-value 0.05]

The fraud test treats the null hypothesis as "the node is fraudulent" (mismatches
each nonce with probability >= p_mismatch). If P(X <= observed_mismatches | Binomial(N, p_mismatch))
is below p_value, the fraud hypothesis is rejected and the two sets are consensus-consistent (PASS).
"""
import argparse
import base64
import json
import statistics
import struct
from math import comb, sqrt


def load(path):
    d = json.load(open(path))
    return {a["nonce"]: a["vector_b64"] for a in d["artifacts"]}


def vec(b64):
    return struct.unpack("<12e", base64.b64decode(b64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--dist-threshold", type=float, default=0.40)
    ap.add_argument("--p-mismatch", type=float, default=0.10)
    ap.add_argument("--p-value", type=float, default=0.05)
    args = ap.parse_args()

    a, b = load(args.a), load(args.b)
    common = sorted(set(a) & set(b))
    if not common:
        raise SystemExit("no common nonces")

    dists = sorted(
        sqrt(sum((x - y) ** 2 for x, y in zip(vec(a[n]), vec(b[n])))) for n in common
    )
    n = len(dists)
    k = sum(1 for d in dists if d > args.dist_threshold)
    p_lower = sum(
        comb(n, i) * (args.p_mismatch ** i) * ((1 - args.p_mismatch) ** (n - i))
        for i in range(0, k + 1)
    )
    verdict = "PASS" if p_lower < args.p_value else "FRAUD"

    print(f"common nonces      : {n}")
    print(f"L2 median/mean/p95 : {dists[n // 2]:.6f} / {statistics.mean(dists):.6f} / {dists[int(n * 0.95)]:.6f}")
    print(f"L2 max             : {dists[-1]:.6f}")
    print(f"mismatches (>{args.dist_threshold}) : {k}/{n} ({100 * k / n:.2f}%)")
    print(f"binomial P(X<=k)   : {p_lower:.3e}  (threshold {args.p_value})")
    print(f"VERDICT            : {verdict}")


if __name__ == "__main__":
    main()
