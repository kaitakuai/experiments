#!/usr/bin/env python3
"""Cross-implementation L2 + binomial fraud test for two PoC-v2 nonce sets.

Each input is a nonces file as produced by collect_artifacts.py:
    {"block_hash": ..., "seq_len": 1024, "k_dim": 12,
     "artifacts": [{"nonce": int, "vector_b64": "<32-char base64 = 12 fp16 LE>"}, ...]}

Usage:
    python3 l2_crossval.py A.json B.json [--dist-threshold 0.40] [--p-mismatch 0.10] [--p-value 0.05]

Fraud test — identical to the chain rule in gonka-ai/vllm `vllm/poc/data.py::fraud_test`:

    H0  : the node is honest, i.e. the per-nonce mismatch rate equals p_mismatch.
    H1  : the mismatch rate is *greater* than p_mismatch.
    stat: p_value = P(X >= observed_mismatches | Binomial(N, p_mismatch))   (upper tail)
    rule: FRAUD when p_value < p_value_threshold, otherwise PASS.

So the burden of proof is on the accusation: an inconclusive result reads as "not proven
fraudulent", never as fraud. This mirrors production exactly.

NOTE — this file previously used the *opposite* convention (H0 = "the node is fraudulent",
lower tail P(X <= k), PASS only when that was significant). That inverted the meaning of an
inconclusive result: production would have called such a node honest while this script
called it fraud. The two disagreed for any mismatch rate between roughly 8.5 % and 11.6 %
at N=1000, p_mismatch=0.10. No measurement in this repository fell in that band, so no
reported verdict changes — but the convention is now aligned.
"""
import argparse
import base64
import json
import statistics

import numpy as np
from scipy.stats import binomtest

# Parameters used for the V4 measurements in this repository.
# The chain passes these per model; they are not fixed constants.
DEFAULT_DIST_THRESHOLD = 0.40
DEFAULT_P_MISMATCH = 0.10
DEFAULT_FRAUD_THRESHOLD = 0.05


def load(path):
    d = json.load(open(path))
    return {a["nonce"]: a["vector_b64"] for a in d["artifacts"]}


def vec(b64):
    """Identical to vllm/poc/data.py::decode_vector — fp16 LE decoded into fp32."""
    return np.frombuffer(base64.b64decode(b64), dtype="<f2").astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--dist-threshold", type=float, default=DEFAULT_DIST_THRESHOLD)
    ap.add_argument("--p-mismatch", type=float, default=DEFAULT_P_MISMATCH)
    ap.add_argument("--p-value", type=float, default=DEFAULT_FRAUD_THRESHOLD)
    args = ap.parse_args()

    a, b = load(args.a), load(args.b)
    common = sorted(set(a) & set(b))
    if not common:
        raise SystemExit("no common nonces")

    # float(np.linalg.norm(...)) in fp32 — same arithmetic as vllm/poc/data.py::is_mismatch
    dists = sorted(
        float(np.linalg.norm(vec(a[n]) - vec(b[n]))) for n in common
    )
    n = len(dists)
    k = sum(1 for d in dists if d > args.dist_threshold)
    # Chain rule (vllm/poc/data.py::fraud_test): one-sided binomial, upper tail.
    # H0 = honest. Reject H0 -> fraud. Inconclusive -> PASS.
    p_value = float(
        binomtest(k, n, args.p_mismatch, alternative="greater").pvalue
    ) if n else 1.0
    verdict = "FRAUD" if p_value < args.p_value else "PASS"

    print(f"common nonces      : {n}")
    print(f"L2 median/mean/p95 : {dists[n // 2]:.6f} / {statistics.mean(dists):.6f} / {dists[int(n * 0.95)]:.6f}")
    print(f"L2 max             : {dists[-1]:.6f}")
    print(f"mismatches (>{args.dist_threshold}) : {k}/{n} ({100 * k / n:.2f}%)")
    print(f"binomial P(X>=k)   : {p_value:.3e}  (fraud if < {args.p_value})")
    print(f"VERDICT            : {verdict}")


if __name__ == "__main__":
    main()
