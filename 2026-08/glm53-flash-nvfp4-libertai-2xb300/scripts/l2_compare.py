#!/usr/bin/env python3
"""L2 distances between honest FP8 and NVFP4 nonce fingerprints for GLM-5.3-Flash.

Reproduces the numbers in this folder's README from the committed artifacts:

    python3 scripts/l2_compare.py

Each nonce is a 12-dim fp16 vector (`vector_b64`, 24 bytes) produced by a single
1024-token prefill. Only nonce sets generated from the SAME seed pair
(block_hash / public_key) are comparable — different seeds feed different inputs and
land on the ~1.41 asymptote of the metric.
"""
import base64
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "..", "artifacts")
SEEDS = ("s1", "s2", "s3")


def load(path):
    d = json.load(open(path))
    arts = d["artifacts"] if isinstance(d, dict) else d
    return {a["nonce"]: np.frombuffer(base64.b64decode(a["vector_b64"]),
                                      dtype="<f2").astype(np.float32) for a in arts}


def dists(pa, pb):
    a, b = load(pa), load(pb)
    common = sorted(set(a) & set(b))
    return np.array([np.linalg.norm(a[k] - b[k]) for k in common])


def main():
    print(f"{'comparison':38} {'n':>6} {'median':>8} {'>0.40':>8} {'bit-exact':>10}")
    pooled = []
    for s in SEEDS:
        d = dists(f"{ART}/ref_nonces_fp8_{s}.json", f"{ART}/nonces_nvfp4_{s}.json")
        pooled.append(d)
        print(f"{'honest FP8 vs NVFP4 ' + s:38} {len(d):6d} {np.median(d):8.4f} "
              f"{(d > 0.4).mean() * 100:7.1f}% {(d == 0).mean() * 100:9.1f}%")
    p = np.concatenate(pooled)
    print(f"{'POOLED':38} {len(p):6d} {np.median(p):8.4f} {(p > 0.4).mean() * 100:7.1f}%")

    # Scale reference: different seeds feed different inputs, so this is the ceiling of
    # the metric — the distance you get from a completely unrelated nonce set.
    a = dists(f"{ART}/ref_nonces_fp8_s1.json", f"{ART}/ref_nonces_fp8_s2.json")
    b = dists(f"{ART}/ref_nonces_fp8_s1.json", f"{ART}/ref_nonces_fp8_s3.json")
    print(f"\nscale ceiling (honest, different seeds): s1-s2 {np.median(a):.4f}  "
          f"s1-s3 {np.median(b):.4f}")


if __name__ == "__main__":
    main()
