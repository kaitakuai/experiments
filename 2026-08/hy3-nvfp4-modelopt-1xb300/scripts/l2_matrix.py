#!/usr/bin/env python3
"""
Reproduce the L2 tables in this experiment's README from the committed artifacts.

Each artifact file is the raw output of `collect_artifacts.py`: a JSON object with an
`artifacts` list of {"nonce": int, "vector_b64": str}. `vector_b64` is 12 little-endian
fp16 values (24 bytes) — the PoC v2 prefill fingerprint for that nonce.

Two runs are comparable only if they used the SAME seed pair (block_hash / public_key);
seeds live in poc_seeds.json and are encoded in the file names (s1/s2/s3).

Usage:
    python3 l2_matrix.py ../artifacts a.json b.json      # one pair
    python3 l2_matrix.py ../artifacts                    # all pairs sharing a seed
"""

import base64
import json
import math
import os
import statistics as st
import struct
import sys

THRESHOLD = 0.40  # reference gate used by earlier campaigns; see README for why it is
                  # NOT a valid pass/fail gate for Hy3


def load(path):
    with open(path) as fh:
        data = json.load(fh)
    return {a["nonce"]: a["vector_b64"] for a in data["artifacts"]}


def vec(b64):
    return struct.unpack("<12e", base64.b64decode(b64))


def compare(path_a, path_b):
    a, b = load(path_a), load(path_b)
    common = sorted(set(a) & set(b))
    if not common:
        raise SystemExit(f"no common nonces between {path_a} and {path_b}")
    identical = sum(1 for n in common if a[n] == b[n])
    diff = [n for n in common if a[n] != b[n]]
    l2 = sorted(math.dist(vec(a[n]), vec(b[n])) for n in diff) if diff else [0.0]
    return {
        "n": len(common),
        "identical_pct": 100.0 * identical / len(common),
        "median": st.median(l2),
        "p95": l2[max(0, int(0.95 * len(l2)) - 1)],
        "max": l2[-1],
        "over_threshold_pct": 100.0 * sum(1 for x in l2 if x > THRESHOLD) / len(common),
        "first_in_batch_share": (
            sum(1 for n in diff if n % 32 == 0) / len(diff) if diff else 0.0
        ),
    }


def seed_of(name):
    """s1 / s2 / s3 from a file name like nonces_b200_fp8_s1_r2.json"""
    stem = name[: -len(".json")]
    for part in stem.split("_"):
        if part in ("s1", "s2", "s3"):
            return part
    return None


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    directory = sys.argv[1]
    files = sys.argv[2:]

    if len(files) == 2:
        pairs = [(files[0], files[1])]
    else:
        names = sorted(f for f in os.listdir(directory) if f.endswith(".json"))
        pairs = [
            (x, y)
            for i, x in enumerate(names)
            for y in names[i + 1:]
            if seed_of(x) and seed_of(x) == seed_of(y)
        ]

    print(f"| pair | identical | L2 median | p95 | max | >{THRESHOLD} |")
    print("|---|---:|---:|---:|---:|---:|")
    for x, y in pairs:
        r = compare(os.path.join(directory, x), os.path.join(directory, y))
        label = f"{x.replace('nonces_', '').replace('.json', '')} ↔ " \
                f"{y.replace('nonces_', '').replace('.json', '')}"
        print(
            f"| {label} | {r['identical_pct']:.1f} % | {r['median']:.4f} | "
            f"{r['p95']:.4f} | {r['max']:.4f} | {r['over_threshold_pct']:.1f} % |"
        )


if __name__ == "__main__":
    main()
