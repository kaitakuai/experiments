#!/usr/bin/env python3
"""
Rebuild `l2-distances-data.json` — the dataset behind `l2-distances.html` — from the
nonce artifacts committed in the sibling Hy3 experiment folders.

Every group pools several comparisons so the chart shows a population rather than a
single pair. Two runs are comparable only when they used the same seed (s1/s2/s3),
which is encoded in the file names.

Usage (from this directory):
    python3 scripts/build_l2_data.py > l2-distances-data.json
"""

import base64
import json
import math
import os
import statistics as st
import struct
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

BINS, LO, HI = 60, 0.0, 1.4
THRESHOLDS = [round(0.10 + 0.02 * i, 2) for i in range(36)]


def art(folder, name):
    return os.path.join(ROOT, folder, "artifacts", name)


def load(path):
    with open(path) as fh:
        data = json.load(fh)
    return {a["nonce"]: a["vector_b64"] for a in data["artifacts"]}


def vec(b64):
    return struct.unpack("<12e", base64.b64decode(b64))


def dists(p1, p2):
    """L2 distances over the nonces the two runs share, skipping bit-identical ones."""
    a, b = load(p1), load(p2)
    return [
        math.dist(vec(a[n]), vec(b[n]))
        for n in sorted(set(a) & set(b))
        if a[n] != b[n]
    ]


# Each group is a list of (folder_a, file_a, folder_b, file_b) comparisons.
GROUPS = {
    "Честный пол (разные машины и повторы)": [
        ("hy3-fp8-4xh200", "nonces_fp8_s1.json", "hy3-fp8-4xh200", "nonces_fp8_s1_r2.json"),
        ("hy3-fp8-8xh100", "nonces_fp8_s1.json", "hy3-fp8-8xh100", "nonces_fp8_s1_r2.json"),
        ("hy3-fp8-4xh200", "redo_nonces_fp8_s1.json", "hy3-fp8-4xh200", "redo_nonces_fp8_s1_r2.json"),
        ("hy3-fp8-8xh100", "nonces_fp8_s1.json", "hy3-fp8-2xb300", "nonces_fp8_s1.json"),
        ("hy3-fp8-4xb200", "nonces_fp8_s1.json", "hy3-fp8-2xb300", "nonces_fp8_s1.json"),
    ],
    "INT4 W4A16 · cyankiwi": [
        ("hy3-int4-cyankiwi-4xh200", f"ref_nonces_fp8_{s}.json",
         "hy3-int4-cyankiwi-4xh200", f"nonces_int4_{s}.json")
        for s in ("s1", "s2", "s3")
    ],
    "NVFP4 · RedHatAI": [
        ("hy3-nvfp4-redhatai-4xb200", f"ref_nonces_fp8_{s}.json",
         "hy3-nvfp4-redhatai-4xb200", f"nonces_nvfp4_{s}.json")
        for s in ("s1", "s2", "s3")
    ],
    "NVFP4 · r0b0tlab": [
        ("hy3-nvfp4-r0b0tlab-4xb200", f"ref_nonces_fp8_{s}.json",
         "hy3-nvfp4-r0b0tlab-4xb200", f"nonces_nvfp4_{s}.json")
        for s in ("s1", "s2", "s3")
    ],
}


def main():
    out = {"bins": {"lo": LO, "hi": HI, "n": BINS}, "thresholds": THRESHOLDS,
           "groups": [], "gates": []}
    pools = {}

    for name, comparisons in GROUPS.items():
        vals = []
        for fa, na, fb, nb in comparisons:
            vals += dists(art(fa, na), art(fb, nb))
        pools[name] = vals

        hist = [0] * BINS
        for x in vals:
            idx = min(BINS - 1, max(0, int((x - LO) / (HI - LO) * BINS)))
            hist[idx] += 1
        n = len(vals)
        out["groups"].append({
            "name": name,
            "n": n,
            "median": round(st.median(vals), 4),
            "p95": round(sorted(vals)[int(0.95 * n) - 1], 4),
            "density": [round(c / n * BINS / (HI - LO), 4) for c in hist],
            "share040": round(100 * sum(1 for x in vals if x > 0.40) / n, 1),
        })

    for group in out["groups"]:
        vals = pools[group["name"]]
        out["gates"].append({
            "name": group["name"],
            "curve": [round(100 * sum(1 for x in vals if x > t) / group["n"], 2)
                      for t in THRESHOLDS],
        })

    json.dump(out, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
