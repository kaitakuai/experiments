#!/usr/bin/env python3
"""Regenerate every table in README.md from the committed artifacts.

Usage:  python3 scripts/summarize.py artifacts > artifacts/summary.json

The honest reference sets are committed alongside the fraud sets (`ref_nonces_honest_*.json`)
so this folder is self-contained; they are byte-identical copies of the ones in
`../glm53-flash-fp8-4xh200/artifacts/`.

Reported per seed:

1. **Fraud distance** — NVFP4 (ModelOpt, LibertAI) against the honest arm on the same box.
2. **The batch-boundary split** — nonces at `index % 16 == 0` behave differently from the rest
   and are broken out, because they fail cross-hardware comparison even for honest nodes and
   would otherwise inflate the fraud signal.
3. **Control** — two different seeds, which must be incomparable (~1.41).

L2 arithmetic follows the chain (`vllm/poc/data.py`): fp16 little-endian -> fp32, fp64 norm,
strict `>` against the threshold.
"""
import base64
import glob
import json
import math
import os
import struct
import sys

THRESHOLD = 0.40
BATCH = 16  # collection batch size used for every run in this folder


def load(path):
    with open(path) as fh:
        doc = json.load(fh)
    arts = doc["artifacts"]
    k = len(base64.b64decode(arts[0]["vector_b64"])) // 2
    vecs = {}
    for a in arts:
        raw = base64.b64decode(a["vector_b64"])
        vecs[a["nonce"]] = [float(x) for x in struct.unpack("<%de" % k, raw)]
    return doc, vecs


def l2_map(a, b):
    return {n: math.dist(a[n], b[n]) for n in sorted(set(a) & set(b))}


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0.0


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p / 100))] if xs else 0.0


def stats(d):
    vals = list(d.values())
    over = [n for n, v in d.items() if v > THRESHOLD]
    return {
        "n": len(vals),
        "mean": round(sum(vals) / len(vals), 4),
        "median": round(median(vals), 4),
        "p25": round(pct(vals, 25), 4),
        "p95": round(pct(vals, 95), 4),
        "max": round(max(vals), 4),
        "over_threshold": len(over),
        "over_threshold_pct": round(100 * len(over) / len(vals), 2),
    }


def batch_split(d, batch=BATCH):
    out = {"batch_size": batch}
    for name, grp in (("first_in_batch", {n: v for n, v in d.items() if n % batch == 0}),
                      ("rest", {n: v for n, v in d.items() if n % batch != 0})):
        if not grp:
            continue
        over = sum(1 for v in grp.values() if v > THRESHOLD)
        out[name] = {
            "count": len(grp),
            "median": round(median(list(grp.values())), 4),
            "over_threshold": over,
            "over_threshold_pct": round(100 * over / len(grp), 1),
        }
    return out


def main(art_dir):
    out = {"threshold": THRESHOLD, "batch_size": BATCH, "integrity": {}, "pairs": {}}
    sets = {}
    for path in sorted(glob.glob(os.path.join(art_dir, "*nonces_*.json"))):
        name = os.path.basename(path)[:-len(".json")].replace("nonces_", "")
        doc, vecs = load(path)
        sets[name] = vecs
        arts = doc["artifacts"]
        out["integrity"][name] = {
            "nonces": len(arts),
            "non_empty": sum(1 for a in arts if a.get("vector_b64")),
            "unique": len({a["vector_b64"] for a in arts}),
            "block_hash_prefix": (doc.get("block_hash") or "")[:12],
        }

    def pair(label, a, b):
        if a in sets and b in sets:
            d = l2_map(sets[a], sets[b])
            out["pairs"][label] = {"a": a, "b": b, **stats(d), "by_batch_position": batch_split(d)}

    for s in ("s1", "s2", "s3"):
        pair("nvfp4_vs_honest_%s" % s, "ref_honest_%s" % s, "nvfp4_%s" % s)
    pair("control_different_seeds", "ref_honest_s1", "ref_honest_s2")

    json.dump(out, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "artifacts")
