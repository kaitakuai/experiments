#!/usr/bin/env python3
"""Full pairwise L2 matrix across every GLM-5.3-Flash arm measured so far.

Usage:  python3 scripts/matrix.py > artifacts/matrix.json

Reads the committed nonce sets from the other experiment folders — nothing is duplicated
here, so the matrix cannot drift from the reports it summarises.

Every pair is computed on all three shared seeds (s1, s2, s3) and reported as the median of
the per-seed medians plus the mismatch rate against the chain's 0.40 gate.

Two things this script is careful about:

* **Basenames.** `compare_nonces.py` labels pairs by file basename and silently compares a
  file with itself when two inputs share a name. This script keys by (folder, arm) instead
  and never relies on filenames.
* **Confounds.** Arms differ in more than one dimension at a time. Each pair carries a
  `varies` field listing what actually changes, so a reader cannot mistake a
  hardware+build comparison for a hardware one.
"""
import base64
import json
import math
import os
import struct
import sys

THRESHOLD = 0.40
BATCH = 16
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(HERE, "..", "..")

# arm key -> (path template, hardware, image, TP, kind)
ARMS = {
    "b300_honest_aug": ("2026-08/glm53-flash-fp8-2xb300/artifacts/nonces_fp8_%s.json",
                        "2xB300", "old (FlashInfer 0.6.17)", 2, "honest"),
    "b300_nvfp4_aug": ("2026-08/glm53-flash-nvfp4-libertai-2xb300/artifacts/nonces_nvfp4_%s.json",
                       "2xB300", "old (FlashInfer 0.6.17)", 2, "fraud/nvfp4"),
    "h200_honest": ("2026-09/glm53-flash-fp8-4xh200/artifacts/nonces_honest_%s.json",
                    "4xH200", "k3 (FlashInfer 0.6.18)", 4, "honest"),
    "h200_reap50": ("2026-09/glm53-flash-reap50-patrickbdevaney-4xh200/artifacts/nonces_reap50_%s.json",
                    "4xH200", "k3 (FlashInfer 0.6.18)", 4, "fraud/reap50"),
    "b200_honest": ("2026-09/glm53-flash-fp8-4xb200/artifacts/nonces_honest_%s.json",
                    "4xB200", "k3 (FlashInfer 0.6.18)", 4, "honest"),
    "b200_nvfp4": ("2026-09/glm53-flash-nvfp4-libertai-4xb200/artifacts/nonces_nvfp4_%s.json",
                   "4xB200", "k3 (FlashInfer 0.6.18)", 4, "fraud/nvfp4"),
    "h100_honest": ("2026-09/glm53-flash-fp8-8xh100/artifacts/nonces_honest_%s.json",
                    "8xH100", "k3 (FlashInfer 0.6.18)", 8, "honest"),
}

SEEDS = ("s1", "s2", "s3")


def load(path):
    with open(path) as fh:
        doc = json.load(fh)
    out = {}
    for a in doc["artifacts"]:
        raw = base64.b64decode(a["vector_b64"])
        out[a["nonce"]] = struct.unpack("<%de" % (len(raw) // 2), raw)
    return out, (doc.get("block_hash") or "")[:12]


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0.0


def varies(a, b):
    """What actually differs between two arms — the confound list."""
    _, hw_a, img_a, tp_a, kind_a = ARMS[a]
    _, hw_b, img_b, tp_b, kind_b = ARMS[b]
    out = []
    if hw_a != hw_b:
        out.append("hardware")
    if img_a != img_b:
        out.append("image")
    if tp_a != tp_b:
        out.append("TP")
    if kind_a != kind_b:
        out.append("checkpoint")
    return out or ["nothing (same arm)"]


def main():
    sets, seedcheck = {}, {}
    for key, (tmpl, *_rest) in ARMS.items():
        for s in SEEDS:
            p = os.path.normpath(os.path.join(ROOT, tmpl % s))
            if not os.path.exists(p):
                print("отсутствует набор: %s" % p, file=sys.stderr)
                continue
            sets[(key, s)], seedcheck[(key, s)] = load(p)

    # каждое семя обязано совпадать по block_hash во всех плечах, иначе сравнение бессмысленно
    for s in SEEDS:
        hashes = {seedcheck[(k, s)] for k in ARMS if (k, s) in seedcheck}
        if len(hashes) > 1:
            raise SystemExit("семя %s имеет разные block_hash: %s" % (s, hashes))

    keys = list(ARMS)
    out = {"threshold": THRESHOLD, "batch_size": BATCH, "seeds": list(SEEDS),
           "arms": {k: dict(zip(("hardware", "image", "tp", "kind"), ARMS[k][1:])) for k in ARMS},
           "pairs": {}}

    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            per_seed, rates, first_rates = [], [], []
            for s in SEEDS:
                if (a, s) not in sets or (b, s) not in sets:
                    continue
                va, vb = sets[(a, s)], sets[(b, s)]
                common = sorted(set(va) & set(vb))
                d = {n: math.dist(va[n], vb[n]) for n in common}
                per_seed.append(median(list(d.values())))
                rates.append(100.0 * sum(1 for v in d.values() if v > THRESHOLD) / len(d))
                firsts = [n for n in common if n % BATCH == 0]
                first_rates.append(100.0 * sum(1 for n in firsts if d[n] > THRESHOLD) / len(firsts))
            if not per_seed:
                continue
            out["pairs"]["%s__vs__%s" % (a, b)] = {
                "median_l2": round(median(per_seed), 4),
                "per_seed_median": [round(x, 4) for x in per_seed],
                "past_threshold_pct": round(sum(rates) / len(rates), 1),
                "per_seed_past_pct": [round(x, 1) for x in rates],
                "first_in_batch_past_pct": round(sum(first_rates) / len(first_rates), 1),
                "varies": varies(a, b),
            }

    json.dump(out, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
