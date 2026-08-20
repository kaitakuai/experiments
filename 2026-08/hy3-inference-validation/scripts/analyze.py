#!/usr/bin/env python3
"""Hy3 inference validation: separation of each fraud arm from honest.

Reads the validator's replay files and reports, per logprobs mode:
  mean distance2 and its ratio to the honest floor,
  best F1 over all thresholds, and TP at a 5% false-positive rate.

Answers shorter than 100 tokens are also reported separately: distance2 pins
its denominator at max(100, n) * top_k + 1, so short answers collapse toward
1/401 for honest and fraud alike and only add noise to the threshold.
"""

import json
import os
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/out"
ARMS = ["nvfp4", "int4"]
MODES = ["processed_logprobs", "raw_logprobs"]


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def best_f1(honest, fraud):
    """Max F1 over thresholds; fraud is positive."""
    pts = sorted({*honest, *fraud})
    best = (0.0, None, 0, 0)
    for t in pts:
        tp = sum(1 for d in fraud if d >= t)
        fp = sum(1 for d in honest if d >= t)
        fn = len(fraud) - tp
        if tp == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best[0]:
            best = (f1, t, prec, rec)
    return best


def tp_at_fp(honest, fraud, fp_rate=0.05):
    """Recall at the threshold admitting at most fp_rate false positives."""
    k = int(len(honest) * fp_rate)
    thr = sorted(honest, reverse=True)[k] if k < len(honest) else max(honest)
    tp = sum(1 for d in fraud if d > thr)
    return tp / len(fraud), thr


def stats(rows, min_tokens=0):
    out = []
    for r in rows:
        n = len(r["inference_result"]["results"])
        if n >= min_tokens and r.get("distance") is not None:
            out.append(r["distance"])
    return out


for mode in MODES:
    hp = os.path.join(OUT, f"val_honest_{mode}.jsonl")
    if not os.path.exists(hp):
        continue
    honest_rows = load(hp)
    print(f"\n{'=' * 72}\n{mode}\n{'=' * 72}")
    for min_tok, label in ((0, "all answers"), (100, "answers >= 100 tokens")):
        h = stats(honest_rows, min_tok)
        if not h:
            continue
        floor = sum(h) / len(h)
        print(f"\n  {label}:  honest n={len(h)}  mean={floor:.6f}")
        for arm in ARMS:
            fp_path = os.path.join(OUT, f"val_{arm}_{mode}.jsonl")
            if not os.path.exists(fp_path):
                continue
            fr = stats(load(fp_path), min_tok)
            if not fr:
                continue
            m = sum(fr) / len(fr)
            f1, thr, prec, rec = best_f1(h, fr)
            tpr, thr2 = tp_at_fp(h, fr)
            print(
                f"    {arm:6s} n={len(fr):4d} mean={m:.6f} ratio={m / floor:.2f}x  "
                f"F1={f1:.3f}@{thr:.4f} (P={prec:.3f} R={rec:.3f})  "
                f"TP@FP5%={100 * tpr:.1f}% @{thr2:.4f}"
            )

# length mismatches -- must be zero everywhere for the replay to be trusted
print(f"\n{'=' * 72}\nlength mismatches (must be 0)\n{'=' * 72}")
for arm in ["honest"] + ARMS:
    for mode in MODES:
        p = os.path.join(OUT, f"val_{arm}_{mode}.jsonl")
        if os.path.exists(p):
            rows = load(p)
            mm = sum(1 for r in rows if r.get("len_mismatch"))
            print(f"  {arm:6s} {mode:20s} n={len(rows):4d} mismatches={mm}")
