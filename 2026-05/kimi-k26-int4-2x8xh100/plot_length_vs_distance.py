#!/usr/bin/env python3
"""Build the gonka-style scatter (length vs distance2) for our H100 TP=16 run.

Inputs:
  - artifacts/validation_report.json  (23 logged sample points with index + distance2)
  - data/honest_b200_h200.jsonl       (lookup: index -> prompt)

Output:
  - length_vs_distance.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HERE = Path(__file__).parent
REPORT = HERE / "artifacts" / "validation_report.json"
HONEST = HERE / "data" / "honest_b200_h200.jsonl"
OUT = HERE / "length_vs_distance.png"

# Gonka color/marker scheme (from scripts/plot_distance_vs_length.py)
HONEST_COLOR = "#0B3D91"   # dark blue
LANG_MARKER = {"sp": "^", "en": "o", "ch": "s", "ar": "D", "hi": "P"}
LANG_NAME = {"sp": "Spanish", "en": "English", "ch": "Chinese", "ar": "Arabic", "hi": "Hindi"}


def main():
    if not REPORT.exists():
        sys.exit(f"missing {REPORT}")
    if not HONEST.exists():
        sys.exit(f"missing {HONEST}")

    report = json.loads(REPORT.read_text())
    items = report["per_item_logged"]

    # Load honest dataset, indexed by line number
    print(f"loading {HONEST}...", flush=True)
    honest = []
    with HONEST.open(encoding="utf-8") as f:
        for line in f:
            honest.append(json.loads(line))
    print(f"loaded {len(honest)} honest items", flush=True)

    # Match each item by index, attach prompt length
    points = []
    for it in items:
        idx = it["index"]
        if idx >= len(honest):
            print(f"  skip idx={idx} (out of range)", flush=True)
            continue
        honest_item = honest[idx]
        prompt = honest_item["prompt"]
        lang = it["language"]
        points.append({
            "lang": lang,
            "length_chars": len(prompt),
            "distance2": it["distance2"],
            "index": idx,
            "mismatches": it["token_mismatches"],
        })

    print(f"plotting {len(points)} points", flush=True)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Group by language
    by_lang = {}
    for p in points:
        by_lang.setdefault(p["lang"], []).append(p)

    for lang, pts in by_lang.items():
        marker = LANG_MARKER.get(lang, "o")
        xs = [p["length_chars"] for p in pts]
        ys = [p["distance2"] for p in pts]
        ax.scatter(xs, ys, s=36, alpha=0.7, marker=marker,
                   c=HONEST_COLOR, edgecolors="white", linewidths=0.4,
                   label=f"{LANG_NAME.get(lang, lang)} ({len(pts)})")

    # Reference lines
    baseline = report["summary"]["honest_baseline_exp2"]["mean"]
    our_mean = report["summary"]["distance2_mean"]
    gate = 0.2
    soft = 0.05
    ax.axhline(baseline, color="gray", linestyle=":", linewidth=1, alpha=0.7,
               label=f"honest exp2 baseline mean ({baseline:.4f})")
    ax.axhline(our_mean, color="green", linestyle="--", linewidth=1, alpha=0.7,
               label=f"our mean ({our_mean:.4f})")
    ax.axhline(soft, color="orange", linestyle="--", linewidth=1, alpha=0.5,
               label=f"soft pass mean ≤ {soft}")
    ax.axhline(gate, color="red", linestyle="--", linewidth=1, alpha=0.5,
               label=f"production gate mean ≤ {gate}")

    ax.set_xlabel("Length (characters)", fontsize=11)
    ax.set_ylabel("distance2", fontsize=11)
    ax.set_title("Kimi-K2.6 INT4 — length vs distance2  (16×H100 TP=16, vs honest exp2 dataset)",
                 fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Tight & save
    plt.tight_layout()
    plt.savefig(OUT, bbox_inches="tight", dpi=300)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
