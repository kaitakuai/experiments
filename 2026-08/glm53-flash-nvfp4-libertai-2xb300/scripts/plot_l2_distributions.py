#!/usr/bin/env python3
"""L2 distance distribution for GLM-5.3-Flash PoC nonces.

Same presentation as the DeepSeek-V4 and Hy3 L2 charts in this repository: filled
histograms, y in % of nonces, medians in the legend, no threshold lines.

    python3 scripts/plot_l2_distributions.py   ->  artifacts/l2_distributions_glm53.png

Two curves:
  * honest FP8 vs NVFP4 fraud, 3 seeds pooled (3000 pairs) — the detection signal;
  * honest FP8 vs honest FP8 on DIFFERENT seeds — the ceiling of the metric, i.e. what
    a completely unrelated nonce set looks like. It is the scale, not a fraud arm.

There is deliberately no honest-vs-honest same-seed curve here: this box measured a
single honest run, so the honest floor of this model is not established by this
experiment (see README, "What this experiment does not answer").
"""
import base64
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "..", "artifacts")
SEEDS = ("s1", "s2", "s3")


def load(p):
    d = json.load(open(p))
    a = d["artifacts"] if isinstance(d, dict) else d
    return {x["nonce"]: np.frombuffer(base64.b64decode(x["vector_b64"]),
                                      dtype="<f2").astype(np.float32) for x in a}


def dists(pa, pb):
    a, b = load(pa), load(pb)
    common = sorted(set(a) & set(b))
    return np.array([np.linalg.norm(a[k] - b[k]) for k in common])


def pooled(pairs):
    out = [dists(x, y) for x, y in pairs if os.path.exists(x) and os.path.exists(y)]
    return np.concatenate(out) if out else np.array([])


series = [
    ("ФРОД NVFP4  (LibertAIDAI / ModelOpt)",
     [(f"{ART}/ref_nonces_fp8_{s}.json", f"{ART}/nonces_nvfp4_{s}.json") for s in SEEDS],
     "#A87BC7"),
    ("ПОТОЛОК ШКАЛЫ: честный FP8, РАЗНЫЕ семена",
     [(f"{ART}/ref_nonces_fp8_s1.json", f"{ART}/ref_nonces_fp8_s2.json"),
      (f"{ART}/ref_nonces_fp8_s1.json", f"{ART}/ref_nonces_fp8_s3.json")],
     "#5A8FC7"),
]

XMAX = 1.8
fig, ax = plt.subplots(figsize=(15, 7.2))
bins = np.arange(0, XMAX + 0.018, 0.018)
stats = []
for label, pairs, colour in series:
    d = pooled(pairs)
    if not len(d):
        print("SKIP (no data):", label)
        continue
    ax.hist(d, bins=bins, weights=np.full(len(d), 100.0 / len(d)),
            color=colour, alpha=0.62, edgecolor="none",
            label=f"{label}      med={np.median(d):.3f}")
    stats.append((label, len(d), float(np.median(d)), float((d > 0.4).mean() * 100)))

ax.set_xlabel("L2 distance", fontsize=10)
ax.set_ylabel("% нонсов", fontsize=10)
ax.set_title("GLM-5.3-Flash · L2 между нонсами · 1000 нонсов на пару · dim=12 · seq_len=1024",
             fontsize=13, pad=12)
ax.legend(fontsize=10.5, loc="upper right", framealpha=0.95)
ax.grid(alpha=0.25, linewidth=0.7)
ax.set_axisbelow(True)
ax.set_xlim(0, XMAX)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
out = os.path.join(ART, "l2_distributions_glm53.png")
fig.savefig(out, dpi=150)
print("saved", out)
print(f"\n{'comparison':50} {'n':>6} {'median':>8} {'>0.4 %':>8}")
for name, n, med, pct in stats:
    print(f"{name[:50]:50} {n:6d} {med:8.3f} {pct:8.2f}")
