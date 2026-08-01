#!/usr/bin/env python3
"""L2 distance distributions for DeepSeek-V4-Flash-0731 PoC nonces.

Every curve is 1000 nonces per seed, pooled over the seeds available for that
comparison. Same-card repeats are excluded (they are a spike at zero and only
flatten the axis); no threshold line, no annotations.
"""
import base64
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
H200 = f"{R}/2026-08/deepseek-v4-flash-0731-2xh200/artifacts"
H100 = f"{R}/2026-08/deepseek-v4-flash-0731-dspark-4xh100/artifacts"
B300 = f"{R}/2026-08/deepseek-v4-flash-0731-nvfp4-1xb300/artifacts"
OLD = f"{R}/2026-07/deepseek-v4-flash-2xh200/artifacts"


def load(p):
    return {a["nonce"]: a["vector_b64"] for a in json.load(open(p))["artifacts"]}


def vec(b):
    return np.frombuffer(base64.b64decode(b), dtype="<f2").astype(np.float32)


def dists(pa, pb):
    a, b = load(pa), load(pb)
    common = sorted(set(a) & set(b))
    return np.array([np.linalg.norm(vec(a[k]) - vec(b[k])) for k in common])


def pooled(pairs):
    out = []
    for pa, pb in pairs:
        if os.path.exists(pa) and os.path.exists(pb):
            out.append(dists(pa, pb))
    return np.concatenate(out) if out else np.array([])


S = ("s1", "s2", "s3")
series = [
    ("Честный шум\nразные модели GPU (H100 ↔ H200)",
     [(f"{H100}/nonces_dspark_off_{s}.json", f"{H200}/nonces_v2_off_{s}.json") for s in S],
     "#4C78A8"),
    ("DSpark вкл. против выкл.\nодна машина",
     [(f"{H100}/nonces_dspark_off_{s}.json", f"{H100}/nonces_dspark_on_{s}.json") for s in S],
     "#54A24B"),
    ("NVFP4-фродер\nпротив честной, одна карта",
     [(f"{B300}/nonces_official_dspark_off_{s}.json",
       f"{B300}/nonces_nvfp4_dspark_off_{s}.json") for s in S],
     "#E45756"),
    ("Подмена чекпоинта\nстарый -Flash вместо 0731",
     [(f"{H200}/nonces_v1_off_legacyseed.json", f"{OLD}/nonces_eager.json")],
     "#B279A2"),
    ("Разные семена\n(потолок шкалы)",
     [(f"{H200}/nonces_v2_off_s1.json", f"{H200}/nonces_v2_off_s2.json")],
     "#9D755D"),
]

fig, ax = plt.subplots(figsize=(11, 6.2))
bins = np.linspace(0, 1.8, 150)
stats = []
for label, pairs, colour in series:
    d = pooled(pairs)
    if not len(d):
        print("SKIP (no data):", label.replace("\n", " "))
        continue
    ax.hist(d, bins=bins, density=True, histtype="stepfilled",
            alpha=0.32, color=colour)
    ax.hist(d, bins=bins, density=True, histtype="step",
            linewidth=2.0, color=colour,
            label=f"{label}   медиана {np.median(d):.3f}")
    stats.append((label.replace("\n", " "), len(d), float(np.median(d)),
                  float((d > 0.4).mean() * 100)))

ax.set_xlabel("L2-расстояние между парой нонсов", fontsize=11)
ax.set_ylabel("плотность", fontsize=11)
ax.set_title("DeepSeek-V4-Flash-0731 · распределение L2 по 1000 нонсов на семя\n"
             "PoC v2, seq_len 1024, k_dim 12", fontsize=12.5, pad=14)
ax.legend(fontsize=9.2, loc="upper right", framealpha=0.94)
ax.grid(alpha=0.22, linewidth=0.7)
ax.set_xlim(0, 1.8)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts", "l2_distributions_0731.png")
fig.savefig(out, dpi=170)
print("saved", out)
print(f"\n{'comparison':52} {'n':>6} {'median':>8} {'>0.4 %':>8}")
for name, n, med, pct in stats:
    print(f"{name[:52]:52} {n:6d} {med:8.3f} {pct:8.2f}")
