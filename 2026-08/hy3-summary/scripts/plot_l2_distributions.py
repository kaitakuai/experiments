#!/usr/bin/env python3
"""L2 distance distributions for Hy3 PoC nonces.

Every curve pools 1000 nonces per seed over the seeds available for that comparison.
Blackwell same-machine repeats are excluded — they are a spike at exactly zero
(1000/1000 bit-identical) and only flatten the axis. Hopper repeats are NOT a spike:
they sit on the honest floor with everything else, so they stay in the honest curve.
"""
import base64
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
H200 = f"{R}/hy3-fp8-4xh200/artifacts"
H100 = f"{R}/hy3-fp8-8xh100/artifacts"
B300 = f"{R}/hy3-fp8-2xb300/artifacts"
B200 = f"{R}/hy3-fp8-4xb200/artifacts"
INT4 = f"{R}/hy3-int4-cyankiwi-4xh200/artifacts"
NVR = f"{R}/hy3-nvfp4-r0b0tlab-4xb200/artifacts"
NVH = f"{R}/hy3-nvfp4-redhatai-4xb200/artifacts"


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
    ("Честный шум\nразные машины и повторы на Hopper",
     [(f"{H100}/nonces_fp8_s1.json", f"{H200}/nonces_fp8_s1.json"),
      (f"{H100}/nonces_fp8_s1.json", f"{B300}/nonces_fp8_s1.json"),
      (f"{B200}/nonces_fp8_s1.json", f"{B300}/nonces_fp8_s1.json"),
      (f"{H200}/nonces_fp8_s1.json", f"{H200}/nonces_fp8_s1_r2.json"),
      (f"{H100}/nonces_fp8_s1.json", f"{H100}/nonces_fp8_s1_r2.json")],
     "#4C78A8"),
    ("INT4 W4A16 · cyankiwi\nпротив честной, одна машина",
     [(f"{INT4}/ref_nonces_fp8_{s}.json", f"{INT4}/nonces_int4_{s}.json") for s in S],
     "#54A24B"),
    ("NVFP4 · RedHatAI\nпротив честной, одна машина",
     [(f"{NVH}/ref_nonces_fp8_{s}.json", f"{NVH}/nonces_nvfp4_{s}.json") for s in S],
     "#E45756"),
    ("NVFP4 · r0b0tlab\nпротив честной, одна машина",
     [(f"{NVR}/ref_nonces_fp8_{s}.json", f"{NVR}/nonces_nvfp4_{s}.json") for s in S],
     "#B279A2"),
    ("Разные семена\n(потолок шкалы)",
     [(f"{H200}/nonces_fp8_s1.json", f"{H200}/nonces_fp8_s2.json")],
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
ax.set_title("Hy3 · распределение L2 по 1000 нонсов на семя\n"
             "PoC v2, seq_len 1024, k_dim 12", fontsize=12.5, pad=14)
ax.legend(fontsize=9.2, loc="upper right", framealpha=0.94)
ax.grid(alpha=0.22, linewidth=0.7)
ax.set_xlim(0, 1.8)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts",
                   "l2_distributions_hy3.png")
fig.savefig(out, dpi=170)
print("saved", out)
print(f"\n{'comparison':52} {'n':>6} {'median':>8} {'>0.4 %':>8}")
for name, n, med, pct in stats:
    print(f"{name[:52]:52} {n:6d} {med:8.3f} {pct:8.2f}")
