"""Гистограммы L2 между нонсами GLM-5.3-Flash: честный пол против фрод-плеч.

Повторяет формат графика по Hy3: одна честная кривая как базовая линия и фрод-плечи поверх.
Честных линий две: внутри одной архитектуры (H100 против H200, обе Hopper) и между поколениями.
Они лежат почти друг на друге — это и есть результат, что одинаковая архитектура почти ничего
не даёт, расходится любое иное железо. Повтор на одном боксе в график не входит: там 93% нонсов совпадают
побитово, это вырожденный столбик на нуле, а не распределение.

Палитра — Окабэ–Ито, различима при всех типах дальтонизма.
"""
import base64
import json
import math
import os
import struct

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E = os.path.normpath(os.path.join(HERE, ".."))   # каталог 2026-09

SEEDS = ("s1", "s2", "s3")


def vecs(path):
    with open(path) as fh:
        doc = json.load(fh)
    return {a["nonce"]: struct.unpack("<12e", base64.b64decode(a["vector_b64"]))
            for a in doc["artifacts"]}


def dists(pairs):
    """pairs: список (путь_A, путь_B) — вернуть все попарные L2."""
    out = []
    for pa, pb in pairs:
        A, B = vecs(pa), vecs(pb)
        for n in sorted(set(A) & set(B)):
            out.append(math.dist(A[n], B[n]))
    return np.array(out)


def p(folder, name, seed):
    return os.path.join(E, folder, "artifacts", "%s_%s.json" % (name, seed))


H = lambda s: p("glm53-flash-fp8-4xh200", "nonces_honest", s)
H1 = lambda s: p("glm53-flash-fp8-8xh100", "nonces_honest", s)
B2 = lambda s: p("glm53-flash-fp8-4xb200", "nonces_honest", s)
B3 = lambda s: p("glm53-flash-fp8-2xb300", "nonces_honest", s)
REAP = lambda s: p("glm53-flash-reap50-patrickbdevaney-4xh200", "nonces_reap50", s)
NV2 = lambda s: p("glm53-flash-nvfp4-libertai-4xb200", "nonces_nvfp4", s)
NV3 = lambda s: p("glm53-flash-nvfp4-libertai-2xb300", "nonces_nvfp4", s)

SERIES = [
    ("ЧЕСТНЫЙ ↔ ЧЕСТНЫЙ, одна архитектура: 8×H100 ↔ 4×H200 (3 пары)",
     [(H1(s), H(s)) for s in SEEDS], "#56B4E9"),
    ("ЧЕСТНЫЙ ↔ ЧЕСТНЫЙ, разные поколения карт (15 пар)",
     [(H(s), B2(s)) for s in SEEDS] + [(B3(s), H(s)) for s in SEEDS]
     + [(B3(s), B2(s)) for s in SEEDS] + [(H1(s), B2(s)) for s in SEEDS]
     + [(H1(s), B3(s)) for s in SEEDS], "#0072B2"),
    ("ФРОД NVFP4 (LibertAI / ModelOpt) на 4×B200",
     [(B2(s), NV2(s)) for s in SEEDS], "#E69F00"),
    ("ФРОД REAP50 (прунинг экспертов) на 4×H200",
     [(H(s), REAP(s)) for s in SEEDS], "#009E73"),
    ("ФРОД NVFP4 (LibertAI / ModelOpt) на 2×B300",
     [(B3(s), NV3(s)) for s in SEEDS], "#CC79A7"),
]

fig, ax = plt.subplots(figsize=(14.0, 6.8), dpi=150)
bins = np.linspace(0.0, 1.4, 71)

handles = []
for label, pairs, colour in SERIES:
    d = dists(pairs)
    med = float(np.median(d))
    w = np.ones_like(d) * 100.0 / len(d)
    ax.hist(d, bins=bins, weights=w, color=colour, alpha=0.55,
            label="%-58s med=%.3f" % (label, med))

ax.axvline(0.40, color="#444444", lw=1.4, ls="--", zorder=5)
ax.text(0.408, ax.get_ylim()[1] * 0.96, "порог цепочки 0.40",
        fontsize=9, color="#444444", va="top")

ax.set_title("GLM-5.3-Flash · L2 между нонсами · 1000 нонсов на пару · dim=12 · seq_len=1024",
             fontsize=13, pad=12)
ax.set_xlabel("L2 distance")
ax.set_ylabel("% нонсов")
ax.set_xlim(0, 1.4)
ax.grid(axis="y", color="#DDDDDD", lw=0.6)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
leg = ax.legend(loc="upper right", fontsize=9, framealpha=0.95, prop={"family": "monospace"})
leg.get_frame().set_edgecolor("#CCCCCC")

fig.tight_layout()
out = os.path.join(HERE, "artifacts", "l2_distributions.png")
fig.savefig(out, bbox_inches="tight")
print("сохранено:", out)

for label, pairs, _ in SERIES:
    d = dists(pairs)
    print("%-52s N=%5d  med=%.4f  за 0.40 = %5.1f%%" % (
        label, len(d), float(np.median(d)), 100.0 * float((d > 0.40).mean())))
