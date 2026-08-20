#!/usr/bin/env python3
"""
Render `l2-distances.png` from `l2-distances-data.json`.

Run `build_l2_data.py` first if the artifacts changed:

    python3 scripts/build_l2_data.py > l2-distances-data.json
    python3 scripts/render_chart.py

Requires matplotlib. Colours are the four leading slots of the validated categorical
palette (blue / orange / aqua / violet); every curve is also labelled in the legend, so
identity never rests on colour alone.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
LABELS = ["честный пол", "INT4 · cyankiwi", "NVFP4 · RedHatAI", "NVFP4 · r0b0tlab"]

GATES = (0.28, 0.40)


def main():
    with open(os.path.join(ROOT, "l2-distances-data.json")) as fh:
        data = json.load(fh)

    groups, gates, thresholds = data["groups"], data["gates"], data["thresholds"]
    lo, hi, nb = data["bins"]["lo"], data["bins"]["hi"], data["bins"]["n"]
    centres = [lo + (i + 0.5) * (hi - lo) / nb for i in range(nb)]
    ymax = max(max(g["density"]) for g in groups) * 1.08

    plt.rcParams.update({
        "font.size": 11, "axes.edgecolor": "#c9c8c3", "axes.linewidth": 0.8,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 9.4), gridspec_kw={"height_ratios": [1.25, 1], "hspace": 0.34})

    # --- density ---
    for g, colour, label in zip(groups, COLORS, LABELS):
        ax1.plot(centres, g["density"], color=colour, lw=2,
                 label=f"{label}  (медиана {g['median']:.3f})")
        ax1.fill_between(centres, g["density"], color=colour, alpha=0.10)
    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(0, ymax)
    for gate in GATES:
        ax1.axvline(gate, color="#8a8983", ls=(0, (4, 4)), lw=1.3)
        ax1.text(gate, ymax * 0.995, f"{gate:.2f}", ha="center", va="top",
                 fontsize=10, color="#6b6a65")
    ax1.set_xlabel("L2-расстояние между отпечатками одного нонса")
    ax1.set_ylabel("плотность")
    ax1.set_title("Hy3 · PoC v2 — распределение L2-расстояний\n"
                  "честный пол против трёх фрод-чекпоинтов",
                  loc="left", fontsize=14, fontweight="bold", pad=14)
    ax1.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.97))
    ax1.grid(axis="y", color="#ececea", lw=0.8)
    ax1.set_axisbelow(True)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)
    ax1.annotate("честное расхождение\nприжато к 0.20", xy=(0.235, 4.35), xytext=(0.40, 3.55),
                 fontsize=10.5, color="#3a3936",
                 arrowprops=dict(arrowstyle="->", color="#8a8983", lw=1.1))
    ax1.annotate("два разных кванта\nлегли друг на друга", xy=(0.34, 3.45), xytext=(0.60, 2.35),
                 fontsize=10.5, color="#3a3936",
                 arrowprops=dict(arrowstyle="->", color="#8a8983", lw=1.1))

    # --- share above gate ---
    for g, colour, label in zip(gates, COLORS, LABELS):
        ax2.plot(thresholds, [v / 100 for v in g["curve"]], color=colour, lw=2, label=label)
    for gate in GATES:
        ax2.axvline(gate, color="#8a8983", ls=(0, (4, 4)), lw=1.3)
    ax2.set_xlim(thresholds[0], thresholds[-1])
    ax2.set_ylim(0, 1.02)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.set_xlabel("порог L2")
    ax2.set_ylabel("доля нонсов выше порога")
    ax2.set_title("Почему поднимать порог вредно: справа сигнал исчезает вместе "
                  "с ложными срабатываниями", loc="left", fontsize=12, pad=10)
    ax2.grid(axis="y", color="#ececea", lw=0.8)
    ax2.set_axisbelow(True)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax2.annotate("порог 0.28:\nчестный 18 % · фрод 80 %", xy=(0.28, 0.80), xytext=(0.44, 0.90),
                 fontsize=10, color="#3a3936",
                 arrowprops=dict(arrowstyle="->", color="#8a8983", lw=1))
    ax2.annotate("порог 0.40:\nчестный 3.5 % · фрод 42 %", xy=(0.40, 0.42), xytext=(0.55, 0.56),
                 fontsize=10, color="#3a3936",
                 arrowprops=dict(arrowstyle="->", color="#8a8983", lw=1))

    fig.text(0.008, 0.012,
             "Повтор одной конфигурации на одной машине Blackwell даёт ровно 0 на всех 1000 "
             "нонсах — там детекция точная, а не статистическая.   "
             "n: честный пол 5000, каждый фрод 3000 (три семени).",
             fontsize=9.5, color="#6b6a65")

    out = os.path.join(ROOT, "l2-distances.png")
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
