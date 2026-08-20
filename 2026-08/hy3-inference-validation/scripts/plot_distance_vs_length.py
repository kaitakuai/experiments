#!/usr/bin/env python3
"""
Scatter plot: Length (characters) vs Distance — exact gonka format.
Uses distance2 from gonka validation.utils.
"""

import json
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


# --------------- Gonka distance2 (exact copy) ---------------

def token_distance2_pos(inf_lp, val_lp):
    dist = 0.0
    n_matches = 0
    if not val_lp:
        return len(inf_lp), 0
    sorted_vals = sorted(val_lp.values())
    if len(sorted_vals) >= 2:
        min1, min2 = sorted_vals[0], sorted_vals[1]
    else:
        min1 = sorted_vals[0]
        min2 = min1 - 1.0
    for token, inf_logprob in inf_lp.items():
        if token in val_lp:
            val_logprob = val_lp[token]
            n_matches += 1
        else:
            val_logprob = min1 - (min2 - min1)
        denom = 1e-10 + abs(inf_logprob) + abs(val_logprob)
        dist += abs(inf_logprob - val_logprob) / denom / 2.0
    return dist, n_matches


def distance2(inf_result, val_result):
    inf_tokens = inf_result["results"]
    val_tokens = val_result["results"]
    n = min(len(inf_tokens), len(val_tokens))
    if n == 0:
        return 0.0
    top_k = len(inf_tokens[0]["logprobs"])
    total_dist = 0.0
    total_matches = 0
    for i in range(n):
        d, m = token_distance2_pos(inf_tokens[i]["logprobs"], val_tokens[i]["logprobs"])
        total_dist += d
        total_matches += m
    total_dist = (total_dist + 1.0) / (max(100, n) * top_k + 1.0)
    return total_dist


# --------------- Helpers ---------------

def detect_language(item):
    lang = item.get("language", "")
    if lang:
        return lang
    # Auto-detect from prompt text
    prompt = item.get("prompt", "")
    if not prompt:
        return "en"
    # Check for CJK characters (Chinese)
    for ch in prompt[:200]:
        if '\u4e00' <= ch <= '\u9fff':
            return "ch"
    # Check for Arabic
    for ch in prompt[:200]:
        if '\u0600' <= ch <= '\u06ff':
            return "ar"
    # Check for Devanagari (Hindi)
    for ch in prompt[:200]:
        if '\u0900' <= ch <= '\u097f':
            return "hi"
    # Check for Spanish markers
    sp_markers = ['¿', '¡', 'ñ', 'á', 'é', 'í', 'ó', 'ú']
    if any(c in prompt[:200].lower() for c in sp_markers):
        return "sp"
    return "en"


def load_jsonl(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


# --------------- Exact gonka plot ---------------

def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_distance_vs_length.py <honest.jsonl> <fraud.jsonl> [--title TITLE] [--output FILE] [--bounds LOWER,UPPER]")
        sys.exit(1)

    honest_path = sys.argv[1]
    fraud_path = sys.argv[2]

    title = "Qwen235B - Length vs Distance Comparison"
    output = None
    bounds = None
    for i, arg in enumerate(sys.argv):
        if arg == "--title" and i + 1 < len(sys.argv):
            title = sys.argv[i + 1]
        if arg == "--output" and i + 1 < len(sys.argv):
            output = sys.argv[i + 1]
        if arg == "--bounds" and i + 1 < len(sys.argv):
            parts = sys.argv[i + 1].split(",")
            bounds = (float(parts[0]), float(parts[1]))

    honest_items = load_jsonl(honest_path)
    fraud_items = load_jsonl(fraud_path)

    print(f"Honest: {len(honest_items)} prompts")
    print(f"Fraud:  {len(fraud_items)} prompts")

    # Compute distances and lengths
    honest_distances = []
    honest_lengths = []
    honest_langs = []
    for item in honest_items:
        inf = item["inference_result"]
        val = item.get("validation_result") or item.get("new_validation_result")
        honest_distances.append(distance2(inf, val))
        honest_lengths.append(len(inf.get("text", "")))
        honest_langs.append(detect_language(item))

    fraud_distances = []
    fraud_lengths = []
    fraud_langs = []
    for item in fraud_items:
        inf = item["inference_result"]
        val = item.get("validation_result") or item.get("new_validation_result")
        fraud_distances.append(distance2(inf, val))
        fraud_lengths.append(len(inf.get("text", "")))
        fraud_langs.append(detect_language(item))

    # --- Exact gonka plot style ---
    plt.figure(figsize=(10, 6))
    marker_size = 36

    honest_color = '#0B3D91'
    fraud_color = plt.cm.Reds(0.8)

    fixed_marker_map = {
        'sp': '^',
        'en': 'o',
        'ch': 's',
        'ar': 'D',
        'hi': 'P',
    }
    fallback_markers = ['v', '*', 'X', 'h', '<', '>', '1', '2', '3', '4']

    # Collect unique languages in order
    seen = set()
    unique_languages = []
    for lang in honest_langs + fraud_langs:
        if lang not in seen:
            seen.add(lang)
            unique_languages.append(lang)

    unknown_langs = sorted([l for l in unique_languages if l not in fixed_marker_map])
    unknown_marker_map = {l: fallback_markers[i % len(fallback_markers)] for i, l in enumerate(unknown_langs)}
    marker_map = {**fixed_marker_map, **unknown_marker_map}

    # Plot honest
    for lang in unique_languages:
        idxs = [i for i, l in enumerate(honest_langs) if l == lang]
        if not idxs:
            continue
        plt.scatter(
            [honest_lengths[i] for i in idxs],
            [honest_distances[i] for i in idxs],
            c=honest_color,
            marker=marker_map.get(lang, 'o'),
            alpha=0.6,
            s=marker_size,
            label=None,
        )

    # Plot fraud
    for lang in unique_languages:
        idxs = [i for i, l in enumerate(fraud_langs) if l == lang]
        if not idxs:
            continue
        plt.scatter(
            [fraud_lengths[i] for i in idxs],
            [fraud_distances[i] for i in idxs],
            c=fraud_color,
            marker=marker_map.get(lang, 'o'),
            alpha=0.6,
            s=marker_size,
            label=None,
        )

    # Bounds lines
    if bounds is not None:
        lower, upper = bounds
        plt.axhline(lower, color='blue', linestyle='--')
        plt.axhline(upper, color='purple', linestyle='--')

    # Group legend
    honest_key = os.path.splitext(os.path.basename(honest_path))[0]
    fraud_key = os.path.splitext(os.path.basename(fraud_path))[0]
    group_handles = [
        Line2D([0], [0], marker='o', color=honest_color, linestyle='None', markersize=8, label=f"Honest - {honest_key}"),
        Line2D([0], [0], marker='o', color=fraud_color, linestyle='None', markersize=8, label=f"Fraud - {fraud_key}"),
    ]
    legend1 = plt.legend(handles=group_handles, title='Groups', loc='upper left')
    plt.gca().add_artist(legend1)

    # Language legend
    language_name_map = {'sp': 'Spanish', 'en': 'English', 'ch': 'Chinese', 'ar': 'Arabic', 'hi': 'Hindi', 'unk': 'Unknown'}
    lang_handles = [
        Line2D([0], [0], marker=marker_map[lang], color='black', linestyle='None', markersize=8,
               label=language_name_map.get(lang, str(lang)))
        for lang in ['sp', 'en', 'ch', 'ar', 'hi', 'unk']
        if lang in marker_map
    ]
    legend2 = plt.legend(handles=lang_handles, title='Languages', loc='upper right')

    if bounds is not None:
        plt.gca().add_artist(legend2)
        lower, upper = bounds
        bounds_handles = [
            Line2D([0], [0], color='blue', linestyle='--', linewidth=2, label=f'Lower: {lower:.6f}'),
            Line2D([0], [0], color='purple', linestyle='--', linewidth=2, label=f'Upper: {upper:.6f}'),
        ]
        plt.legend(handles=bounds_handles, title='Bounds', loc='lower right')

    plt.title(f'{title} - Length vs Distance Comparison')
    plt.xlabel('Length (characters)')
    plt.ylabel('Distance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if not output:
        output = os.path.splitext(honest_path)[0] + "_plot.png"
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()


if __name__ == "__main__":
    main()
