#!/usr/bin/env python3
"""Canonical Gonka PoC v2 nonces comparison.

Implements the EXACT same logic as `vllm.poc.data` and `vllm.poc.validation`
(from gonka's vllm-pocv2 fork), so a comparison run here yields the same
verdict the chain validator would produce.

Source references:
  vllm/poc/data.py        — decode_vector, is_mismatch, fraud_test
  vllm/poc/validation.py  — validate_artifacts, run_validation

Inputs are two JSON files in the artifact-collection format:
  {"artifacts": [{"nonce": int, "vector_b64": str}, ...]}
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import binomtest


# Canonical defaults from vllm/poc/data.py
DEFAULT_DIST_THRESHOLD = 0.02
DEFAULT_P_MISMATCH = 0.001
DEFAULT_FRAUD_THRESHOLD = 0.01

# Threshold scenarios reported by default. p_mismatch=0.02 reflects the
# calibrated honest baseline from the official PoC v2 validation report.
DEFAULT_SCENARIOS = [
    ("vllm strict (self-validation default)", 0.02, 0.001),
    ("PoC report calibrated honest 98pct",    0.173663, 0.02),
    ("chain proto default (production)",      0.4, 0.001),
    ("chain proto default + calibrated p_mis", 0.4, 0.02),
]


# --- Canonical functions, lifted byte-for-byte from vllm/poc/data.py ---

def decode_vector(b64: str) -> np.ndarray:
    """Decode base64 FP16 little-endian to FP32 (exact gonka decoder)."""
    data = base64.b64decode(b64)
    f16 = np.frombuffer(data, dtype="<f2")
    return f16.astype(np.float32)


def is_mismatch(
    computed_vector: np.ndarray,
    received_b64: str,
    dist_threshold: float = DEFAULT_DIST_THRESHOLD,
) -> bool:
    """Per-nonce mismatch test (exact gonka rule)."""
    received = decode_vector(received_b64)
    if not np.all(np.isfinite(received)):
        return True
    distance = float(np.linalg.norm(computed_vector - received))
    return distance > dist_threshold


def fraud_test(
    n_mismatch: int,
    n_total: int,
    p_mismatch: float = DEFAULT_P_MISMATCH,
    fraud_threshold: float = DEFAULT_FRAUD_THRESHOLD,
) -> Tuple[float, bool]:
    """One-sided binomial fraud test (exact gonka rule)."""
    if n_total == 0:
        return 1.0, False
    p_value = binomtest(
        n_mismatch, n_total, p_mismatch, alternative="greater"
    ).pvalue
    return p_value, p_value < fraud_threshold


# --- Canonical validation loop, lifted from vllm/poc/validation.py ---

def validate_artifacts(
    computed_artifacts: List[Dict],
    validation_map: Dict[int, str],
    dist_threshold: float = DEFAULT_DIST_THRESHOLD,
    k_dim: int = 12,
) -> Tuple[int, List[int], np.ndarray]:
    """Compare computed artifacts against received vectors (gonka rule).

    Returns (n_mismatch, mismatch_nonces, distances) where distances is the
    raw per-nonce L2 array — useful for histograms / informational stats
    even though gonka's pass/fail uses only `n_mismatch`.
    """
    n_mismatch = 0
    mismatch_nonces: List[int] = []
    distances: List[float] = []
    for artifact in computed_artifacts:
        nonce = artifact["nonce"]
        received_b64 = validation_map.get(nonce)
        if not received_b64:
            continue
        computed_vec = decode_vector(artifact["vector_b64"])
        received_vec = decode_vector(received_b64)
        if received_vec.shape != (k_dim,):
            n_mismatch += 1
            mismatch_nonces.append(nonce)
            distances.append(float("inf"))
            continue
        if not np.all(np.isfinite(received_vec)):
            n_mismatch += 1
            mismatch_nonces.append(nonce)
            distances.append(float("inf"))
            continue
        distance = float(np.linalg.norm(computed_vec - received_vec))
        distances.append(distance)
        if distance > dist_threshold:
            n_mismatch += 1
            mismatch_nonces.append(nonce)
    return n_mismatch, mismatch_nonces, np.array(distances)


# --- Loading + reporting helpers (NOT canonical, just glue) ---

def load_artifacts(path: str) -> List[Dict]:
    """Load artifacts list from a `nonces_*.json` file."""
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    if "artifacts" not in d or not isinstance(d["artifacts"], list):
        raise ValueError(f"{path}: missing 'artifacts' list")
    return d["artifacts"]


def to_validation_map(arts: List[Dict]) -> Dict[int, str]:
    return {a["nonce"]: a["vector_b64"] for a in arts}


def fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"


def report_pair(
    label_a: str,
    arts_a: List[Dict],
    label_b: str,
    arts_b: List[Dict],
    scenarios: List[Tuple[str, float, float]],
    fraud_threshold: float,
    k_dim: int,
) -> Dict:
    """Compute distances + run all scenarios, return a result dict."""
    map_b = to_validation_map(arts_b)
    keys_common = sorted(
        set(a["nonce"] for a in arts_a) & set(map_b.keys()), key=int
    )
    arts_a_filtered = [a for a in arts_a if a["nonce"] in set(keys_common)]
    n_total = len(arts_a_filtered)

    # Distances from baseline scenario (threshold 0 → all distances)
    _, _, distances = validate_artifacts(
        arts_a_filtered, map_b, dist_threshold=0.0, k_dim=k_dim
    )

    print(f"\n{'-'*78}")
    print(f"PAIR: {label_a}")
    print(f"  vs: {label_b}")
    print(f"  N_common = {n_total}, k_dim = {k_dim}")
    if n_total == 0:
        print("  (no overlap, skipping)")
        return {"n_total": 0}

    print(
        f"  L2 stats: mean={distances.mean():.4f}  "
        f"median={np.median(distances):.4f}  max={distances.max():.4f}"
    )
    print(
        f"  L2 percentiles: p25={np.percentile(distances,25):.4f}  "
        f"p75={np.percentile(distances,75):.4f}  "
        f"p95={np.percentile(distances,95):.4f}  "
        f"p99={np.percentile(distances,99):.4f}"
    )
    print()
    print(
        f"  {'scenario':<46} {'thr':>6} {'p_mis':>7} "
        f"{'n_mis':>7} {'rate':>7} {'p_value':>11} {'verdict':>8}"
    )

    scenario_results = []
    for name, thr, p_mis in scenarios:
        n_mis, mis_nonces, _ = validate_artifacts(
            arts_a_filtered, map_b, dist_threshold=thr, k_dim=k_dim
        )
        p_val, fraud = fraud_test(n_mis, n_total, p_mis, fraud_threshold)
        verdict = "FRAUD" if fraud else "PASS"
        rate = n_mis / n_total
        print(
            f"  {name:<46} {thr:>6.3f} {p_mis:>7.3f} "
            f"{n_mis:>4d}/{n_total:<3d} {fmt_pct(rate):>7} "
            f"{p_val:>10.3g} {verdict:>8}"
        )
        scenario_results.append(
            {
                "name": name,
                "dist_threshold": thr,
                "p_mismatch": p_mis,
                "n_mismatch": n_mis,
                "rate": rate,
                "p_value": p_val,
                "fraud_detected": fraud,
            }
        )

    return {
        "n_total": n_total,
        "label_a": label_a,
        "label_b": label_b,
        "stats": {
            "mean": float(distances.mean()),
            "median": float(np.median(distances)),
            "p25": float(np.percentile(distances, 25)),
            "p75": float(np.percentile(distances, 75)),
            "p95": float(np.percentile(distances, 95)),
            "p99": float(np.percentile(distances, 99)),
            "max": float(distances.max()),
        },
        "scenarios": scenario_results,
        "distances": distances.tolist(),
    }


def make_histogram(results: List[Dict], out_path: str) -> None:
    """Optional: render canonical gonka-style histogram with threshold lines."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[WARN] matplotlib not available, skipping {out_path}", file=sys.stderr)
        return

    n = len([r for r in results if r.get("n_total", 0) > 0])
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), squeeze=False)
    for ax, r in zip(axes[:, 0], [r for r in results if r.get("n_total", 0) > 0]):
        d = np.array(r["distances"])
        ax.hist(d, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(d.mean(), color="red", ls="--", label=f"Mean = {d.mean():.4f}")
        ax.axvline(
            np.median(d), color="orange", ls="--",
            label=f"Median = {np.median(d):.4f}",
        )
        ax.axvline(0.02, color="green", ls=":", label="threshold=0.02 (vllm strict)")
        ax.axvline(0.2, color="purple", ls=":", label="threshold=0.2")
        ax.axvline(0.4, color="black", ls=":", label="threshold=0.4 (chain default)")
        ax.set_title(
            f"{r['label_a']}\nvs {r['label_b']}\n"
            f"({len(d)} vectors, dim=12)"
        )
        ax.set_xlabel("L2 Distance")
        ax.set_ylabel("Count")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nHistogram saved: {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Canonical Gonka PoC v2 nonces L2 comparison."
    )
    ap.add_argument(
        "files", nargs="+",
        help="Two or more nonces_*.json files. "
             "All pairwise comparisons are reported.",
    )
    ap.add_argument(
        "--label", action="append", default=[],
        help="Label for each file (repeat in same order). "
             "Defaults to file basenames.",
    )
    ap.add_argument(
        "--scenario", action="append", default=[],
        help="Custom 'name,threshold,p_mismatch' scenario "
             "(may repeat). Overrides defaults if any provided.",
    )
    ap.add_argument(
        "--fraud-threshold", type=float, default=DEFAULT_FRAUD_THRESHOLD,
        help=f"p-value cutoff for fraud_detected "
             f"(default {DEFAULT_FRAUD_THRESHOLD}).",
    )
    ap.add_argument(
        "--k-dim", type=int, default=12,
        help="Expected vector dimension (default 12).",
    )
    ap.add_argument(
        "--json-out", type=str, default=None,
        help="Optional path to dump full structured results as JSON.",
    )
    ap.add_argument(
        "--plot", type=str, default=None,
        help="Optional path to save canonical L2 histogram PNG "
             "(stacked subplots, one per pair).",
    )
    args = ap.parse_args()

    if len(args.files) < 2:
        ap.error("need at least 2 files to compare")

    # Resolve labels
    labels = list(args.label)
    while len(labels) < len(args.files):
        # default label = file basename (no extension)
        import os
        base = os.path.basename(args.files[len(labels)])
        labels.append(os.path.splitext(base)[0])

    # Resolve scenarios
    if args.scenario:
        scenarios = []
        for s in args.scenario:
            parts = s.split(",")
            if len(parts) != 3:
                ap.error(f"--scenario must be 'name,thr,p_mis', got: {s}")
            scenarios.append((parts[0], float(parts[1]), float(parts[2])))
    else:
        scenarios = list(DEFAULT_SCENARIOS)

    print("=" * 78)
    print("CANONICAL GONKA PoC v2 L2 COMPARISON")
    print("=" * 78)
    print(f"Files ({len(args.files)}):")
    for label, path in zip(labels, args.files):
        try:
            arts = load_artifacts(path)
            print(f"  {label:<35}  {path}  ({len(arts)} nonces)")
        except Exception as exc:
            print(f"  {label:<35}  {path}  ERROR: {exc}")
            return 1

    print(f"\nScenarios ({len(scenarios)}):")
    for n, t, p in scenarios:
        print(f"  • {n}: thr={t}, p_mis={p}")

    # Load all
    loaded = {}
    for label, path in zip(labels, args.files):
        loaded[label] = load_artifacts(path)

    # Pairwise
    from itertools import combinations
    results = []
    for la, lb in combinations(labels, 2):
        r = report_pair(
            la, loaded[la],
            lb, loaded[lb],
            scenarios=scenarios,
            fraud_threshold=args.fraud_threshold,
            k_dim=args.k_dim,
        )
        results.append(r)

    if args.json_out:
        # Strip distances from JSON dump unless small (keep file readable)
        slim = []
        for r in results:
            if not r.get("n_total"):
                slim.append(r)
                continue
            slim.append({k: v for k, v in r.items() if k != "distances"})
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(
                {"results": slim, "scenarios": scenarios},
                fh, indent=2, default=str,
            )
        print(f"\nJSON results saved: {args.json_out}")

    if args.plot:
        make_histogram(results, args.plot)

    print("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
