"""Fit and evaluate length-normalizations f(N) for the replay distance.

Reproduces every number in ../README.md from the inference-validation
artifacts. The honest curve is summarized as per-log-bin medians; candidate
f(N) forms are fitted to those bins in log space; each candidate is scored
by TP at FP=5% (threshold = 95th percentile of honest R = D/f(N)).

Inputs (LFS): the exp2-raw artifacts of the 0731 campaign (fit + score) and
the July V4-Flash campaign (independent transfer check, same hardware pair).

Usage:
    python fit_normalization.py \
        --fit-honest  .../0731/exp2-raw-logprobs/artifacts/honest_b300_h200.jsonl \
        --fit-fraud   .../0731/exp2-raw-logprobs/artifacts/fraud_nvfp4_b300_h200.jsonl \
        --val-honest  .../july/exp2-raw-logprobs/artifacts/honest_b300_h200.jsonl \
        --val-fraud   .../july/exp2-raw-logprobs/artifacts/fraud_nvfp4_b300_h200.jsonl
"""

import argparse
import json

import numpy as np
from scipy.optimize import curve_fit, minimize

LOOKUP5_KNOTS = np.array([5.0, 25.0, 120.0, 500.0, 1800.0])


def load(path):
    D, N = [], []
    for line in open(path):
        r = json.loads(line)
        if r.get("len_mismatch") or r.get("n_tokens", 0) < 2 or "distance" not in r:
            continue
        D.append(r["distance"])
        N.append(r["n_tokens"])
    return np.array(D), np.array(N)


def bin_curve(D, N, nbins=25, min_per_bin=8):
    """Per-log-bin medians of the honest cloud — the empirical f(N)."""
    bins = np.logspace(np.log10(2), np.log10(N.max() + 1), nbins)
    bx, by = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (N >= lo) & (N < hi)
        if m.sum() >= min_per_bin:
            bx.append(np.median(N[m]))
            by.append(np.median(D[m]))
    return np.array(bx), np.array(by)


def tp_at_fp(f, Dh, Nh, Df, Nf, fp=0.05):
    thr = np.quantile(Dh / f(Nh), 1 - fp)
    return float(np.mean(Df / f(Nf) > thr)), float(thr)


def log_fit(fn, bx, by, p0):
    p, _ = curve_fit(
        lambda N, *a: np.log(np.clip(fn(N, *a), 1e-9, None)),
        bx, np.log(by), p0=p0, maxfev=300000,
    )
    return list(p)


def make_lookup(knots, values):
    v = np.clip(np.asarray(values, dtype=float), 1e-5, None)
    return lambda N: np.exp(
        np.interp(np.log(np.clip(N, 2, 4000)), np.log(knots), np.log(v))
    )


# Candidate forms, keyed by the names used in the README.
FN4 = lambda N, A, l, B, d: A * (1 - np.exp(-N / l)) - B * np.log(N) + d
FN5 = lambda N, A, l, B, C, d: (
    A * (1 - np.exp(-N / l)) - B * np.log(N) - C * np.log(N) ** 2 + d
)
FN6 = lambda N, A, m, s, E, l, d: (
    A * np.exp(-((np.log(N) - m) ** 2) / (2 * s**2))
    + E * (1 - np.exp(-N / l))
    + d
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-honest", required=True)
    ap.add_argument("--fit-fraud", required=True)
    ap.add_argument("--val-honest")
    ap.add_argument("--val-fraud")
    args = ap.parse_args()

    Dh, Nh = load(args.fit_honest)
    Df, Nf = load(args.fit_fraud)
    bx, by = bin_curve(Dh, Nh)

    candidates = {}
    candidates["0p const"] = lambda N: np.full_like(
        np.asarray(N, float), np.median(Dh)
    )
    p4 = log_fit(FN4, bx, by, [0.074, 56, 0.01, 0.02])
    candidates["4p sat-lnN"] = lambda N: np.clip(FN4(N, *p4), 1e-4, None)
    p5 = log_fit(FN5, bx, by, [0.074, 56, 0.01, 0.0, 0.02])
    candidates["5p sat-lnN-quad"] = lambda N: np.clip(FN5(N, *p5), 1e-4, None)
    p6 = log_fit(FN6, bx, by, [0.02, 5.3, 1.0, 0.02, 30, 0.004])
    candidates["6p lognorm+sat"] = lambda N: np.clip(FN6(N, *p6), 1e-4, None)

    v0 = np.exp(np.interp(np.log(LOOKUP5_KNOTS), np.log(bx), np.log(by)))
    res = minimize(
        lambda v: np.mean(
            (np.log(make_lookup(LOOKUP5_KNOTS, v)(bx)) - np.log(by)) ** 2
        ),
        v0, method="Nelder-Mead", options={"maxiter": 8000},
    )
    candidates["lookup-5"] = make_lookup(LOOKUP5_KNOTS, res.x)
    candidates["lookup-20 (ceiling)"] = make_lookup(bx, by)

    print(f"{'candidate':<22}{'TP@FP5% (fit set)':>18}", end="")
    has_val = args.val_honest and args.val_fraud
    if has_val:
        DhV, NhV = load(args.val_honest)
        DfV, NfV = load(args.val_fraud)
        print(f"{'TP (val set)':>14}")
    else:
        print()
    for name, f in candidates.items():
        tp, _ = tp_at_fp(f, Dh, Nh, Df, Nf)
        line = f"{name:<22}{tp * 100:>17.1f}%"
        if has_val:
            tpv, _ = tp_at_fp(f, DhV, NhV, DfV, NfV)
            line += f"{tpv * 100:>13.1f}%"
        print(line)

    print("\nfitted parameters:")
    print("  4p:", np.round(p4, 5).tolist())
    print("  5p:", np.round(p5, 5).tolist())
    print("  6p:", np.round(p6, 5).tolist())
    print("  lookup-5 knots:", LOOKUP5_KNOTS.tolist(),
          "values:", np.round(res.x, 5).tolist())


if __name__ == "__main__":
    main()
