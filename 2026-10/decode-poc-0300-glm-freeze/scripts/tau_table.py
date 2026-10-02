#!/usr/bin/env python3
"""Separability tables for GLM-5.3-Flash, rebuilt from the validation npz files in artifacts/ (no GPU).

Layout: artifacts/validations/<counters>/validator_<card>/prover_<card>/npz_<honest|fraud>_hNN_v200.npz, 16 cells.
  glm          snap margin: counts[i, n] is the number of mismatching points of nonce n whose snap margin (top1 - top2
               of the validator's query) is >= taus[i]; steps: the points compared per nonce (257: the prompt position
               and 256 decode steps).
  glm-claimed  claimed margin: counts[i, n] is 1 when nonce n has a mismatching point and the largest claimed margin
               among its mismatching points (top1 - score of the reflection vector the prover claimed; the plugin's
               mismatch_margin_max) is > taus[i], else 0.
A nonce is flagged at tau when counts[i, n] >= 1.

Points %: mismatching points / all points (snap counters), mean over the ten hashes [min-max].
Room by points: per validator, the highest honest hash and the lowest fraud hash at tau = 0 over every prover validated
there; gap and ratio are computed from the unrounded values.
Nonces %: share of nonces flagged, mean over hashes. Groups by hardware family of the pair: Hopper (H200, H100) and
Blackwell (B300, B200).
Threshold table: partitions of 200 nonces, alpha 1%; p_mismatch* is the smallest value on the grid 0.001 ... 0.994
(step 0.001) at which the worst honest hash of all cells passes with probability >= 99%; power is against the lowest
fraud hash.
Run from the experiment folder: python3 scripts/tau_table.py > artifacts/tau_matrix.md
"""
import os
from functools import lru_cache
from math import comb
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = os.path.join(ROOT, 'artifacts', 'validations')
TREES = {'claimed': 'glm-claimed', 'snap': 'glm'}
RULE = {'claimed': 'claimed margin (top1 - score of the claimed reflection vector) > tau, the rule in the image\'s plugin; counters in glm-claimed/',
        'snap': 'snap margin (top1 - top2) >= tau; counters in glm/'}
CARDS = ['b300', 'h200', 'h100', 'b200']
NAME = {c: c.upper() for c in CARDS}
HOPPER = {'h200', 'h100'}
HASHES = [f'h{i:02d}' for i in range(1, 11)]
TAUS = [0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040]
GROUP_TAUS = [0.010, 0.015, 0.020, 0.025, 0.030, 0.035]
PART, ALPHA = 200, 0.01
P_GRID = [round(0.001 * i, 3) for i in range(1, 995)]
CELLS = [(p, v) for v in CARDS for p in CARDS]
GROUPS = [('hopper', 'H200 · H100', 'honest, Hopper ↔ Hopper (4 cells, 2 of them self-validation)'),
          ('blackwell', 'B300 · B200', 'honest, Blackwell ↔ Blackwell (4 cells, 2 of them self-validation)'),
          ('mixed', 'all four', 'honest, Hopper ↔ Blackwell (8 cells)')]

_cache = {}


def corpus(tree, v, p, role, h):
    key = (tree, v, p, role, h)
    if key not in _cache:
        z = np.load(os.path.join(V, TREES[tree], f'validator_{v}', f'prover_{p}', f'npz_{role}_{h}_v200.npz'))
        _cache[key] = (z['taus'], z['counts'], int(z['steps']))
    return _cache[key]


def ti(taus, t): return int(np.argmin(np.abs(taus - t)))


def points(v, p, role, h, tau=0.0):
    taus, c, steps = corpus('snap', v, p, role, h)
    return 100.0 * c[ti(taus, tau)].sum() / (steps * c.shape[1])


def flagged(tree, v, p, role, h, tau):
    taus, c, _ = corpus(tree, v, p, role, h)
    return 100.0 * float((c[ti(taus, tau)] >= 1).mean())


def group_of(p, v):
    if p in HOPPER and v in HOPPER: return 'hopper'
    if p not in HOPPER and v not in HOPPER: return 'blackwell'
    return 'mixed'


@lru_cache(maxsize=None)
def cdf(k, n, q): return sum(comb(n, i) * q ** i * (1 - q) ** (n - i) for i in range(k + 1))


@lru_cache(maxsize=None)
def crit(n, q, alpha):
    for k in range(n + 1):
        if 1 - cdf(k - 1, n, q) <= alpha: return k
    return n + 1


def rng(xs):
    lo, hi = f'{min(xs):.2f}', f'{max(xs):.2f}'
    return lo if lo == hi else f'{lo}–{hi}'


P = print
P('# GLM-5.3-Flash: separability tables\n')
P('Output of `scripts/tau_table.py` on the counters in `artifacts/validations/`. Validators B300, H200, H100, B200 '
  '(TP 2, 4, 8, 4); provers the same four; a cell is one prover\'s corpora (ten per arm: h01-h10, 250 nonces each) '
  'validated on one validator; prover = validator is self-validation on another boot.\n')

# ---- points
P('## Points at tau = 0, %, mean over ten hashes [min-max]; validators in columns\n')
P('| prover -> validator | ' + ' | '.join(NAME[c] for c in CARDS) + ' |'); P('|---|' + '---:|' * len(CARDS))
for p in CARDS:
    row = []
    for v in CARDS:
        x = [points(v, p, 'honest', h) for h in HASHES]
        row.append(f'{np.mean(x):.2f} [{min(x):.2f}-{max(x):.2f}]' + (' *' if p == v else ''))
    P(f'| honest {NAME[p]} | ' + ' | '.join(row) + ' |')
row = []
for v in CARDS:
    x = [points(v, p, 'fraud', h) for p in CARDS for h in HASHES]
    row.append(f'{np.mean(x):.2f} [{min(x):.2f}-{max(x):.2f}]')
P('| fraud W4A16 (four provers) | ' + ' | '.join(row) + ' |')
P('\n\\* self-validation (the same configuration validating its own corpora on another boot)\n')

P('## Room by points at tau = 0, per validator (every prover validated there)\n')
P('| validator | highest honest | lowest fraud | worst-case gap | ratio |'); P('|---|---:|---:|---:|---:|')
his, los, ratios = [], [], []
for v in CARDS:
    hv, hp, hh = max((points(v, p, 'honest', h), p, h) for p in CARDS for h in HASHES)
    fv, fp, fh = min((points(v, p, 'fraud', h), p, h) for p in CARDS for h in HASHES)
    his.append(hv); los.append(fv); ratios.append(fv / hv)
    P(f'| {NAME[v]} | {hv:.2f}% (prover {NAME[hp]}{", self-validation" if hp == v else ""}, {hh}) | '
      f'{fv:.2f}% (prover {NAME[fp]}{", self-validation" if fp == v else ""}, {fh}) | {fv - hv:.2f} pp | {fv / hv:.3f} |')
P(f'\nAcross the four validators: between {max(his):.2f}% and {min(los):.2f}%; ratio {min(ratios):.3f}-{max(ratios):.3f}.\n')

P('## Points by snap margin, %, mean over the cells and hashes of each group\n')
P('| prover set | ' + ' | '.join(f'tau {t:.3f}' for t in [0.0] + TAUS) + ' |'); P('|---|' + '---:|' * (len(TAUS) + 1))
for key, _, lab in GROUPS:
    P(f'| {lab} | ' + ' | '.join(f'{np.mean([points(v, p, "honest", h, t) for p, v in CELLS if group_of(p, v) == key for h in HASHES]):.5f}'
                                for t in [0.0] + TAUS) + ' |')
P('| fraud W4A16 (16 cells) | ' + ' | '.join(f'{np.mean([points(v, p, "fraud", h, t) for p, v in CELLS for h in HASHES]):.5f}'
                                          for t in [0.0] + TAUS) + ' |')

# ---- nonces, both statistics
for tree in ('claimed', 'snap'):
    hon_all = lambda t: [flagged(tree, v, p, 'honest', h, t) for p, v in CELLS for h in HASHES]
    fr_all = lambda t: [flagged(tree, v, p, 'fraud', h, t) for p, v in CELLS for h in HASHES]
    P(f'\n## Nonces flagged, {RULE[tree]}\n')
    P('### By group: range of cell means over the cells of the row, %; [worst honest hash / lowest fraud hash]\n')
    P('| validator | prover set | ' + ' | '.join(f'tau {t:.3f}' for t in GROUP_TAUS) + ' |'); P('|---|---|' + '---:|' * len(GROUP_TAUS))
    for key, vl, lab in GROUPS:
        ks = [(p, v) for p, v in CELLS if group_of(p, v) == key]
        P(f'| {vl} | {lab} | ' + ' | '.join(
            f'{rng([np.mean([flagged(tree, v, p, "honest", h, t) for h in HASHES]) for p, v in ks])} '
            f'[{max(flagged(tree, v, p, "honest", h, t) for p, v in ks for h in HASHES):.1f}]' for t in GROUP_TAUS) + ' |')
    P('| all four | fraud W4A16 (16 cells) | ' + ' | '.join(
        f'{rng([np.mean([flagged(tree, v, p, "fraud", h, t) for h in HASHES]) for p, v in CELLS])} '
        f'[{min(flagged(tree, v, p, "fraud", h, t) for p, v in CELLS for h in HASHES):.1f}]' for t in GROUP_TAUS) + ' |')
    P(f'\nAt tau = 0 every one of the 160 honest validations (40 corpora on four validators) is {min(hon_all(0.0)):.1f}% flagged.')
    P('Fraud / honest, all 16 cells: ' + '; '.join(
        f'tau {t:.3f}: {np.mean(fr_all(t)) / np.mean(hon_all(t)):.1f} by means, {np.mean(fr_all(t)) / max(hon_all(t)):.1f} against the worst honest hash'
        for t in (0.015, 0.020)) + '.')

    P('\n### By cell: mean over hashes, %; [worst honest hash / lowest fraud hash]\n')
    P('| cell | ' + ' | '.join(f'tau {t:.3f}' for t in TAUS) + ' |'); P('|---|' + '---:|' * len(TAUS))
    for role in ('honest', 'fraud'):
        for p, v in CELLS:
            s = np.array([[flagged(tree, v, p, role, h, t) for t in TAUS] for h in HASHES])
            ext = s.max(0) if role == 'honest' else s.min(0)
            P(f'| {role} {NAME[p]} -> {NAME[v]}{" (self-validation)" if p == v else ""} | ' +
              ' | '.join(f'{s[:, i].mean():.2f} [{ext[i]:.1f}]' for i in range(len(TAUS))) + ' |')

    P(f'\n### One (tau, p_mismatch) for the fleet: partitions of {PART}, alpha {ALPHA}, all 16 cells\n')
    P('| tau | honest: mean / worst hash | fraud: mean / lowest hash | p_mismatch* | reject at >= n of 200 | power vs lowest fraud hash |')
    P('|---:|---|---|---:|---:|---:|')
    for t in TAUS:
        hon = [x / 100 for x in hon_all(t)]; fr = [x / 100 for x in fr_all(t)]
        hw, fb = max(hon), min(fr); pstar = None
        for cand in P_GRID:
            k = crit(PART, cand, ALPHA)
            if 1 - cdf(k - 1, PART, hw) <= ALPHA: pstar = cand; break
        head = f'| {t:.3f} | {100 * np.mean(hon):.2f} / {100 * hw:.1f}% | {100 * np.mean(fr):.2f} / {100 * fb:.1f}% |'
        if pstar is None: P(head + ' - | - | - |'); continue
        k = crit(PART, pstar, ALPHA); power = 1 - cdf(k - 1, PART, fb)
        P(head + f' {pstar:.3f}{" (grid floor)" if pstar == P_GRID[0] else ""} | {k} | {100 * power:.1f}% |')
P(f'\n{P_GRID[0]:.3f} is the lowest p_mismatch on the grid of the rule: where it appears, the worst honest hash passes at the grid floor.')
