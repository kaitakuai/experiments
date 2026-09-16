#!/usr/bin/env python3
"""Separability tables for GLM-5.3-Flash, rebuilt from the validation npz files in artifacts/ (no GPU).

Layout: artifacts/validations/glm/validator_<card>/prover_<card>/npz_<honest|fraud>_hNN_v200.npz.
An npz holds counts[i, n]: the number of mismatching points of nonce n at taus[i]; steps: the
points compared per nonce (257: the prompt position and 256 decode steps).

Points %: mismatching points / all points, mean over the ten hashes [min-max].
Nonces %: share of nonces with at least one mismatching point whose snap margin exceeds tau
(the chain statistic), mean over hashes; worst honest hash / lowest fraud hash in brackets.
Threshold table: partitions of 200 nonces; p_mismatch* is the smallest value at which the worst
honest hash passes with probability >= 99%; power is against the lowest fraud hash.
Run from the experiment folder: python3 scripts/tau_table.py > artifacts/tau_matrix.md
"""
import glob, os, numpy as np
from math import comb
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = os.path.join(ROOT, 'artifacts', 'validations', 'glm')
CARDS = ['b300', 'h200', 'h100', 'b200']
TAUS = [0.0, 0.01, 0.02, 0.025, 0.03, 0.04, 0.05]
PART, ALPHA = 200, 0.01

def load(v, p, role):
    P, N, F = [], [], []
    for fn in sorted(glob.glob(os.path.join(V, f'validator_{v}', f'prover_{p}', f'npz_{role}_h*_v200.npz'))):
        z = np.load(fn); c = z['counts']; steps = int(z['steps']); t = z['taus']
        idx = [int(np.argmin(np.abs(t - x))) for x in TAUS]
        P.append([100 * c[i].sum() / (steps * c.shape[1]) for i in idx])
        N.append([100 * float((c[i] >= 1).mean()) for i in idx])
        F.append([(c[i] >= 1) for i in idx])
    return np.array(P), np.array(N), F

cells = {}
for v in CARDS:
    for p in CARDS:
        P, N, F = load(v, p, 'honest')
        if len(P): cells[('honest', p, v)] = (P, N, F)
        P, N, F = load(v, p, 'fraud')
        if len(P): cells[('fraud', p, v)] = (P, N, F)

def i_of(x): return TAUS.index(x)
print('## Points at tau = 0, %, mean over ten hashes [min-max]; validators in columns\n')
print('| prover -> validator | ' + ' | '.join(c.upper() for c in CARDS) + ' |'); print('|---|' + '---:|' * len(CARDS))
for p in CARDS:
    row = []
    for v in CARDS:
        k = ('honest', p, v)
        if k not in cells: row.append('-'); continue
        P = cells[k][0][:, 0]; row.append(f'{P.mean():.2f} [{P.min():.2f}-{P.max():.2f}]' + (' *' if p == v else ''))
    print(f'| honest {p.upper()} | ' + ' | '.join(row) + ' |')
row = []
for v in CARDS:
    ps = [cells[('fraud', p, v)][0][:, 0] for p in CARDS if ('fraud', p, v) in cells]
    allv = np.concatenate(ps); row.append(f'{allv.mean():.2f} [{allv.min():.2f}-{allv.max():.2f}]' + ('' if len(ps) == 4 else f' ({len(ps)} provers)'))
print('| fraud (all provers) | ' + ' | '.join(row) + ' |')
print('\n\\* self-validation (another boot; H200: the validator on a second instance of the same node)\n')

print('## Nonces flagged by the chain statistic (a nonce is flagged if one mismatching point has a snap margin > tau), %, mean over hashes [worst hash]\n')
print('| cell | ' + ' | '.join(f'tau {t}' for t in TAUS[1:]) + ' |'); print('|---|' + '---:|' * (len(TAUS) - 1))
for (role, p, v), (P, N, F) in sorted(cells.items(), key=lambda kv: (kv[0][0] != 'honest', CARDS.index(kv[0][2]), CARDS.index(kv[0][1]))):
    ext = N.max(axis=0) if role == 'honest' else N.min(axis=0)
    print(f'| {role} {p.upper()} -> {v.upper()}{" (self, other boot)" if p == v else ""} | ' + ' | '.join(f'{N[:, i_of(t)].mean():.1f} [{ext[i_of(t)]:.1f}]' for t in TAUS[1:]) + ' |')

def binom_cdf(k, n, q): return sum(comb(n, i) * q ** i * (1 - q) ** (n - i) for i in range(k + 1))
def crit(n, q, alpha):
    for k in range(n + 1):
        if 1 - binom_cdf(k - 1, n, q) <= alpha: return k
    return n + 1
print(f'\n## One (tau, p_mismatch) for the fleet: partitions of {PART}, alpha {ALPHA}, against the worst honest hash and the lowest fraud hash of all cells\n')
print('| tau | honest: mean / worst hash | fraud: mean / lowest hash | p_mismatch* | reject at >= n of 200 | power vs lowest fraud hash |'); print('|---:|---|---|---:|---:|---:|')
for t in TAUS[1:]:
    i = i_of(t)
    hon = [f[i].mean() for (role, p, v), (P, N, F) in cells.items() if role == 'honest' for f in F]
    fr = [f[i].mean() for (role, p, v), (P, N, F) in cells.items() if role == 'fraud' for f in F]
    hw, fb = max(hon), min(fr); pstar = None
    for cand in np.arange(0.005, 0.995, 0.005):
        k = crit(PART, cand, ALPHA)
        if 1 - binom_cdf(k - 1, PART, hw) <= ALPHA: pstar = cand; break
    if pstar is None: print(f'| {t} | {100*np.mean(hon):.1f} / {100*hw:.1f} | {100*np.mean(fr):.1f} / {100*fb:.1f} | - | - | - |'); continue
    k = crit(PART, pstar, ALPHA); power = 1 - binom_cdf(k - 1, PART, fb)
    print(f'| {t} | {100*np.mean(hon):.1f} / {100*hw:.1f} | {100*np.mean(fr):.1f} / {100*fb:.1f} | {pstar:.3f} | {k} | {100*power:.0f}% |')
