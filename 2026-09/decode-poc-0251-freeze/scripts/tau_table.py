#!/usr/bin/env python3
"""Nonce-level tau table for MiniMax, rebuilt from the validation npz files in the devkit (no GPU).

Cell: one prover set validated on one validator. Columns: points % at tau = 0 (mean over hashes
[min-max]); share of nonces with at least one disagreement at tau 0.02 / 0.025 / 0.04 / 0.05
(mean over hashes). counts[i, n] in an npz is the number of disagreeing steps of nonce n at taus[i];
total_points is the number of compared steps of the corpus.

Fraud on H200, H100 and A100: the npz files hold the decayed corpus of 7 September with every nonce;
the report's cells use its nonces without NaN steps (h01-h04), taken here from artifacts/fig_data_*.json.
Run from the experiment folder: python3 scripts/tau_table.py > artifacts/tau_matrix.md
"""
import glob, json, os, re, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = os.path.join(ROOT, 'artifacts', 'validations', 'minimax')
TAUS = [0.02, 0.025, 0.04, 0.05]
M = json.load(open(os.path.join(ROOT, 'artifacts', 'fig_data_matrix.json')))
T = json.load(open(os.path.join(ROOT, 'artifacts', 'fig_data_tau.json')))

def cell(pattern):
    pts, shares = [], {t: [] for t in TAUS}
    for fn in sorted(glob.glob(pattern)):
        d = np.load(fn); taus = d['taus']; c = d['counts']
        pts.append(100.0 * c[0].sum() / max(int(d['total_points']), 1))
        for t in TAUS:
            i = int(np.argmin(np.abs(taus - t))); shares[t].append(100.0 * float((c[i] > 0).mean()))
    if not pts: return None
    return (np.mean(pts), min(pts), max(pts), [np.mean(shares[t]) for t in TAUS], len(pts))

def row(label, r, note=''):
    if r is None: return f'| {label} | — | — | — | — | — | {note} |'
    m, lo, hi, sh, n = r
    return f'| {label} | {m:.2f} [{lo:.2f}–{hi:.2f}] | ' + ' | '.join(f'{x:.0f} %' for x in sh) + f' | {note or f"{n} hashes"} |'

print('# MiniMax-M2.7: nonce-level tau table (rebuilt from the npz counters in artifacts/)\n')
print('Points % at tau = 0: mean over hashes [min–max]. Nonce columns: share of nonces with at least one')
print('disagreement whose margin exceeds tau, mean over hashes. Corpora: 250 nonces per block hash.\n')
print('| validator | prover set | points %, tau = 0 | tau 0.02 | 0.025 | 0.04 | 0.05 | note |')
print('| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |')
for val in ['B300', 'H200', 'H100', 'A100']:
    d = os.path.join(V, f'validator_{val.lower()}')
    print(row(f'{val} | honest, same boot', cell(f'{d}/npz_honest_h*_v250.npz')))
    if val == 'B300':
        print(row('B300 | honest, same card, other boot and plugin build', cell(f'{d}/old_goldens_0907_to_new_code/npz_honest_h*_v250.npz'), 'reference artifacts of 7 Sep on this code'))
    for prov in ['B300', 'H200', 'H100', 'A100']:
        if prov == val: continue
        print(row(f'{val} | honest {prov}', cell(f'{d}/npz_{prov.lower()}_honest_h*_v250.npz')))
    if val == 'B300':
        print(row('B300 | fraud QuantTrio, one boot per hash (10 Sep)', cell(f'{d}/fraud_0910/npz_fraud_h*_v250.npz'), '26 of 2,500 nonces carry a NaN step'))
    print(row(f'{val} | fraud QuantTrio, decayed corpus (7 Sep), every nonce', cell(f'{d}/npz_fraud_h*_v250.npz'), 'later hashes are NaN chains; not used in the artifact'))
    if val != 'B300':
        f = M[val]['fraud']; taus = T['taus']; fr = T[val]['fraud']
        sh = [fr[int(np.argmin(np.abs(np.array(taus) - t)))] for t in TAUS]
        print(f"| {val} | fraud QuantTrio, nonces without NaN steps (h01–h04, 616 nonces) | {f['mean']:.2f} [{f['min']:.2f}–{f['max']:.2f}] | " + ' | '.join(f'{x:.0f} %' for x in sh) + ' | the cell used in the artifact (docs/fig_data_*.json) |')
print('\nSource files: `data/validations/minimax/validator_<card>/npz_<prover>_honest_hNN_v250.npz`,')
print('`npz_honest_hNN_v250.npz` (same boot), `validator_b300/fraud_0910/`, `validator_b300/old_goldens_0907_to_new_code/`.')
