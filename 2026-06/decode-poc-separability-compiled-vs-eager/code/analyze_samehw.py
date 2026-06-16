#!/usr/bin/env python3
# Same-hardware config-coupling analysis from the three prover reference
# trajectories (ref_cc / ref_ee / ref_ec). This is the FLOOR of how much the
# compile config alone perturbs the seal, measured on one GPU before any
# cross-hardware or honest-vs-fraud comparison.
#
# Key trick: ref_ee and ref_ec share an identical EAGER prefill, so they enter
# decode step 1 from a bit-identical state. Their step-1 disagreement is therefore
# a clean compiled-vs-eager DECODE signal with no cascade. ref_cc vs ref_ee step 0
# is the compiled-vs-eager PREFILL signal. genref is free-run, so anything after
# the first disagreement cascades — only the clean steps above are interpretable.
#
# Usage: python3 analyze_samehw.py <dir with ref_cc.json ref_ee.json ref_ec.json>
import json, os, sys
from collections import Counter

D = sys.argv[1] if len(sys.argv) > 1 else "."
L = {c: json.load(open(os.path.join(D, f"ref_{c}.json"))) for c in ("cc", "ee", "ec")}
ks = sorted(L["cc"], key=int)
N = len(ks)
S = len(L["cc"][ks[0]])


def agree_at(a, b, s):
    return sum(1 for k in ks if a[k][s] == b[k][s]) / N


def first_div(a, b):
    return sum(next((i for i in range(S) if a[k][i] != b[k][i]), S) for k in ks) / N


print(f"nonces={N} steps={S}  (k in [0,16))")
print("\n=== determinism / non-degeneracy ===")
for c in ("cc", "ee", "ec"):
    s0 = [L[c][k][0] for k in ks]
    allk = [x for k in ks for x in L[c][k]]
    print(f"  [{c}] step0={s0} distinct={len(set(s0))}/{N}  traj-cells={len(set(allk))}/16")
print(f"  ee vs ec step0 agree = {agree_at(L['ee'], L['ec'], 0):.3f}  (must be 1.000: shared eager prefill)")

print("\n=== same-hw compiled-vs-eager coupling ===")
print(f"  PREFILL  cc vs ee  step0 disagree = {1 - agree_at(L['cc'], L['ee'], 0):.3f}")
print(f"  DECODE   ee vs ec  step1 disagree = {1 - agree_at(L['ee'], L['ec'], 1):.3f}  (clean, no cascade)")
print(f"  DECODE   ee vs ec  step2 disagree = {1 - agree_at(L['ee'], L['ec'], 2):.3f}  (cascade onset)")
print(f"\n  cascade onset (mean first diverging step):")
print(f"    cc~ee={first_div(L['cc'], L['ee']):.2f}  cc~ec={first_div(L['cc'], L['ec']):.2f}  ee~ec={first_div(L['ee'], L['ec']):.2f}")
