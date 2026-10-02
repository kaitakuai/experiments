"""One (tau, p_mismatch) pair for a set of cells by the chain rule, from validation npz counters.

A nonce is flagged at tau when its counter at tau is >= 1. Snap-margin counters (r2_validate.py; artifacts/validations/glm/
in the report, data/validations/ in the devkit) count mismatching points whose snap margin is >= tau; claimed-margin
counters (artifacts/validations/glm-claimed/, data/derived/npz_claimed/) hold 1 when the largest claimed margin of the
nonce is > tau, the rule in the image's plugin. A partition passes or fails by the binomial test of the number of
flagged nonces against p_mismatch.
For every tau of the counters' grid: the share of flagged nonces over the honest cells (mean and worst hash) and over the
fraud cells (mean and lowest hash); then, for a partition of n nonces and level alpha, the smallest p_mismatch on the
grid 0.001 ... 0.994 (step 0.001) at which the worst honest hash passes with probability >= 1-alpha, the rejection
count, and the power against the lowest fraud hash. The last line gives the taus of the grid with power >= 99% and
p_mismatch* <= 0.010.
Run: python3 tau_fleet.py [--part=200] [--alpha=0.01] label=dir [label=dir ...]"""
import sys, glob, os, numpy as np
from math import comb
args=[a for a in sys.argv[1:] if '=' in a and not a.startswith('--')]
opt={a.split('=')[0]:a.split('=')[1] for a in sys.argv[1:] if a.startswith('--')}
PART=int(opt.get('--part',200)); ALPHA=float(opt.get('--alpha',0.01))
P_GRID=[round(0.001*i,3) for i in range(1,995)]
cells=dict(a.split('=',1) for a in args)
data={}; taus=None
for lbl,d in cells.items():
    for f in sorted(glob.glob(os.path.join(d,'npz_*_v*.npz'))):
        z=np.load(f, allow_pickle=True); name=os.path.basename(f)[4:]; role=name.split('_')[0]; h=name.split('_')[1]
        c=z['counts']; c=c if c.ndim==2 else c[None,:]
        data[(lbl,role,h)]=(c>=1)          # nonce flagged at each tau
        taus=z['taus'] if taus is None else taus
_cdf={}
def binom_cdf(k,n,p):
    if (k,n,p) not in _cdf: _cdf[(k,n,p)]=sum(comb(n,i)*p**i*(1-p)**(n-i) for i in range(k+1))
    return _cdf[(k,n,p)]
_crit={}
def crit(n,p,alpha):
    # smallest k with P(X>=k | p) <= alpha: the partition rejection count
    if (n,p,alpha) not in _crit:
        _crit[(n,p,alpha)]=next((k for k in range(n+1) if 1-binom_cdf(k-1,n,p) <= alpha), n+1)
    return _crit[(n,p,alpha)]
print(f"cells {len(cells)}, corpora {len(data)}, partition {PART}, alpha={ALPHA}")
print("tau    | honest: mean / worst hash     | fraud: mean / lowest hash     | p_mismatch* | reject   | power vs lowest fraud hash")
window=[]
for ti,tau in enumerate(taus):
    if tau>0.101: break
    hon=[m[ti].mean() for (l,r,h),m in data.items() if r=='honest']; fr=[m[ti].mean() for (l,r,h),m in data.items() if r=='fraud']
    if not hon or not fr: continue
    hw=max(hon); fb=min(fr)
    # p_mismatch*: smallest p at which the worst honest hash (share hw) is rejected no more often than alpha
    p=None
    for cand in P_GRID:
        k=crit(PART,cand,ALPHA)
        if 1-binom_cdf(k-1,PART,hw) <= ALPHA: p=cand; break
    if p is None: print(f"{tau:.3f}  | {100*np.mean(hon):5.1f} / {100*hw:5.1f}            | {100*np.mean(fr):5.1f} / {100*fb:5.1f}         | —"); continue
    k=crit(PART,p,ALPHA); power=1-binom_cdf(k-1,PART,fb) if k<=PART else 0.0
    print(f"{tau:.3f}  | {100*np.mean(hon):5.1f} / {100*hw:5.1f}            | {100*np.mean(fr):5.1f} / {100*fb:5.1f}         | {p:.3f}       | {k:3d}/{PART}  | {100*power:5.1f} %")
    if k<=PART and power>=0.99 and p<=0.0101: window.append(float(tau))
print("\ntaus of this grid with power >= 99% and p_mismatch* <= 0.010: " + (f"{min(window):.3f}-{max(window):.3f} ({len(window)} grid values)" if window else "none"))
