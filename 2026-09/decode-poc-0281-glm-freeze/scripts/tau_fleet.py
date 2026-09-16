"""One (tau, p_mismatch) pair for the fleet by the chain statistic (gonka-ai/gonka#1743, gonka-ai/gonka-vllm-plugins#8),
from the validation npz files of several cells: a nonce is flagged if at least one of its mismatching points has a snap
margin > tau; a partition passes or fails by the binomial test of the number of flagged nonces against p_mismatch.
For every tau of the grid: the share of flagged nonces over the honest cells (mean and worst hash) and over the fraud
cells (mean and lowest hash); then, for a partition of n nonces and level alpha, the smallest p_mismatch at which the
worst honest hash passes with probability >= 1-alpha, and the power against the lowest fraud hash.
Run: python3 scripts/tau_fleet.py [--part=200] [--alpha=0.01] label=dir [label=dir ...]"""
import sys, glob, os, numpy as np
from math import comb
args=[a for a in sys.argv[1:] if '=' in a and not a.startswith('--')]
opt={a.split('=')[0]:a.split('=')[1] for a in sys.argv[1:] if a.startswith('--')}
PART=int(opt.get('--part',200)); ALPHA=float(opt.get('--alpha',0.01))
cells=dict(a.split('=',1) for a in args)
data={}; taus=None
for lbl,d in cells.items():
    for f in sorted(glob.glob(os.path.join(d,'npz_*_v*.npz'))):
        z=np.load(f, allow_pickle=True); name=os.path.basename(f)[4:]; role=name.split('_')[0]; h=name.split('_')[1]
        c=z['counts']; c=c if c.ndim==2 else c[None,:]
        data[(lbl,role,h)]=(c>=1)          # nonce flagged at each tau
        taus=z['taus'] if taus is None else taus
def binom_cdf(k,n,p): return sum(comb(n,i)*p**i*(1-p)**(n-i) for i in range(k+1))
def crit(n,p,alpha):
    # smallest k with P(X>=k | p) <= alpha: the partition rejection count
    for k in range(n+1):
        if 1-binom_cdf(k-1,n,p) <= alpha: return k
    return n+1
print(f"cells {len(cells)}, corpora {len(data)}, partition {PART}, alpha={ALPHA}")
print("tau    | honest: mean / worst hash     | fraud: mean / lowest hash     | p_mismatch* | reject   | power vs lowest fraud hash")
best=None
for ti,tau in enumerate(taus):
    if tau>0.101: break
    hon=[m[ti].mean() for (l,r,h),m in data.items() if r=='honest']; fr=[m[ti].mean() for (l,r,h),m in data.items() if r=='fraud']
    if not hon or not fr: continue
    hw=max(hon); fb=min(fr)
    # p_mismatch*: smallest p at which the worst honest hash (share hw) is rejected no more often than alpha
    p=None
    for cand in np.arange(0.005,0.995,0.005):
        k=crit(PART,cand,ALPHA)
        if 1-binom_cdf(k-1,PART,hw) <= ALPHA: p=cand; break
    if p is None: print(f"{tau:.3f}  | {100*np.mean(hon):5.1f} / {100*hw:5.1f}            | {100*np.mean(fr):5.1f} / {100*fb:5.1f}         | —"); continue
    k=crit(PART,p,ALPHA); power=1-binom_cdf(k-1,PART,fb)
    print(f"{tau:.3f}  | {100*np.mean(hon):5.1f} / {100*hw:5.1f}            | {100*np.mean(fr):5.1f} / {100*fb:5.1f}         | {p:.3f}       | {k:3d}/{PART}  | {100*power:5.1f} %")
    if best is None or power>best[0]: best=(power,tau,p,k)
if best: print(f"\nbest point by power against the worst case: tau={best[1]:.3f}, p_mismatch={best[2]:.3f}, reject at >= {best[3]} flagged of {PART}, power {100*best[0]:.1f} %")
