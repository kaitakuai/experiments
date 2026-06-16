#!/usr/bin/env python3
# Per-nonce DECODE-mismatch statistics from teacher-forced crosshw trajectories.
# For each config (cc, ee): honest vs fraud per-nonce decode fraction -> gap + 95% CI.
# Treats the NONCE as the independent unit (the correct unit), N up to 64.
import json, os, sys, math, statistics as st

D = sys.argv[1] if len(sys.argv) > 1 else "."
def load(p):
    fp = os.path.join(D, p)
    return json.load(open(fp)) if os.path.exists(fp) else None

def per_nonce_decode_frac(ref, traj):
    # ref/traj: {nonce_str: [k0..k64]}. decode = steps 1.. ; fraction mismatched per nonce.
    out = {}
    for n in ref:
        if n not in traj: continue
        r = ref[n]; t = traj[n]
        m = min(len(r), len(t))
        dec_r, dec_t = r[1:m], t[1:m]              # skip step0 (prefill) -> decode only
        if not dec_r: continue
        mism = sum(1 for a, b in zip(dec_r, dec_t) if a != b)
        out[n] = mism / len(dec_r)
    return out

def ci95(vals):
    n = len(vals); m = st.mean(vals); s = st.pstdev(vals)
    se = s / math.sqrt(n) if n else 0
    return m, s, 1.96 * se

print("="*84)
print("PER-NONCE DECODE разделимость (teacher-forced crosshw, N=64, нонс = единица)")
print("="*84)
results = {}
for cfg in ("cc", "ee"):
    ref = load(f"refN64_{cfg}.json")
    th = load(f"trajectories_honest_{cfg}.json")
    tf = load(f"trajectories_fraud_{cfg}.json")
    if not (ref and th and tf):
        print(f"\n[{cfg}] неполные данные (ref={ref is not None} honest={th is not None} fraud={tf is not None})")
        continue
    h = list(per_nonce_decode_frac(ref, th).values())
    f = list(per_nonce_decode_frac(ref, tf).values())
    hm, hs, hci = ci95(h); fm, fs, fci = ci95(f)
    gap = fm - hm
    gap_se = math.sqrt((hs**2)/len(h) + (fs**2)/len(f))
    gap_ci = 1.96 * gap_se
    results[cfg] = (gap, gap_ci, h, f)
    print(f"\n[{cfg}]  (honest N={len(h)}, fraud N={len(f)})")
    print(f"  honest decode-mismatch: {hm:.3f} ± {hci:.3f} (95%CI)   std={hs:.3f}")
    print(f"  fraud  decode-mismatch: {fm:.3f} ± {fci:.3f} (95%CI)   std={fs:.3f}")
    sig = "ДА (CI не включает 0)" if gap - gap_ci > 0 else "НЕТ (CI включает 0)"
    print(f"  GAP = {gap:+.3f}  ± {gap_ci:.3f} (95%CI)  ->  разделяет фрод: {sig}")

if "cc" in results and "ee" in results:
    gcc, cci, _, _ = results["cc"]; gee, eci, _, _ = results["ee"]
    diff = gcc - gee; diff_ci = math.sqrt(cci**2 + eci**2)  # approx (CIs already 1.96*SE)
    print("\n" + "="*84)
    print(f"compiled gap {gcc:+.3f}±{cci:.3f}  vs  eager gap {gee:+.3f}±{eci:.3f}")
    print(f"разница (cc−ee) = {diff:+.3f} ± {diff_ci:.3f} (95%CI)  ->  "
          + ("compiled РАЗДЕЛЯЕТ СИЛЬНЕЕ (значимо)" if diff - diff_ci > 0 else "разница НЕ значима"))
    print("="*84)
