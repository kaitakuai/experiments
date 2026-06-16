import json, sys, os
D = sys.argv[1] if len(sys.argv)>1 else "."
def prof(tag,cfg):
    p=os.path.join(D,f"prof_{tag}_{cfg}.json")
    return json.load(open(p)) if os.path.exists(p) else None
def m(a): return sum(a)/len(a) if a else None
cfgs={"cc":"compiled prefill + compiled decode","ee":"eager prefill + eager decode","ec":"eager prefill + compiled decode"}
print("="*92)
print("РАЗДЕЛИТЕЛЬНАЯ СПОСОБНОСТЬ honest(FP8 cross-hw) vs fraud(AWQ) — MiniMax, B300↔H100, per-step")
print("="*92)
rows=[]
for cfg in ["cc","ee","ec"]:
    h=prof("honest",cfg); f=prof("fraud",cfg)
    if not h or not f: 
        print(f"\n[{cfg}] {cfgs[cfg]}: данные неполны (honest={h is not None} fraud={f is not None})"); continue
    slices={"full (prefill+decode)":(m(h),m(f)),
            "prefill only (step0)":(h[0],f[0]),
            "decode only (steps 1..)":(m(h[1:]),m(f[1:]))}
    print(f"\n[{cfg}] {cfgs[cfg]}")
    print(f"  {'срез':<26}{'honest':>9}{'fraud':>9}{'gap (fraud-honest)':>20}")
    for name,(hv,fv) in slices.items():
        print(f"  {name:<26}{hv:>9.3f}{fv:>9.3f}{fv-hv:>20.3f}")
        rows.append((cfg,name,hv,fv,fv-hv))
# map to the 7 user cases
print("\n"+"="*92); print("7 ЗАПРОШЕННЫХ СЛУЧАЕВ (gap = разделимость, больше = лучше):"); print("="*92)
cases=[("1) compiled prefill + compiled decode","cc","full (prefill+decode)"),
       ("2) только compiled prefill","cc","prefill only (step0)"),
       ("3) только compiled decode","cc","decode only (steps 1..)"),
       ("4) eager prefill + eager decode","ee","full (prefill+decode)"),
       ("5) только eager prefill","ee","prefill only (step0)"),
       ("6) только eager decode","ee","decode only (steps 1..)"),
       ("7) eager prefill + compiled decode","ec","full (prefill+decode)")]
lk={(c,s):(hv,fv,g) for c,s,hv,fv,g in rows}
for label,cfg,sl in cases:
    v=lk.get((cfg,sl))
    if v: print(f"  {label:<42} honest={v[0]:.3f}  fraud={v[1]:.3f}  gap={v[2]:+.3f}")
    else: print(f"  {label:<42} (нет данных)")
