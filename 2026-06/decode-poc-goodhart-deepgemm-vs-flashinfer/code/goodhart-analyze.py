import re, sys
log = open(sys.argv[1] if len(sys.argv)>1 else "/dev/stdin").read()
NN=ST=None
m=re.search(r'N_NONCES=(\d+) STEPS=(\d+)', log)
if m: NN,ST=int(m.group(1)),int(m.group(2))
res={}
for tag in ("flashinfer","deepgemm"):
    blk=log.split(f"[{tag}] start")
    if len(blk)<2: continue
    b=blk[1]
    el=re.search(r'genref:.*?\[([\d.]+)s\]', b) or re.search(r'\[([\d.]+)s\]', b)
    sv=re.search(r'SERVING:.*?=\s*([\d.]+) tok/s', b)
    P = (NN*ST/float(el.group(1))) if (el and NN) else None
    S = float(sv.group(1)) if sv else None
    res[tag]={"P_steps_s":P,"S_tok_s":S,"genref_s":float(el.group(1)) if el else None}
print("="*72); print("decode-PoC GOODHART TEST — DeepGEMM vs FlashInfer (B300, MiniMax-M2.7)"); print("="*72)
print(f"P = decode-PoC throughput (steps/s, batched genref {NN}x{ST})  |  S = serving tok/s\n")
print(f"  {'backend':<12}{'P (PoC steps/s)':>18}{'S (serv tok/s)':>18}{'genref_s':>12}")
for t in ("flashinfer","deepgemm"):
    r=res.get(t,{})
    print(f"  {t:<12}{(r.get('P_steps_s') or 0):>18.1f}{(r.get('S_tok_s') or 0):>18.1f}{(r.get('genref_s') or 0):>12.2f}")
fi,dg=res.get("flashinfer",{}),res.get("deepgemm",{})
if all(x is not None for x in [fi.get('P_steps_s'),dg.get('P_steps_s'),fi.get('S_tok_s'),dg.get('S_tok_s')]):
    bestP = "deepgemm" if dg['P_steps_s']>fi['P_steps_s'] else "flashinfer"
    bestS = "deepgemm" if dg['S_tok_s']>fi['S_tok_s'] else "flashinfer"
    dP=(dg['P_steps_s']/fi['P_steps_s']-1)*100; dS=(dg['S_tok_s']/fi['S_tok_s']-1)*100
    print(f"\n  DeepGEMM vs FlashInfer:  PoC {dP:+.1f}%   serving {dS:+.1f}%")
    print(f"  PoC prefers: {bestP}   |   serving prefers: {bestS}")
    print("\n  VERDICT:", "GOODHART OVERCOME (PoC and serving agree on backend)" if bestP==bestS
          else "GOODHART PERSISTS (PoC and serving prefer DIFFERENT backends)")
else:
    print("\n  (incomplete data)")
