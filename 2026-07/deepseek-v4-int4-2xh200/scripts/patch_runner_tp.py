import re,os
R="/app/packages/api/src/api/inference/vllm/runner.py"
s=open(R).read(); open(R+".bak-orig","w").write(s)
TP=os.environ.get("TP","1")
for flag,val in [("--tensor-parallel-size",TP),("--max-model-len","400000"),("--max-num-batched-tokens","32768")]:
    pat=re.compile(r"\('%s', '\d+'\)"%re.escape(flag))
    if not pat.search(s): raise SystemExit("НЕ НАЙДЕНО: "+flag)
    s=pat.sub("('%s', '%s')"%(flag,val), s)
open(R,"w").write(s); print("  runner.py patched TP=%s"%TP)
