import sys, time, json, urllib.request, concurrent.futures
MODEL = sys.argv[1]
N    = int(sys.argv[2]) if len(sys.argv) > 2 else 256
CONC = int(sys.argv[3]) if len(sys.argv) > 3 else 64
OUT  = int(sys.argv[4]) if len(sys.argv) > 4 else 256
URL = "http://127.0.0.1:8000/v1/completions"
PROMPT = "In the field of machine learning and large language models, " * 18  # ~140 tok
def one(_):
    body = json.dumps({"model": MODEL, "prompt": PROMPT, "max_tokens": OUT,
                       "ignore_eos": True, "temperature": 0}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)["usage"]["completion_tokens"]
    except Exception as e:
        return 0
one(0)  # warmup
t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(CONC) as ex:
    toks = list(ex.map(one, range(N)))
el = time.time() - t0
tot = sum(toks); ok = sum(1 for t in toks if t > 0)
print(f"SERVING: {tot} out-tok / {el:.2f}s = {tot/el:.1f} tok/s (ok={ok}/{N} conc={CONC} out={OUT})")
