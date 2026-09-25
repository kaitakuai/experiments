#!/usr/bin/env python3
"""Send replay requests built from ORIG_DIR to a stack and store responses.
usage: run_replay.py http://127.0.0.1:8000 ORIG_DIR OUT_DIR [make_replay options...]"""
import json, sys, os, glob, subprocess, urllib.request, urllib.error

def call(url, body):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode(errors="replace")[:600]}

if __name__ == "__main__":
    url, orig_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
    extra = sys.argv[4:]
    os.makedirs(out, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    for f in sorted(glob.glob(f"{orig_dir}/orig_*.json")):
        body = json.loads(subprocess.check_output([sys.executable, f"{here}/make_replay.py", f, *extra]))
        status, resp = call(url, body)
        rec = {"source": f, "request": body, "status": status, "response": resp}
        name = os.path.basename(f).replace("orig_", "replay_")
        json.dump(rec, open(f"{out}/{name}", "w"), ensure_ascii=False)
        ch = (resp.get("choices") or [{}])[0]
        print(f"{name} http={status} finish={ch.get('finish_reason')} tokens={(resp.get('usage') or {}).get('completion_tokens')} positions={len((ch.get('logprobs') or {}).get('content') or [])}")
