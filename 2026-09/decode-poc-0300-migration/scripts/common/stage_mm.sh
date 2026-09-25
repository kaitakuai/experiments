#!/usr/bin/env bash
# Tier A for MiniMax on the running 0.30 server: gates, 3-hash corpus, validation of the old
# reference and of itself, a mining round through /init/generate with the gate check, a replay
# equivalence pass. Env: OUT (server's out dir), REF (old reference corpora dir), TAG, HASHES.
set -u
OUT=${OUT:-/root/rout/mm}; REF=${REF:?old reference dir}; TAG=${TAG:-mm030}; HASHES=${HASHES:-h01,h02,h03}
KIT=${KIT:-/root/r-kit}; VENV=/opt/gv30; PY=$VENV/bin/python; URL=http://127.0.0.1:8000; SERVED=${SERVED:-MiniMaxAI/MiniMax-M2.7}; PART=${PART:-250}; GEN_BATCH=${GEN_BATCH:-250}
LOG=$OUT/stage_${TAG}.log; say(){ echo "[$(date -u +%T)] $*" | tee -a "$LOG"; }
export KIT VENV OUT SERVED_MODEL=$SERVED
# Boot provenance with the KV wall (r_corpora refuses to run without wall_nonces). r_boot writes it only when its
# own readiness check passed; otherwise derive it from the server log: KV tokens / 512, capped at MNS.
python3 - "$OUT" "$TAG" "${CFG:-?}" "${MNS:-704}" "${MODEL_DIR:-?}" <<'PY'
import json, os, re, sys
out, tag, cfg, mns, md = sys.argv[1:6]
p = os.path.join(out, f"kv_{tag}.json"); rec = {}
try: rec = json.load(open(p))
except Exception: pass
if not rec.get("wall_nonces"):
    kv = 0
    for line in open(os.path.join(out, "server.log"), errors="ignore"):
        m = re.search(r"KV cache size: ([0-9,]+) tokens", line)
        if m: kv = int(m.group(1).replace(",", ""))
    wall = min(kv // 512, int(mns)) if kv else int(mns)
    rec.update({"cfg": cfg, "model_dir": md, "wall_nonces": wall, "kv_tokens": kv, "tag": tag,
                "synth": "0.30 tier A: wall derived from the server log", "vllm": "0.30.0", "plugin": "0.2.0"})
    json.dump(rec, open(p, "w"))
json.dump(rec, open(os.path.join(out, "current_boot.json"), "w"))
print(f"provenance: wall {rec.get('wall_nonces')} nonces (kv {rec.get('kv_tokens', '?')})")
PY
say "stage start tag=$TAG hashes=$HASHES ref=$REF"
say "--- 1. gates"; (cd $KIT/scripts && $PY r1_gates.py $TAG 2>&1 | tail -6 | tee -a "$LOG")
say "--- 2. corpora (GEN_BATCH=250)"; (cd $KIT/scripts && GEN_BATCH=$GEN_BATCH $PY r_corpora.py honest $TAG "$HASHES" 2>&1 | tail -6 | tee -a "$LOG")
ls -la $OUT/corp_honest_h0*_gen.json 2>/dev/null | awk '{print "   ", $5, $9}' | tee -a "$LOG"
for sub in vs_old vs_self; do mkdir -p $OUT/$sub; cp $OUT/current_boot.json $OUT/$sub/; done
say "--- 3. validate the OLD reference corpora with this server (partition $PART)"
(cd $KIT/scripts && OUT=$OUT/vs_old CORP_DIR=$REF PART=$PART RUN_LABEL=old $PY r2_validate.py "honest_h*" $TAG 2>&1 | grep -vE "^\s*$" | tail -14 | tee -a "$LOG")
say "--- 4. validate this server's own corpora (same boot)"
(cd $KIT/scripts && CORP_DIR=$OUT OUT=$OUT/vs_self PART=$PART RUN_LABEL=self $PY r2_validate.py "honest_h*" $TAG 2>&1 | grep -vE "^\s*$" | tail -14 | tee -a "$LOG")
say "--- 5. mining round through /init/generate + gate"
chat(){ curl -s -m 60 -o /dev/null -w "%{http_code}" -X POST $URL/v1/chat/completions -H 'content-type: application/json' -d '{"model":"'"$SERVED"'","messages":[{"role":"user","content":"Say hi"}],"max_tokens":8}'; }
say "   chat before round: HTTP $(chat)"
INIT=$(curl -s -m 30 -X POST $URL/api/v1/pow/init/generate -H 'content-type: application/json' -d '{"block_hash":"deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef","block_height":1,"public_key":"cafebabecafebabecafebabecafebabe","node_id":0,"node_count":1,"batch_size":250,"params":{"model":"'"$SERVED"'","seq_len":256,"k_dim":12,"scheme":"decode","max_tokens":256}}')
say "   init/generate: $(echo "$INIT" | cut -c1-160)"; sleep 20
say "   status: $(curl -s -m 10 $URL/api/v1/pow/status | cut -c1-200)"
say "   chat during round: HTTP $(chat)  (503 expected)"
say "   stop: $(curl -s -m 60 -X POST $URL/api/v1/pow/stop | cut -c1-160)"; sleep 3
say "   chat after stop: HTTP $(chat)  (200 expected)"
say "--- 6. replay equivalence (24 originals, replay on the same server)"
mkdir -p $OUT/replay; (cd /root/replay-kit && MODEL_NAME=$SERVED NOTHINK=1 python3 gen_originals.py $URL $OUT/replay/orig 24 2>&1 | tail -2; python3 run_replay.py $URL $OUT/replay/orig $OUT/replay/self 2>&1 | tail -2; python3 report.py $OUT/replay/self --threshold 0.95 2>&1 | tail -4) | tee -a "$LOG"
say "stage done"
