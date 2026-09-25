#!/usr/bin/env bash
# Tier A for DeepSeek-V4-Flash on the running 0.30 server (d-kit): gates, validation of the old
# reference artifact set (10 hashes x 250), a 3-hash self set + self-validation, mining round + gate,
# replay equivalence. Env: OUT, REFXQ (old xq_g10 file), TAG, QUANT, SERVED.
set -u
OUT=${OUT:-/root/rout/ds}; REFXQ=${REFXQ:?old xq_g10 artifact set}; TAG=${TAG:-ds030}; QUANT=${QUANT:-nvfp4}
SERVED=${SERVED:-deepseek-ai/DeepSeek-V4-Flash-0731}; KIT=/root/d-kit; VENV=/opt/gv30; PY=$VENV/bin/python; URL=http://127.0.0.1:8000
LOG=$OUT/stage_${TAG}.log; say(){ echo "[$(date -u +%T)] $*" | tee -a "$LOG"; }
export KIT VENV OUT QUANT SERVED_MODEL=$SERVED
[ -f $OUT/current_boot.json ] || cp $OUT/kv_${TAG}.json $OUT/current_boot.json 2>/dev/null || echo '{"tag":"'"$TAG"'"}' > $OUT/current_boot.json
say "stage start tag=$TAG quant=$QUANT ref=$REFXQ"
say "--- 1. gates"; (cd $KIT/scripts && $PY r1_gates.py $TAG 2>&1 | tail -6 | tee -a "$LOG")
say "--- 2. validate the OLD reference artifact set (teacher-forced, all hashes)"
cp "$REFXQ" $OUT/xq_old.json; (cd $KIT/scripts && $PY d_cross_quant.py val old $TAG 2>&1 | tail -22 | tee -a "$LOG")
say "--- 3. own artifact set, 3 hashes x 250, then self-validation (same boot)"
(cd $KIT/scripts && NONCES=250 HASHES=h01,h02,h03 $PY d_cross_quant.py gen self $TAG 2>&1 | tail -5 | tee -a "$LOG"; $PY d_cross_quant.py val self $TAG 2>&1 | tail -14 | tee -a "$LOG")
say "--- 4. mining round through /init/generate + gate"
chat(){ curl -s -m 90 -o /dev/null -w "%{http_code}" -X POST $URL/v1/chat/completions -H 'content-type: application/json' -d '{"model":"'"$SERVED"'","messages":[{"role":"user","content":"Say hi"}],"max_tokens":8}'; }
say "   chat before round: HTTP $(chat)"
INIT=$(curl -s -m 30 -X POST $URL/api/v1/pow/init/generate -H 'content-type: application/json' -d '{"block_hash":"deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef","block_height":1,"public_key":"cafebabecafebabecafebabecafebabe","node_id":0,"node_count":1,"batch_size":250,"params":{"model":"'"$SERVED"'","seq_len":256,"k_dim":12,"scheme":"decode","max_tokens":256}}')
say "   init/generate: $(echo "$INIT" | cut -c1-160)"; sleep 30
say "   status: $(curl -s -m 10 $URL/api/v1/pow/status | cut -c1-200)"
say "   chat during round: HTTP $(chat)  (503 expected)"
say "   stop: $(curl -s -m 60 -X POST $URL/api/v1/pow/stop | cut -c1-160)"; sleep 3
say "   chat after stop: HTTP $(chat)  (200 expected)"
say "--- 5. replay equivalence"
mkdir -p $OUT/replay; (cd /root/replay-kit && MODEL_NAME=$SERVED NOTHINK=1 python3 gen_originals.py $URL $OUT/replay/orig 24 2>&1 | tail -2; python3 run_replay.py $URL $OUT/replay/orig $OUT/replay/self 2>&1 | tail -2; python3 report.py $OUT/replay/self --threshold 0.95 2>&1 | tail -4) | tee -a "$LOG"
say "stage done"
