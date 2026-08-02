#!/bin/bash
# Runs INSIDE the v4box container. 1xB300 TP=1.
# maxnbt 32768 so PoC batch 32 fits its metadata buffer; 275 GB per card leaves
# ample room, so gmu stays at 0.90 as on H200.
set -u
M=${MODEL:-deepseek-ai/DeepSeek-V4-Flash-0731}
TAGP=${TAGPREFIX:-official}
O=/root/out_$TAGP
mkdir -p $O
BASE='"--tensor-parallel-size","2","--gpu-memory-utilization","0.90","--max-model-len","400000","--max-num-batched-tokens","32768","--kv-cache-dtype","fp8","--logprobs-mode","processed_logprobs","--worker-extension-cls","gonka_poc.worker.PoCWorkerExtension","--trust-remote-code"'
SPEC='"--speculative-config","{\"method\":\"dspark\",\"num_speculative_tokens\":7,\"draft_sample_method\":\"greedy\"}"'

up () {
  local ARGS="$BASE"; [ -n "$1" ] && ARGS="$BASE,$1"
  curl -s -X POST http://127.0.0.1:8081/api/v1/inference/down -m 90 >/dev/null; sleep 15
  curl -s -X POST http://127.0.0.1:8081/api/v1/inference/up/async -H 'Content-Type: application/json' \
    -d "{\"model\":\"$M\",\"dtype\":\"auto\",\"additional_args\":[$ARGS]}" -m 60 >/dev/null
  for i in $(seq 1 360); do
    S=$(curl -s -m 6 http://127.0.0.1:8081/api/v1/inference/up/status 2>/dev/null)
    echo "$S" | grep -q '"is_running":true' && { echo "ENGINE_UP after $((i*15))s"; return 0; }
    echo "$S" | grep -q '"status":"failed"' && { echo "ENGINE_FAILED"; echo "$S" | head -c 250; return 1; }
    sleep 15
  done
  echo "ENGINE_TIMEOUT"; return 1
}

arm () {
  local T=$1
  grep -a "non-default args" /root/api.log | tail -1 | cut -c1-360 > $O/engine_args_$T.txt
  grep -aE "Using V2 Model Runner|GPU KV cache size|DSpark speculator" /root/api.log | tail -3 >> $O/engine_args_$T.txt
  cat $O/engine_args_$T.txt
  curl -s -m 900 -X POST http://127.0.0.1:8081/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$M\",\"prompt\":\"Warm up:\",\"max_tokens\":48,\"temperature\":0}" >/dev/null
  echo "=== ${T}_SWEEP ==="
  MODEL=$M HOST_IP=127.0.0.1 python3 -u /root/run_pow_generation.py --phase 3 --skip-check 2>&1 \
    | tee $O/sweep_$T.log | grep -E "Nonces/min|│|Best" | tail -8
  echo "=== ${T}_NONCES ==="
  while read -r SID BH PK; do
    [ -z "${SID:-}" ] && continue
    timeout 1200 python3 -u /root/collect_artifacts.py --url http://127.0.0.1:8081 --model "$M" \
      --output-dir $O/arts_${T}_$SID --nonces 1000 --batch-size 32 --logprobs-count 0 \
      --block-hash "$BH" --public-key "$PK" --gpu 2xB200-SXM-TP2 --vllm-version "0.25.1-k10-$T" 2>&1 \
      | grep -aE "PoC seeds|Collected|Saved|Error" | head -3
    cp $O/arts_${T}_$SID/nonces_1000.json $O/nonces_${T}_${SID}.json 2>/dev/null \
      && echo "  OK $SID" || echo "  MISSING $SID"
  done < /root/seeds.env
  echo "=== ${T}_SERVING ==="
  python3 -u /root/serving_bench.py --url http://127.0.0.1:8081 --model "$M" --tag ${TAGP}_$T \
    --out $O/serving_$T.json > $O/serving_$T.log 2>&1
  grep -E '"scenario"|OUTPUT_TOK_PER_S|TPOT|TOKENS_PER_CHUNK|FAILED_REQ' $O/serving_$T.log | tail -20
  echo "=== ${T}_DONE ==="
}

python3 - <<'PYEOF' > /root/seeds.env
import json
for v in json.load(open('/root/poc_seeds.json'))['seeds'][:3]:
    print(f"{v['id']} {v['block_hash']} {v['public_key']}")
PYEOF

echo "=== ARM_OFF ==="; up "" && arm dspark_off
if [ "${SKIP_DSPARK:-0}" != "1" ]; then echo "=== ARM_ON ==="; up "$SPEC" && arm dspark_on; fi
echo "=== ALL_DONE ==="
