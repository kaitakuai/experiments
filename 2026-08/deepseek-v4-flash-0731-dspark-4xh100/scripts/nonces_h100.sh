#!/bin/bash
# Collect PoC nonce vectors on 4xH100 at batch 16.
# Lessons applied: the collector runs FIRST on a clean engine (no sweep before
# it), and max-num-batched-tokens stays at 32768 - the value under which the
# collector works on 2xH200. Batch 16 is fine: nonce values are batch-invariant
# (median L2 0.000 between b8/b16/b32 sets, see deepseek-v4-seed-stability-1xb300).
set -u
M=deepseek-ai/DeepSeek-V4-Flash-0731
O=${OUT_DIR:-/root/nonces}
mkdir -p $O
COMMON='"--tensor-parallel-size","4","--max-model-len","400000","--max-num-batched-tokens","32768","--kv-cache-dtype","fp8","--logprobs-mode","processed_logprobs","--worker-extension-cls","gonka_poc.worker.PoCWorkerExtension","--trust-remote-code"'
SPEC='"--speculative-config","{\"method\":\"dspark\",\"num_speculative_tokens\":7,\"draft_sample_method\":\"greedy\"}"'

up () {  # $1 = gmu, $2 = extra
  local ARGS="\"--gpu-memory-utilization\",\"$1\",$COMMON"; [ -n "$2" ] && ARGS="$ARGS,$2"
  curl -s -X POST http://127.0.0.1:8081/api/v1/inference/down -m 90 >/dev/null; sleep 15
  curl -s -X POST http://127.0.0.1:8081/api/v1/inference/up/async -H 'Content-Type: application/json' \
    -d "{\"model\":\"$M\",\"dtype\":\"auto\",\"additional_args\":[$ARGS]}" -m 60 >/dev/null
  for i in $(seq 1 90); do
    S=$(curl -s -m 6 http://127.0.0.1:8081/api/v1/inference/up/status 2>/dev/null)
    echo "$S" | grep -q '"is_running":true' && { echo "ENGINE_UP gmu=$1 after $((i*15))s"; return 0; }
    echo "$S" | grep -q '"status":"failed"' && { echo "ENGINE_FAILED gmu=$1"; return 1; }
    sleep 15
  done
  echo "ENGINE_TIMEOUT"; return 1
}

collect () {  # $1 = tag
  local T=$1
  grep -aE "GPU KV cache size|Using V2 Model Runner" /root/api.log | tail -2
  while read -r SID BH PK; do
    [ -z "${SID:-}" ] && continue
    echo "--- $T $SID ---"
    timeout 900 python3 -u /root/collect_artifacts.py --url http://127.0.0.1:8081 --model "$M" \
      --output-dir $O/arts_${T}_$SID --nonces 1000 --batch-size 16 --logprobs-count 0 \
      --block-hash "$BH" --public-key "$PK" --gpu 4xH100-SXM-TP4 --vllm-version "0.25.1-k10-$T-b16" 2>&1 \
      | grep -aE "PoC seeds|Collected|Saved|Total|Error|error" | head -4
    cp $O/arts_${T}_$SID/nonces_1000.json $O/nonces_${T}_${SID}.json 2>/dev/null \
      && echo "  OK $O/nonces_${T}_${SID}.json" || echo "  MISSING for $SID"
  done < /root/seeds.env
}

python3 - <<'PYEOF' > /root/seeds.env
import json
for v in json.load(open('/root/poc_seeds.json'))['seeds'][:3]:
    print(f"{v['id']} {v['block_hash']} {v['public_key']}")
PYEOF

echo "=== ARM_OFF (no speculation: loosest memory, closest to the July 4xH100 run) ==="
up 0.90 "" && collect dspark_off

echo "=== ARM_ON (DSpark; lower gmu to leave room for graph capture) ==="
if up 0.80 "$SPEC"; then collect dspark_on
else echo "gmu 0.80 failed, retrying at 0.75"; up 0.75 "$SPEC" && collect dspark_on; fi

echo "=== RESULT ==="; ls -la $O/nonces_*.json 2>/dev/null
echo "=== ALL_DONE ==="
