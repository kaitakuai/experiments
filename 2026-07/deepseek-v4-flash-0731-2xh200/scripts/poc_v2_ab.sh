#!/bin/bash
# Does the model runner / DSpark change PoC nonce VALUES?
# A prover with speculative decoding and a validator without it must produce the
# same vectors, or honest nodes get flagged. Arm 1: V2 runner, DSpark off
# (isolates V1->V2). Arm 2: V2 runner, DSpark on (isolates DSpark).
set -u
M=deepseek-ai/DeepSeek-V4-Flash-0731
O=${OUT_DIR:-/root/out/pocv2}
mkdir -p $O

run_poc () {   # $1 = label
  local T=$1
  echo "=== POC_${T}_SWEEP ==="
  MODEL=$M HOST_IP=127.0.0.1 python3 -u /root/run_pow_generation.py --phase 3 --skip-check 2>&1 \
    | tee $O/sweep_$T.log | grep -E "Nonces/min|│|Best" | tail -8
  echo "=== POC_${T}_NONCES ==="
  while read -r SID BH PK; do
    [ -z "${SID:-}" ] && continue
    python3 -u /root/collect_artifacts.py --url http://127.0.0.1:8081 --model "$M" \
      --output-dir $O/arts_${T}_$SID --nonces 1000 --batch-size 32 --logprobs-count 0 \
      --block-hash "$BH" --public-key "$PK" --gpu 2xH200-SXM-TP2 --vllm-version "0.25.1-k9-$T" 2>&1 \
      | grep -aE "PoC seeds|Collected|Saved|Error" | head -3
    cp $O/arts_${T}_$SID/nonces_1000.json $O/nonces_${T}_${SID}.json 2>/dev/null
  done < /root/seeds.env
  echo "=== POC_${T}_DONE ==="
}

python3 - <<'PYEOF' > /root/seeds.env
import json
for v in json.load(open('/root/poc_seeds.json'))['seeds'][:3]:
    print(f"{v['id']} {v['block_hash']} {v['public_key']}")
PYEOF

run_poc v2_off

echo "=== POC_SWITCH_DSPARK ==="
curl -s -X POST http://127.0.0.1:8081/api/v1/inference/down -m 90 >/dev/null; sleep 25
SPEC='"--speculative-config","{\"method\":\"dspark\",\"num_speculative_tokens\":7,\"draft_sample_method\":\"greedy\"}"'
curl -s -X POST http://127.0.0.1:8081/api/v1/inference/up/async -H 'Content-Type: application/json' \
  -d "{\"model\":\"$M\",\"dtype\":\"auto\",\"additional_args\":[$SPEC]}" -m 60 >/dev/null
for i in $(seq 1 60); do
  S=$(curl -s -m 6 http://127.0.0.1:8081/api/v1/inference/up/status 2>/dev/null)
  echo "$S" | grep -q '"is_running":true' && { echo "DSPARK_UP after $((i*15))s"; break; }
  echo "$S" | grep -q '"status":"failed"' && { echo "DSPARK_FAILED"; exit 1; }
  sleep 15
done
curl -s -m 900 -X POST http://127.0.0.1:8081/v1/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$M\",\"prompt\":\"Warm up:\",\"max_tokens\":32,\"temperature\":0}" >/dev/null

run_poc v2_dspark
echo "=== POC_V2_AB_ALL_DONE ==="
