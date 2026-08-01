#!/bin/bash
# DSpark A/B, both arms on the V2 model runner: the only difference is the
# --speculative-config flag. V1-vs-V2 is deliberately NOT part of the contrast.
set -u
M=deepseek-ai/DeepSeek-V4-Flash-0731
O=${OUT_DIR:-/root/out/v2}
mkdir -p $O

echo "=== AB_DSPARK_ON_SERVING ==="
python3 -u /root/serving_bench.py --url http://127.0.0.1:8081 --model "$M" \
  --tag dspark_on --out $O/serving_dspark_on.json > $O/serving_dspark_on.log 2>&1
grep -E "TTFT\"|TPOT|OUTPUT_TOK_PER_S|TOKENS_PER_CHUNK|scenario" $O/serving_dspark_on.log | tail -20
echo "=== AB_DSPARK_ON_DONE ==="

echo "=== AB_SWITCH_OFF ==="
curl -s -X POST http://127.0.0.1:8081/api/v1/inference/down -m 90 >/dev/null; sleep 25
curl -s -X POST http://127.0.0.1:8081/api/v1/inference/up/async -H 'Content-Type: application/json' \
  -d "{\"model\":\"$M\",\"dtype\":\"auto\",\"additional_args\":[]}" -m 60 >/dev/null
for i in $(seq 1 60); do
  S=$(curl -s -m 6 http://127.0.0.1:8081/api/v1/inference/up/status 2>/dev/null)
  echo "$S" | grep -q '"is_running":true' && { echo "OFF_ENGINE_UP after $((i*15))s"; break; }
  echo "$S" | grep -q '"status":"failed"' && { echo "OFF_ENGINE_FAILED"; exit 1; }
  sleep 15
done
grep -a "Using V2 Model Runner" /root/api.log | tail -1

curl -s -m 900 -X POST http://127.0.0.1:8081/v1/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$M\",\"prompt\":\"Warm up. Count to five:\",\"max_tokens\":48,\"temperature\":0}" >/dev/null
echo "=== AB_DSPARK_OFF_GREEDY ==="
python3 -u /root/greedy_probe.py http://127.0.0.1:8081 "$M" $O/greedy_dspark_off.json

echo "=== AB_DSPARK_OFF_SERVING ==="
python3 -u /root/serving_bench.py --url http://127.0.0.1:8081 --model "$M" \
  --tag dspark_off --out $O/serving_dspark_off.json > $O/serving_dspark_off.log 2>&1
grep -E "TTFT\"|TPOT|OUTPUT_TOK_PER_S|TOKENS_PER_CHUNK|scenario" $O/serving_dspark_off.log | tail -20
echo "=== AB_ALL_DONE ==="
