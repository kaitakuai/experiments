#!/bin/bash
# Recover the phases lost when scenario s4 crashed the engine.
# s4 (45k prompt x concurrency 20) kills the engine on 4xH100 with AND without
# DSpark, so it is excluded here; everything downstream of it was 502.
set -u
M=deepseek-ai/DeepSeek-V4-Flash-0731
O=${OUT_DIR:-/root/final4}
mkdir -p $O
BASE='"--tensor-parallel-size","4","--gpu-memory-utilization","0.85","--max-model-len","400000","--max-num-batched-tokens","16384","--kv-cache-dtype","fp8","--logprobs-mode","processed_logprobs","--worker-extension-cls","gonka_poc.worker.PoCWorkerExtension","--trust-remote-code"'
SPEC='"--speculative-config","{\"method\":\"dspark\",\"num_speculative_tokens\":7,\"draft_sample_method\":\"greedy\"}"'

up () {
  local EXTRA="$1"; local ARGS="$BASE"; [ -n "$EXTRA" ] && ARGS="$BASE,$EXTRA"
  curl -s -X POST http://127.0.0.1:8081/api/v1/inference/down -m 90 >/dev/null; sleep 20
  curl -s -X POST http://127.0.0.1:8081/api/v1/inference/up/async -H 'Content-Type: application/json' \
    -d "{\"model\":\"$M\",\"dtype\":\"auto\",\"additional_args\":[$ARGS]}" -m 60 >/dev/null
  for i in $(seq 1 90); do
    S=$(curl -s -m 6 http://127.0.0.1:8081/api/v1/inference/up/status 2>/dev/null)
    echo "$S" | grep -q '"is_running":true' && { echo "ENGINE_UP after $((i*15))s"; return 0; }
    echo "$S" | grep -q '"status":"failed"' && { echo "ENGINE_FAILED"; return 1; }
    sleep 15
  done
  echo "ENGINE_TIMEOUT"; return 1
}

arm () {
  local T=$1
  curl -s -m 900 -X POST http://127.0.0.1:8081/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$M\",\"prompt\":\"Warm up:\",\"max_tokens\":48,\"temperature\":0}" >/dev/null
  echo "=== ${T}_SWEEP ==="
  MODEL=$M HOST_IP=127.0.0.1 python3 -u /root/run_pow_generation.py --phase 3 --skip-check 2>&1 \
    | tee $O/sweep_$T.log | grep -E "Nonces/min|│|Best" | tail -8
  echo "=== ${T}_GEN ==="
  python3 -u /root/equiv_probe.py gen --url http://127.0.0.1:8081 --model "$M" --out $O/gen_$T.json
  echo "=== ${T}_SERVING ==="
  python3 -u /root/serving_bench.py --url http://127.0.0.1:8081 --model "$M" --tag $T \
    --only s1,s2,s3,s4 --out $O/serving_$T.json > $O/serving_$T.log 2>&1
  grep -E '"scenario"|OUTPUT_TOK_PER_S|TPOT|TTFT"|TOKENS_PER_CHUNK' $O/serving_$T.log | tail -16
  echo "=== ${T}_DONE ==="
}

echo "=== ARM_ON ==="
up "$SPEC" || exit 1
arm dspark_on
echo "=== ARM_OFF ==="
up "" || exit 1
arm dspark_off
echo "=== EQUIV ==="
python3 -u /root/equiv_probe.py check --url http://127.0.0.1:8081 --model "$M" \
  --records $O/gen_dspark_on.json  --out $O/equiv_dspark_on.json
python3 -u /root/equiv_probe.py check --url http://127.0.0.1:8081 --model "$M" \
  --records $O/gen_dspark_off.json --out $O/equiv_control.json
echo "=== ALL_DONE ==="
