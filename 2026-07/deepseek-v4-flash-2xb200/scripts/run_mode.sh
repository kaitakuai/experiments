#!/bin/bash
# Полный прогон одного режима: PoC sweep + 1000 нонсов + compressa-perf инференс.
# Usage: bash run_mode.sh <eager|compiled> <gpu-label> <hf-model>
set -x
exec 2>&1
MODE=${1:?eager|compiled}
GPULABEL=${2:-unknown-gpu}
OUT=/root/out; mkdir -p $OUT
API=http://127.0.0.1:8081
MODEL=${MODEL:-${3:-${2:-deepseek-ai/DeepSeek-V4-Flash}}}

# режим-специфичные аргументы (НЕ в forced-списке runner.py => доезжают как есть)
if [ "$MODE" = "eager" ]; then
  EXTRA='"--enforce-eager"'
else
  EXTRA='"--compilation-config", "{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"]}"'
fi

echo "=== [$MODE] inference/up ==="
curl -s -X POST "$API/api/v1/inference/up/async" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"dtype\":\"auto\",\"additional_args\":[$EXTRA]}" -m 60 | head -c 300; echo

echo "=== [$MODE] ждём готовности ==="
for i in $(seq 1 120); do
  st=$(curl -s -m 6 "$API/api/v1/inference/up/status" | python3 -c "import sys,json;print(json.load(sys.stdin).get('is_running'))" 2>/dev/null)
  [ "$st" = "True" ] && { echo "=== ${MODE}_VLLM_UP after $((i*15))s ==="; break; }
  sleep 15
done

echo "=== [$MODE] фактически разрешённый режим компиляции (из vLLM-лога) ==="
grep -ihE "CompilationMode|cudagraph_mode|enforce_eager|Capturing cudagraph|compilation_config" /app/logs/*.log /root/*.log 2>/dev/null | tail -6

echo "=== [$MODE] PoC SWEEP ==="
HOST_IP=127.0.0.1 python3 -u /root/run_pow_generation.py --phase 3 --skip-check 2>&1 | tee $OUT/${MODE}_poc_sweep.log | tail -25
echo "=== ${MODE}_SWEEP_DONE ==="

echo "=== [$MODE] COLLECT 1000 нонсов ==="
python3 -u /root/collect_artifacts.py --url $API --model $MODEL \
  --output-dir $OUT/${MODE}_arts --nonces 1000 --batch-size 32 --logprobs-count 0 \
  --gpu "$GPULABEL" --vllm-version "0.25.1-$MODE" 2>&1 | tee $OUT/${MODE}_collect.log | tail -6
cp $OUT/${MODE}_arts/nonces_1000.json $OUT/${MODE}_nonces_1000.json 2>/dev/null
echo "=== ${MODE}_COLLECT_DONE ==="

echo "=== [$MODE] COMPRESSA-PERF инференс ==="
cd /root && /root/cpvenv/bin/compressa-perf measure-from-yaml --no-sign \
  --account_address 0x0000000000000000000000000000000000000000 \
  --node_url $API --model_name $MODEL /root/compressa_config.yml 2>&1 | tee $OUT/${MODE}_compressa.log | tail -20
/root/cpvenv/bin/compressa-perf list --show-metrics 2>&1 | tail -40 >> $OUT/${MODE}_compressa.log
echo "=== ${MODE}_COMPRESSA_DONE ==="

echo "=== [$MODE] inference/down ==="
curl -s -X POST "$API/api/v1/inference/down" -m 60 | head -c 200; echo
sleep 20
echo "=== ${MODE}_ALL_DONE ==="
