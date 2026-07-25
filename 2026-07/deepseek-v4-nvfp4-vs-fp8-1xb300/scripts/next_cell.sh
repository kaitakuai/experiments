#!/bin/bash
# Следующая ячейка матрицы: опустить движок, поднять с нужным режимом/моделью, снять serving.
# Аргументы: <метка> <модель> <eager|graphs>
set -x
C=nvfp4ab; LBL=$1; M=$2; MODE=$3; API=http://127.0.0.1:8081
if [ "$MODE" = "eager" ]; then EXTRA='"--enforce-eager"'
else EXTRA='"--compilation-config","{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"]}"'; fi
echo "=== NC_${LBL}_DOWN ==="
docker exec $C curl -s -X POST $API/api/v1/inference/down -m 90 | head -c 80; echo
sleep 40
echo "=== NC_${LBL}_UP ==="
docker exec $C curl -s -X POST $API/api/v1/inference/up/async -H 'Content-Type: application/json' \
  -d "{\"model\":\"$M\",\"dtype\":\"auto\",\"additional_args\":[$EXTRA]}" -m 60 | head -c 120; echo
for i in $(seq 1 240); do
  st=$(docker exec $C curl -s -m 6 $API/api/v1/inference/up/status 2>/dev/null \
       | python3 -c "import sys,json;print(json.load(sys.stdin).get('is_running'))" 2>/dev/null)
  [ "$st" = "True" ] && { echo "=== NC_${LBL}_UP_OK after $((i*15))s ==="; break; }
  sleep 15
done
# прогрев: форсирует ленивую компиляцию, иначе первый сценарий меряет JIT
docker exec $C curl -s -m 300 -X POST $API/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"warmup\"}],\"max_tokens\":16}" | head -c 60; echo
echo "=== NC_${LBL}_WARM ==="
bash /root/serving.sh $LBL "$M"
