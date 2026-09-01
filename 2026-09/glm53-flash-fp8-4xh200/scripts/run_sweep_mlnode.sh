#!/bin/bash
# Штатный свип PoC через mlnode. Окно 120 c на батч + прогрев (значения по умолчанию
# в самом скрипте, снаружи их не трогаем).
#
# HOST_IP=127.0.0.1 обязателен: на Vast инстанс сам себе контейнер, и если колбэк уйдёт
# на адрес докерного шлюза, свип покажет ровные нули.
set -u
exec >/root/sweep.log 2>&1
echo "=== СВИП $(date +%T) ==="
ulimit -n 524288

for i in $(seq 1 240); do
  ST=$(curl -s -m 8 http://127.0.0.1:8081/api/v1/inference/up/status | python3 -c "import sys,json;print(json.load(sys.stdin).get('is_running'))" 2>/dev/null)
  [ "$ST" = "True" ] && { echo "движок готов $(date +%T)"; break; }
  sleep 15
done
[ "$ST" = "True" ] || { echo "движок не поднялся"; echo SWEEP_DONE; exit 1; }

grep -aE "attention backend|GPU KV cache size" /root/api.log | tail -2 | cut -c1-170

BH=$(python3 -c "import json;print(json.load(open('/root/poc_seeds.json'))['seeds'][0]['block_hash'])")
PK=$(python3 -c "import json;print(json.load(open('/root/poc_seeds.json'))['seeds'][0]['public_key'])")
[ -n "$BH" ] || { echo "ПУСТОЕ СЕМЯ"; echo SWEEP_DONE; exit 1; }
echo "семя s1: ${BH:0:12}…"

curl -s -m 20 -X POST http://127.0.0.1:8081/api/v1/inference/pow/stop >/dev/null 2>&1
sleep 5

HOST_IP=127.0.0.1 BATCH_SIZES=8,16,24,32 \
  POC_BLOCK_HASH="$BH" POC_PUBLIC_KEY="$PK" \
  python3 -u /root/run_pow_generation.py --phase 3 --skip-check
echo "IMA=$(grep -ac 'illegal memory' /root/api.log)"
echo SWEEP_DONE
