#!/bin/bash
# Свип PoC по обоим плечам на B200, окно 120 с на батч.
#
# Запускается ПОСЛЕ основной ячейки: свип может уронить движок на большом батче, и если
# гонять его раньше, потеряются нонсы.
#
# Батчи 8/16/32/48: на Hopper потолок между 16 и 24 (падает DeepGEMM), но на Blackwell
# августовский эталон снимался на батче 32 — значит граница другая, и её надо нащупать.
#
# Здесь vLLM поднят напрямую на 8081, поэтому измеритель туда и направляется: маршруты
# /api/v1/pow/* принадлежат самому движку. Через mlnode пришлось бы целиться в порт 5000.
set -u
: "${BATCHES:=8,16,32,48}"
exec >/root/sweep_b200.log 2>&1
echo "=== СВИП $(date +%T) батчи=$BATCHES ==="

BH=$(python3 -c "import json;print(json.load(open('/root/poc_seeds.json'))['seeds'][0]['block_hash'])")
PK=$(python3 -c "import json;print(json.load(open('/root/poc_seeds.json'))['seeds'][0]['public_key'])")
[ -n "$BH" ] && [ -n "$PK" ] || { echo "ПУСТОЕ СЕМЯ"; echo SWEEP_DONE; exit 1; }

sweep_arm() {  # $1 = метка плеча
  local ARM=$1 SNAP
  echo "### свип: $ARM $(date +%T) ###"
  curl -s -m 5 -o /dev/null http://127.0.0.1:8081/health || { echo "движок не отвечает"; return 1; }
  SNAP=$(curl -s -m 10 http://127.0.0.1:8081/v1/models | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])")
  echo "модель по версии движка: $SNAP"
  curl -s -m 20 -X POST http://127.0.0.1:8081/api/v1/pow/stop -H "Content-Type: application/json" -d '{}' >/dev/null 2>&1
  sleep 5
  MODEL="$SNAP" MLNODE_URL=http://127.0.0.1:8081 HOST_IP=127.0.0.1 \
    BATCH_SIZES="$BATCHES" POC_BLOCK_HASH="$BH" POC_PUBLIC_KEY="$PK" \
    timeout 1800 python3 -u /root/run_pow_generation.py --phase 3 --skip-check
  echo "IMA после $ARM: $(grep -ac 'illegal memory' /root/vllm.log)"
}

# 1) плечо, которое сейчас поднято (после ячейки это NVFP4)
sweep_arm nvfp4

# 2) перезапуск на честные веса и свип по ним
echo "### перезапуск на честные веса ###"
for P in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$P" 2>/dev/null; done
pkill -f "[g]onka-vllm-serve" 2>/dev/null; pkill -f "[V]LLM::" 2>/dev/null
sleep 10
for i in $(seq 1 60); do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
  [ "${U:-9999}" -lt 2000 ] && break
  sleep 5
done
SNAP=$(ls -d /root/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash/snapshots/*/ | head -1)
sed -i "s|--model \"[^\"]*\"|--model \"$SNAP\"|" /root/serve.sh
: > /root/vllm.log
nohup bash /root/serve.sh >/root/vllm.log 2>&1 &
RD=no
for i in $(seq 1 320); do
  sleep 15
  curl -s -m 5 -o /dev/null http://127.0.0.1:8081/health 2>/dev/null && { RD=yes; break; }
done
echo "честный движок готов=$RD $(date +%T)"
[ "$RD" = yes ] && sweep_arm honest || tail -20 /root/vllm.log

echo SWEEP_DONE
