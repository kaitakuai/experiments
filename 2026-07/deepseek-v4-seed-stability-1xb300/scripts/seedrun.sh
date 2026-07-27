#!/bin/bash
# Устойчивость L2 к семени: честный FP8 и NVFP4 по 5 семенам + проверка периодичности артефакта.
set -x
C=seedbox
API=http://127.0.0.1:8081; OUT=/root/seedtest; mkdir -p $OUT
SEEDS=/root/poc_seeds.json  # читается на ХОСТЕ

up () {  # $1=модель $2=режим
  M=$1; MODE=$2
  if [ "$MODE" = "eager" ]; then EXTRA='"--enforce-eager"'
  else EXTRA='"--compilation-config","{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"]}"'; fi
  curl -s -X POST $API/api/v1/inference/up/async -H 'Content-Type: application/json' \
    -d "{\"model\":\"$M\",\"dtype\":\"auto\",\"additional_args\":[$EXTRA]}" -m 60 >/dev/null
  for i in $(seq 1 260); do
    st=$(docker exec $C curl -s -m 6 $API/api/v1/inference/up/status 2>/dev/null|python3 -c "import sys,json;print(json.load(sys.stdin).get('is_running'))" 2>/dev/null)
    [ "$st" = "True" ] && { echo "=== SR_UP_OK $M/$MODE after $((i*15))s ==="; break; }
    er=$(docker exec $C curl -s -m 6 $API/api/v1/inference/up/status 2>/dev/null|grep -o '"error":"[^"]*"'|head -1)
    [ -n "$er" ] && { echo "=== SR_FAILED $M/$MODE $er ==="; return 1; }
    sleep 15
  done
  curl -s -m 900 -X POST $API/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"warmup\"}],\"max_tokens\":8}" >/dev/null
  echo "=== SR_WARM $M/$MODE ==="
}

collect () {  # $1=модель $2=тег $3=seed_id $4=batch
  M=$1; TAG=$2; SID=$3; BS=${4:-32}
  BH=$(python3 -c "import json;print([s['block_hash'] for s in json.load(open('$SEEDS'))['seeds'] if s['id']=='$SID'][0])")
  PK=$(python3 -c "import json;print([s['public_key'] for s in json.load(open('$SEEDS'))['seeds'] if s['id']=='$SID'][0])")
  python3 -u /root/collect_artifacts.py --url $API --model "$M" --output-dir /root/a_${TAG}_${SID}_b${BS} \
    --nonces 1000 --batch-size $BS --logprobs-count 0 --block-hash "$BH" --public-key "$PK" \
    --gpu b300 --vllm-version "0.25.1-$TAG" 2>&1 | grep -E "Nonces saved|Error" | head -2
  docker cp $C:/root/a_${TAG}_${SID}_b${BS}/nonces_1000.json $OUT/${TAG}_${SID}_b${BS}.json 2>/dev/null
  echo "=== SR_DONE ${TAG}_${SID}_b${BS} ==="
}

echo "=== SR_START ==="
up deepseek-ai/DeepSeek-V4-Flash eager || exit 1
for s in s1 s2 s3 s4 s5; do collect deepseek-ai/DeepSeek-V4-Flash honest $s 32; done
# периодичность артефакта: те же семена, другие батчи
collect deepseek-ai/DeepSeek-V4-Flash honest s1 8
collect deepseek-ai/DeepSeek-V4-Flash honest s1 16
echo "=== SR_HONEST_EAGER_DONE ==="
docker exec $C curl -s -X POST $API/api/v1/inference/down -m 90 >/dev/null; sleep 30

up deepseek-ai/DeepSeek-V4-Flash graphs || exit 1
for s in s1 s2 s3; do collect deepseek-ai/DeepSeek-V4-Flash honestg $s 32; done
collect deepseek-ai/DeepSeek-V4-Flash honestg s1 8
echo "=== SR_HONEST_GRAPHS_DONE ==="
docker exec $C curl -s -X POST $API/api/v1/inference/down -m 90 >/dev/null; sleep 30

up nvidia/DeepSeek-V4-Flash-NVFP4 eager || exit 1
for s in s1 s2 s3 s4 s5; do collect nvidia/DeepSeek-V4-Flash-NVFP4 nvfp4 $s 32; done
echo "=== SR_NVFP4_DONE ==="
docker exec $C curl -s -X POST $API/api/v1/inference/down -m 90 >/dev/null
echo "=== SR_ALL_DONE ==="
