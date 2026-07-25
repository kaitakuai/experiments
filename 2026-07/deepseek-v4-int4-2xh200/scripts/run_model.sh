#!/bin/bash
# Прогон одной модели на Vast-боксе: свип + 1000 нонсов + serving, в двух режимах.
# Аргументы: <модель> <тег> [режимы: "eager graphs" по умолчанию]
set -x
M=$1; TAG=$2; MODES=${3:-"eager graphs"}; API=http://127.0.0.1:8081; OUT=/root/out/$TAG
mkdir -p $OUT
echo "=== RM_${TAG}_START ==="

for MODE in $MODES; do
  if [ "$MODE" = "eager" ]; then EXTRA='"--enforce-eager"'
  else EXTRA='"--compilation-config","{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"]}"'; fi

  echo "=== RM_${TAG}_${MODE}_UP ==="
  curl -s -X POST $API/api/v1/inference/up/async -H 'Content-Type: application/json' \
    -d "{\"model\":\"$M\",\"dtype\":\"auto\",\"additional_args\":[$EXTRA]}" -m 60 | head -c 200; echo

  for i in $(seq 1 260); do
    st=$(curl -s -m 6 $API/api/v1/inference/up/status 2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin).get('is_running'))" 2>/dev/null)
    [ "$st" = "True" ] && { echo "=== RM_${TAG}_${MODE}_UP_OK after $((i*15))s ==="; break; }
    er=$(curl -s -m 6 $API/api/v1/inference/up/status 2>/dev/null | grep -o '"error":"[^"]*"' | head -1)
    [ -n "$er" ] && { echo "=== RM_${TAG}_${MODE}_LOAD_FAILED $er ==="
      cp /root/api.log $OUT/${MODE}_api.log
      echo "--- первопричина ---"
      grep -aE "RuntimeError|Error:|Unsupported|assert|KeyError|AttributeError|ValueError|OutOfMemory" $OUT/${MODE}_api.log \
        | grep -viE "watcher|common.manager|inference.manager" | head -6
      break; }
    sleep 15
  done
  [ "$st" = "True" ] || continue

  # бэкенды — то, чего не хватает в таблице
  echo "=== RM_${TAG}_${MODE}_BACKENDS ==="
  grep -aoE "moe_backend='[a-z_]+'|Using [A-Za-z0-9_]+ backend[^,]{0,30}|Using [A-Za-z]*MoE[A-Za-z]*|attention backend[^,\"]{0,40}|FlashMLA[A-Za-z]*|FLASH_ATTN[A-Z_]*|TRITON[A-Z_]*|CUTLASS[A-Z_]*|FlashInfer[A-Za-z]*|Marlin[A-Za-z]*|DeepGEMM" /root/api.log | sort -u | head -12 | tee $OUT/${MODE}_backends.txt

  curl -s -m 600 -X POST $API/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"warmup\"}],\"max_tokens\":16}" | head -c 80; echo
  echo "=== RM_${TAG}_${MODE}_WARM ==="

  HOST_IP=127.0.0.1 MODEL="$M" python3 -u /root/run_pow_generation.py --phase 3 --skip-check 2>&1 \
    | tee $OUT/${MODE}_poc_sweep.log | tail -18
  echo "=== RM_${TAG}_${MODE}_SWEEP_DONE ==="

  python3 -u /root/collect_artifacts.py --url $API --model "$M" --output-dir /root/arts_${TAG}_${MODE} \
    --nonces 1000 --batch-size 32 --logprobs-count 0 --gpu "$TAG" --vllm-version "0.25.1-$MODE" 2>&1 \
    | tee $OUT/${MODE}_collect.log | grep -E "Nonces saved|Error" | head -2
  cp /root/arts_${TAG}_${MODE}/nonces_1000.json $OUT/${MODE}_nonces.json 2>/dev/null
  echo "=== RM_${TAG}_${MODE}_COLLECT_DONE ==="

  python3 - <<PY
import re
s=open('/root/compressa_config.yml').read().replace('MODEL_PLACEHOLDER','$M')
b=[x for x in re.split(r'\n(?=- model_name:)', s) if x.strip().startswith('- model_name')]
for i,x in enumerate(b,1): open('/root/sc%d.yml'%i,'w').write(x)
PY
  for n in 1 2 3 4; do
    /root/cpvenv/bin/compressa-perf measure-from-yaml --db /root/${TAG}_${MODE}.sqlite /root/sc${n}.yml 2>&1 \
      | grep -E "Number of failed|Error" | head -2
  done
  /root/cpvenv/bin/python /root/metrics.py /root/${TAG}_${MODE}.sqlite > $OUT/${MODE}_serving.json 2>/dev/null
  cp /root/api.log $OUT/${MODE}_api.log
  echo "=== RM_${TAG}_${MODE}_SERVING_DONE ==="

  curl -s -X POST $API/api/v1/inference/down -m 90 | head -c 60; echo
  sleep 35
done
echo "=== RM_${TAG}_DONE ==="
