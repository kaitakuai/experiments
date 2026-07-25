#!/bin/bash
# Полный тест одной модели: оба режима × (свип + 1000 нонсов + serving по 4 сценариям).
# Аргументы: <модель> <тег>
set -x
M=$1; TAG=$2; C=${TAG}c; API=http://127.0.0.1:8081; OUT=/root/${TAG}out
IMG=ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash:0.2.13-vllm0.25.1-overlay-k4
mkdir -p $OUT
echo "=== FT_${TAG}_START ==="
docker rm -f $C 2>/dev/null
docker run -d --name $C --gpus '"device=0"' --shm-size=32g \
  -v /root/.cache/huggingface:/root/.cache/huggingface --entrypoint sleep $IMG infinity
sleep 5
for f in run_pow_generation.py collect_artifacts.py patch_runner.py metrics.py compressa_config_027.yml; do
  docker cp /root/$f $C:/root/ ; done
docker exec $C mv /root/compressa_config_027.yml /root/compressa_config.yml
docker exec $C python3 /root/patch_runner.py
docker exec $C pip install --no-cache-dir --quiet toml accelerate fire fastrlock h2 termcolor typer-slim setuptools-scm tenacity 2>&1|tail -1
docker exec $C python3 -m venv /root/cpvenv 2>&1|tail -1
docker exec $C /root/cpvenv/bin/pip install --quiet compressa-perf 2>&1|tail -1
docker exec -d $C bash -c '
export WATCHER_MAX_UNHEALTHY_COUNT=9999
export PYTHONPATH="/app:/app/packages/api/src:/app/packages/pow/src:/app/packages/train/src:/app/packages/common/src"
exec python3 -m uvicorn api.app:app --host 0.0.0.0 --port 8081 --app-dir /app/packages/api/src > /root/api.log 2>&1'
for i in $(seq 1 40); do docker exec $C curl -s -m 5 $API/health >/dev/null 2>&1 && break; sleep 5; done
echo "=== FT_${TAG}_API_READY ==="

run () {
  MODE=$1
  if [ "$MODE" = "eager" ]; then EXTRA='"--enforce-eager"'
  else EXTRA='"--compilation-config","{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"]}"'; fi
  echo "=== FT_${TAG}_${MODE}_UP ==="
  docker exec $C curl -s -X POST $API/api/v1/inference/up/async -H 'Content-Type: application/json' \
    -d "{\"model\":\"$M\",\"dtype\":\"auto\",\"additional_args\":[$EXTRA]}" -m 60 | head -c 250; echo
  for i in $(seq 1 240); do
    st=$(docker exec $C curl -s -m 6 $API/api/v1/inference/up/status 2>/dev/null \
         | python3 -c "import sys,json;print(json.load(sys.stdin).get('is_running'))" 2>/dev/null)
    [ "$st" = "True" ] && { echo "=== FT_${TAG}_${MODE}_UP_OK after $((i*15))s ==="; break; }
    er=$(docker exec $C curl -s -m 6 $API/api/v1/inference/up/status 2>/dev/null | grep -o '"error":"[^"]*"' | head -1)
    [ -n "$er" ] && { echo "=== FT_${TAG}_${MODE}_LOAD_FAILED $er ==="
      docker cp $C:/root/api.log $OUT/${MODE}_api.log 2>/dev/null
      echo "--- первопричина ---"; grep -aE "RuntimeError|Error:|Unsupported|assert|KeyError|AttributeError|ValueError" $OUT/${MODE}_api.log | grep -viE "watcher|common.manager|inference.manager" | head -6
      return 1; }
    sleep 15
  done
  # прогрев: форсирует ленивый JIT, иначе свип ловит 502
  docker exec $C curl -s -m 600 -X POST $API/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"warmup\"}],\"max_tokens\":16}" | head -c 80; echo
  echo "=== FT_${TAG}_${MODE}_WARM ==="
  docker exec -e HOST_IP=127.0.0.1 -e MODEL="$M" $C python3 -u /root/run_pow_generation.py --phase 3 --skip-check 2>&1 \
    | tee $OUT/${MODE}_poc_sweep.log | tail -18
  echo "=== FT_${TAG}_${MODE}_SWEEP_DONE ==="
  docker exec $C python3 -u /root/collect_artifacts.py --url $API --model "$M" \
    --output-dir /root/${MODE}_arts --nonces 1000 --batch-size 32 --logprobs-count 0 \
    --gpu "B300-$TAG" --vllm-version "0.25.1-$MODE" 2>&1 | tee $OUT/${MODE}_collect.log | grep -E "Nonces saved|Error" | head -2
  docker cp $C:/root/${MODE}_arts/nonces_1000.json $OUT/${MODE}_nonces.json
  echo "=== FT_${TAG}_${MODE}_COLLECT_DONE ==="
  docker exec $C bash -c "python3 - <<'PY'
import re
s=open('/root/compressa_config.yml').read().replace('MODEL_PLACEHOLDER','$M')
b=[x for x in re.split(r'\n(?=- model_name:)', s) if x.strip().startswith('- model_name')]
for i,x in enumerate(b,1): open('/root/sc%d.yml'%i,'w').write(x)
PY"
  for n in 1 2 3 4; do
    docker exec $C /root/cpvenv/bin/compressa-perf measure-from-yaml --db /root/${MODE}.sqlite /root/sc${n}.yml 2>&1 \
      | grep -E "Number of failed|Error" | head -2
  done
  docker exec $C /root/cpvenv/bin/python /root/metrics.py /root/${MODE}.sqlite > $OUT/${MODE}_serving.json 2>/dev/null
  echo "=== FT_${TAG}_${MODE}_SERVING_DONE ==="
  docker cp $C:/root/api.log $OUT/${MODE}_api.log 2>/dev/null
  docker exec $C curl -s -X POST $API/api/v1/inference/down -m 90 | head -c 60; echo
  sleep 35
}
run eager
run graphs
docker cp $C:/root/api.log $OUT/final_api.log 2>/dev/null
docker rm -f $C
echo "=== FT_${TAG}_DONE ==="
