#!/bin/bash
# V4 eager-vs-CUDA-graph A/B на ОДНОЙ карте прод-B300, всё внутри контейнера.
# Прод не трогаем: только GPU 0, отдельный контейнер, отдельный DOCKER_CONFIG.
set -x
IMG=ghcr.io/kaitakuai/mlnode-b300-deepseek-v4-flash:0.2.13-vllm0.25.1-overlay-k4
C=v4ab
OUT=/root/v4out
mkdir -p $OUT

echo "=== V4AB_START ==="

docker rm -f $C 2>/dev/null
docker run -d --name $C --gpus '"device=0"' --shm-size=32g \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint sleep $IMG infinity
sleep 5
docker cp /root/run_pow_generation.py $C:/root/
docker cp /root/collect_artifacts.py  $C:/root/

echo "=== V4AB_PATCH_RUNNER ==="
docker cp /root/patch_runner.py $C:/root/patch_runner.py
docker exec $C python3 /root/patch_runner.py
docker exec $C grep -A 8 "_forced = \[" /app/packages/api/src/api/inference/vllm/runner.py | head -10

echo "=== V4AB_DEPS ==="
docker exec $C pip install --no-cache-dir --quiet \
  toml accelerate fire fastrlock h2 termcolor typer-slim setuptools-scm tenacity 2>&1 | tail -3
docker exec $C python3 -c "import toml, fire, tenacity; print('deps OK')"

echo "=== V4AB_API_UP ==="
docker exec -d $C bash -c '
export WATCHER_MAX_UNHEALTHY_COUNT=9999
export PYTHONPATH="/app:/app/packages/api/src:/app/packages/pow/src:/app/packages/train/src:/app/packages/common/src"
exec python3 -m uvicorn api.app:app --host 0.0.0.0 --port 8081 --app-dir /app/packages/api/src > /root/api.log 2>&1'
for i in $(seq 1 40); do
  docker exec $C curl -s -m 5 http://127.0.0.1:8081/health >/dev/null 2>&1 && break; sleep 5
done
docker exec $C curl -s -m 6 http://127.0.0.1:8081/health | head -c 200; echo
echo "=== V4AB_API_READY ==="

run_mode () {
  MODE=$1
  if [ "$MODE" = "eager" ]; then EXTRA='"--enforce-eager"'
  else EXTRA='"--compilation-config","{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"]}"'; fi

  echo "=== V4AB_UP_$MODE ==="
  docker exec $C curl -s -X POST http://127.0.0.1:8081/api/v1/inference/up/async \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"deepseek-ai/DeepSeek-V4-Flash\",\"dtype\":\"auto\",\"additional_args\":[$EXTRA]}" -m 60 | head -c 200; echo

  for i in $(seq 1 220); do
    st=$(docker exec $C curl -s -m 6 http://127.0.0.1:8081/api/v1/inference/up/status 2>/dev/null \
         | python3 -c "import sys,json;print(json.load(sys.stdin).get('is_running'))" 2>/dev/null)
    [ "$st" = "True" ] && { echo "=== V4AB_${MODE}_VLLM_UP after $((i*15))s ==="; break; }
    sleep 15
  done

  echo "=== V4AB_${MODE}_GRAPHFLAGS ==="
  docker exec $C grep -aiE "BREAKABLE_CUDAGRAPH|Breakable CUDA graph|torch.compile pipeline|enforce.eager|GPU KV cache size" /root/api.log | tail -6

  echo "=== V4AB_${MODE}_SWEEP ==="
  docker exec -e HOST_IP=127.0.0.1 $C python3 -u /root/run_pow_generation.py --phase 3 --skip-check 2>&1 \
    | tee $OUT/${MODE}_poc_sweep.log | tail -20
  echo "=== V4AB_${MODE}_SWEEP_DONE ==="

  echo "=== V4AB_${MODE}_COLLECT ==="
  docker exec $C python3 -u /root/collect_artifacts.py --url http://127.0.0.1:8081 \
    --model deepseek-ai/DeepSeek-V4-Flash --output-dir /root/${MODE}_arts \
    --nonces 1000 --batch-size 32 --logprobs-count 0 \
    --gpu "B300-prod" --vllm-version "0.25.1-$MODE" 2>&1 | tee $OUT/${MODE}_collect.log | tail -5
  docker cp $C:/root/${MODE}_arts/nonces_1000.json $OUT/${MODE}_nonces_1000.json 2>/dev/null
  echo "=== V4AB_${MODE}_COLLECT_DONE ==="

  docker exec $C curl -s -X POST http://127.0.0.1:8081/api/v1/inference/down -m 60 | head -c 120; echo
  sleep 30
  echo "=== V4AB_${MODE}_ALL_DONE ==="
}

run_mode eager
run_mode compiled

docker exec $C nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "=== V4AB_DONE ==="
