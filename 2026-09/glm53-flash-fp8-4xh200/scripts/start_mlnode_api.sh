#!/bin/bash
# Старт mlnode API и подъём vLLM через него — тот же путь, что в проде.
#
# WATCHER_MAX_UNHEALTHY_COUNT задран: штатный наблюдатель убивает API во время долгого
# холодного старта, а у GLM-5.3 на Hopper он занимает больше десяти минут.
#
# Порт 8081, потому что 8080 на Vast занят их порталом.
#
# После подъёма скрипт печатает ФАКТИЧЕСКУЮ командную строку vLLM из лога и сверяет её с
# тем, что зашито в runner.py: с Kimi цепочка передавливала образные значения, и полагаться
# на «мы же попросили» нельзя.
set -u
exec >/root/mlnode_api.log 2>&1
echo "START $(date +%T)"
ulimit -n 524288

cd /app/packages/api/src
export WATCHER_MAX_UNHEALTHY_COUNT=9999
# mlnode разложен по четырём пакетам, в PYTHONPATH нужны все: api импортирует common,
# а измеритель PoC — pow
export PYTHONPATH=/app/packages/api/src:/app/packages/common/src:/app/packages/pow/src:/app/packages/train/src:${PYTHONPATH:-}
setsid nohup python3 -m uvicorn api.app:app --host 0.0.0.0 --port 8081 \
  --app-dir /app/packages/api/src </dev/null >/root/api.log 2>&1 &
echo "API запущен, pid=$!"

for i in $(seq 1 40); do
  sleep 5
  curl -s -m 6 http://127.0.0.1:8081/health >/dev/null 2>&1 && break
done
curl -s -m 6 http://127.0.0.1:8081/health | head -c 300; echo

SNAP=$(ls -d /root/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash/snapshots/*/ | head -1)
echo "модель: $SNAP"
echo "=== поднимаю vLLM через mlnode ==="
curl -s -X POST http://127.0.0.1:8081/api/v1/inference/up/async \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SNAP\",\"dtype\":\"auto\"}" -m 60 | head -c 300; echo

for i in $(seq 1 240); do
  ST=$(curl -s -m 8 http://127.0.0.1:8081/api/v1/inference/up/status \
       | python3 -c "import sys,json;print(json.load(sys.stdin).get('is_running'))" 2>/dev/null)
  [ "$ST" = "True" ] && { echo "VLLM_UP через $((i*15))c $(date +%T)"; break; }
  sleep 15
done

echo "=== фактическая командная строка vLLM ==="
grep -aoE "gonka-vllm-serve.*|vllm serve.*" /root/api.log | tail -1 | tr ' ' '\n' | paste -sd' ' - | cut -c1-900
echo
echo "=== ключевые параметры, как их видит движок ==="
grep -aE "tensor_parallel_size=|kv_cache_dtype=|max_num_seqs=|block_size=" /root/api.log | tail -1 | cut -c1-400
grep -aE "attention backend|GPU KV cache size" /root/api.log | tail -2 | cut -c1-170
echo API_DONE
