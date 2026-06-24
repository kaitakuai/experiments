#!/usr/bin/env bash
ROLE="${1:-fp8}"; MODEL="${2:-/dev/shm/GLM-5.2-FP8}"; OUT="${3:-/root/work/2026-06/glm/artifacts/nonces_honest}"
pkill -9 -f coll.py 2>/dev/null
curl -s -X POST http://localhost:8081/api/v1/inference/pow/stop >/dev/null 2>&1
fuser -k 9998/tcp 2>/dev/null
sleep 5
sed 's|/api/v1/pow/|/api/v1/inference/pow/|g' /root/work/tools/collect_artifacts.py > /tmp/coll.py
exec python3 /tmp/coll.py --url http://localhost:8081 --model "$MODEL" --output-dir "$OUT" --nonces 1000 --batch-size 16 --gpu B300
