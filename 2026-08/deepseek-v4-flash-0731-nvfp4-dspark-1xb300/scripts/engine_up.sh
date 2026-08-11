#!/bin/bash
set -u
curl -s -X POST http://127.0.0.1:8081/api/v1/inference/up/async \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/nvfp4","dtype":"auto","additional_args":[]}' -m 60
echo
for i in $(seq 1 360); do
  S=$(curl -s -m 6 http://127.0.0.1:8081/api/v1/inference/up/status 2>/dev/null)
  echo "$S" | grep -q '"is_running":true' && { echo "ENGINE_UP after $((i*15))s"; exit 0; }
  echo "$S" | grep -q '"status":"failed"' && { echo "ENGINE_FAILED"; echo "$S" | head -c 400; exit 1; }
  sleep 15
done
echo ENGINE_TIMEOUT
