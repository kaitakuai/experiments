#!/usr/bin/env bash
# Native start of the b200-glm-5-2 image: DeepGEMM + cudagraph (image default). No runner patch.
MODE="${1:-cudagraph}"   # cudagraph | eager
API_DIR=/app/packages/api
PORT=8081
# clean
[ -f /tmp/mlnode_glm_api.pid ] && kill -9 "$(cat /tmp/mlnode_glm_api.pid)" 2>/dev/null
pkill -9 -f "uvicorn api.app" 2>/dev/null
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p; done
sleep 4
# extra args for eager mode (override baked default which is compiled/cudagraph)
EXTRA=""
[ "$MODE" = "eager" ] && EXTRA='"--enforce-eager"'
# start API detached with DeepGEMM forced; engine inherits env
setsid bash -c "
  source ${API_DIR}/.venv/bin/activate
  export PYTHONPATH=${API_DIR}/src
  export VLLM_USE_DEEP_GEMM=1 VLLM_MOE_USE_DEEP_GEMM=1 VLLM_USE_FLASHINFER_MOE_FP8=0
  exec uvicorn api.app:app --host 0.0.0.0 --port ${PORT}
" </dev/null &>/tmp/mlnode_glm_api.log &
echo $! > /tmp/mlnode_glm_api.pid
echo "API pid $(cat /tmp/mlnode_glm_api.pid), mode=$MODE"
for i in $(seq 1 60); do curl -sf http://localhost:${PORT}/health >/dev/null 2>&1 && break; sleep 2; done
# bring engine up
BODY="{\"model\": \"/dev/shm/GLM-5.2-FP8\", \"dtype\": \"auto\""
[ "$MODE" = "eager" ] && BODY="${BODY}, \"additional_args\": [\"--enforce-eager\"]"
BODY="${BODY}}"
echo "up request: $BODY"
curl -sf -X POST http://localhost:${PORT}/api/v1/inference/up/async -H 'Content-Type: application/json' -d "$BODY"
echo; echo "UPSENT"
