#!/bin/bash
# Prepare the k4 container for the NVFP4 checkpoint on a 1×B300 card.
#
# NOTE: no image fixes are applied here. The k4 image needs none — libnvrtc
# links, VLLM_USE_V2_MODEL_RUNNER is absent so vLLM selects the V2 runner on
# its own, and the venv carries the API deps. That is a result of this run, not
# an assumption: the diagnostics below print it.
set -u
C=${C:-nvfp4a}
IMAGE=${IMAGE:?}
GPUS=${GPUS:-'"device=0"'}
WEIGHTS=${WEIGHTS:-/root/nvfp4}

echo "=== CONTAINER ==="
docker rm -f $C >/dev/null 2>&1
docker run -d --name $C --gpus "$GPUS" --shm-size=32g \
  -v "$WEIGHTS":/models/nvfp4:ro --entrypoint sleep "$IMAGE" infinity
sleep 3
docker image inspect "$IMAGE" --format '{{index .RepoDigests 0}}' 2>/dev/null || true

echo "=== DIAGNOSTICS (nothing is fixed here) ==="
docker exec $C bash -lc '
echo -n "nvrtc_link: "; echo "int main(){return 0;}" > /tmp/t.c
gcc /tmp/t.c -lnvrtc -o /tmp/t 2>/dev/null && echo OK || echo FAIL
echo -n "V2_env: "; env | grep VLLM_USE_V2_MODEL_RUNNER || echo ABSENT
echo -n "vllm: "; /app/packages/api/.venv/bin/python -c "import vllm;print(vllm.__version__)"'

echo "=== API ==="
# The image sets PYTHONPATH=/app:/app/packages/api/src, but `common` lives in
# /app/packages/common/src; the venv .pth files add it, so the venv must be
# activated (this is what /app/entrypoint.sh does in production).
cat > /tmp/start_api.sh <<'APIEOF'
cd /app
source /app/packages/api/.venv/bin/activate
export WATCHER_MAX_UNHEALTHY_COUNT=9999
exec uvicorn api.app:app --host=0.0.0.0 --port=8081
APIEOF
docker cp /tmp/start_api.sh $C:/root/start_api.sh   # heredocs do not survive `docker exec`
docker exec -d $C bash -lc 'setsid bash /root/start_api.sh > /root/api.log 2>&1 < /dev/null'
sleep 22
docker exec $C curl -s -m 6 http://127.0.0.1:8081/health | head -c 90; echo
echo "=== SETUP_DONE ==="
