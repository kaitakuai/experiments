#!/bin/bash
# Prepare the k10 container on our own B300 box (TP=1).
# Everything runs INSIDE the container: on our own machines the image is just a
# docker image, the host has no uvicorn/hf.
set -u
C=v4box
echo "=== WAIT_CONTAINER ==="
for i in $(seq 1 90); do docker ps --format '{{.Names}}' | grep -qx $C && break; sleep 10; done
docker ps --format '{{.Names}}\t{{.Status}}' | grep $C

echo "=== FIXES ==="
docker exec $C bash -lc '
L=/usr/local/cuda/targets/x86_64-linux/lib
ln -sf $L/libnvrtc.so.13 $L/libnvrtc.so 2>/dev/null
ln -sf /usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libnvrtc.so.13 /usr/local/lib/libnvrtc.so 2>/dev/null
ldconfig
echo "int main(){return 0;}" > /tmp/t.c && gcc /tmp/t.c -lnvrtc -o /tmp/t && echo NVRTC_OK || echo NVRTC_FAIL
sed -i "s/^VLLM_USE_V2_MODEL_RUNNER=0/VLLM_USE_V2_MODEL_RUNNER=1/" /etc/environment
grep V2_MODEL /etc/environment
pip install --no-cache-dir -q toml accelerate fire fastrlock h2 termcolor typer-slim setuptools-scm tenacity aiohttp 2>&1 | tail -1
python3 -c "import toml, fire, tenacity, aiohttp; print(\"DEPS_OK\")"'

echo "=== WEIGHTS (official 0731) ==="
docker exec $C bash -lc 'export HF_XET_HIGH_PERFORMANCE=1
hf download deepseek-ai/DeepSeek-V4-Flash-0731 --revision 9e165c30e2704aec5d9d593cce3eebd58bbef1cb 2>&1 | tail -2'
docker exec $C du -sh /root/.cache/huggingface/hub

echo "=== API ==="
docker exec $C bash -lc 'cat > /root/start_api.sh <<APIEOF
export WATCHER_MAX_UNHEALTHY_COUNT=9999
export VLLM_USE_V2_MODEL_RUNNER=1
export PYTHONPATH="/app:/app/packages/api/src:/app/packages/pow/src:/app/packages/train/src:/app/packages/common/src"
exec python3 -m uvicorn api.app:app --host 0.0.0.0 --port 8081 --app-dir /app/packages/api/src
APIEOF
chmod +x /root/start_api.sh'
docker exec -d $C bash -lc 'setsid bash /root/start_api.sh > /root/api.log 2>&1 < /dev/null'
sleep 18
docker exec $C curl -s -m 6 http://127.0.0.1:8081/health | head -c 60; echo
echo "=== SETUP_DONE ==="
