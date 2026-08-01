#!/bin/bash
# Prepare a Vast 4xH100 box running the k10 image for the 0731 campaign.
set -u
echo "=== SETUP_START ==="

# k10 still ships only the versioned libnvrtc; without the plain .so FlashInfer
# cannot link its sm90 kernel and the worker dies on the first forward.
L=/usr/local/cuda/targets/x86_64-linux/lib
ln -sf $L/libnvrtc.so.13 $L/libnvrtc.so 2>/dev/null
ln -sf /usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libnvrtc.so.13 /usr/local/lib/libnvrtc.so 2>/dev/null
ldconfig
echo "int main(){return 0;}" > /tmp/t.c
gcc /tmp/t.c -lnvrtc -o /tmp/t && echo "NVRTC_LINK_OK" || echo "NVRTC_LINK_FAIL"

# k10 still ships VLLM_USE_V2_MODEL_RUNNER=0, which disables DSpark entirely.
sed -i 's/^VLLM_USE_V2_MODEL_RUNNER=0/VLLM_USE_V2_MODEL_RUNNER=1/' /etc/environment
grep V2_MODEL /etc/environment

pip install --no-cache-dir -q toml accelerate fire fastrlock h2 termcolor \
  typer-slim setuptools-scm tenacity aiohttp 2>&1 | tail -1
python3 -c "import toml, fire, tenacity, aiohttp; print('DEPS_OK')"

cat > /root/dl.sh <<'DLEOF'
export HF_XET_HIGH_PERFORMANCE=1
hf download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --revision 9e165c30e2704aec5d9d593cce3eebd58bbef1cb > /root/dl.log 2>&1
echo "DL_EXIT=$?" >> /root/dl.log
DLEOF
setsid bash /root/dl.sh > /dev/null 2>&1 < /dev/null &

cat > /root/start_api.sh <<'APIEOF'
export WATCHER_MAX_UNHEALTHY_COUNT=9999
export VLLM_USE_V2_MODEL_RUNNER=1
export PYTHONPATH="/app:/app/packages/api/src:/app/packages/pow/src:/app/packages/train/src:/app/packages/common/src"
exec python3 -m uvicorn api.app:app --host 0.0.0.0 --port 8081 --app-dir /app/packages/api/src
APIEOF
setsid bash /root/start_api.sh > /root/api.log 2>&1 < /dev/null &
sleep 15
curl -s -m 6 http://127.0.0.1:8081/health | head -c 60; echo

for i in $(seq 1 90); do
  grep -q "DL_EXIT=0" /root/dl.log 2>/dev/null && { echo "WEIGHTS_OK after $((i*20))s"; break; }
  grep -q "DL_EXIT=[1-9]" /root/dl.log 2>/dev/null && { echo "WEIGHTS_FAIL"; tail -4 /root/dl.log; break; }
  sleep 20
done
du -sh /root/.cache/huggingface/hub
echo "=== SETUP_DONE ==="
