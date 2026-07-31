#!/bin/bash
# Full setup of a Vast 2xH200 box for the DeepSeek-V4-Flash-0731 DSpark A/B.
# Every step here is a fix for a failure we already hit once; see FIX comments.
set -u
echo "=== SETUP_START ==="

# FIX 1: the k9 image ships no unversioned libnvrtc.so, so FlashInfer's JIT of the
# Hopper kernel fp8_blockscale_gemm_sm90 fails at link time ("cannot find -lnvrtc")
# and the worker dies on the first forward. Blackwell never takes this path.
L=/usr/local/cuda/targets/x86_64-linux/lib
ln -sf $L/libnvrtc.so.13 $L/libnvrtc.so
ln -sf /usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libnvrtc.so.13 /usr/local/lib/libnvrtc.so
ldconfig
echo "int main(){return 0;}" > /tmp/t.c
gcc /tmp/t.c -lnvrtc -o /tmp/t && echo "NVRTC_LINK_OK" || echo "NVRTC_LINK_FAIL"

# FIX 2: the mlnode API imports zeroband, which needs deps absent from the image.
pip install --no-cache-dir -q toml accelerate fire fastrlock h2 termcolor \
  typer-slim setuptools-scm tenacity aiohttp 2>&1 | tail -1
python3 -c "import toml, fire, tenacity, aiohttp; print('DEPS_OK')"

# FIX 3: k9 forces TP=1 / 200k context. Restore the exact args of the earlier
# 2xH200 -Flash run so the comparison stays 1:1.
python3 - <<'PYEOF'
p = "/app/packages/api/src/api/inference/vllm/runner.py"
s = open(p).read()
o = s
s = s.replace("('--tensor-parallel-size', '1')", "('--tensor-parallel-size', '2')")
s = s.replace("('--max-model-len', '200000')", "('--max-model-len', '400000')")
s = s.replace("('--max-num-batched-tokens', '16384')", "('--max-num-batched-tokens', '32768')")
assert s != o, "RUNNER_PATCH_FAILED"
open(p, "w").write(s)
print("RUNNER_PATCHED")
PYEOF
grep -A8 "_b300_dsv4_plugin_forced = \[" /app/packages/api/src/api/inference/vllm/runner.py | head -9

# weights (Xet pulls ~167 GB in about a minute on this host)
cat > /root/dl.sh <<'EOF'
export HF_XET_HIGH_PERFORMANCE=1
hf download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --revision 9e165c30e2704aec5d9d593cce3eebd58bbef1cb > /root/dl.log 2>&1
echo "DL_EXIT=$?" >> /root/dl.log
EOF
setsid bash /root/dl.sh > /dev/null 2>&1 < /dev/null &

cat > /root/start_api.sh <<'EOF'
export WATCHER_MAX_UNHEALTHY_COUNT=9999
export PYTHONPATH="/app:/app/packages/api/src:/app/packages/pow/src:/app/packages/train/src:/app/packages/common/src"
exec python3 -m uvicorn api.app:app --host 0.0.0.0 --port 8081 --app-dir /app/packages/api/src
EOF
setsid bash /root/start_api.sh > /root/api.log 2>&1 < /dev/null &
sleep 15
curl -s -m 6 http://127.0.0.1:8081/health | head -c 60; echo

# wait for weights
for i in $(seq 1 60); do
  grep -q "DL_EXIT=0" /root/dl.log 2>/dev/null && { echo "WEIGHTS_OK after $((i*20))s"; break; }
  grep -q "DL_EXIT=[1-9]" /root/dl.log 2>/dev/null && { echo "WEIGHTS_FAIL"; tail -5 /root/dl.log; break; }
  sleep 20
done
du -sh /root/.cache/huggingface/hub
echo "=== SETUP_DONE ==="
