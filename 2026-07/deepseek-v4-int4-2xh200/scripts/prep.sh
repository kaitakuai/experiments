#!/bin/bash
# Подготовка Vast-бокса под замеры V4. Аргументы: <TP> [--patch]
set -x
TP=${1:?TP}; PATCH=${2:-}
echo "=== HW ==="
nvidia-smi --query-gpu=index,name,memory.total,power.max_limit,driver_version --format=csv,noheader
nvidia-smi topo -m | head -6
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
echo "compute_cap=$CAP"

echo "=== libnvrtc (нужен на sm_90, FlashInfer JIT линкует -lnvrtc) ==="
for D in /usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvrtc/lib /usr/local/cuda/lib64; do
  [ -d "$D" ] || continue
  SRC=$(ls $D/libnvrtc.so.1[0-9]* 2>/dev/null | grep -vE "builtins|alt" | head -1)
  [ -n "$SRC" ] && ln -sf "$SRC" "$D/libnvrtc.so" && echo "  $D/libnvrtc.so -> $(basename $SRC)"
done
ldconfig 2>/dev/null

echo "=== зависимости mlnode ==="
pip install --no-cache-dir --quiet toml accelerate fire fastrlock h2 termcolor typer-slim setuptools-scm tenacity 2>&1|tail -1
python3 -c "import toml,fire,tenacity; print('  deps OK')"

if [ "$PATCH" = "--patch" ]; then
  echo "=== накатываю PR #45645 (INT4) ==="
  cd /usr/local/lib/python3.12/dist-packages && patch -p1 --fuzz=5 < /root/pr45645_vllm_only.diff 2>&1|tail -4
  python3 -c "
import py_compile
for f in ['vllm/models/deepseek_v4/nvidia/flashmla.py','vllm/models/deepseek_v4/nvidia/model.py','vllm/model_executor/layers/quantization/utils/mxfp4_utils.py']:
    py_compile.compile('/usr/local/lib/python3.12/dist-packages/'+f, doraise=True)
print('  патч применён, файлы компилируются')"
fi

echo "=== runner.py: TP=$TP, 400k, 32768 ==="
TP=$TP python3 /root/patch_runner_tp.py

echo "=== compressa в отдельный venv ==="
python3 -m venv /root/cpvenv 2>&1|tail -1
/root/cpvenv/bin/pip install --quiet compressa-perf 2>&1|tail -1

echo "=== API на 8081 ==="
mkdir -p /root/out
WATCHER_MAX_UNHEALTHY_COUNT=9999 \
PYTHONPATH="/app:/app/packages/api/src:/app/packages/pow/src:/app/packages/train/src:/app/packages/common/src" \
setsid python3 -m uvicorn api.app:app --host 0.0.0.0 --port 8081 --app-dir /app/packages/api/src \
  </dev/null > /root/api.log 2>&1 &
for i in $(seq 1 40); do curl -s -m 5 http://127.0.0.1:8081/health >/dev/null 2>&1 && break; sleep 5; done
curl -s -m 6 http://127.0.0.1:8081/health | head -c 160; echo
echo "=== PREP_DONE ==="
