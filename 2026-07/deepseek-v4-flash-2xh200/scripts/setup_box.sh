#!/bin/bash
# Полная подготовка бокса под V4-бенчмарк. Учтены все грабли H100-прогона.
set -x
exec 2>&1

echo "=== 1) HW-верификация (не доверять лейблам) ==="
nvidia-smi --query-gpu=index,name,memory.total,power.max_limit,driver_version --format=csv,noheader
nvidia-smi topo -m 2>/dev/null | head -5
df -h / | tail -1

echo "=== 2) sm_90-фикс: libnvrtc.so (FlashInfer JIT линкует -lnvrtc) ==="
for D in /usr/local/cuda/lib64 /usr/local/cuda/targets/x86_64-linux/lib; do
  SRC=$(ls $D/libnvrtc.so.1[0-9]* 2>/dev/null | grep -vE "builtins|alt" | head -1)
  [ -n "$SRC" ] && ln -sf "$SRC" "$D/libnvrtc.so" && echo "  $D/libnvrtc.so -> $(basename $SRC)"
done
ldconfig

echo "=== 3) mlnode-зависимости (иначе api.app падает на toml) ==="
pip install -q toml accelerate fire fastrlock h2 termcolor typer-slim setuptools-scm tenacity 2>&1 | tail -1

echo "=== 4) compressa-perf в ОТДЕЛЬНЫЙ venv (в системном он ломает openai для vLLM) ==="
python3 -m venv /root/cpvenv
/root/cpvenv/bin/pip install -q git+https://github.com/product-science/compressa-perf.git 2>&1 | tail -1
/root/cpvenv/bin/compressa-perf --help >/dev/null 2>&1 && echo "  compressa OK"
python3 -c "import vllm; from vllm.entrypoints.openai.cli_args import make_arg_parser; print('  vLLM CLI OK', vllm.__version__)"

echo "=== 5) runner.py: только параметры (TP=2 для H200) ==="
R=/app/packages/api/src/api/inference/vllm/runner.py
cp $R $R.bak-orig
python3 - <<'PY'
R="/app/packages/api/src/api/inference/vllm/runner.py"
s=open(R).read()
pairs=[("('--tensor-parallel-size', '1')","('--tensor-parallel-size', '2')"),
       ("('--max-model-len', '200000')","('--max-model-len', '400000')"),
       ("('--max-num-batched-tokens', '16384')","('--max-num-batched-tokens', '32768')")]
for o,n in pairs:
    assert o in s, "НЕ НАЙДЕНО: "+o
    s=s.replace(o,n)
open(R,"w").write(s)
print("  runner.py patched")
PY
grep -A 8 "_b300_dsv4_plugin_forced = \[" $R | head -9

echo "=== 6) старт mlnode API (порт 8081, толерантный watcher) ==="
cat > /root/start_api.sh <<'EOS'
#!/bin/bash
export WATCHER_MAX_UNHEALTHY_COUNT=9999
export PYTHONPATH="/app:/app/packages/api/src:/app/packages/pow/src:/app/packages/train/src:/app/packages/common/src"
cd /app
exec python3 -m uvicorn api.app:app --host 0.0.0.0 --port 8081 --app-dir /app/packages/api/src
EOS
chmod +x /root/start_api.sh
setsid bash /root/start_api.sh </dev/null > /root/api.log 2>&1 &
sleep 18
curl -s -m 6 http://127.0.0.1:8081/health | python3 -c "import sys,json;d=json.load(sys.stdin);print('  API:',d['status'],'| GPU:',d['gpu']['count'])"

echo "=== 7) скачивание весов (setsid, переживёт дисконнект) ==="
setsid env HF_HUB_ENABLE_HF_TRANSFER=1 hf download deepseek-ai/DeepSeek-V4-Flash --max-workers 16 </dev/null > /root/dl.log 2>&1 &
echo "=== SETUP_DONE (загрузка идёт в фоне) ==="
