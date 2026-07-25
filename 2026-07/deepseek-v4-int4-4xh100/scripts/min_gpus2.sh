#!/bin/bash
# Минимум карт для INT4. Ключевое отличие: API перезапускается ПОСЛЕ правки runner.py,
# иначе модуль уже импортирован и правка не действует.
set -x
M=Intel/DeepSeek-V4-Flash-W4A16-AutoRound; API=http://127.0.0.1:8081; OUT=/root/out/mingpu
mkdir -p $OUT
try () {
  TP=$1; GMU=$2; MAXLEN=$3
  echo "=== MG2_TRY tp=$TP gmu=$GMU maxlen=$MAXLEN ==="
  for p in $(ps -eo pid,args|grep -F "uvicorn api.ap""p"|grep -v grep|awk '{print $1}'); do kill $p 2>/dev/null; done
  sleep 10
  python3 - <<PY
import re
R="/app/packages/api/src/api/inference/vllm/runner.py"
s=open(R+".bak-orig").read()
for f,v in [("--tensor-parallel-size","$TP"),("--max-model-len","$MAXLEN"),
            ("--max-num-batched-tokens","32768"),("--gpu-memory-utilization","$GMU")]:
    p=re.compile(r"\('%s', '[0-9.]+'\)"%re.escape(f))
    if p.search(s): s=p.sub("('%s', '%s')"%(f,v), s)
open(R,"w").write(s); print("  runner: TP=$TP gmu=$GMU maxlen=$MAXLEN")
PY
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((TP-1))) \
  WATCHER_MAX_UNHEALTHY_COUNT=9999 \
  PYTHONPATH="/app:/app/packages/api/src:/app/packages/pow/src:/app/packages/train/src:/app/packages/common/src" \
  setsid python3 -m uvicorn api.app:app --host 0.0.0.0 --port 8081 --app-dir /app/packages/api/src \
    </dev/null > /root/api_tp${TP}.log 2>&1 &
  for i in $(seq 1 30); do curl -s -m 5 $API/health >/dev/null 2>&1 && break; sleep 5; done
  curl -s -m 6 $API/api/v1/inference/up/async -X POST -H 'Content-Type: application/json' \
    -d "{\"model\":\"$M\",\"dtype\":\"auto\",\"additional_args\":[\"--enforce-eager\"]}" -m 60|head -c 130; echo
  ok=""
  for i in $(seq 1 100); do
    st=$(curl -s -m 6 $API/api/v1/inference/up/status 2>/dev/null|python3 -c "import sys,json;print(json.load(sys.stdin).get('is_running'))" 2>/dev/null)
    [ "$st" = "True" ] && { ok=1; echo "=== MG2_UP_OK tp=$TP after $((i*15))s ==="; break; }
    er=$(curl -s -m 6 $API/api/v1/inference/up/status 2>/dev/null|grep -o '"error":"[^"]*"'|head -1)
    [ -n "$er" ] && { echo "=== MG2_FAILED tp=$TP $er ==="
      grep -aE "OutOfMemory|No available memory|RuntimeError" /root/api_tp${TP}.log|grep -viE "watcher|manager"|head -2
      break; }
    sleep 15
  done
  if [ -n "$ok" ]; then
    echo "--- проверка что TP реально $TP ---"
    grep -aoE "tensor_parallel_size=[0-9]+" /root/api_tp${TP}.log|tail -1
    grep -aoE "Model loading took [0-9.]+ GiB|GPU KV cache size: [0-9,]+" /root/api_tp${TP}.log|tail -2
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader|tr '\n' ' '; echo
    curl -s -m 600 -X POST $API/v1/chat/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}"|head -c 50; echo
    HOST_IP=127.0.0.1 MODEL="$M" python3 -u /root/run_pow_generation.py --phase 3 --skip-check 2>&1 \
      | tee $OUT/tp${TP}_sweep.log | grep -E "Result:|Best batch"
    echo "=== MG2_SWEEP_DONE tp=$TP ==="
  fi
  curl -s -X POST $API/api/v1/inference/down -m 90 >/dev/null 2>&1; sleep 25
}
try 1 0.96 200000
try 2 0.96 200000
echo "=== MG2_DONE ==="
