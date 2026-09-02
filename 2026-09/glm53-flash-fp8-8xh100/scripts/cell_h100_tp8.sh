#!/bin/bash
# 8×H100 на образе k3, TP=8 — единственная топология, которой в кампании нет.
#
# Зачем восемь карт: у H100 по 80 ГБ, при TP=4 веса займут 76 ГБ и на KV-кэш не останется
# ничего. На H200 (141 ГБ) и B200 (183 ГБ) четырёх карт хватает, поэтому там TP=4 — рабочая
# топология, а не урезанная; зашитый в образ TP=8 рассчитан как раз на 80-гигабайтные карты.
#
# Здесь снимаем только честное плечо: нужна третья архитектура в матрице (H100 отсутствует
# вовсе) и первый замер при TP=8.
#
# Зачем именно TP=2: августовский эталон B300 снят на TP=2 и на СТАРОМ образе, а наш
# сентябрьский B200 — на TP=4 и на k3. Этот прогон стоит ровно между ними и раскладывает
# расхождение по NVFP4 на два независимых множителя:
#
#   этот B300 (k3) против августовского B300 (0.6.17) → изолирует СБОРКУ
#   этот B300 (k3) против сентябрьского B200 (k3)     → изолирует АРХИТЕКТУРУ
#
# Сейчас в наблюдаемых 0.74 между двумя прогонами одного NVFP4-чекпойнта эти два фактора
# смешаны, и по отдельности не измерены ни разу.
#
# Переменные NCCL — из B300-CONTAINER.md. На контейнерном Blackwell без них движок не
# поднимается вовсе: воркеры крутят по ядру, карты и диск простаивают, лог молчит часами.
set -u
exec >/root/cell_h100.log 2>&1
echo "=== СТАРТ $(date +%T) ==="

nvidia-smi --query-gpu=index,name,memory.total,power.max_limit,driver_version --format=csv,noheader
nvidia-smi topo -m | head -6
python3 -c "import vllm,flashinfer;print('vllm',vllm.__version__,'flashinfer',flashinfer.__version__)"
python3 -c "import gonka_poc;print('gonka_poc',gonka_poc.__version__)"
echo "патч индексера (ожидается >=2): $(grep -c 'torch.full' \
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py)"
echo "мягкий лимит дескрипторов: $(ulimit -Sn)"
df -h / | tail -1

sed -i 's|/api/v1/inference/pow|/api/v1/pow|g' /root/collect_artifacts.py
python3 - <<'PY'
p = "/root/collect_artifacts.py"; s = open(p).read()
old = "if elapsed > 600:  # 10 min timeout"
new = 'if elapsed > float(os.environ.get("POC_COLLECT_TIMEOUT", 600)):'
if "POC_COLLECT_TIMEOUT" not in s and old in s:
    s = s.replace(old, new, 1)
    if "\nimport os\n" not in s:
        s = s.replace("import json", "import json\nimport os", 1)
    import ast; ast.parse(s); open(p, "w").write(s); print("таймаут сборщика настраивается")
PY

# Честные веса качаем синхронно — они нужны первыми. Фрод-набор уходит в фон: пока идёт
# подъём движка и честная тройка семян (это 20-30 минут), он успевает приехать сам.
# Раньше скрипт ждал обе закачки перед стартом и терял на этом полчаса аренды.
echo "DOWNLOAD honest START $(date +%T)"
hf download zai-org/GLM-5.3-Flash --max-workers 16 > /root/dl_honest.log 2>&1
echo "DOWNLOAD honest END $(date +%T)"


for M in zai-org--GLM-5.3-Flash; do
  SNAP=$(ls -d /root/.cache/huggingface/hub/models--$M/snapshots/*/ 2>/dev/null | head -1)
  echo -n "$M: "
  python3 - "$SNAP" <<'PY'
import json, glob, os, sys
s = sys.argv[1]
if not s: print("снапшота нет"); raise SystemExit
idx = os.path.join(s, "model.safetensors.index.json")
if not os.path.exists(idx): print("индекса нет"); raise SystemExit
need = set(json.load(open(idx))["weight_map"].values())
have = {os.path.basename(p) for p in glob.glob(s + "/*.safetensors")}
print("%d/%d %s" % (len(have), len(need), "OK" if not need - have else "НЕТ"))
PY
done
echo "недокачанных файлов: $(find /root/.cache/huggingface -name '*.incomplete' | wc -l)"

start_engine() {  # $1 = каталог модели в кэше
  local REPO=$1 SNAP RD=no
  for P in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$P" 2>/dev/null; done
  pkill -f "[g]onka-vllm-serve" 2>/dev/null; pkill -f "[V]LLM::" 2>/dev/null
  sleep 10
  for i in $(seq 1 60); do
    U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    [ "${U:-9999}" -lt 2000 ] && break
    sleep 5
  done
  SNAP=$(ls -d /root/.cache/huggingface/hub/$REPO/snapshots/*/ | head -1)
  cat > /root/serve.sh <<EOF
#!/bin/bash
ulimit -n 524288
export VLLM_ENGINE_READY_TIMEOUT_S=3600
exec gonka-vllm-serve \\
  --model "$SNAP" --served-model-name glm53 \\
  --disable-custom-all-reduce \\
  --tensor-parallel-size 8 \\
  --kv-cache-dtype fp8 --block-size 2304 --max-num-seqs 256 \\
  --max-num-batched-tokens 65536 \\
  --no-enable-flashinfer-autotune --logprobs-mode processed_logprobs \\
  --worker-extension-cls gonka_poc.worker.PoCWorkerExtension \\
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \\
  --host 0.0.0.0 --port 8081
EOF
  chmod +x /root/serve.sh
  : > /root/vllm.log
  nohup bash /root/serve.sh >/root/vllm.log 2>&1 &
  for i in $(seq 1 320); do
    sleep 15
    curl -s -m 5 -o /dev/null http://127.0.0.1:8081/health 2>/dev/null && { RD=yes; break; }
  done
  echo "движок $REPO готов=$RD $(date +%T)"
  [ "$RD" = yes ] && grep -aE "attention backend|GPU KV cache size" /root/vllm.log | tail -2 | cut -c1-170
  [ "$RD" = yes ]
}

run_seed() {  # $1=метка $2=семя
  local LABEL=$1 SEED=$2 BH PK OUT CP
  BH=$(python3 -c "import json;print([x for x in json.load(open('/root/poc_seeds.json'))['seeds'] if x['id']=='$SEED'][0]['block_hash'])" 2>/dev/null)
  PK=$(python3 -c "import json;print([x for x in json.load(open('/root/poc_seeds.json'))['seeds'] if x['id']=='$SEED'][0]['public_key'])" 2>/dev/null)
  [ -n "$BH" ] && [ -n "$PK" ] || { echo "ПУСТОЕ СЕМЯ $SEED — прерываю"; return 1; }
  curl -s -m 20 -X POST http://127.0.0.1:8081/api/v1/pow/stop -H "Content-Type: application/json" -d '{}' >/dev/null 2>&1
  sleep 5
  OUT=/root/out_${LABEL}_$SEED; rm -rf "$OUT"; mkdir -p "$OUT"
  nohup python3 -u /root/collect_artifacts.py --url http://127.0.0.1:8081 --model glm53 \
    --output-dir "$OUT" --block-hash "$BH" --public-key "$PK" \
    --nonces 1000 --batch-size 16 > "$OUT/c.log" 2>&1 &
  CP=$!
  for i in $(seq 1 300); do
    grep -qa "Nonces saved" "$OUT/c.log" && break
    kill -0 "$CP" 2>/dev/null || break
    sleep 5
  done
  kill -9 "$CP" 2>/dev/null   # до фаз, которые роняют движок
  for p in $(ls /proc | grep -E '^[0-9]+$'); do
    grep -qa collect_artifacts /proc/$p/cmdline 2>/dev/null && kill -9 "$p" 2>/dev/null
  done
  sleep 3
  echo "${LABEL}/${SEED}: $(python3 - "$OUT" <<'PY'
import glob, json, sys
fs = glob.glob(sys.argv[1] + "/nonces_*.json")
if not fs: print("файла нет")
else:
    d = json.load(open(fs[0])); a = d.get("artifacts", []); t = d.get("generation_time_sec") or 0
    print("%d нонсов за %.1f c = %.0f нонсов/мин" % (len(a), t, (len(a)/t*60) if t else 0))
PY
) | IMA=$(grep -ac 'illegal memory' /root/vllm.log)"
}

load_test() {  # $1 = метка плеча
  echo "--- инференс под нагрузкой ($1) ---"
  python3 - <<'PY'
import concurrent.futures as cf, time, requests
def run(n, mx=800):
    P={"model":"glm53","messages":[{"role":"user","content":"Explain paged attention in detail."}],
       "max_tokens":mx,"temperature":0.7}
    def one(i):
        t=time.monotonic()
        r=requests.post("http://127.0.0.1:8081/v1/chat/completions",json=P,timeout=900)
        r.raise_for_status(); return time.monotonic()-t, r.json()["usage"]["completion_tokens"]
    t0=time.monotonic()
    with cf.ThreadPoolExecutor(n) as ex: res=list(ex.map(one,range(n)))
    el=time.monotonic()-t0; tok=sum(x[1] for x in res); lat=sorted(x[0] for x in res)
    print("параллельно %2d: %.1f c, %d токенов, %.1f ток/с, задержка med %.1f c" %
          (n, el, tok, tok/el, lat[len(lat)//2]), flush=True)
for n in (1, 8, 20):
    try: run(n)
    except Exception as e: print("параллельно %d: ОШИБКА %r" % (n, e), flush=True)
PY
}

export POC_COLLECT_TIMEOUT=1800

echo "### честное плечо FP8, TP=8 — третья архитектура в матрице и первый замер при TP=8 ###"
if start_engine models--zai-org--GLM-5.3-Flash; then
  run_seed honest s4            # прогрев: первый прогон после старта занижен на ~11%; s4 не входит в отчётную тройку
  for S in s1 s2 s3; do run_seed honest "$S"; done
  run_seed honestrep s1             # повтор s1 = честный пол на этом железе
  load_test honest
else
  tail -25 /root/vllm.log
fi

echo "ИТОГ IMA=$(grep -ac 'illegal memory' /root/vllm.log) $(date +%T)"
echo CELL_DONE
