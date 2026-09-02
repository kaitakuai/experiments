#!/bin/bash
# Сбор нонсов по УЖЕ поднятому движку: прогрев, тройка семян, повтор s1 на пол, нагрузка.
#
# Нужен отдельно, потому что фоновый запуск большого скрипта через ssh не переживает
# закрытие сессии: движок стартует своим nohup и выживает, а сам скрипт ячейки умирает.
# Здесь наоборот — движок уже работает, скрипту остаётся только собрать.
set -u
: "${BATCH:=16}" "${LABEL:=honest}"
exec >>/root/seeds.log 2>&1
echo "=== СБОР $LABEL батч=$BATCH $(date +%T) ==="

curl -s -m 5 -o /dev/null http://127.0.0.1:8081/health || { echo "движок не отвечает"; echo SEEDS_DONE; exit 1; }
grep -aE "attention backend|GPU KV cache size" /root/vllm.log | tail -2 | cut -c1-170
echo "маршрут: $(curl -s -m 10 http://127.0.0.1:8081/api/v1/pow/versions | cut -c1-200)"

export POC_COLLECT_TIMEOUT=1800

run_seed() {  # $1=метка $2=семя
  local L=$1 S=$2 BH PK OUT CP
  BH=$(python3 -c "import json;print([x for x in json.load(open('/root/poc_seeds.json'))['seeds'] if x['id']=='$S'][0]['block_hash'])" 2>/dev/null)
  PK=$(python3 -c "import json;print([x for x in json.load(open('/root/poc_seeds.json'))['seeds'] if x['id']=='$S'][0]['public_key'])" 2>/dev/null)
  [ -n "$BH" ] && [ -n "$PK" ] || { echo "ПУСТОЕ СЕМЯ $S — прерываю"; return 1; }
  curl -s -m 20 -X POST http://127.0.0.1:8081/api/v1/pow/stop -H "Content-Type: application/json" -d '{}' >/dev/null 2>&1
  sleep 5
  OUT=/root/out_${L}_$S; rm -rf "$OUT"; mkdir -p "$OUT"
  nohup python3 -u /root/collect_artifacts.py --url http://127.0.0.1:8081 --model glm53 \
    --output-dir "$OUT" --block-hash "$BH" --public-key "$PK" \
    --nonces 1000 --batch-size "$BATCH" > "$OUT/c.log" 2>&1 &
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
  echo "${L}/${S}: $(python3 - "$OUT" <<'PY'
import glob, json, sys
fs = glob.glob(sys.argv[1] + "/nonces_*.json")
if not fs: print("файла нет")
else:
    d = json.load(open(fs[0])); a = d.get("artifacts", []); t = d.get("generation_time_sec") or 0
    print("%d нонсов за %.1f c = %.0f нонсов/мин" % (len(a), t, (len(a)/t*60) if t else 0))
PY
) | IMA=$(grep -ac 'illegal memory' /root/vllm.log)"
}

run_seed "$LABEL" s4                      # прогрев, в отчёт не идёт
for S in s1 s2 s3; do run_seed "$LABEL" "$S"; done
run_seed "${LABEL}rep" s1                 # повтор s1 = честный пол на этом железе

echo "--- инференс под нагрузкой ---"
python3 - <<'PY'
import concurrent.futures as cf, time, requests
def run(n, mx=800, tag=""):
    P={"model":"glm53","messages":[{"role":"user","content":"Explain paged attention in detail."}],
       "max_tokens":mx,"temperature":0.7}
    def one(i):
        t=time.monotonic()
        r=requests.post("http://127.0.0.1:8081/v1/chat/completions",json=P,timeout=900)
        r.raise_for_status(); return time.monotonic()-t, r.json()["usage"]["completion_tokens"]
    t0=time.monotonic()
    with cf.ThreadPoolExecutor(n) as ex: res=list(ex.map(one,range(n)))
    el=time.monotonic()-t0; tok=sum(x[1] for x in res); lat=sorted(x[0] for x in res)
    print("%sпараллельно %2d: %.1f c, %d токенов, %.1f ток/с, задержка med %.1f c" %
          (tag, n, el, tok, tok/el, lat[len(lat)//2]), flush=True)
try: run(1, tag="прогрев: ")          # первый запрос после PoC оплачивает холодные ядра
except Exception as e: print("прогрев: ОШИБКА %r" % e, flush=True)
for n in (1, 8, 20):
    try: run(n)
    except Exception as e: print("параллельно %d: ОШИБКА %r" % (n, e), flush=True)
PY
echo "ИТОГ IMA=$(grep -ac 'illegal memory' /root/vllm.log) $(date +%T)"
echo SEEDS_DONE
