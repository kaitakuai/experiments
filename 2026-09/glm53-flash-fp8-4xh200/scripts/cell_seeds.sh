#!/bin/bash
# Прогон семян по УЖЕ поднятому движку.
#
# cell_honest.sh сам генерирует serve_cell.sh и стартует движок; здесь это нельзя —
# движок поднимался руками после починки NCCL, и перезапуск стоил бы ещё получаса.
# Скрипт ждёт /health и делает ровно оставшуюся часть ячейки: 3 семени + нагрузка.
set -u
: "${BATCH:=16}" "${LABEL:=honest}"
exec >>/root/cell_${LABEL}_seeds.log 2>&1
echo "=== СЕМЕНА $LABEL батч=$BATCH $(date +%T) ==="

for i in $(seq 1 240); do
  curl -s -m 5 -o /dev/null http://127.0.0.1:8081/health 2>/dev/null && break
  sleep 15
done
curl -s -m 5 -o /dev/null http://127.0.0.1:8081/health || { echo "движок не поднялся"; echo CELL_DONE; exit 1; }
echo "движок готов $(date +%T)"
grep -aE "attention backend|GPU KV cache size" /root/vllm.log | tail -2 | cut -c1-170
echo "маршрут: $(curl -s -m 10 http://127.0.0.1:8081/api/v1/pow/versions | cut -c1-200)"

export POC_COLLECT_TIMEOUT=1800
for SEED in s1 s2 s3; do
  BH=$(python3 -c "import json;print([x for x in json.load(open('/root/poc_seeds.json'))['seeds'] if x['id']=='$SEED'][0]['block_hash'])")
  PK=$(python3 -c "import json;print([x for x in json.load(open('/root/poc_seeds.json'))['seeds'] if x['id']=='$SEED'][0]['public_key'])")
  if [ -z "$BH" ] || [ -z "$PK" ]; then echo "ПУСТОЕ СЕМЯ $SEED — прерываю"; break; fi
  curl -s -m 20 -X POST http://127.0.0.1:8081/api/v1/pow/stop -H "Content-Type: application/json" -d '{}' >/dev/null 2>&1
  sleep 5
  OUT=/root/out_${LABEL}_$SEED; rm -rf "$OUT"; mkdir -p "$OUT"
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
  echo "$SEED: $(python3 - "$OUT" <<'PY'
import glob, json, sys
fs = glob.glob(sys.argv[1] + "/nonces_*.json")
if not fs: print("файла нет")
else:
    d = json.load(open(fs[0])); a = d.get("artifacts", []); t = d.get("generation_time_sec") or 0
    print("%d нонсов за %.1f c = %.0f нонсов/мин" % (len(a), t, (len(a)/t*60) if t else 0))
PY
) | IMA=$(grep -ac 'illegal memory' /root/vllm.log)"
done

echo "=== инференс под нагрузкой ==="
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
echo "ИТОГ IMA=$(grep -ac 'illegal memory' /root/vllm.log) $(date +%T)"
echo CELL_DONE
