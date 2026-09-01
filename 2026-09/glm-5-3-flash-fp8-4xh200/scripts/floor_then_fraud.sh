#!/bin/bash
# Две вещи подряд, обе обязательны для осмысленного вывода про фрод:
#
# 1) ПОЛ. Повтор семени s1 на тех же честных весах. Расстояние между двумя честными
#    прогонами — та планка, ниже которой различать нечего. Без него число «фрод дал L2 X»
#    ничего не значит: на Hopper бит-идентичных нонсов не бывает даже при сравнении машины
#    с собой, и часть расхождения — свойство железа, а не подлога.
#
# 2) ФРОД. Тот же движок, те же флаги, другие веса: REAP50 — 144 эксперта из 288 при
#    неизменном topk=8. Схема квантования та же (fp8-блоки 128x128), маршрут тот же,
#    поэтому разница в нонсах относится к прунингу, а не к смене ядер.
set -u
exec >/root/floor_fraud.log 2>&1
echo "=== СТАРТ $(date +%T) ==="

run_seed() {  # $1=метка $2=id семени
  local LABEL=$1 SEED=$2
  local BH PK OUT CP
  BH=$(python3 -c "import json;print([x for x in json.load(open('/root/poc_seeds.json'))['seeds'] if x['id']=='$SEED'][0]['block_hash'])")
  PK=$(python3 -c "import json;print([x for x in json.load(open('/root/poc_seeds.json'))['seeds'] if x['id']=='$SEED'][0]['public_key'])")
  if [ -z "$BH" ] || [ -z "$PK" ]; then echo "ПУСТОЕ СЕМЯ $SEED — прерываю"; return 1; fi
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
  kill -9 "$CP" 2>/dev/null
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

export POC_COLLECT_TIMEOUT=1800

echo "--- 1. пол: повтор s1 на честных весах ---"
run_seed honestrep s1

echo "--- 2. перевод движка на REAP50 ---"
for P in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$P" 2>/dev/null; done
pkill -f "[g]onka-vllm-serve" 2>/dev/null
pkill -f "[V]LLM::" 2>/dev/null
sleep 10
for i in $(seq 1 60); do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
  [ "${U:-9999}" -lt 2000 ] && break
  sleep 5
done
echo "карты освобождены: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"

mv /root/vllm.log /root/vllm_honest.log
MODEL_REPO=models--patrickbdevaney--GLM-5.3-Flash-REAP50-FP8 TP=4 \
  setsid nohup bash /root/serve_h200_cell.sh </dev/null >/root/vllm.log 2>&1 &
RD=no
for i in $(seq 1 320); do
  sleep 15
  curl -s -m 5 -o /dev/null http://127.0.0.1:8081/health 2>/dev/null && { RD=yes; break; }
done
echo "фрод-движок готов=$RD $(date +%T)"
[ "$RD" = yes ] || { tail -25 /root/vllm.log; echo FRAUD_DONE; exit 1; }
grep -aE "attention backend|GPU KV cache size" /root/vllm.log | tail -2 | cut -c1-170

echo "--- 3. фрод: три семени ---"
for S in s1 s2 s3; do run_seed fraud "$S"; done

echo "--- 4. инференс под нагрузкой на фроде ---"
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
echo FRAUD_DONE
