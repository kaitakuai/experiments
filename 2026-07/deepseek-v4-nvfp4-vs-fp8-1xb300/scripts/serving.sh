#!/bin/bash
# Serving-замер: каждый сценарий отдельным запуском, репортер 0.2.7 падает после каждого — это ок,
# метрики уже в базе. Аргументы: <метка> <модель>
set -x
C=nvfp4ab; LBL=$1; M=$2; API=http://127.0.0.1:8081; OUT=/root/nvfp4out
mkdir -p $OUT
for i in $(seq 1 120); do
  st=$(docker exec $C curl -s -m 6 $API/api/v1/inference/up/status 2>/dev/null \
       | python3 -c "import sys,json;print(json.load(sys.stdin).get('is_running'))" 2>/dev/null)
  [ "$st" = "True" ] && break; sleep 15
done
echo "=== SV_${LBL}_START ==="
docker exec $C bash -c "
python3 - <<'PY'
import re
s=open('/root/compressa_config.yml').read().replace('MODEL_PLACEHOLDER','$M')
blocks=[b for b in re.split(r'\n(?=- model_name:)', s) if b.strip().startswith('- model_name')]
for i,b in enumerate(blocks,1):
    open('/root/sc%d.yml'%i,'w').write(b)
print('сценариев:',len(blocks))
PY"
for n in 1 2 3 4; do
  echo "--- сценарий $n ---"
  docker exec $C /root/cpvenv/bin/compressa-perf measure-from-yaml \
    --db /root/${LBL}.sqlite /root/sc${n}.yml 2>&1 | grep -E "Experiment|Number of failed|TTFT|Error" | head -4
done
docker exec $C /root/cpvenv/bin/compressa-perf list --db /root/${LBL}.sqlite --show-metrics 2>&1 \
  | head -160 > $OUT/${LBL}_compressa.log
echo "=== SV_${LBL}_SCENARIOS: $(docker exec $C /root/cpvenv/bin/compressa-perf list --db /root/${LBL}.sqlite 2>/dev/null | grep -cE '\| s[0-9]_') ==="
echo "=== SV_${LBL}_DONE ==="
