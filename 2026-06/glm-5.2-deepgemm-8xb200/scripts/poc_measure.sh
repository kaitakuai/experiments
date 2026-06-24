#!/usr/bin/env bash
U=http://localhost:8081/api/v1/inference/pow
curl -s -X POST $U/stop >/dev/null 2>&1; sleep 4
B='{"block_hash":"LIVE","block_height":1,"public_key":"benchpub","node_id":0,"node_count":1,"group_id":0,"n_groups":1,"batch_size":16,"params":{"model":"/dev/shm/GLM-5.2-FP8","seq_len":1024,"k_dim":12},"poc_stronger_rng":false}'
echo "init: $(curl -s -X POST $U/init/generate -H 'Content-Type: application/json' -d "$B" | head -c 90)"
prev=0; prevt=0
for t in 10 20 30 40 50 60 70 80; do
  sleep 10
  S=$(curl -s $U/status)
  echo "$S" | python3 -c "
import sys,json
d=json.load(sys.stdin)
s=d.get('stats') or (d.get('backends',[{}])[0].get('stats')) or {}
tp=s.get('total_processed',0); nps=s.get('nonces_per_second',0)
print(f'  t=${t}s  total={tp}  cum_nonces/min={nps*60:.0f}')
"
done
curl -s -X POST $U/stop >/dev/null 2>&1
echo LIVEDONE
