#!/bin/bash
# Collect 1000 PoC nonces per seed (s1/s2/s3) at batch 32.
# ARM only names the output dirs so honest and fraud artifacts never collide.
ARM=${ARM:-fp8}
API=http://127.0.0.1:8081
OUT=/root/out
mkdir -p $OUT
cd /root
for SID in s1 s2 s3; do
  BH=$(python3 -c "
import json;d=json.load(open('/root/poc_seeds.json'));s=d.get('seeds',d)
print([x for x in s if x['id']=='$SID'][0]['block_hash'])")
  PK=$(python3 -c "
import json;d=json.load(open('/root/poc_seeds.json'));s=d.get('seeds',d)
print([x for x in s if x['id']=='$SID'][0]['public_key'])")
  echo "=== seed $SID ==="
  python3 -u collect_artifacts.py \
    --url $API --model glm53 \
    --output-dir $OUT/${ARM}_${SID} \
    --nonces 1000 --batch-size 32 \
    --block-hash "$BH" --public-key "$PK" \
    --gpu "2xB300-SXM6" --vllm-version "0.28.0.dev0+glm53.gonka.sampler1"
  echo "SEED_${SID}_DONE"
done
echo "COLLECT_DONE"
