#!/bin/bash
# genarm.sh <arm-name> <n-prompts>
# Generates both logprobs modes against whatever model the engine currently serves.
ARM=$1
N=${2:-1000}
cd /root
MODEL=$(curl -s http://127.0.0.1:8081/v1/models | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
echo "[$(date +%H:%M)] arm=$ARM n=$N model=$MODEL"
for MODE in processed_logprobs raw_logprobs; do
  echo "### $ARM $MODE ###"
  python3 /root/v4val.py --mode generate \
    --url http://127.0.0.1:8081/v1 --model "$MODEL" \
    --prompts /root/prompts_multilingual_1000.jsonl --n-prompts "$N" \
    --output "/root/out/gen_${ARM}_${MODE}.jsonl" \
    --logprobs-mode "$MODE" --workers 16 --tag "${ARM}-b300" 2>&1 | tail -2
done
echo "[$(date +%H:%M)] ${ARM}_GEN_DONE"
wc -l /root/out/gen_${ARM}_*.jsonl
