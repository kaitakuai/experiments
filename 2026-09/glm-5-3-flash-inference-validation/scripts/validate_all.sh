#!/bin/bash
# Replay every generated set against the honest model on this box.
# Executor was 2xB300; this is the Hopper side of the pair.
N=${1:-500}
cd /root
MODEL=$(curl -s http://127.0.0.1:5001/v1/models | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
echo "[$(date +%H:%M)] validator model: $MODEL  n=$N"

for ARM in honest fraud; do
  for MODE in processed_logprobs raw_logprobs; do
    IN="/root/out/gen_${ARM}_${MODE}.jsonl"
    OUT="/root/out/val_${ARM}_${MODE}.jsonl"
    [ -f "$IN" ] || { echo "MISSING $IN"; continue; }
    echo "### $ARM $MODE ###"
    python3 /root/v4val.py --mode replay \
      --url http://127.0.0.1:5001/v1 --model "$MODEL" \
      --input "$IN" --output "$OUT" --n-prompts "$N" \
      --logprobs-mode "$MODE" --workers 16 --tag "${ARM}-h200" 2>&1 | tail -2
  done
done
echo "[$(date +%H:%M)] VALIDATION_DONE"
wc -l /root/out/val_*.jsonl
