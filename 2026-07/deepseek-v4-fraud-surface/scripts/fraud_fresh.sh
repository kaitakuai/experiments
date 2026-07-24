#!/bin/bash
# Фрод-прогон на чистом боксе: setup -> скачать фрод-модель -> поднять -> 1000 нонсов.
# Usage: bash fraud_fresh.sh <TP> <hf-fraud-model> <label>
set -x; exec 2>&1
TP=${1:?TP}; FM=${2:?fraud model}; LBL=${3:-fraud}
API=http://127.0.0.1:8081; OUT=/root/out; mkdir -p $OUT

echo "=== FRAUD_SETUP ==="
bash /root/setup_box.sh "$TP" "$FM" 2>&1 | tail -25
echo "=== FRAUD_WAIT_WEIGHTS ==="
for i in $(seq 1 160); do
  INC=$(find /root/.cache/huggingface -name "*.incomplete" 2>/dev/null | wc -l)
  D=$(grep -c "Downloaded" /root/dl.log 2>/dev/null || echo 0)
  [ "$INC" = "0" ] && [ "$D" -ge 1 ] && break
  sleep 30
done
du -sh /root/.cache/huggingface | tail -1
echo "=== FRAUD_QUANT_CONFIG (что объявляет модель) ==="
SNAP=$(ls -d /root/.cache/huggingface/hub/models--*/snapshots/*/ 2>/dev/null | head -1)
for f in config.json hf_quant_config.json quantization_config.json quant_metadata.json; do
  [ -f "$SNAP/$f" ] && { echo "--- $f ---"; python3 -c "
import json,sys
d=json.load(open('$SNAP/$f'))
for k in ['quantization','quantization_config','quant_method','producer','quant_algo','kv_cache_quant_algo','weight_block_size','torch_dtype','model_type','architectures']:
    if k in d: print('   ',k,'=',json.dumps(d[k])[:160])
" 2>/dev/null; }
done
echo "=== какие quant-методы знает vLLM ==="
python3 -c "
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
print('   ', sorted(QUANTIZATION_METHODS)[:40])" 2>&1 | tail -2

echo "=== FRAUD_UP ==="
curl -s -X POST "$API/api/v1/inference/up/async" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$FM\",\"dtype\":\"auto\",\"additional_args\":[\"--enforce-eager\"]}" -m 60 | head -c 200; echo
for i in $(seq 1 140); do
  S=$(curl -s -m 6 "$API/api/v1/inference/up/status" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('status'),d.get('is_running'))" 2>/dev/null)
  case "$S" in
    *True*) echo "=== FRAUD_VLLM_UP after $((i*15))s ==="; break;;
    failed*) echo "=== FRAUD_VLLM_FAILED — модель не грузится ==="
             grep -iE "error|assert|not supported|quant|unsupported|Traceback|KeyError" /root/api.log | tail -15 | cut -c1-150; break;;
  esac
  sleep 15
done
if curl -s -m 6 "$API/api/v1/inference/up/status" | grep -q '"is_running":true'; then
  echo "=== FRAUD_COLLECT ==="
  python3 -u /root/collect_artifacts.py --url $API --model "$FM" \
    --output-dir $OUT/${LBL}_arts --nonces 1000 --batch-size 32 --logprobs-count 0 \
    --gpu "$LBL" --vllm-version "0.25.1-fraud" 2>&1 | tail -5
  cp $OUT/${LBL}_arts/nonces_1000.json $OUT/${LBL}_nonces_1000.json 2>/dev/null
  python3 -c "import json;d=json.load(open('$OUT/${LBL}_nonces_1000.json'));print('=== FRAUD_NONCES',d['total_nonces'],'@',round(d.get('nonces_per_min',0)),'n/min ===')"
fi
echo "=== FRAUD_ALL_DONE ==="
