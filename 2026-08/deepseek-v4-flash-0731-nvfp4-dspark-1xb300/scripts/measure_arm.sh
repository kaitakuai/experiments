#!/bin/bash
# One measurement arm, run inside the container. TAG names the output dir.
# The engine comes up with additional_args=[] so every flag is the image's own
# baked profile; the only thing that changes between arms is quant_config.py.
set -u
M=${MODEL:-/models/nvfp4}
TAG=${TAG:?}
O=/root/out_$TAG; mkdir -p $O

bash /root/scripts/engine_up.sh | tee $O/up.log
grep -q ENGINE_UP $O/up.log || { echo "ARM_ABORTED"; exit 1; }

echo "=== [$TAG] ENGINE FACTS ==="
{
  grep -a "non-default args" /root/api.log | tail -1 | cut -c1-600
  grep -aoE "quantization=[a-z0-9_]+|expert_dtype resolved to .[a-z0-9]+.|GPU KV cache size: [0-9,]+|Using V2 Model Runner|DSpark draft model loaded: [0-9]+ params" \
    /root/api.log | sort -u
} | tee $O/engine_facts.txt

echo "=== [$TAG] ENFORCED TOKENS (validation replay) ==="
python3 /root/scripts/enforced_test.py gen   | tee    $O/enforced.txt
python3 /root/scripts/enforced_test.py replay | tee -a $O/enforced.txt

echo "=== [$TAG] SERVING / ACCEPTANCE ==="
python3 -u /root/scripts/serving_bench.py --url http://127.0.0.1:8081 \
  --model "$M" --tag "$TAG" --out $O/serving.json > $O/serving.log 2>&1
grep -E '"scenario"|TOKENS_PER_CHUNK|OUTPUT_TOK_PER_S|FAILED_REQ' $O/serving.log

echo "=== [$TAG] POC SWEEP (x2; the first after bring-up reads low) ==="
for R in 1 2; do
  MODEL=$M HOST_IP=127.0.0.1 MLNODE_URL=http://127.0.0.1:8081 \
    python3 -u /root/scripts/run_pow_generation.py --phase 3 --skip-check 2>&1 \
    | tee $O/sweep_$R.log | grep -E "│|Best" | tail -5
done
echo "=== [$TAG] ARM_DONE ==="
