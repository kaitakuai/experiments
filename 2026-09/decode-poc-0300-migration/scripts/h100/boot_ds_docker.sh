#!/usr/bin/env bash
# DeepSeek-V4-Flash FP8 on 4xH100 (devices 0-3) inside vllm/vllm-openai:v0.30.0 + residual overlay + plugin.
# d-kit h100 profile: TP4, gmu 0.85, MNS 1024, MNBT 16384, fp8 KV, capture 608 (reference boot of the fp8_h100 goldens).
set -u
NAME=${NAME:-ds030}; OUT=${OUT:-/root/rout/ds}; MNS=${MNS:-1024}; CAPTURE=${CAPTURE:-608}; GMU=${GMU:-0.85}; MNBT=${MNBT:-16384}; DEVS=${DEVS:-0,1,2,3}
mkdir -p "$OUT"; docker rm -f "$NAME" >/dev/null 2>&1
docker run -d --name "$NAME" --gpus "\"device=$DEVS\"" --network host --shm-size 32g --ulimit memlock=-1 --ulimit nofile=1048576:1048576 \
  -e VLLM_CACHE_ROOT=/gout/.vllm_cache_ds -v /root/models:/models -v /root/rout:/rout -v /root/res030:/src/res030 -v /root/plug030:/src/plug030 \
  --entrypoint bash vllm/vllm-openai:v0.30.0 -lc "sleep infinity" >/dev/null
VD=$(docker exec "$NAME" python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
docker exec "$NAME" bash -lc "cd /src/res030 && while read f; do install -D \"\$f\" \"$VD/\${f#vllm/}\"; done < residual_files.txt && echo overlaid \$(wc -l < residual_files.txt) files"
docker exec "$NAME" bash -lc "pip install -q --no-deps /src/plug030 && pip install -q scipy && python3 -c 'import importlib.metadata as m; print(\"gonka-poc\", m.version(\"gonka-poc\"))'"
docker exec -d "$NAME" bash -lc "vllm serve /models/DeepSeek-V4-Flash-0731 --served-model-name deepseek-ai/DeepSeek-V4-Flash-0731 --host 127.0.0.1 --port 8000 \
  --trust-remote-code --enable-auto-tool-choice --tensor-parallel-size 4 --gpu-memory-utilization $GMU --max-model-len 400000 \
  --max-num-batched-tokens $MNBT --kv-cache-dtype fp8 --logprobs-mode processed_logprobs --worker-extension-cls gonka_poc.worker.PoCWorkerExtension \
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --no-enable-prefix-caching --max-num-seqs $MNS \
  --compilation-config '{\"max_cudagraph_capture_size\":$CAPTURE}' ${EXTRA:-} > /rout/$(basename $OUT)/server.log 2>&1"
echo "$NAME booting; log $OUT/server.log"
