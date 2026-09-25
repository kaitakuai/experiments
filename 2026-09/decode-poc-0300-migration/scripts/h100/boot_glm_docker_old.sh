#!/usr/bin/env bash
# The 15.09 freeze stack: official image vllm/vllm-openai:glm53-flash (0.28.1rc1.dev580+g385dce36b) + the 21-file residual @316e2779 + plugin @7617c0d.
set -u
NAME=${NAME:-glmold}; OUT=${OUT:-/root/rout/glm_old}; N=${N:-256}; CAPTURE=${CAPTURE:-$N}; GMU=${GMU:-0.95}; MNBT=${MNBT:-8192}; KVDT=${KVDT:-fp8}
mkdir -p "$OUT"; docker rm -f "$NAME" >/dev/null 2>&1
docker run -d --name "$NAME" --gpus all --network host --shm-size 32g --ulimit memlock=-1 --ulimit nofile=1048576:1048576 \
  -e POC_BATCH_SIZE_DEFAULT=${POC_BATCH:-16} -e VLLM_CACHE_ROOT=/gout/.vllm_cache_old -v /root/models:/models -v /root/rout:/rout -v /root/res028:/src/res028 -v /root/plug028:/src/plug028 \
  --entrypoint bash vllm/vllm-openai:glm53-flash -lc "sleep infinity" >/dev/null
VD=$(docker exec "$NAME" python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
docker exec "$NAME" bash -lc "cd /src/res028 && while read f; do install -D \"\$f\" \"$VD/\${f#vllm/}\"; done < residual_files.txt && echo overlaid \$(wc -l < residual_files.txt) files into $VD"
docker exec "$NAME" bash -lc "pip install -q --no-deps /src/plug028 && pip install -q scipy && python3 -c 'import importlib.metadata as m; print(\"gonka-poc\", m.version(\"gonka-poc\"))'"
docker exec -d "$NAME" bash -lc "vllm serve /models/GLM-5.3-Flash --served-model-name zai-org/GLM-5.3-Flash --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 8 --gpu-memory-utilization $GMU --kv-cache-dtype $KVDT --block-size 2304 --max-num-seqs $N \
  --max-num-batched-tokens $MNBT --logprobs-mode processed_logprobs --worker-extension-cls gonka_poc.worker.PoCWorkerExtension --enable-trace-replay \
  --tool-call-parser glm47 --reasoning-parser glm45 --trust-remote-code --enable-auto-tool-choice --no-enable-flashinfer-autotune \
  --no-enable-prefix-caching --no-disable-hybrid-kv-cache-manager --disable-custom-all-reduce \
  --compilation-config '{\"max_cudagraph_capture_size\":$CAPTURE}' ${EXTRA:-} > /rout/$(basename $OUT)/server.log 2>&1"
echo "$NAME booting; log $OUT/server.log"
