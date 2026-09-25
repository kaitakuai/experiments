#!/usr/bin/env bash
# GLM-5.3-Flash on 2xB300 inside the vLLM 0.30 image, the freeze's B300 profile: TP2, gmu 0.9, KV auto (bf16),
# block 2304, N 256, MNBT 32768, capture 256, autotune off, POC batch 32. EXTRA appends flags.
set -u
OUT=${OUT:-/root/rout/glm}; N=${N:-256}; CAPTURE=${CAPTURE:-$N}; GMU=${GMU:-0.9}; MNBT=${MNBT:-32768}; KVDT=${KVDT:-auto}
mkdir -p "$OUT"; export POC_BATCH_SIZE_DEFAULT=${POC_BATCH:-32} VLLM_CACHE_ROOT=/root/.vllm_cache
eval "set -- ${EXTRA:-}"
nohup vllm serve /root/models/GLM-5.3-Flash --served-model-name zai-org/GLM-5.3-Flash --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 2 --gpu-memory-utilization $GMU --kv-cache-dtype $KVDT --block-size 2304 --max-num-seqs $N \
  --max-num-batched-tokens $MNBT --logprobs-mode processed_logprobs --worker-extension-cls gonka_poc.worker.PoCWorkerExtension \
  --tool-call-parser glm47 --reasoning-parser glm45 --trust-remote-code --enable-auto-tool-choice --no-enable-flashinfer-autotune \
  --no-enable-prefix-caching --no-disable-hybrid-kv-cache-manager \
  --compilation-config "{\"max_cudagraph_capture_size\":$CAPTURE}" "$@" > "$OUT/server.log" 2>&1 < /dev/null &
echo "vllm serve pid $! ; log $OUT/server.log ; extra: ${EXTRA:-none}"
