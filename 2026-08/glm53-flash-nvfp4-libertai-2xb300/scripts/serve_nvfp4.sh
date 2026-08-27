#!/bin/bash
# NVFP4 fraud arm. Same engine config as the honest arm — only the checkpoint differs.
TP=${TP:-2}
SNAP=$(ls -d /root/.cache/huggingface/hub/models--LibertAIDAI--GLM-5.3-Flash-NVFP4/snapshots/*/ | head -1)
export VLLM_ENGINE_READY_TIMEOUT_S=3600
exec gonka-vllm-serve \
  --model "$SNAP" \
  --served-model-name glm53 \
  --tensor-parallel-size "$TP" \
  --kv-cache-dtype fp8 \
  --max-model-len 131072 \
  --max-num-batched-tokens 65536 \
  --max-num-seqs 128 \
  --worker-extension-cls gonka_poc.worker.PoCWorkerExtension \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 --enable-auto-tool-choice \
  --host 0.0.0.0 --port 8081
