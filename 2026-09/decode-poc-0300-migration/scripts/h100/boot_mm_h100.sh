#!/usr/bin/env bash
# MiniMax-M2.7 on 4xH100 (GPUs 0-3), the frozen 9 September profile: TP4, triton MoE, gmu 0.94, N 752, MNBT 32768, capture 768.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export KIT=/root/r-kit VENV=/opt/gv30 OUT=${OUT:-/root/rout/mm} CFG=h100 TP=4 GMU=${GMU:-0.94} MNS=${MNS:-752} CAPTURE=${CAPTURE:-768}
export MODEL_DIR=/root/models/MiniMax-M2.7 VLLM_CACHE_ROOT=/root/.cache/vllm_030_mm
export EXTRA_SERVE_FLAGS="--max-num-batched-tokens ${MNBT:-32768} --logprobs-mode processed_logprobs --worker-extension-cls gonka_poc.worker.PoCWorkerExtension ${EXTRA:-}"
mkdir -p "$OUT"; cd /root && bash /root/r-kit/scripts/r_boot.sh "${1:-mm030}"
echo "r_boot rc=$?"
