#!/usr/bin/env bash
# DeepSeek-V4-Flash FP8 on 4xH100 (GPUs 0-3), the boot of the 9 September fp8_h100 goldens: TP4, MNS 1024, gmu 0.85, MNBT 16384, capture 608.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export KIT=/root/d-kit VENV=/opt/gv30 OUT=${OUT:-/root/rout/ds} CFG=h100 QUANT=fp8 TP=4 MNS=${MNS:-1024} MNBT=${MNBT:-16384} GMU=${GMU:-0.85} CAPTURE=${CAPTURE:-608} MOE_BACKEND=auto
export MODEL_DIR=/root/models/DeepSeek-V4-Flash-0731 VLLM_CACHE_ROOT=/root/.cache/vllm_030_ds
mkdir -p "$OUT"; cd /root && bash /root/d-kit/scripts/d_boot.sh "${1:-ds030}"
echo "d_boot rc=$?"
