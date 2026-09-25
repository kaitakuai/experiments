#!/usr/bin/env bash
# GLM-5.3-Flash FP8 on 8xH100 with the g-kit profile (TP8, fp8 KV, block 2304, N 256, MNBT 8192, gmu 0.95, --disable-custom-all-reduce)
# run against the 0.30 venv instead of the container. VLLM_VER=0.30.0 selects no extra line flags.
export KIT=/root/g-kit OUT=${OUT:-/root/rout/glm} PY=/opt/gv30/bin/python VLLM=/opt/gv30/bin/vllm VENV=/opt/gv30
export MODEL_DIR=/root/models/GLM-5.3-Flash CFG=h100 ARM=fp8 N=${N:-256} VLLM_VER=0.30.0 VLLM_CACHE_ROOT=/root/.cache/vllm_030_glm
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
mkdir -p "$OUT"; cd /root && bash /root/g-kit/scripts/g_boot.sh "${1:-glm030}"
echo "g_boot rc=$?"
