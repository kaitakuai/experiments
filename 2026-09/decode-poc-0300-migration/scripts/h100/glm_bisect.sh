#!/usr/bin/env bash
# Bisect the GLM 0.30-vs-0.28.1 validation gap on 8xH100: one boot per candidate, re-validate the 10 old H100 corpora.
LOG=/root/glm_bisect.log; say(){ echo "[$(date -u +%T)] $*" | tee -a "$LOG"; }
healthy(){ curl -s -m 3 -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health 2>/dev/null | grep -q 200; }
export KIT=/root/g-kit VENV=/opt/gv30 SERVED_MODEL=zai-org/GLM-5.3-Flash
run_cell(){ # $1 tag, rest: env assignments for the boot
  local tag=$1; shift; local out=/root/rout/glm_$tag
  docker rm -f glmkda glmmarlin glm030 glm_$tag >/dev/null 2>&1; sleep 5; mkdir -p $out/vs_old
  env NAME=glm_$tag OUT=$out "$@" bash /root/boot_glm_docker2.sh 2>&1 | tail -1 | tee -a "$LOG"
  for i in $(seq 1 90); do healthy && break; sleep 20; done
  healthy || { say "$tag boot FAILED: $(grep -aE 'Error|error' $out/server.log | grep -v error_ | tail -2 | cut -c1-220)"; docker rm -f glm_$tag >/dev/null 2>&1; return 1; }
  say "$tag healthy: $(grep -aoE 'Enabled custom fusions: [a-z_, ]+|Initialized FlashInfer Allreduce[^\"]{0,40}|Using .{0,30}MoE backend|KV cache size: [0-9,]+ tokens' $out/server.log | sort -u | tr '\n' ' ')"
  python3 -c "import json; json.dump({'cfg':'h100','tag':'glm_$tag','wall_nonces':256,'model_dir':'/root/models/GLM-5.3-Flash','synth':'0.30 bisect $tag'}, open('$out/current_boot.json','w'))"; cp $out/current_boot.json $out/vs_old/
  (cd /root/g-kit/scripts && CORP_DIR=/root/ref/glm/h100 OUT=$out/vs_old PART=200 RUN_LABEL=old /opt/gv30/bin/python r2_validate.py "honest_h*" glm_$tag 2>&1 | grep -E "корпусов|итог" | tee -a "$LOG")
  /opt/gv30/bin/python /root/tau_cells.py $out/vs_old | tee -a "$LOG"
  docker rm -f glm_$tag >/dev/null 2>&1
}
say "=== A: allreduce+rms fusion off"
run_cell noarfuse CCJSON='{"max_cudagraph_capture_size":256,"pass_config":{"fuse_allreduce_rms":false}}'
say "=== B: tilelang removed (mHC torch/triton fallback)"
run_cell notilelang PRE='pip uninstall -y -q tilelang && python3 -c "from vllm.model_executor.layers.mhc import HAS_TILELANG_MHC; print(\"HAS_TILELANG_MHC\", HAS_TILELANG_MHC)"'
say "bisect done"
