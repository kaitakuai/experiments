#!/usr/bin/env bash
# After the bisect: boot the 15.09 freeze stack (0.28.1) on this box and use it as the validator for
# (1) the old H100 refs (sanity: expect the same-card band), (2) the 3+3 corpora generated on 0.30 (triton and marlin boots).
LOG=/root/glm_old.log; say(){ echo "[$(date -u +%T)] $*" | tee -a "$LOG"; }
healthy(){ curl -s -m 3 -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health 2>/dev/null | grep -q 200; }
for i in $(seq 1 200); do grep -q "bisect done" /root/glm_bisect.log 2>/dev/null && break; sleep 30; done
grep -q "bisect done" /root/glm_bisect.log || { say "bisect never finished; abort"; exit 1; }
docker rm -f glm_noarfuse glm_notilelang glmkda >/dev/null 2>&1; sleep 5
OUT=/root/rout/glm_old; mkdir -p $OUT/vs_old $OUT/vs_030triton $OUT/vs_030marlin
NAME=glmold OUT=$OUT bash /root/boot_glm_docker_old.sh 2>&1 | tail -1 | tee -a "$LOG"
for i in $(seq 1 90); do healthy && break; sleep 20; done
healthy || { say "old-stack boot FAILED: $(grep -aE 'Error|error' $OUT/server.log | grep -v error_ | tail -3 | cut -c1-220)"; exit 1; }
say "old stack healthy: $(grep -aoE 'Initializing a V1 LLM engine \([^)]*\)|Enabled custom fusions: [a-z_, ]+|Initialized FlashInfer Allreduce[^\"]{0,40}|Using .{0,30}MoE backend|KV cache size: [0-9,]+ tokens' $OUT/server.log | sort -u | tr '\n' ' ')"
export KIT=/root/g-kit VENV=/opt/gv30 SERVED_MODEL=zai-org/GLM-5.3-Flash
python3 -c "import json; json.dump({'cfg':'h100','tag':'glmold','wall_nonces':256,'model_dir':'/root/models/GLM-5.3-Flash','synth':'0.28.1 freeze stack as validator'}, open('$OUT/current_boot.json','w'))"
for sub in vs_old vs_030triton vs_030marlin; do cp $OUT/current_boot.json $OUT/$sub/; done
(cd /root/g-kit/scripts && CORP_DIR=/root/ref/glm/h100 OUT=$OUT/vs_old PART=200 RUN_LABEL=old /opt/gv30/bin/python r2_validate.py "honest_h*" glmold 2>&1 | grep -E "корпусов|итог" | tee -a "$LOG")
(cd /root/g-kit/scripts && CORP_DIR=/root/rout/glm OUT=$OUT/vs_030triton PART=200 RUN_LABEL=t030 /opt/gv30/bin/python r2_validate.py "honest_h0*" glmold 2>&1 | grep -E "корпусов|итог" | tee -a "$LOG")
(cd /root/g-kit/scripts && CORP_DIR=/root/rout/glm_marlin OUT=$OUT/vs_030marlin PART=200 RUN_LABEL=m030 /opt/gv30/bin/python r2_validate.py "honest_h0*" glmold 2>&1 | grep -E "корпусов|итог" | tee -a "$LOG")
/opt/gv30/bin/python /root/tau_cells.py $OUT/vs_old $OUT/vs_030triton $OUT/vs_030marlin | tee -a "$LOG"
say "old check done"
