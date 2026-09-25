#!/usr/bin/env bash
# GLM on H100 with the marlin MoE backend (the kit's H100 perf profile) instead of the engine default (triton):
# re-validate the 10 old H100 corpora and the 3 own corpora of the triton boot, to see whether the MoE kernel explains the gap.
LOG=/root/glm_marlin.log; say(){ echo "[$(date -u +%T)] $*" | tee -a "$LOG"; }
healthy(){ curl -s -m 3 -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health 2>/dev/null | grep -q 200; }
docker rm -f glm030 dsperf ds030 >/dev/null 2>&1; sleep 5; mkdir -p /root/rout/glm_marlin
NAME=glmmarlin OUT=/root/rout/glm_marlin EXTRA="--moe-backend marlin" bash /root/boot_glm_docker.sh 2>&1 | tail -1 | tee -a "$LOG"
for i in $(seq 1 90); do healthy && break; sleep 20; done; healthy || { say "marlin boot FAILED: $(grep -aE 'Error|error' /root/rout/glm_marlin/server.log | grep -v error_ | tail -2 | cut -c1-200)"; exit 1; }
say "healthy: $(grep -aoE 'Using .{0,40}MoE backend|KV cache size: [0-9,]+ tokens' /root/rout/glm_marlin/server.log | sort -u | tr '\n' ' ')"
export KIT=/root/g-kit VENV=/opt/gv30 SERVED_MODEL=zai-org/GLM-5.3-Flash
python3 -c "import json; json.dump({'cfg':'h100','tag':'glmmarlin','wall_nonces':256,'model_dir':'/root/models/GLM-5.3-Flash','synth':'0.30 marlin MoE check'}, open('/root/rout/glm_marlin/current_boot.json','w'))"
for sub in vs_old vs_triton; do mkdir -p /root/rout/glm_marlin/$sub; cp /root/rout/glm_marlin/current_boot.json /root/rout/glm_marlin/$sub/; done
(cd /root/g-kit/scripts && CORP_DIR=/root/ref/glm/h100 OUT=/root/rout/glm_marlin/vs_old PART=200 RUN_LABEL=old /opt/gv30/bin/python r2_validate.py "honest_h*" glmmarlin 2>&1 | grep -E "корпусов|итог" | tee -a "$LOG")
(cd /root/g-kit/scripts && CORP_DIR=/root/rout/glm OUT=/root/rout/glm_marlin/vs_triton PART=200 RUN_LABEL=triton /opt/gv30/bin/python r2_validate.py "honest_h0*" glmmarlin 2>&1 | grep -E "корпусов|итог" | tee -a "$LOG")
/opt/gv30/bin/python /root/tau_cells.py /root/rout/glm_marlin/vs_old /root/rout/glm_marlin/vs_triton | tee -a "$LOG"
say "marlin check done"
