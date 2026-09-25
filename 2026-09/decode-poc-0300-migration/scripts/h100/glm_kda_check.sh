#!/usr/bin/env bash
# GLM on 8xH100, vLLM 0.30 with the KDA prefill kernel pinned to triton (the 0.28.1 path) instead of the 0.30 default FlashKDA.
# Re-validate the 10 old H100 corpora (0.28.1 refs) and the 3 own corpora of the 0.30 triton-MoE boot (generated under FlashKDA).
LOG=/root/glm_kda.log; say(){ echo "[$(date -u +%T)] $*" | tee -a "$LOG"; }
healthy(){ curl -s -m 3 -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health 2>/dev/null | grep -q 200; }
docker rm -f glmmarlin glm030 dsperf ds030 >/dev/null 2>&1; sleep 5; mkdir -p /root/rout/glm_kda
NAME=glmkda OUT=/root/rout/glm_kda EXTRA="--kda-prefill-backend triton" bash /root/boot_glm_docker.sh 2>&1 | tail -1 | tee -a "$LOG"
for i in $(seq 1 90); do healthy && break; sleep 20; done; healthy || { say "kda-triton boot FAILED: $(grep -aE 'Error|error' /root/rout/glm_kda/server.log | grep -v error_ | tail -2 | cut -c1-200)"; exit 1; }
say "healthy: $(grep -aoE 'Using .{0,40}MoE backend|KV cache size: [0-9,]+ tokens|kda_prefill_backend[^,]{0,20}' /root/rout/glm_kda/server.log | sort -u | tr '\n' ' ')"
export KIT=/root/g-kit VENV=/opt/gv30 SERVED_MODEL=zai-org/GLM-5.3-Flash
python3 -c "import json; json.dump({'cfg':'h100','tag':'glmkda','wall_nonces':256,'model_dir':'/root/models/GLM-5.3-Flash','synth':'0.30 kda_prefill_backend=triton check'}, open('/root/rout/glm_kda/current_boot.json','w'))"
for sub in vs_old vs_flashkda_gen; do mkdir -p /root/rout/glm_kda/$sub; cp /root/rout/glm_kda/current_boot.json /root/rout/glm_kda/$sub/; done
(cd /root/g-kit/scripts && CORP_DIR=/root/ref/glm/h100 OUT=/root/rout/glm_kda/vs_old PART=200 RUN_LABEL=old /opt/gv30/bin/python r2_validate.py "honest_h*" glmkda 2>&1 | grep -E "корпусов|итог" | tee -a "$LOG")
(cd /root/g-kit/scripts && CORP_DIR=/root/rout/glm OUT=/root/rout/glm_kda/vs_flashkda_gen PART=200 RUN_LABEL=flashkda_gen /opt/gv30/bin/python r2_validate.py "honest_h0*" glmkda 2>&1 | grep -E "корпусов|итог" | tee -a "$LOG")
/opt/gv30/bin/python /root/tau_cells.py /root/rout/glm_kda/vs_old /root/rout/glm_kda/vs_flashkda_gen | tee -a "$LOG"
say "kda-triton check done"
