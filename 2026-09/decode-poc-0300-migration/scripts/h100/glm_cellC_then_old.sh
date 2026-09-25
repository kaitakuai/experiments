#!/usr/bin/env bash
# Cell C after the bisect: vLLM 0.30 with sparse MLA forced to the MQA path for prefill rows (the 0.28.1 behaviour,
# where no dense-MHA prefill backend supported GLM's (256,0,256) MLA dims). Then the 0.28.1 freeze-stack check.
LOG=/root/glm_bisect.log; say(){ echo "[$(date -u +%T)] $*" | tee -a "$LOG"; }
healthy(){ curl -s -m 3 -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health 2>/dev/null | grep -q 200; }
export KIT=/root/g-kit VENV=/opt/gv30 SERVED_MODEL=zai-org/GLM-5.3-Flash
for i in $(seq 1 200); do grep -q "bisect done" "$LOG" 2>/dev/null && break; sleep 30; done
grep -q "bisect done" "$LOG" || { say "bisect never finished; abort C"; exit 1; }
tag=forcemqa; out=/root/rout/glm_$tag; mkdir -p $out/vs_old $out/vs_030gen
docker rm -f glm_noarfuse glm_notilelang >/dev/null 2>&1; sleep 5
say "=== C: sparse_mla_force_mqa=true (prefill rows via fp8-KV MQA, as in 0.28.1)"
NAME=glm_$tag OUT=$out EXTRA="--attention-config '{\"sparse_mla_force_mqa\": true}'" bash /root/boot_glm_docker2.sh 2>&1 | tail -1 | tee -a "$LOG"
for i in $(seq 1 90); do healthy && break; sleep 20; done
if healthy; then
  say "$tag healthy: $(grep -aoE 'Using FLASH_ATTN MLA prefill backend|No MLA prefill backend[^\"]{0,60}|sparse_mla_force_mqa[^,]{0,10}|KV cache size: [0-9,]+ tokens' $out/server.log | sort -u | tr '\n' ' ')"
  python3 -c "import json; json.dump({'cfg':'h100','tag':'glm_$tag','wall_nonces':256,'model_dir':'/root/models/GLM-5.3-Flash','synth':'0.30 bisect $tag'}, open('$out/current_boot.json','w'))"; cp $out/current_boot.json $out/vs_old/; cp $out/current_boot.json $out/vs_030gen/
  (cd /root/g-kit/scripts && CORP_DIR=/root/ref/glm/h100 OUT=$out/vs_old PART=200 RUN_LABEL=old /opt/gv30/bin/python r2_validate.py "honest_h*" glm_$tag 2>&1 | grep -E "корпусов|итог" | tee -a "$LOG")
  (cd /root/g-kit/scripts && CORP_DIR=/root/rout/glm OUT=$out/vs_030gen PART=200 RUN_LABEL=g030 /opt/gv30/bin/python r2_validate.py "honest_h0*" glm_$tag 2>&1 | grep -E "корпусов|итог" | tee -a "$LOG")
  /opt/gv30/bin/python /root/tau_cells.py $out/vs_old $out/vs_030gen | tee -a "$LOG"
else
  say "$tag boot FAILED: $(grep -aE 'Error|error' $out/server.log | grep -v error_ | tail -2 | cut -c1-220)"
fi
docker rm -f glm_$tag >/dev/null 2>&1
say "cell C done"
bash /root/glm_old_check.sh
