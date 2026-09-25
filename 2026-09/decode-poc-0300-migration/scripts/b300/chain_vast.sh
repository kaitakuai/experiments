#!/usr/bin/env bash
# After prep: A) default 0.30 boot -> tier A (gates, corpora, old B300 refs, self, round, replay) + perf;
#             B) boot with both prefill flags pinned -> old B300 refs + perf.
LOG=/root/chain.log; say(){ echo "[$(date -u +%T)] $*" | tee -a "$LOG"; }
healthy(){ curl -s -m 3 -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health 2>/dev/null | grep -q 200; }
wait_health(){ for i in $(seq 1 150); do healthy && return 0; pgrep -f "vllm serve" >/dev/null || { [ $i -gt 6 ] && return 1; }; sleep 20; done; return 1; }
stop_server(){ pkill -f "vllm serve" 2>/dev/null; sleep 15; pkill -9 -f "vllm serve" 2>/dev/null; pkill -9 -f "VLLM::EngineCore\|VLLM::Worker" 2>/dev/null; sleep 5; }
for i in $(seq 1 400); do grep -q "PREP DONE" /root/prep.log 2>/dev/null && break; sleep 30; done
grep -q "PREP DONE" /root/prep.log || { say "prep never finished; abort"; exit 1; }
export KIT=/root/g-kit VENV=/opt/gv30 SERVED_MODEL=zai-org/GLM-5.3-Flash
say "=== A. default 0.30 boot"
OUT=/root/rout/glm bash /root/boot_glm_vast.sh | tee -a "$LOG"
if wait_health; then
  say "A healthy: $(grep -aoE 'KV cache size: [0-9,]+ tokens|Using .{0,40}MoE backend|Using [A-Z_]+ attention backend|Using FLASH_ATTN MLA prefill backend|No MLA prefill backend[^"]{0,40}' /root/rout/glm/server.log | sort -u | tr '\n' ' ')"
  cd /root && KIT=/root/g-kit SERVED=zai-org/GLM-5.3-Flash PART=200 GEN_BATCH=256 OUT=/root/rout/glm REF=/root/ref/glm/b300 TAG=glm030 CFG=b300 MNS=256 MODEL_DIR=/root/models/GLM-5.3-Flash bash /root/stage_mm.sh > /root/stage_glm_b300.out 2>&1
  say "A stage done: $(grep -c 'stage done' /root/stage_glm_b300.out)"
  python3 /root/tau_cells.py /root/rout/glm/vs_old /root/rout/glm/vs_self | tee -a "$LOG"
  OUT=/root/rout/glm KITDIR=/root/g-kit TAG=glm030 NPOC=1000 CHAT_CONC=256 SERVED=zai-org/GLM-5.3-Flash bash /root/perf_poc.sh 2>&1 | grep -E "PoC:|chat \(" | tee -a "$LOG"
else
  say "A boot FAILED: $(grep -aE 'Error|error' /root/rout/glm/server.log | grep -v error_ | tail -3 | cut -c1-220)"
fi
stop_server
say "=== B. both prefill flags pinned (kda triton + sparse_mla_force_mqa)"
mkdir -p /root/rout/glm_both/vs_old
OUT=/root/rout/glm_both EXTRA="--kda-prefill-backend triton --attention-config '{\"sparse_mla_force_mqa\": true}'" bash /root/boot_glm_vast.sh | tee -a "$LOG"
if wait_health; then
  say "B healthy: $(grep -aoE 'KV cache size: [0-9,]+ tokens|sparse_mla_force_mqa=[A-Za-z]+|kda_prefill_backend[^,]{0,12}' /root/rout/glm_both/server.log | sort -u | tr '\n' ' ')"
  python3 -c "import json; json.dump({'cfg':'b300','tag':'glmboth','wall_nonces':256,'model_dir':'/root/models/GLM-5.3-Flash','synth':'0.30 both prefill flags pinned'}, open('/root/rout/glm_both/current_boot.json','w'))"; cp /root/rout/glm_both/current_boot.json /root/rout/glm_both/vs_old/
  (cd /root/g-kit/scripts && CORP_DIR=/root/ref/glm/b300 OUT=/root/rout/glm_both/vs_old PART=200 RUN_LABEL=old /opt/gv30/bin/python r2_validate.py "honest_h*" glmboth 2>&1 | grep -E "корпусов|итог" | tee -a "$LOG")
  python3 /root/tau_cells.py /root/rout/glm_both/vs_old | tee -a "$LOG"
  OUT=/root/rout/glm_both KITDIR=/root/g-kit TAG=glmboth NPOC=1000 CHAT_CONC=256 SERVED=zai-org/GLM-5.3-Flash bash /root/perf_poc.sh 2>&1 | grep -E "PoC:|chat \(" | tee -a "$LOG"
else
  say "B boot FAILED: $(grep -aE 'Error|error' /root/rout/glm_both/server.log | grep -v error_ | tail -3 | cut -c1-220)"
fi
stop_server
say "CHAIN DONE"
