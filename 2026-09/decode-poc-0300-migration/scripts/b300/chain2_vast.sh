#!/usr/bin/env bash
# Re-measure the chat sweep (failed: r_chat_sweep calls /opt/gv30/bin/vllm, which the symlink prep did not create)
# plus a second PoC sample on both boots; caches are warm now.
LOG=/root/chain2.log; say(){ echo "[$(date -u +%T)] $*" | tee -a "$LOG"; }
healthy(){ curl -s -m 3 -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health 2>/dev/null | grep -q 200; }
wait_health(){ for i in $(seq 1 90); do healthy && return 0; pgrep -f "vllm serve" >/dev/null || { [ $i -gt 6 ] && return 1; }; sleep 20; done; return 1; }
stop_server(){ pkill -f "vllm serve" 2>/dev/null; sleep 15; pkill -9 -f "vllm serve" 2>/dev/null; pkill -9 -f "VLLM::EngineCore\|VLLM::Worker" 2>/dev/null; sleep 5; }
ln -sf "$(command -v vllm)" /opt/gv30/bin/vllm; ls -la /opt/gv30/bin/ | awk '{print $9, $10, $11}' | tee -a "$LOG"
export KIT=/root/g-kit VENV=/opt/gv30 SERVED_MODEL=zai-org/GLM-5.3-Flash
say "=== A2. default 0.30 boot: chat c=256 + PoC 1000 (2nd sample)"
OUT=/root/rout/glm_a2 bash /root/boot_glm_vast.sh | tee -a "$LOG"
if wait_health; then
  say "A2 healthy: $(grep -aoE 'KV cache size: [0-9,]+ tokens' /root/rout/glm_a2/server.log | tail -1)"
  OUT=/root/rout/glm_a2 KITDIR=/root/g-kit TAG=glma2 NPOC=1000 CHAT_CONC=256 SERVED=zai-org/GLM-5.3-Flash bash /root/perf_poc.sh 2>&1 | grep -E "PoC:|chat \(|rror" | tee -a "$LOG"
else say "A2 boot FAILED: $(grep -aE 'Error|error' /root/rout/glm_a2/server.log | grep -v error_ | tail -2 | cut -c1-200)"; fi
stop_server
say "=== B2. both prefill flags: chat c=256 + PoC 1000 (2nd sample)"
OUT=/root/rout/glm_b2 EXTRA="--kda-prefill-backend triton --attention-config '{\"sparse_mla_force_mqa\": true}'" bash /root/boot_glm_vast.sh | tee -a "$LOG"
if wait_health; then
  say "B2 healthy: $(grep -aoE 'sparse_mla_force_mqa=[A-Za-z]+' /root/rout/glm_b2/server.log | tail -1)"
  OUT=/root/rout/glm_b2 KITDIR=/root/g-kit TAG=glmb2 NPOC=1000 CHAT_CONC=256 SERVED=zai-org/GLM-5.3-Flash bash /root/perf_poc.sh 2>&1 | grep -E "PoC:|chat \(|rror" | tee -a "$LOG"
else say "B2 boot FAILED: $(grep -aE 'Error|error' /root/rout/glm_b2/server.log | grep -v error_ | tail -2 | cut -c1-200)"; fi
stop_server
say "CHAIN2 DONE"
