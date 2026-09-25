#!/usr/bin/env bash
# H100 chain: wait for docker+image, GLM tier A + perf in the container, then DeepSeek FP8 tier A + perf in the container.
LOG=/root/chain_h100.log; say(){ echo "[$(date -u +%T)] $*" | tee -a "$LOG"; }
healthy(){ curl -s -m 3 -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health 2>/dev/null | grep -q 200; }
wait_health(){ local name=$1 n=${2:-90}; for i in $(seq 1 $n); do healthy && return 0; docker ps --format '{{.Names}}' | grep -q "^$name$" || { say "container $name gone"; return 1; }; sleep 20; done; say "health timeout"; return 1; }
for i in $(seq 1 60); do grep -q "DOCKER READY rc=0" /root/install_docker.log && break; sleep 20; done
grep -q "DOCKER READY rc=0" /root/install_docker.log || { say "docker not ready: $(tail -2 /root/install_docker.log)"; exit 1; }
say "=== A. GLM tier A in the container"
pkill -9 -f "vllm serv[e]" 2>/dev/null; pkill -9 -f "VLLM:[:]" 2>/dev/null; sleep 3
OUT=/root/rout/glm bash /root/boot_glm_docker.sh 2>&1 | tail -3 | tee -a "$LOG"
if wait_health glm030 90; then
  say "GLM healthy: $(grep -aoE 'KV cache size: [0-9,]+ tokens|Using .{0,40}MoE backend|PoC native attach: [^\"]{0,60}|attention backend[^\"]{0,40}' /root/rout/glm/server.log | sort -u | tr '\n' ' ')"
  cd /root && KIT=/root/g-kit SERVED=zai-org/GLM-5.3-Flash PART=200 GEN_BATCH=256 OUT=/root/rout/glm REF=/root/ref/glm/h100 TAG=glm030 CFG=h100 MNS=256 MODEL_DIR=/root/models/GLM-5.3-Flash bash /root/stage_mm.sh > /root/stage_glm_h100.out 2>&1
  say "GLM stage done: $(grep -c 'stage done' /root/stage_glm_h100.out)"
  say "=== GLM perf on the same boot (N 256, MNBT 8192): PoC 1000 at once, chat c=256"
  OUT=/root/rout/glm KITDIR=/root/g-kit TAG=glm030 NPOC=1000 CHAT_CONC=256 SERVED=zai-org/GLM-5.3-Flash bash /root/perf_poc.sh 2>&1 | grep -E "PoC:|chat \(" | tee -a "$LOG"
else
  say "GLM boot FAILED: $(grep -aE 'Error|error' /root/rout/glm/server.log | grep -v 'error_' | tail -3 | cut -c1-220)"
fi
docker rm -f glm030 >/dev/null 2>&1; sleep 5
say "=== B. DeepSeek FP8 tier A in the container (waits for the weights)"
for i in $(seq 1 120); do grep -q "deepseek fp8 done rc=0" /root/dl_ds_fp8.log && break; sleep 30; done
grep -q "deepseek fp8 done rc=0" /root/dl_ds_fp8.log || { say "DeepSeek FP8 weights not complete: $(tail -c 200 /root/dl_ds_fp8.log | tr '\r' '\n' | tail -1)"; exit 1; }
OUT=/root/rout/ds bash /root/boot_ds_docker.sh 2>&1 | tail -3 | tee -a "$LOG"
if wait_health ds030 120; then
  say "DeepSeek healthy: $(grep -aoE 'KV cache size: [0-9,]+ tokens|Using .{0,40}MoE backend|PoC native attach: [^\"]{0,60}' /root/rout/ds/server.log | sort -u | tr '\n' ' ')"
  OUT=/root/rout/ds REFXQ=/root/ref/deepseek/chains/xq_g10_fp8_h100.json TAG=ds030 QUANT=fp8 SERVED=deepseek-ai/DeepSeek-V4-Flash-0731 bash /root/stage_ds.sh > /root/stage_ds_h100.out 2>&1
  say "DeepSeek stage done: $(grep -c 'stage done' /root/stage_ds_h100.out)"
  say "=== DeepSeek perf: reboot at the frozen 4xH100 profile (gmu 0.90, N 640, MNBT 16384, capture 640), warm-up 1280"
  docker rm -f ds030 >/dev/null 2>&1; sleep 5; mkdir -p /root/rout/ds_perf
  NAME=dsperf OUT=/root/rout/ds_perf GMU=0.90 MNS=640 CAPTURE=640 bash /root/boot_ds_docker.sh 2>&1 | tail -1 | tee -a "$LOG"
  wait_health dsperf 120 && { OUT=/root/rout/ds_perf KITDIR=/root/d-kit TAG=dsperf NPOC=3000 WARM=1280 CHAT_CONC=768 SERVED=deepseek-ai/DeepSeek-V4-Flash-0731 bash /root/perf_poc.sh 2>&1 | grep -E "PoC:|chat \(" | tee -a "$LOG"; } || say "DeepSeek perf boot FAILED"
else
  say "DeepSeek boot FAILED: $(grep -aE 'Error|error' /root/rout/ds/server.log | grep -v 'error_' | tail -3 | cut -c1-220)"
fi
say "=== chain done"
