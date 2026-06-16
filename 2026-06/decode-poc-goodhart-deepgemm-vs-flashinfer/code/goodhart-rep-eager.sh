#!/bin/bash
# EAGER repeat-measure: per backend 3x genref(P) + 3x serv-bench(S) on one server -> error bars.
LOG=/root/hf/gh-rep-eager.log; echo "REP_START eager $(date -u +%H:%M:%S)" > $LOG
PP=/usr/local/lib/python3.12/dist-packages/vllm/poc
docker exec dpoc bash -lc "cp /patch/*.py $PP/ 2>/dev/null; cp /patch/driver.py /root/driver.py; cp /patch/serv-bench.py /root/serv-bench.py"
NN=48; ST=128
run_backend () { # $1=tag $2=env
  echo "[$1] start $(date +%T)" >> $LOG
  nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null
  docker exec dpoc bash -lc "pkill -9 -f vllm 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null"; sleep 12
  cat > /root/hf/launch.sh <<LAUNCH
#!/bin/bash
export GONKA_POC_SPHERE_CODEBOOK=/hf/poc-tests/codebook_b300.pt
$2
exec python3 -m vllm.entrypoints.openai.api_server --model MiniMaxAI/MiniMax-M2.7 --dtype auto --tensor-parallel-size 1 --enforce-eager --gpu-memory-utilization 0.90 --max-model-len 8192 --port 8000 --host 127.0.0.1 --trust-remote-code
LAUNCH
  docker exec -d dpoc bash -lc "bash /hf/launch.sh > /hf/gh-rep-eager-$1.log 2>&1"
  for i in $(seq 1 200); do [ "$(curl -s -o /dev/null -w %{http_code} http://127.0.0.1:8000/v1/models 2>/dev/null)" = "200" ] && break; sleep 10; done
  [ "$(curl -s -o /dev/null -w %{http_code} http://127.0.0.1:8000/v1/models)" = "200" ] || { echo "[$1] SERVER FAIL" >> $LOG; return 1; }
  for rep in 1 2 3; do
    g=$(docker exec dpoc bash -lc "cd /root && N_NONCES=$NN STEPS=$ST python3 driver.py genref /hf/poc-tests/gh_rep_ref.json" 2>&1 | grep -oE '\[[0-9.]+s\]')
    echo "[$1] P$rep $g" >> $LOG
    s=$(docker exec dpoc bash -lc "cd /root && python3 /root/serv-bench.py MiniMaxAI/MiniMax-M2.7 256 64 256" 2>&1 | grep -oE '= [0-9.]+ tok/s')
    echo "[$1] S$rep $s" >> $LOG
  done
  echo "[$1] done $(date +%T)" >> $LOG
}
run_backend flashinfer "export VLLM_USE_FLASHINFER_MOE_FP8=1; export VLLM_USE_DEEP_GEMM=0"
run_backend deepgemm   "export VLLM_USE_FLASHINFER_MOE_FP8=0; export VLLM_USE_DEEP_GEMM=1"
echo "REP_DONE eager $(date -u +%H:%M:%S)" >> $LOG
