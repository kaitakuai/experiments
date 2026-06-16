#!/bin/bash
# H100 TP=4: per-step profile для 3 конфигов против ref_<cfg>. $1=MODEL_PATH $2=TAG(honest|fraud)
MODEL="$1"; TAG="$2"
IMG=<MLNODE_IMAGE>
LOG=/root/hf/matrix-$TAG.log; echo "MPROF_START $TAG model=$MODEL $(date -u +%H:%M:%S)" > $LOG
P=/usr/local/lib/python3.12/dist-packages/vllm/poc
docker rm -f dpoc >/dev/null 2>&1
docker run -d --name dpoc --gpus all --network host --shm-size 32g -v /root/hf:/hf -v /root/dpoc-merged:/patch -e HF_HOME=/hf --entrypoint bash $IMG -c "sleep infinity" >/dev/null 2>&1
docker exec dpoc bash -lc "cp /patch/*.py $P/ && cp /patch/driver.py /root/driver.py && sed -i 's#MODEL = \"MiniMaxAI/MiniMax-M2.7\"#MODEL = \"$MODEL\"#' /root/driver.py && grep -m1 'MODEL =' /root/driver.py" >> $LOG 2>&1
run_cfg () { # $1=cfg $2=extra-env
  echo "[$TAG/$1] start $(date +%T)" >> $LOG
  nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; sleep 8
  cat > /root/hf/launch.sh <<LAUNCH
#!/bin/bash
export GONKA_POC_SPHERE_CODEBOOK=/hf/poc-tests/codebook.pt $2
exec python3 -m vllm.entrypoints.openai.api_server --model $MODEL --dtype auto --tensor-parallel-size 4 --compilation-config '{"custom_ops": ["all"]}' --gpu-memory-utilization 0.90 --max-model-len 8192 --port 8000 --host 127.0.0.1 --trust-remote-code
LAUNCH
  docker exec -d dpoc bash -lc "bash /hf/launch.sh > /hf/vllm-$TAG-$1.log 2>&1"
  for i in $(seq 1 180); do [ "$(curl -s -o /dev/null -w %{http_code} http://127.0.0.1:8000/v1/models 2>/dev/null)" = "200" ] && { echo "[$TAG/$1] ready ${i}0s" >> $LOG; break; }; sleep 10; done
  [ "$(curl -s -o /dev/null -w %{http_code} http://127.0.0.1:8000/v1/models)" = "200" ] || { echo "[$TAG/$1] SERVER FAIL" >> $LOG; tail -20 /hf/vllm-$TAG-$1.log >> $LOG; return 1; }
  docker exec dpoc bash -lc "cd /root && N_NONCES=8 STEPS=64 python3 driver.py profile /hf/poc-tests/ref_$1.json 2>&1 | grep -E 'PROFILE|step|slope|steps'" >> $LOG 2>&1
  docker exec dpoc bash -lc "cp /hf/poc-tests/profile.json /hf/poc-tests/prof_${TAG}_$1.json"
  echo "[$TAG/$1] profile done $(date +%T)" >> $LOG
}
run_cfg cc ""
run_cfg ee "GONKA_POC_PREFILL_EAGER=1 GONKA_POC_DECODE_EAGER=1"
run_cfg ec "GONKA_POC_PREFILL_EAGER=1"
echo "MPROF_DONE $TAG $(date -u +%H:%M:%S)" >> $LOG
