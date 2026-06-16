#!/bin/bash
# B300: genref FP8 для 3 конфигов компилируемости (cc/ee/ec). custom_ops server + env-флаги.
IMG=<MLNODE_IMAGE>
LOG=/root/hf/matrix-gen.log; echo "MGEN_START $(date -u +%H:%M:%S)" > $LOG
P=/usr/local/lib/python3.12/dist-packages/vllm/poc
docker rm -f dpoc >/dev/null 2>&1
docker run -d --name dpoc --gpus all --network host --shm-size 32g -v /root/hf:/hf -v /root/dpoc-merged:/patch -e HF_HOME=/hf --entrypoint bash $IMG -c "sleep infinity" >/dev/null 2>&1
docker exec dpoc bash -lc "cp /patch/*.py $P/ && cp /patch/driver.py /root/driver.py && grep -c GONKA_POC_PREFILL_EAGER $P/poc_model_runner.py" >> $LOG 2>&1
run_cfg () { # $1=cfg  $2=extra-env
  echo "[$1] start $(date +%T)" >> $LOG
  nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; sleep 8
  cat > /root/hf/launch.sh <<LAUNCH
#!/bin/bash
export GONKA_POC_SPHERE_CODEBOOK=/hf/poc-tests/codebook.pt $2
exec python3 -m vllm.entrypoints.openai.api_server --model MiniMaxAI/MiniMax-M2.7 --dtype auto --tensor-parallel-size 1 --compilation-config '{"custom_ops": ["all"]}' --gpu-memory-utilization 0.90 --max-model-len 8192 --port 8000 --host 127.0.0.1 --trust-remote-code
LAUNCH
  docker exec -d dpoc bash -lc "bash /hf/launch.sh > /hf/vllm-$1.log 2>&1"
  for i in $(seq 1 180); do [ "$(curl -s -o /dev/null -w %{http_code} http://127.0.0.1:8000/v1/models 2>/dev/null)" = "200" ] && { echo "[$1] ready ${i}0s" >> $LOG; break; }; sleep 10; done
  [ "$(curl -s -o /dev/null -w %{http_code} http://127.0.0.1:8000/v1/models)" = "200" ] || { echo "[$1] SERVER FAIL" >> $LOG; tail -20 /hf/vllm-$1.log >> $LOG; return 1; }
  docker exec dpoc bash -lc "cd /root && N_NONCES=8 STEPS=64 python3 driver.py genref /hf/poc-tests/ref_$1.json" >> $LOG 2>&1
  echo "[$1] genref done $(date +%T)" >> $LOG
}
run_cfg cc ""
run_cfg ee "GONKA_POC_PREFILL_EAGER=1 GONKA_POC_DECODE_EAGER=1"
run_cfg ec "GONKA_POC_PREFILL_EAGER=1"
echo "MGEN_DONE $(date -u +%H:%M:%S)" >> $LOG
