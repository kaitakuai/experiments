#!/bin/bash
# vast validator: crosshw (per-nonce, teacher-forced) for cc+ee vs refN64. $1=MODEL $2=TAG(honest|fraud)
MODEL="$1"; TAG="$2"
LOG=/hf/crosshw-$TAG.log; echo "XH_START $TAG model=$MODEL $(date -u +%H:%M:%S)" > $LOG
P=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))")/poc
cp /root/dpoc-merged/*.py "$P"/ && cp /root/dpoc-merged/driver.py /root/driver.py
sed -i "s#MODEL = \"MiniMaxAI/MiniMax-M2.7\"#MODEL = \"$MODEL\"#" /root/driver.py
echo "poc sha: $(sha256sum $P/poc_model_runner.py | cut -c1-16)  model: $(grep -m1 'MODEL =' /root/driver.py)" >> $LOG
run_cfg () { # $1=cfg $2=env
  echo "[$TAG/$1] start $(date +%T)" >> $LOG
  nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null
  pkill -9 -f "vllm.entrypoints" 2>/dev/null; pkill -9 -f "VLLM::" 2>/dev/null; sleep 14
  cat > /root/launch.sh <<LAUNCH
#!/bin/bash
export HF_HOME=/hf
export GONKA_POC_SPHERE_CODEBOOK=/hf/poc-tests/codebook_b300.pt $2
exec python3 -m vllm.entrypoints.openai.api_server --model $MODEL --dtype auto --tensor-parallel-size 4 --compilation-config '{"custom_ops": ["all"]}' --gpu-memory-utilization 0.90 --max-model-len 8192 --port 8000 --host 127.0.0.1 --trust-remote-code
LAUNCH
  nohup bash /root/launch.sh > /hf/xh-$TAG-$1.log 2>&1 &
  for i in $(seq 1 220); do [ "$(curl -s -o /dev/null -w %{http_code} http://127.0.0.1:8000/v1/models 2>/dev/null)" = "200" ] && { echo "[$TAG/$1] ready ${i}0s" >> $LOG; break; }; sleep 10; done
  [ "$(curl -s -o /dev/null -w %{http_code} http://127.0.0.1:8000/v1/models)" = "200" ] || { echo "[$TAG/$1] SERVER FAIL" >> $LOG; tail -25 /hf/xh-$TAG-$1.log >> $LOG; return 1; }
  cd /root && N_NONCES=64 STEPS=64 python3 driver.py crosshw /hf/poc-tests/refN64_$1.json ${TAG}_$1 2>&1 | grep -E 'CROSSHW|TOTAL' >> $LOG
  echo "[$TAG/$1] done $(date +%T)" >> $LOG
}
run_cfg cc ""
run_cfg ee "GONKA_POC_PREFILL_EAGER=1 GONKA_POC_DECODE_EAGER=1"
echo "XH_DONE $TAG $(date -u +%H:%M:%S)" >> $LOG
