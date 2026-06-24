#!/usr/bin/env bash
M=/dev/shm/GLM-5.2-FP8
curl -s -X POST http://localhost:8081/api/v1/inference/pow/stop >/dev/null 2>&1; sleep 3
/usr/bin/python3 -m vllm.entrypoints.cli.main bench serve \
  --backend openai-chat --base-url http://localhost:5001 --endpoint /v1/chat/completions \
  --model "$M" --served-model-name "$M" \
  --dataset-name random --num-prompts 200 --random-input-len 1024 --random-output-len 256 \
  --max-concurrency 32 --ignore-eos
echo BENCH_DONE
