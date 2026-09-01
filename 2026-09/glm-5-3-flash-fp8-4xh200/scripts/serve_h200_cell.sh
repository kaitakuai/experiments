#!/bin/bash
# Запуск GLM-5.3-Flash на 4×H200 под ячейку кампании.
#
# Параметры через окружение:
#   MODEL_REPO — каталог модели в кэше HF (по умолчанию честная сборка)
#   TP         — размер тензорного параллелизма (по умолчанию 4)
#
# ulimit поднимаем всегда: мягкий лимит Vast = 1024, а NCCL раскладывает десятки каналов
# P2P/IPC и на Blackwell из-за этого вставал намертво (см. B300-CONTAINER.md). На Hopper
# каналов меньше, но лимит всё равно тесный — дешевле поднять, чем ловить.
#
# --limit-mm-per-prompt: зрительная башня убивала воркера на профилировании памяти на B300.
# На Hopper этого не наблюдалось, но картинки и видео нам не нужны ни в PoC, ни в текстовом
# нагрузочном тесте, а профилирование без них короче и предсказуемее.
#
# Всё остальное — проверенная fp8-конфигурация из RUNBOOK.md.
ulimit -n 524288
export VLLM_ENGINE_READY_TIMEOUT_S=3600

: "${MODEL_REPO:=models--zai-org--GLM-5.3-Flash}"
: "${TP:=4}"
SNAP=$(ls -d /root/.cache/huggingface/hub/$MODEL_REPO/snapshots/*/ | head -1)
[ -n "$SNAP" ] || { echo "снапшот $MODEL_REPO не найден"; exit 1; }
echo "модель: $SNAP  TP=$TP"

# compat нужен только при драйвере < 580; на 595 он подсовывает библиотеки от 580
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
[ "${DRV:-999}" -lt 580 ] && [ -d /usr/local/cuda/compat ] && \
  export LD_LIBRARY_PATH=/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}

exec gonka-vllm-serve \
  --model "$SNAP" --served-model-name glm53 \
  --tensor-parallel-size "$TP" \
  --kv-cache-dtype fp8 \
  --block-size 2304 \
  --max-num-seqs 256 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --no-enable-flashinfer-autotune \
  --logprobs-mode processed_logprobs \
  --worker-extension-cls gonka_poc.worker.PoCWorkerExtension \
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
  --host 0.0.0.0 --port 8081
