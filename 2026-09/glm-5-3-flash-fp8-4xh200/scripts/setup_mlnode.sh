#!/bin/bash
# Подготовка бокса под запуск ЧЕРЕЗ mlnode — то есть по тому же пути, что и в проде.
#
# Раньше мы поднимали gonka-vllm-serve напрямую: удобно для перебора флагов, но меряли
# конфигурацию, которой в парке нет, и лишались штатного свипа с колбэками.
#
# Этот скрипт НИЧЕГО не патчит в runner.py заранее. Сначала он показывает, какие аргументы
# зашиты в образ, — правки делаем осознанно, увидев исходное. С Kimi мы уже нарывались на
# то, что цепочка передавливает образные значения, и вслепую патчить нельзя.
set -u
exec >/root/setup_mlnode.log 2>&1
echo "START $(date +%T)"

nvidia-smi --query-gpu=index,name,memory.total,power.max_limit,driver_version --format=csv,noheader
python3 -c "import vllm,flashinfer;print('vllm',vllm.__version__,'flashinfer',flashinfer.__version__)"
python3 -c "import gonka_poc;print('gonka_poc',gonka_poc.__version__)"
echo "мягкий лимит дескрипторов: $(ulimit -Sn)"
df -h / | tail -1

echo "=== зависимости mlnode ==="
pip install --quiet toml accelerate fire fastrlock h2 termcolor typer-slim setuptools-scm tenacity 2>&1 | tail -2
echo "установлены"

R=/app/packages/api/src/api/inference/vllm/runner.py
echo "=== аргументы, зашитые в образ ($R) ==="
if [ -f "$R" ]; then
  grep -nE "^\s*\(?'--|\"--" "$R" | head -40
else
  echo "runner.py НЕ НАЙДЕН — mlnode в этом образе отсутствует, путь через API невозможен"
fi

echo "=== маршруты сборщика и измерителя ==="
# В ЭТОМ образе PoC-v2 живёт внутри vLLM и отвечает по /api/v1/pow/* (без префикса
# inference). Префикс inference на порту 8081 — это проксирование, оно отдаёт 502, а
# /api/v1/pow/* на 8081 — legacy PoW-сервис, он даёт 409 при поднятом инференсе.
# Свип направлять на порт 5000 (прокси mlnode перед vLLM), см. run_sweep_mlnode.sh.
for F in /root/collect_artifacts.py /root/run_pow_generation.py; do
  [ -f "$F" ] || continue
  sed -i 's|/api/v1/inference/pow|/api/v1/pow|g' "$F"
  sed -i 's|/inference/pow/|/pow/|g' "$F"   # в измерителе путь собирается из фрагментов
  echo "$F: маршруты /api/v1/pow"
done

echo "DOWNLOAD START $(date +%T)"
hf download zai-org/GLM-5.3-Flash --max-workers 16 > /root/dl.log 2>&1
echo "DOWNLOAD END $(date +%T)"
SNAP=$(ls -d /root/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash/snapshots/*/ | head -1)
python3 - "$SNAP" <<'PY'
import json, glob, os, sys
s = sys.argv[1]
need = set(json.load(open(os.path.join(s, "model.safetensors.index.json")))["weight_map"].values())
have = {os.path.basename(p) for p in glob.glob(s + "/*.safetensors")}
print("шарды: %d/%d %s" % (len(have), len(need), "OK" if not need - have else "НЕТ"))
PY

echo SETUP_DONE
