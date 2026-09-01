#!/bin/bash
# Подготовка 4×H200 под кампанию: проверки, правка сборщика, закачка ОБОИХ наборов весов.
#
# Диск 900 ГБ взят специально, чтобы честные веса (328 ГБ) и фрод REAP50 (~160 ГБ) лежали
# одновременно: на вчерашнем боксе диск не расширялся живьём, и фрод-плечо потребовало бы
# второй аренды. Здесь качаем сразу оба, канал 2874 Мбит это позволяет.
set -u
exec >/root/setup.log 2>&1
echo "START $(date +%T)"

nvidia-smi --query-gpu=index,name,memory.total,power.max_limit,driver_version --format=csv,noheader
nvidia-smi topo -m | head -6
python3 -c "import vllm,flashinfer;print('vllm',vllm.__version__,'flashinfer',flashinfer.__version__)"
python3 -c "import gonka_poc;print('gonka_poc',gonka_poc.__version__)"
echo "патч индексера (ожидается >=2): $(grep -c 'torch.full' \
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py)"
echo "мягкий лимит дескрипторов: $(ulimit -Sn) (поднимаем в скрипте запуска)"
df -h / | tail -1

# маршруты PoC идут без префикса inference, и таймаут сборщика должен настраиваться
sed -i 's|/api/v1/inference/pow|/api/v1/pow|g' /root/collect_artifacts.py
python3 - <<'PY'
p = "/root/collect_artifacts.py"; s = open(p).read()
old = "if elapsed > 600:  # 10 min timeout"
new = 'if elapsed > float(os.environ.get("POC_COLLECT_TIMEOUT", 600)):'
if "POC_COLLECT_TIMEOUT" not in s and old in s:
    s = s.replace(old, new, 1)
    if "\nimport os\n" not in s:
        s = s.replace("import json", "import json\nimport os", 1)
    import ast; ast.parse(s); open(p, "w").write(s); print("таймаут сборщика настраивается")
PY

echo "DOWNLOAD HONEST START $(date +%T)"
hf download zai-org/GLM-5.3-Flash --max-workers 16 > /root/dl_honest.log 2>&1
echo "DOWNLOAD HONEST END $(date +%T)"

# фрод-плечо: половина экспертов выброшена (144 из 288), тот же fp8-маршрут, те же ядра
echo "DOWNLOAD FRAUD START $(date +%T)"
hf download patrickbdevaney/GLM-5.3-Flash-REAP50-FP8 --max-workers 16 > /root/dl_fraud.log 2>&1
echo "DOWNLOAD FRAUD END $(date +%T)"

for M in zai-org--GLM-5.3-Flash patrickbdevaney--GLM-5.3-Flash-REAP50-FP8; do
  SNAP=$(ls -d /root/.cache/huggingface/hub/models--$M/snapshots/*/ 2>/dev/null | head -1)
  echo -n "$M: "
  python3 - "$SNAP" <<'PY'
import json, glob, os, sys
s = sys.argv[1]
if not s: print("снапшота нет"); raise SystemExit
idx = os.path.join(s, "model.safetensors.index.json")
if not os.path.exists(idx): print("индекса нет"); raise SystemExit
need = set(json.load(open(idx))["weight_map"].values())
have = {os.path.basename(p) for p in glob.glob(s + "/*.safetensors")}
print("%d/%d %s" % (len(have), len(need), "OK" if not need - have else "НЕТ " + str(sorted(need - have)[:3])))
PY
done
echo "недокачанных файлов: $(find /root/.cache/huggingface -name '*.incomplete' | wc -l)"
echo SETUP_DONE
