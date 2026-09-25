#!/usr/bin/env bash
# Vast 2xB300, image vllm/vllm-openai:v0.30.0: residual overlay + plugin + kits + refs + GLM weights.
set -u; LOG=/root/prep.log; say(){ echo "[$(date -u +%T)] $*" | tee -a "$LOG"; }
mkdir -p /root/models /root/rout /root/ref/glm/b300 /opt/gv30/bin; ln -sf "$(command -v python3)" /opt/gv30/bin/python
say "python $(python3 --version) vllm $(python3 -c 'import vllm;print(vllm.__version__)')"
pip install -q hf_transfer scipy 2>&1 | grep -v "already satisfied\|notice" | tail -2
VD=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
mkdir -p /root/res030 && tar -xf /root/res030.tar -C /root/res030
(cd /root/res030 && while read f; do install -D "$f" "$VD/${f#vllm/}"; done < residual_files.txt) && say "overlaid $(wc -l < /root/res030/residual_files.txt) files into $VD"
mkdir -p /root/plug030 && tar -xf /root/plug030.tar -C /root/plug030 && pip install -q --no-deps /root/plug030 2>&1 | grep -v notice | tail -1; say "plugin $(python3 -c 'import importlib.metadata as m; print(m.version("gonka-poc"))')"
python3 -c "import gonka_poc, vllm.poc.poc_params" && say "imports ok"
tar -xf /root/gkit.tar -C /root; tar -xf /root/replay.tar -C /root && rm -rf /root/replay-kit && mv /root/replay-equivalence /root/replay-kit && cp /root/run_replay.py /root/replay-kit/ 2>/dev/null
tar -xf /root/refs_b300.tar -C /root/ref/glm/b300 && say "refs: $(ls /root/ref/glm/b300/corp_honest_h*_gen.json | wc -l) corpora"
say "downloading GLM-5.3-Flash @eb9eb208"
HF_HUB_ENABLE_HF_TRANSFER=1 python3 - <<'PY' 2>&1 | tail -3 | tee -a "$LOG"
from huggingface_hub import snapshot_download
p = snapshot_download("zai-org/GLM-5.3-Flash", revision="eb9eb208eb0d988989d07a6a12d0fdeb5f52574a", local_dir="/root/models/GLM-5.3-Flash", max_workers=16)
print("downloaded to", p)
PY
python3 /root/ckpt_id.py /root/models/GLM-5.3-Flash 2>&1 | head -3 | tee -a "$LOG"
say "PREP DONE"
