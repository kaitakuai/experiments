#!/usr/bin/env bash
# Overlay the 0.30 residual into the venv's vllm and install the plugin. Idempotent.
set -euo pipefail
VENV=${VENV:-/opt/gv30}; SP=$($VENV/bin/python -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
echo "vllm at $SP ($($VENV/bin/python -c 'import vllm;print(vllm.__version__)'))"
cd /root/res030 && while read f; do install -D "$f" "$SP/${f#vllm/}"; done < residual_files.txt && echo "residual: $(wc -l < residual_files.txt) files overlaid"
$VENV/bin/pip install -q --no-deps --force-reinstall /root/plug030 && echo "plugin: $($VENV/bin/python -c 'import importlib.metadata as m;print(m.version("gonka-poc"))')"
cd /tmp && $VENV/bin/python - <<'PY'
import importlib
for m in ("vllm.poc.poc_params","vllm.v1.worker.gpu.sample.replay","vllm.entrypoints.launchers.app","gonka_poc.plugin","gonka_poc.mixed.bridge","gonka_poc.entrypoint.api_router"): importlib.import_module(m)
from gonka_poc import _compat; print("compat:", _compat.current().__name__)
from vllm.v1.core.sched import scheduler; print("scheduler poc seam:", hasattr(scheduler, "_replays_enforced_tokens"))
PY
echo "setup030 OK"
