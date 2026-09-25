#!/usr/bin/env bash
echo "start $(date -u +%T)"
/opt/gv30/bin/pip install -q -U pip
/opt/gv30/bin/pip install -q vllm==0.30.0 scipy 2>&1 | tail -3
/opt/gv30/bin/python -c "import vllm, torch; print(vllm.__version__, torch.__version__, torch.cuda.is_available())"
echo "venv done rc=$? $(date -u +%T)"
