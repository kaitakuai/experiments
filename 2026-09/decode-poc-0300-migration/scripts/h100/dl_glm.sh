#!/usr/bin/env bash
echo "start $(date -u +%T)"
/root/hf/bin/hf download zai-org/GLM-5.3-Flash --revision eb9eb208eb0d988989d07a6a12d0fdeb5f52574a --local-dir /root/models/GLM-5.3-Flash --max-workers 16
echo "glm done rc=$? $(date -u +%T)"
