#!/usr/bin/env bash
echo "start $(date -u +%T)"
/root/hf/bin/hf download MiniMaxAI/MiniMax-M2.7 --revision d494266a4affc0d2995ba1fa35c8481cbd84294b --local-dir /root/models/MiniMax-M2.7 --max-workers 16
echo "minimax done rc=$? $(date -u +%T)"
