#!/bin/bash
# PoC throughput sweep. 120 s measurement window per batch (see CONTRIBUTING: 30 s
# windows carry +-17 % bulk-delivery noise). HOST_IP must be pinned: the autodetect
# returns the docker gateway 172.18.0.1, which does not exist on a Vast box, and the
# sweep then silently reports 0 nonces.
export MODEL=glm53
export HOST_IP=127.0.0.1
export POC_BLOCK_HASH=bce35dd795a5a622a9092ec1234eecbc157f4d03bdf117527b8a877bec6489b3
export POC_PUBLIC_KEY=4dfe8269e29777a44cd5ec412aabed42afd9a410cef2dbefe6b5cd6b983d9d5c
export BATCH_SIZES=${BATCH_SIZES:-16,32}
export GENERATION_DURATION_S=120
cd /root
python3 -u run_pow_generation.py --phase 3 --skip-check
echo "SWEEP_DONE"
