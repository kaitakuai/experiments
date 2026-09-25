#!/usr/bin/env bash
set -x
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq docker.io >/dev/null 2>&1 && systemctl enable --now docker >/dev/null 2>&1
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq nvidia-container-toolkit >/dev/null 2>&1
nvidia-ctk runtime configure --runtime=docker >/dev/null 2>&1; systemctl restart docker
docker info 2>/dev/null | grep -iE "runtimes|nvidia" | head -2
docker pull vllm/vllm-openai:v0.30.0 2>&1 | tail -1
docker run --rm --gpus all --entrypoint nvidia-smi vllm/vllm-openai:v0.30.0 --query-gpu=name --format=csv,noheader | head -1
echo "DOCKER READY rc=$?"
