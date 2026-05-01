# RUNBOOK — reproduce Kimi-K2.6 INT4 PoC v2 nonces + inference validation on 2×8×H100

This is a step-by-step replay. All commands are copyable and idempotent unless noted. Replace IPs / hostnames as needed.

---

## 0. Prerequisites

You need **two bare-metal nodes** that share an InfiniBand fabric and **the same IB partition (PKEY)**. If your provider gives you nodes in different PKEYs, ask them to put both into one partition before you start — without that the rest of this doc won't work.

Per node:
- 8× H100 80GB (SXM5 with NVSwitch is what we tested)
- ≥600 GB free disk for the model
- IB ports active and routable to the other node (`ibstat`, then `ping` on `ib0` IP)
- root / sudo
- Docker 25+ and `nvidia-container-toolkit`
- NVIDIA driver **≥580** (the mlnode image needs CUDA 12.9 PTX → 570 won't work)

Verify before going further:

```bash
# On both nodes
nvidia-smi --query-gpu=driver_version,name --format=csv | head -3
# Expect: 580.x.x, NVIDIA H100 80GB HBM3
ibstat | grep -E "CA \"|State|Rate" | head
# Expect: 8× "Rate: 400" with "State: Active"
```

Cross-node IB ping (from node-A to one of node-B's IB IPs):
```bash
ssh user@<node-B> 'ip -br addr show | grep ib0'
# pick the 192.168.x.x address, then on node-A:
ping -I ib0 -c 2 <node-B-ib0-ip>
# Expect: 0% loss
```

If ping fails, check:
- `cat /sys/class/net/ib0/pkey` is the same on both nodes
- `dpkg -l | grep fabricmanager` is installed (NVSwitch needs `nvidia-fabricmanager-580`)

---

## 1. Storage & filesystem

The data NVMe disk is unmounted by default. Mount it as `/data`:

```bash
# Find the unformatted big NVMe (XFS once formatted)
sudo blkid /dev/nvme*n1 | grep xfs
# If nothing — pick an unformatted one, format:
sudo mkfs.xfs -f /dev/nvme1n1   # adjust device
sudo mkdir -p /data
sudo mount /dev/nvme1n1 /data
sudo chmod 1777 /data
df -h /data
# Expect: ~3.5 TB available
```

> **Caveat after reboot**: NVMe device names get re-enumerated. After reboot, `lsblk` and `blkid` again — the disk that was `/dev/nvme1n1` may now be `/dev/nvme0n1`.

Create a workspace:

```bash
mkdir -p /data/hf /data/work
```

---

## 2. Driver upgrade (if you're on stock 570)

The mlnode image embeds CUDA 12.9 PTX in its Marlin kernel; driver 570.158 (CUDA 12.8 ceiling) errors out at `gptq_marlin_repack` with `cudaErrorUnsupportedPtxVersion`. Driver ≥580 fixes it.

On both nodes:

```bash
sudo apt-get update
# Purge old 570 packages cleanly (apt's auto-resolver can't always do it)
sudo dpkg --purge --force-all libnvidia-cfg1-570 libnvidia-compute-570 \
  libnvidia-decode-570 libnvidia-encode-570 libnvidia-extra-570 \
  libnvidia-fbc1-570 nvidia-compute-utils-570 nvidia-driver-570-server \
  nvidia-fabricmanager-570 libnvidia-nscq-570 nvidia-utils-570 \
  nvidia-firmware-570 xserver-xorg-video-nvidia-570 libnvidia-gl-570 nvidia-persistenced
# Install 580 with --force-overwrite to bypass libnvidia-gl file conflict
sudo apt-get install -yqo Dpkg::Options::='--force-overwrite' \
  nvidia-driver-580-server nvidia-fabricmanager-580 \
  libnvidia-nscq-580 libnvidia-gl-580-server nvidia-persistenced
sudo apt-get install -yf
sudo dpkg --configure -a
sudo reboot
```

After reboot (wait ~5-10 min on bare metal for memory training to finish):

```bash
nvidia-smi | head -3
# Expect: Driver Version: 580.126.20
systemctl is-active nvidia-fabricmanager
# Expect: active
```

---

## 3. Pull the mlnode image

The image is public — `docker pull` handles the anonymous bearer-token dance automatically.

```bash
sudo docker pull ghcr.io/product-science/mlnode:3.0.13-alpha5
sudo docker images | grep mlnode
# Expect: ~43.4 GB
```

---

## 4. Download the model (one shot per node, in parallel)

Set up a tiny venv with `huggingface_hub` + `hf_transfer` (no GPU needed):

```bash
sudo apt-get install -yq python3-venv python3-pip
python3 -m venv /data/work/venv
/data/work/venv/bin/pip install -q huggingface_hub hf_transfer
```

Trigger the download (≈540 GB on disk, ~7-12 min on Hyperbolic public 25 GbE per node):

```bash
ulimit -n 65536   # the resolver opens many concurrent fds; default 1024 will OOM with `Too many open files`
HF_HUB_ENABLE_HF_TRANSFER=1 \
  /data/work/venv/bin/hf download moonshotai/Kimi-K2.6 \
    --local-dir /data/hf/kimi-k26 --max-workers 8
```

If it stalls on the last few shards, **just re-run** the same command — already-downloaded shards are skipped.

Sanity-check the result:

```bash
ls /data/hf/kimi-k26/*.safetensors | wc -l   # expect 64
ls /data/hf/kimi-k26/{config.json,generation_config.json,model.safetensors.index.json}
```

---

## 5. Bring up the Ray cluster

Note IB IPs (`ib0`) for both nodes — you'll wire them into the docker run commands.

```bash
# On both nodes
ip -br addr show ib0
```

Say node-A is `192.168.243.158` and node-B is `192.168.242.154`.

### 5a. Ray HEAD on node-A

```bash
sudo docker rm -f mlnode 2>/dev/null
sudo docker run -d --name mlnode \
  --network host --ipc host --shm-size 64g \
  --gpus all --privileged \
  -v /data/hf:/root/.cache/huggingface \
  -v /data/work:/work \
  -e VLLM_HOST_IP=192.168.243.158 \
  -e NCCL_SOCKET_IFNAME=ib0 \
  -e GLOO_SOCKET_IFNAME=ib0 \
  -e NCCL_IB_HCA=mlx5_ib0,mlx5_ib1,mlx5_ib2,mlx5_ib3,mlx5_ib4,mlx5_ib5,mlx5_ib6,mlx5_ib7 \
  -e RAY_USAGE_STATS_ENABLED=0 \
  ghcr.io/product-science/mlnode:3.0.13-alpha5 \
  bash -c "ray start --head --port=6379 --node-ip-address=192.168.243.158 \
            --num-gpus=8 --dashboard-host=0.0.0.0 --block"
```

Wait ~5 sec, then on **node-B**:

### 5b. Ray WORKER on node-B

```bash
sudo docker rm -f mlnode 2>/dev/null
sudo docker run -d --name mlnode \
  --network host --ipc host --shm-size 64g \
  --gpus all --privileged \
  -v /data/hf:/root/.cache/huggingface \
  -v /data/work:/work \
  -e VLLM_HOST_IP=192.168.242.154 \
  -e NCCL_SOCKET_IFNAME=ib0 \
  -e GLOO_SOCKET_IFNAME=ib0 \
  -e NCCL_IB_HCA=mlx5_ib0,mlx5_ib1,mlx5_ib2,mlx5_ib3,mlx5_ib4,mlx5_ib5,mlx5_ib6,mlx5_ib7 \
  -e RAY_USAGE_STATS_ENABLED=0 \
  ghcr.io/product-science/mlnode:3.0.13-alpha5 \
  bash -c "ray start --address=192.168.243.158:6379 \
            --node-ip-address=192.168.242.154 --num-gpus=8 --block"
```

### 5c. Verify cluster

```bash
ssh user@<node-A> 'sudo docker exec mlnode ray status'
```

Expect:
```
Active:
 1 node_xxxxx   (node-A)
 1 node_yyyyy   (node-B)
Resources
  0.0/16.0 GPU
  0.0/256.0 CPU
  0B/3.57TiB memory
```

---

## 6. Launch vLLM with TP=16

Bypass the MLNode runner (it doesn't handle multi-node TP) and call vLLM directly inside the Ray-head container:

```bash
ssh user@<node-A> 'sudo docker exec -d mlnode bash -c "cd /work && rm -f vllm.log && \
  VLLM_HOST_IP=192.168.243.158 \
  NCCL_SOCKET_IFNAME=ib0 \
  GLOO_SOCKET_IFNAME=ib0 \
  NCCL_IB_HCA=mlx5_ib0,mlx5_ib1,mlx5_ib2,mlx5_ib3,mlx5_ib4,mlx5_ib5,mlx5_ib6,mlx5_ib7 \
  python3 -m vllm.entrypoints.openai.api_server \
    --model /root/.cache/huggingface/kimi-k26 \
    --served-model-name moonshotai/Kimi-K2.6 \
    --tensor-parallel-size 16 \
    --distributed-executor-backend ray \
    --host 0.0.0.0 \
    --port 5001 \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 64 \
    --logprobs-mode processed_logprobs \
    --trust-remote-code \
    --attention-backend FLASHMLA \
    > /work/vllm.log 2>&1"'
```

Notes:
- `--attention-backend FLASHMLA` **is required** — the default FLASHINFER doesn't support MLA. (On Blackwell you'd use `CUTLASS_MLA` instead.)
- `--logprobs-mode processed_logprobs` is what gonka production uses.
- `--max-model-len 262144` is K2.6's native context (256K). It fits with `gpu-memory-utilization 0.92` on 16 H100×80GB.

Watch the startup:

```bash
ssh user@<node-A> 'sudo docker exec mlnode tail -F /work/vllm.log' \
  | grep -E "Application startup complete|ERROR|Traceback|Loaded model|Engine core"
```

Expected timeline:
- 0:00 — Ray init, NCCL setup (~1 min)
- 0:30 — `Initializing a V1 LLM engine`
- 1:00 — `Loading safetensors checkpoint shards: 0%`
- 4-10 min — `100% Completed | 64/64`
- 11-13 min — `Application startup complete.`

If you see `cudaErrorUnsupportedPtxVersion` → driver upgrade didn't take, see §2.
If you see `FLASHINFER ... MLA not supported` → forgot `--attention-backend FLASHMLA`.

Sanity:

```bash
ssh user@<node-A> 'curl -sX POST http://127.0.0.1:5001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"moonshotai/Kimi-K2.6\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi\"}],\"max_tokens\":20,\"temperature\":0.7}"'
```

---

## 7. Collect 1000 PoC v2 nonces + 5-language inference probe

Copy `collect_artifacts.py` (in this folder) into `/data/work/` on node-A:

```bash
scp collect_artifacts.py user@<node-A>:/data/work/collect_artifacts.py
```

Run inside the container (gives results in `/work/artifacts/`):

```bash
ssh user@<node-A> 'sudo docker exec -d mlnode bash -c "cd /work && \
  python3 collect_artifacts.py \
    --url http://127.0.0.1:5001 \
    --model moonshotai/Kimi-K2.6 \
    --output-dir /work/artifacts \
    --nonces 1000 \
    --batch-size 8 \
    --gpu \"16x H100 80GB SXM5\" \
    --vllm-version 0.15.1 \
    --startup-cmd \"vllm TP=16 FLASHMLA processed_logprobs ctx=262144\" \
    --logprobs-count 0 \
    > /work/collect.log 2>&1"'
```

Watch progress:

```bash
ssh user@<node-A> 'sudo docker exec mlnode tail -F /work/collect.log' \
  | grep -E "nonces/min|saved|Done|ERROR"
```

Total ~1 minute @ batch=8. Output:
- `/work/artifacts/config.json` — run config
- `/work/artifacts/nonces_1000.json` — the 1000 nonces + `nonces_per_min`
- `/work/artifacts/inference_5langs.json` — the 5-language eyeball probe (sentinels visible)

### Optional — sweep batches

```bash
ssh user@<node-A> 'sudo docker exec -d mlnode bash -c "cd /work && \
  for B in 8 16 32; do
    python3 collect_artifacts.py --url http://127.0.0.1:5001 --model moonshotai/Kimi-K2.6 \
      --output-dir /work/artifacts_b\${B} --nonces 1000 --batch-size \$B \
      --gpu \"16x H100 80GB SXM5\" --vllm-version 0.15.1 --logprobs-count 0 \
      --lang5-max-tokens 8 2>&1 | grep -E \"nonces/min|saved|Done\"
    sleep 5
  done > /work/sweep.log 2>&1"'
```

Don't push past `batch=32` at full 256K context — the PoC engine OOMs and gets stuck (need vLLM restart). To go higher, drop `--max-model-len` to ~120K first.

---

## 8. Cross-validate inference logprobs against honest dataset

Download the honest dataset (Git LFS) once:

```bash
ssh user@<node-A> 'curl -sL -o /data/work/honest_b200_h200.jsonl \
  https://media.githubusercontent.com/media/kaitakuai/experiments/main/2026-04/kimi-k26-inference-validation/exp2-top-k-40-sentinels/artifacts/honest_b200_h200.jsonl'
ssh user@<node-A> 'wc -l /data/work/honest_b200_h200.jsonl'
# Expect: 1000
```

Copy the validator (`validate_against_honest.py` in this folder):

```bash
scp validate_against_honest.py user@<node-A>:/data/work/validate_against_honest.py
```

Run (200 items takes ~30 min sequentially; 100 is enough for a sanity check):

```bash
ssh user@<node-A> 'sudo docker exec -d mlnode bash -c "cd /work && \
  python3 validate_against_honest.py \
    --url http://127.0.0.1:5001 \
    --honest /work/honest_b200_h200.jsonl \
    --n 200 \
    --output /work/artifacts/validation_report.json \
    > /work/validate.log 2>&1"'

ssh user@<node-A> 'sudo docker exec mlnode tail -F /work/validate.log' \
  | grep -E "ETA|SUMMARY|GATE|SOFT|mean|Report saved|ERROR"
```

The validator replays each honest prompt with the **same sampling params** (temp=0.7 seed=1 top_k=40 top_p=0.95 repetition_penalty=1.2) and the **same `enforced_tokens`** so logprobs land on identical positions — then computes gonka's exact `distance2`.

Production gate is **mean distance2 ≤ 0.2**. Soft pass is **mean ≤ 0.05**.

Report → `/work/artifacts/validation_report.json` with per-item distances, percentiles, and the gate verdict.

---

## 9. Save artifacts and tear down

```bash
# Pull artifacts to your local box
mkdir -p ./out
scp -r user@<node-A>:/data/work/artifacts ./out/
scp -r user@<node-A>:/data/work/{collect.log,sweep.log,validate.log,vllm.log} ./out/logs/

# Stop the cluster
ssh user@<node-A> 'sudo docker rm -f mlnode'
ssh user@<node-B> 'sudo docker rm -f mlnode'
```

You can shut down the bare-metal nodes — model and venv on `/data` survive (the disk is XFS on a single NVMe). The mount itself doesn't survive a reboot, so step §1 is needed each time. Don't bother adding to fstab unless you're going to keep the nodes for days.

---

## Common gotchas (each one cost us at least 10 minutes during the original run)

1. **PKEY mismatch on Hyperbolic** — first pair of nodes had different PKEYs and IB cross-node was silently broken. Provider re-provisioned both into one PKEY.
2. **Driver 570 vs CUDA 12.9 PTX** — `cudaErrorUnsupportedPtxVersion` at `gptq_marlin_repack`. Upgrade driver, see §2.
3. **`libnvidia-gl-580-server` apt conflict** — overwrites a file owned by libnvidia-gl-570. Use `-o Dpkg::Options::='--force-overwrite'`.
4. **MLNode runner.py only does single-node TP** — local `torch.cuda.device_count()` doesn't see Ray's remote GPUs. Skip the MLNode API and run `vllm.entrypoints.openai.api_server` directly inside the container.
5. **FLASHINFER doesn't support MLA** — explicit `--attention-backend FLASHMLA` (Hopper) or `CUTLASS_MLA` (Blackwell).
6. **Ray placement groups stick around** — if you kill vLLM and try to relaunch with a different TP/PP, the old group still holds GPUs. Easiest cure: `docker rm -f mlnode` on both nodes, restart Ray cluster.
7. **`--max-model-len 262144` + batch ≥ 64 = PoC engine stuck.** No nonces emitted; `vllm.log` keeps spinning. Drop ctx to ~120K to test bigger batches.
8. **NVMe device names re-enumerate after reboot.** Find the disk by `blkid | grep xfs` — don't hardcode `/dev/nvmeXn1`.
9. **`hf download` opens many fds** — bump `ulimit -n 65536` in the same shell or it dies at ~half progress with `Too many open files`.
10. **`pkill -f <pattern>` matches its own bash command line** — kills your ssh session. Use `pgrep -f <pattern> | xargs -r kill -9` with a more specific pattern (e.g. `api_server` rather than `vllm`).
11. **Windows `cmd.exe` strips single quotes from `ssh user@host '...'`** — use double quotes or no quotes when running these from `cmd.exe`.
