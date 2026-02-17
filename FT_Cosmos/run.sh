#!/usr/bin/env bash
set -euo pipefail

# how many GPUs
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}

if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "No GPUs detected. Set NUM_GPUS=1 to run on CPU (not recommended) or ensure NVIDIA drivers are available."
  exit 1
fi

echo "Using ${NUM_GPUS} GPUs"

# Safer defaults for multi-GPU rendezvous
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

if [[ "${NUM_GPUS}" -eq 1 ]]; then
  # Avoid torch.distributed rendezvous for single-GPU runs (can segfault in some envs).
  python train_cosmosvl_video_json.py "$@"
else
  # Internal multiprocessing launcher avoids torchrun/elastic rendezvous.
  export INTERNAL_SPAWN=1
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export MASTER_PORT="${MASTER_PORT:-29500}"
  python train_cosmosvl_video_json.py "$@"
fi