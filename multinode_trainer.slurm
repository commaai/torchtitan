#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

JOB_ENV_FILE="${1:?Usage: sbatch multinode_trainer.slurm JOB_ENV_FILE}"
source "$JOB_ENV_FILE"

export HOME="/home/$(whoami)"
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.trainer_venv}"
export LOG_RANK="${LOG_RANK:-0}"
export TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE:-http://localhost:29510}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export LOGLEVEL="${LOGLEVEL:-INFO}"
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0,en,eth,em,bond}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-2097152}"
export FI_EFA_SET_CUDA_SYNC_MEMOPS="${FI_EFA_SET_CUDA_SYNC_MEMOPS:-0}"

set_default_gloo_socket_ifname() {
  [[ -z "${GLOO_SOCKET_IFNAME:-}" ]] || return 0

  local route_ifname=""
  if command -v ip >/dev/null 2>&1; then
    route_ifname="$(
      ip route get 1.1.1.1 2>/dev/null \
        | awk '{for (i = 1; i < NF; i++) if ($i == "dev") {print $(i + 1); exit}}' \
        || true
    )"
  fi

  if [[ -n "$route_ifname" ]]; then
    export GLOO_SOCKET_IFNAME="$route_ifname"
  fi
}

set_default_gloo_socket_ifname

if [[ -d /opt/amazon/efa/lib ]]; then
  export FI_PROVIDER="${FI_PROVIDER:-efa}"
  export LD_LIBRARY_PATH="/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-}"
fi
export LD_LIBRARY_PATH="/usr/local/lib/:${LD_LIBRARY_PATH:-}"

output_file="/var/log/slurm/${SLURM_JOB_NAME}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}.out"
error_file="/var/log/slurm/${SLURM_JOB_NAME}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}.err"

run_logged() {
  "$@" \
    1> >(tee -a "$output_file") \
    2> >(tee -a "$error_file" >&2)
}

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is not installed on $(hostname). Install uv before running TorchTitan Slurm jobs." >&2
  exit 1
fi

mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
head_node="${nodes[0]}"
rdzv_host="${MASTER:-$head_node}"
rdzv_endpoint="${rdzv_host}:${MASTER_PORT:-12355}"

echo "Head node: $head_node"
echo "Rendezvous endpoint: $rdzv_endpoint"
echo "GLOO_SOCKET_IFNAME: ${GLOO_SOCKET_IFNAME:-unset}"
echo "Training command: $COMMAND_TO_LOG"

cd "$PROJECT_CACHE"

if command -v dcgmi >/dev/null 2>&1; then
  dcgmi profile --pause || true
fi

if [[ -n "${COMM_MODE:-}" ]]; then
  run_logged srun --export=ALL --nodes=1 --ntasks=1 --kill-on-bad-exit=1 \
    uv run --project "$PROJECT_CACHE" --frozen \
    python -m torchtitan.train \
    --module "$MODULE" --config "$CONFIG" \
    "${TORCHTITAN_ARGS[@]}" --comm.mode="$COMM_MODE" --training.steps 1
else
  run_logged srun --export=ALL --ntasks="$SLURM_NNODES" --kill-on-bad-exit=1 \
    uv run --project "$PROJECT_CACHE" --frozen \
    python -m torch.distributed.run \
    --nnodes "$SLURM_NNODES" \
    --nproc_per_node "$NGPU" \
    --rdzv_id "$SLURM_JOB_ID" \
    --rdzv_backend c10d \
    --rdzv_endpoint "$rdzv_endpoint" \
    --local-ranks-filter "$LOG_RANK" \
    --role rank \
    --tee 3 \
    -m torchtitan.train \
    --module "$MODULE" --config "$CONFIG" \
    "${TORCHTITAN_ARGS[@]}"
fi

if command -v dcgmi >/dev/null 2>&1; then
  dcgmi profile --resume || true
fi
