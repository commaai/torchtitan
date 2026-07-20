#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

_tt_generate_uuid() {
  if [[ -z "${1:-}" ]]; then
    cat /proc/sys/kernel/random/uuid
    return
  fi

  local hex
  hex=$(printf '%s' "$1" | sha1sum)
  hex=${hex%% *}
  printf '%s-%s-%s-%s-%s\n' "${hex:0:8}" "${hex:8:4}" "${hex:12:4}" "${hex:16:4}" "${hex:20:12}"
}

tt_prepare() {
  NGPU=${NGPU:-"$(nvidia-smi -L | wc -l)"}
  export LOG_RANK=${LOG_RANK:-0}
  export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
  export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
  export NCCL_SHM_LOCALITY=${NCCL_SHM_LOCALITY:-1}
  COMM_MODE=${COMM_MODE:-""}
  NNODES=${NNODES:-${SLURM_JOB_NUM_NODES:-1}}
  NODE_RANK=${NODE_RANK:-${SLURM_NODEID:-0}}

  TT_ARGS=()
  TT_COMM_SUFFIX=()
  for arg in "$@"; do
    case "$arg" in
      codedir=*) TT_ARGS+=("--codedir=${arg#codedir=}") ;;
      master_addr=*) MASTER_ADDR="${arg#master_addr=}" ;;
      master_port=*) MASTER_PORT="${arg#master_port=}" ;;
      *) TT_ARGS+=("$arg") ;;
    esac
  done

  MASTER_PORT=${MASTER_PORT:-12355}
  TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
  export REPORTERV2_HOST="${REPORTERV2_HOST:-mkv://data-gen.comma.life:3080/reporterv2}"

  if [[ -z "${REPORTERV2_TRAINING_ID:-}" ]]; then
    local reporterv2_seed=""
    if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
      reporterv2_seed="slurm:${SLURM_ARRAY_JOB_ID}:${SLURM_ARRAY_TASK_ID}"
    elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
      reporterv2_seed="slurm:${SLURM_JOB_ID}"
    elif [[ -n "${RDZV_ID:-}" ]]; then
      reporterv2_seed="rdzv:${RDZV_ID}"
    fi
    REPORTERV2_TRAINING_ID="$(_tt_generate_uuid "$reporterv2_seed")"
  fi
  export REPORTERV2_TRAINING_ID
}

tt_launch() {
  local python_module=$1
  shift

  if [[ -n "$COMM_MODE" ]]; then
    echo "Running with comm_mode=${COMM_MODE}"
    NGPU="$NGPU" LOCAL_RANK=0 python3 -m "$python_module" \
      "$@" --comm.mode="$COMM_MODE" "${TT_COMM_SUFFIX[@]}"
    return
  fi

  local rdzv_endpoint="localhost:0"
  if [[ -n "${MASTER_ADDR:-}" ]]; then
    rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}"
  fi

  PYTORCH_ALLOC_CONF="expandable_segments:True" \
  TORCHFT_LIGHTHOUSE="$TORCHFT_LIGHTHOUSE" \
  torchrun --nnodes="$NNODES" --node_rank="$NODE_RANK" --nproc_per_node="$NGPU" \
    --rdzv_id="${RDZV_ID:-${SLURM_JOB_ID:-$(_tt_generate_uuid)}}" --rdzv_backend=c10d \
    --rdzv_endpoint="$rdzv_endpoint" \
    --local-ranks-filter="$LOG_RANK" --role=rank --tee=3 \
    -m "$python_module" "$@"
}
