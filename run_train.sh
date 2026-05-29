#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

help() {
cat <<'EOF'
Usage:       ./run_train.sh [N=nodes PARTITION=partition|NODELIST=nodes] [WAIT=0|1] [launcher args] [TorchTitan args]

Description: Runs TorchTitan locally by default, or submits a Slurm job when N/PARTITION
             or NODELIST/PARTITION are provided. Cluster jobs cache xx to code.nfs,
             use the torchtitan uv project, and reuse ~/.trainer_venv on workers.

Launcher args:
  MODULE|module       TorchTitan module name. Default: llama3
  CONFIG|config       TorchTitan config name. Default: llama3_debugmodel
  NGPU|ngpu|devices   GPUs per node. Default: 8
  LOG_RANK|log_rank   torchrun local ranks to tee. Default: 0
  COMM_MODE|comm_mode TorchTitan comm debug mode for local runs.
  N                  Number of Slurm nodes.
  PARTITION          Slurm partition.
  NODELIST           Slurm nodelist.
  WAIT               0 queues only, 1 waits until start then attaches.
  cpus_per_task      Slurm CPUs per node task. Default: 12 * NGPU.
  gpus_per_task      Optional Slurm GPU request. Unset by default.
  master             Rendezvous host override for Slurm.
  masterport         Rendezvous port. Default: 12355 on Slurm.
  -e KEY=VALUE       Export an extra environment variable to the run.

TorchTitan args:
  Pass native tyro args, e.g. --training.steps 1, or xx-style key=value,
  e.g. training.steps=1. xx-style training args are converted to --key=value.

Examples:
  NGPU=2 MODULE=worldmodel CONFIG=worldmodel_debugmodel ./run_train.sh --training.steps 1
  ./run_train.sh module=worldmodel config=worldmodel_debugmodel ngpu=2 training.steps=1
  ./run_train.sh N=1 PARTITION=tbox module=worldmodel config=worldmodel ngpu=8 training.steps=1000
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  help
  exit 0
fi

COMMAND_TO_LOG="$0 $*"
LOCAL_TORCHTITAN_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
LOCAL_CODEDIR=$(realpath "$LOCAL_TORCHTITAN_DIR/..")
PROJECT_DIR="$LOCAL_TORCHTITAN_DIR"

declare -A env_vars
TORCHTITAN_ARGS=()

NGPU_VALUE="${NGPU:-8}"
LOG_RANK_VALUE="${LOG_RANK:-0}"
MODULE_VALUE="${MODULE:-llama3}"
CONFIG_VALUE="${CONFIG:-llama3_debugmodel}"
COMM_MODE_VALUE="${COMM_MODE:-}"
TORCHFT_LIGHTHOUSE_VALUE="${TORCHFT_LIGHTHOUSE:-http://localhost:29510}"
MASTER_PORT_VALUE="${MASTER_PORT:-12355}"
MASTER_VALUE=""
CPUS_PER_TASK_VALUE=""
GPUS_PER_TASK_VALUE=""
N_VALUE=""
PARTITION_VALUE=""
NODELIST_VALUE=""
WAIT_VALUE=""

append_torchtitan_arg() {
  local arg="$1"
  if [[ "$arg" == --* ]]; then
    TORCHTITAN_ARGS+=("$arg")
  elif [[ "$arg" == *=* ]]; then
    local key="${arg%%=*}"
    local value="${arg#*=}"
    key="${key//_/-}"
    case "${value,,}" in
      true) TORCHTITAN_ARGS+=("--$key") ;;
      false)
        if [[ "$key" == *.* ]]; then
          TORCHTITAN_ARGS+=("--${key%.*}.no-${key##*.}")
        else
          TORCHTITAN_ARGS+=("--no-$key")
        fi
        ;;
      *) TORCHTITAN_ARGS+=("--$key=$value") ;;
    esac
  else
    TORCHTITAN_ARGS+=("$arg")
  fi
}

while [[ $# -gt 0 ]]; do
  if [[ "$1" == "-e" ]]; then
    [[ $# -ge 2 ]] || { echo "Error: -e requires KEY=VALUE"; exit 1; }
    IFS='=' read -r key value <<< "$2"
    [[ -n "$key" && "$2" == *=* ]] || { echo "Error: -e requires KEY=VALUE"; exit 1; }
    env_vars[$key]="$value"
    shift 2
    continue
  fi

  if [[ "$1" == "--module="* ]]; then
    MODULE_VALUE="${1#--module=}"
    shift
    continue
  fi
  if [[ "$1" == "--module" ]]; then
    [[ $# -ge 2 ]] || { echo "Error: --module requires a value"; exit 1; }
    MODULE_VALUE="$2"
    shift 2
    continue
  fi
  if [[ "$1" == "--config="* ]]; then
    CONFIG_VALUE="${1#--config=}"
    shift
    continue
  fi
  if [[ "$1" == "--config" ]]; then
    [[ $# -ge 2 ]] || { echo "Error: --config requires a value"; exit 1; }
    CONFIG_VALUE="$2"
    shift 2
    continue
  fi

  if [[ "$1" == --* ]]; then
    append_torchtitan_arg "$1"
    shift
    continue
  fi

  if [[ "$1" == *=* ]]; then
    IFS='=' read -r key value <<< "$1"
    case "$key" in
      N) N_VALUE="$value" ;;
      PARTITION) PARTITION_VALUE="$value" ;;
      NODELIST) NODELIST_VALUE="$value" ;;
      WAIT) WAIT_VALUE="$value" ;;
      NGPU|ngpu|devices) NGPU_VALUE="$value" ;;
      LOG_RANK|log_rank) LOG_RANK_VALUE="$value" ;;
      MODULE|module) MODULE_VALUE="$value" ;;
      CONFIG|config) CONFIG_VALUE="$value" ;;
      COMM_MODE|comm_mode) COMM_MODE_VALUE="$value" ;;
      TORCHFT_LIGHTHOUSE|torchft_lighthouse) TORCHFT_LIGHTHOUSE_VALUE="$value" ;;
      MASTER|master) MASTER_VALUE="$value" ;;
      MASTER_PORT|masterport) MASTER_PORT_VALUE="$value" ;;
      CPUS_PER_TASK|cpus_per_task) CPUS_PER_TASK_VALUE="$value" ;;
      GPUS_PER_TASK|gpus_per_task) GPUS_PER_TASK_VALUE="$value" ;;
      PROJECT|project|codedir) echo "Error: $key is fixed by run_train.sh and is not supported"; exit 1 ;;
      *) append_torchtitan_arg "$1" ;;
    esac
  else
    append_torchtitan_arg "$1"
  fi
  shift
done

env_vars["PYTHONUNBUFFERED"]=1
env_vars["LOGGABLE_PROGRESS"]=1
env_vars["TRAINING_COMMAND"]="$COMMAND_TO_LOG"

generate_uuid() {
  if [[ -r /proc/sys/kernel/random/uuid ]]; then
    tr -d '\n' < /proc/sys/kernel/random/uuid
  elif command -v uuidgen >/dev/null 2>&1; then
    uuidgen | tr '[:upper:]' '[:lower:]'
  elif command -v python3 >/dev/null 2>&1; then
    python3 -c 'import uuid; print(uuid.uuid4())'
  else
    date +'%s-%N'
  fi
}

if [[ -z "${env_vars[REPORTERV2_HOST]+set}" ]]; then
  env_vars["REPORTERV2_HOST"]="${REPORTERV2_HOST:-http://data-gen.comma.life:3080/reporterv2}"
fi
if [[ -z "${env_vars[REPORTERV2_TRAINING_ID]+set}" ]]; then
  env_vars["REPORTERV2_TRAINING_ID"]="${REPORTERV2_TRAINING_ID:-$(generate_uuid)}"
fi

is_cluster_run=0
if [[ -z "$N_VALUE" && -z "$PARTITION_VALUE" && -z "$NODELIST_VALUE" ]]; then
  echo "N nodes and partition not provided, running locally."
elif [[ -n "$N_VALUE" && -n "$PARTITION_VALUE" && -z "$NODELIST_VALUE" ]]; then
  echo "Running on $N_VALUE node(s) in $PARTITION_VALUE"
  is_cluster_run=1
elif [[ -z "$N_VALUE" && -n "$PARTITION_VALUE" && -n "$NODELIST_VALUE" ]]; then
  echo "Running on $NODELIST_VALUE in $PARTITION_VALUE"
  is_cluster_run=1
else
  echo "Error: provide either N and PARTITION, NODELIST and PARTITION, or neither to run locally"
  exit 1
fi

has_training_steps_arg() {
  local arg
  for arg in "${TORCHTITAN_ARGS[@]}"; do
    [[ "$arg" == "--training.steps" || "$arg" == --training.steps=* ]] && return 0
  done
  return 1
}

run_local() {
  for key in "${!env_vars[@]}"; do
    export "$key=${env_vars[$key]}"
  done
  export LOG_RANK="$LOG_RANK_VALUE"
  export TORCHFT_LIGHTHOUSE="$TORCHFT_LIGHTHOUSE_VALUE"
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

  cd "$PROJECT_DIR"
  if [[ -n "$COMM_MODE_VALUE" ]]; then
    echo "Running with comm_mode=$COMM_MODE_VALUE"
    local -a extra_args=()
    if ! has_training_steps_arg; then
      extra_args+=(--training.steps 1)
    fi
    NGPU="$NGPU_VALUE" LOCAL_RANK=0 uv run --project "$PROJECT_DIR" --frozen \
      python -m torchtitan.train \
      --module "$MODULE_VALUE" --config "$CONFIG_VALUE" \
      "${TORCHTITAN_ARGS[@]}" --comm.mode="$COMM_MODE_VALUE" "${extra_args[@]}"
  else
    uv run --project "$PROJECT_DIR" --frozen \
      python -m torch.distributed.run \
      --nproc_per_node="$NGPU_VALUE" \
      --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
      --local-ranks-filter "$LOG_RANK_VALUE" --role rank --tee 3 \
      -m torchtitan.train --module "$MODULE_VALUE" --config "$CONFIG_VALUE" \
      "${TORCHTITAN_ARGS[@]}"
  fi
}

cache_git_info() {
  local cache_dir="$1"
  local git_info_cache="$LOCAL_CODEDIR/git-info.json"
  mkdir -p "$cache_dir"
  if [[ -f "$git_info_cache" ]]; then
    cp "$git_info_cache" "$cache_dir/git-info.json"
    return
  fi

  local branch commit diff upstream
  commit=$(git -C "$LOCAL_CODEDIR" rev-parse HEAD)
  diff=$(git -C "$LOCAL_CODEDIR" diff HEAD)
  branch=$(git -C "$LOCAL_CODEDIR" rev-parse --abbrev-ref HEAD)
  if [[ "$branch" == "HEAD" ]]; then
    branch="HEAD detached"
  elif upstream=$(git -C "$LOCAL_CODEDIR" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null); then
    branch="$upstream"
  fi

  json_escape() {
    local value="$1"
    value=${value//\\/\\\\}
    value=${value//\"/\\\"}
    value=${value//$'\n'/\\n}
    value=${value//$'\r'/\\r}
    value=${value//$'\t'/\\t}
    printf '%s' "$value"
  }

  printf '{"branch":"%s","commit":"%s","diff":"%s"}' \
    "$(json_escape "$branch")" \
    "$(json_escape "$commit")" \
    "$(json_escape "$diff")" \
    > "$cache_dir/git-info.json"
}

write_job_env() {
  local job_env_file="$1"
  local code_cache="$2"
  local project_cache="$3"
  {
    printf 'CODE_CACHE=%q\n' "$code_cache"
    printf 'PROJECT_CACHE=%q\n' "$project_cache"
    printf 'COMMAND_TO_LOG=%q\n' "$COMMAND_TO_LOG"
    printf 'NGPU=%q\n' "$NGPU_VALUE"
    printf 'LOG_RANK=%q\n' "$LOG_RANK_VALUE"
    printf 'MODULE=%q\n' "$MODULE_VALUE"
    printf 'CONFIG=%q\n' "$CONFIG_VALUE"
    printf 'COMM_MODE=%q\n' "$COMM_MODE_VALUE"
    printf 'MASTER=%q\n' "$MASTER_VALUE"
    printf 'MASTER_PORT=%q\n' "$MASTER_PORT_VALUE"
    printf 'TORCHFT_LIGHTHOUSE=%q\n' "$TORCHFT_LIGHTHOUSE_VALUE"
    printf 'TORCHTITAN_ARGS=('
    local arg
    for arg in "${TORCHTITAN_ARGS[@]}"; do
      printf ' %q' "$arg"
    done
    printf ' )\n'
    local key
    for key in "${!env_vars[@]}"; do
      printf 'export %s=%q\n' "$key" "${env_vars[$key]}"
    done
  } > "$job_env_file"
}

print_job_logs() {
  local job_id="$1"
  local job_name="$2"
  local nodes
  nodes=$(sacct -X -j "$job_id" --noheader --format=nodelist%-100 | xargs || true)
  [[ -n "$nodes" && "$nodes" != "None assigned" ]] || return 0

  local -a node_list
  mapfile -t node_list < <(scontrol show hostnames "$nodes")
  local node file_type log_file
  for node in "${node_list[@]}"; do
    for file_type in out err; do
      log_file="/var/log/slurm/${job_name}.${file_type}"
      echo "===== $node:$log_file ====="
      ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$node" \
        "tail -n 300 '$log_file' 2>/dev/null || true"
    done
  done
}

wait_for_job_start() {
  local job_id="$1"
  local tries=0
  local state=""
  while [[ "$tries" -lt 30 || "$WAIT_VALUE" == "1" ]]; do
    state=$(sacct -X -j "$job_id" --noheader --format=state | head -n 1 | xargs || true)
    [[ "$state" == "RUNNING" || "$state" =~ COMPLETED|FAILED|CANCELLED|NODE_FAIL|TIMEOUT|OUT_OF_MEMORY ]] && return 0
    sleep 1
    tries=$((tries + 1))
  done
  return 1
}

wait_for_job_done() {
  local job_id="$1"
  local state=""
  while true; do
    state=$(sacct -X -j "$job_id" --noheader --format=state | head -n 1 | xargs || true)
    if [[ "$state" =~ COMPLETED|FAILED|CANCELLED|NODE_FAIL|TIMEOUT|OUT_OF_MEMORY ]]; then
      [[ "$state" == "COMPLETED" ]]
      return
    fi
    sleep 5
  done
}

submit_cluster() {
  [[ -z "${env_vars[codedir]:-}" ]] || { echo "Error: codedirs are no longer supported for training"; exit 1; }

  local job_name
  job_name="$(hostname)_$(date +'%Y-%m-%d_%H-%M-%S')"
  echo "JOB NAME: $job_name"

  local code_cache="/code.nfs/branches/caches/$job_name/xx"
  local project_cache="$code_cache/torchtitan"
  if [[ -z "$CPUS_PER_TASK_VALUE" ]]; then
    CPUS_PER_TASK_VALUE=$((NGPU_VALUE * 12))
  fi
  local exclude_from="$LOCAL_CODEDIR/training/.training_cache_exclude"
  local exclude_from_local="$LOCAL_CODEDIR/training/.training_cache_exclude.local"
  [[ -r "$exclude_from_local" ]] || exclude_from_local=/dev/null

  mkdir -p "$code_cache"
  echo "caching $LOCAL_CODEDIR -> $code_cache..."
  rsync -a --max-delete=0 --copy-dest=/xx --info=progress2 \
    --exclude-from="$exclude_from" --exclude-from="$exclude_from_local" \
    "$LOCAL_CODEDIR/" "rsync://app01:1026/code_nfs/${code_cache#/code.nfs}/"

  [[ -f "$project_cache/run_train.sh" ]] || { echo "Error: cached run_train.sh not found: $project_cache/run_train.sh"; exit 1; }
  [[ -f "$project_cache/multinode_trainer.slurm" ]] || { echo "Error: cached Slurm helper not found: $project_cache/multinode_trainer.slurm"; exit 1; }

  cache_git_info "$code_cache"

  local job_env_file="$project_cache/outputs/$job_name.env"
  mkdir -p "$(dirname "$job_env_file")"
  write_job_env "$job_env_file" "$code_cache" "$project_cache"

  local -a sbatch_args=(
    --job-name "$job_name"
    --output=/dev/null
    --exclusive
    --no-requeue
    --export=NIL
    --uid=batman
    --gid=batman
    --chdir=/home
    --cpus-per-task "$CPUS_PER_TASK_VALUE"
  )
  [[ -n "$GPUS_PER_TASK_VALUE" ]] && sbatch_args+=(--gpus-per-task "$GPUS_PER_TASK_VALUE")
  if [[ -n "$N_VALUE" ]]; then
    sbatch_args+=(--nodes "$N_VALUE" --ntasks "$N_VALUE")
  fi
  [[ -n "$PARTITION_VALUE" ]] && sbatch_args+=(--partition "$PARTITION_VALUE")
  if [[ -n "$NODELIST_VALUE" ]]; then
    sbatch_args+=(--nodelist "$NODELIST_VALUE")
    if [[ -z "$N_VALUE" ]] && command -v scontrol >/dev/null 2>&1; then
      local node_count
      node_count=$(scontrol show hostnames "$NODELIST_VALUE" 2>/dev/null | wc -l)
      [[ "$node_count" -gt 0 ]] && sbatch_args+=(--ntasks "$node_count")
    fi
  fi

  local sbatch_output job_id
  sbatch_output=$(sudo sbatch "${sbatch_args[@]}" "$project_cache/multinode_trainer.slurm" "$job_env_file")
  echo "$sbatch_output"
  job_id=$(awk '{print $NF}' <<< "$sbatch_output")
  [[ -n "$job_id" ]] || { echo "Error: unable to parse sbatch job id"; exit 1; }

  [[ "$WAIT_VALUE" == "0" ]] && exit 0

  echo "Waiting for job to start..."
  if ! wait_for_job_start "$job_id"; then
    echo "Job queued but hasn't started yet. Attach with training/show_logs.sh once started."
    return 0
  fi

  if [[ -t 1 ]]; then
    local state
    state=$(sacct -X -j "$job_id" --noheader --format=state | head -n 1 | xargs || true)
    if [[ "$state" == "RUNNING" ]]; then
      sattach "$job_id.0" || print_job_logs "$job_id" "$job_name"
    else
      print_job_logs "$job_id" "$job_name"
    fi
  else
    if wait_for_job_done "$job_id"; then
      print_job_logs "$job_id" "$job_name"
    else
      print_job_logs "$job_id" "$job_name"
      return 1
    fi
  fi
}

if [[ "$is_cluster_run" -eq 1 ]]; then
  submit_cluster
else
  run_local
fi
