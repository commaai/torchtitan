#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/_run_common.sh"

CONFIG=${CONFIG:-"convnext_small"}
ANNEALING_STEPS=${ANNEALING_STEPS:-128}
ANNEALING_DECAY_TYPE=${ANNEALING_DECAY_TYPE:-linear}

if [[ -z "${CHECKPOINT_RUN_ID:-}" ]]; then
  echo "CHECKPOINT_RUN_ID must be set"
  exit 1
fi
if [[ -z "${CHECKPOINT_STEPS:-}" ]]; then
  echo "CHECKPOINT_STEPS must be set"
  exit 1
fi
read -ra CHECKPOINT_STEP_ARGS <<< "$CHECKPOINT_STEPS"

ARGS=()
for arg in "$@"; do
  case "$arg" in
    --metrics.enable_reporterv2=) ARGS+=("--metrics.enable_reporterv2") ;;
    --validator.save_predictions=) ARGS+=("--validator.save_predictions") ;;
    *) ARGS+=("$arg") ;;
  esac
done

tt_prepare "${ARGS[@]}"
tt_launch torchtitan.experiments.path.tuning.anneal \
  --config "$CONFIG" \
  --checkpoint-run-id "$CHECKPOINT_RUN_ID" \
  --checkpoint-steps "${CHECKPOINT_STEP_ARGS[@]}" \
  --annealing-steps "$ANNEALING_STEPS" \
  --annealing-decay-type "$ANNEALING_DECAY_TYPE" \
  "${TT_ARGS[@]}"
