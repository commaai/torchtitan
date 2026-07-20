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

if [[ -z "${RUN_ID:-}" ]]; then
  echo "RUN_ID must be set"
  exit 1
fi
if [[ -z "${EVAL_ID:-}" ]]; then
  echo "EVAL_ID must be set"
  exit 1
fi

CHECKPOINT_ARGS=()
if [[ -n "${CHECKPOINT_STEPS:-}" ]]; then
  read -ra CHECKPOINT_STEP_ARGS <<< "$CHECKPOINT_STEPS"
  CHECKPOINT_ARGS=(--checkpoint-steps "${CHECKPOINT_STEP_ARGS[@]}")
fi

export REPORTERV2_TRAINING_ID="$RUN_ID"
tt_prepare "$@"
tt_launch torchtitan.experiments.path.tuning.eval \
  --config "$CONFIG" \
  --run-id "$RUN_ID" \
  "${CHECKPOINT_ARGS[@]}" \
  --eval-id "$EVAL_ID" \
  "${TT_ARGS[@]}"
