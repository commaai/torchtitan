#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/_run_common.sh"

MODULE=${MODULE:-"llama3"}
CONFIG=${CONFIG:-"llama3_debugmodel"}

tt_prepare "$@"
TT_COMM_SUFFIX=(--training.steps 1)
tt_launch torchtitan.train \
  --module "$MODULE" \
  --config "$CONFIG" \
  "${TT_ARGS[@]}"
