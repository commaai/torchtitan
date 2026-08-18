# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .model import PathHead
from .model_constants import (
    ACTION_LEN,
    DESIRE_LEN,
    LEAD_TRAJECTORY_DIM,
    LL_SIZE,
    META_LEN,
    PLAN_SIZE,
    POSE_OUTSIZE,
    RE_SIZE,
)

PLAN_HEAD_SIZE = 2 * PLAN_SIZE

DRIVING_HEADS = [
    PathHead(name="plan", output_size=PLAN_HEAD_SIZE, mlp=True, scale=True),
    PathHead(name="lead", output_size=3 * (2 * LEAD_TRAJECTORY_DIM), mlp=True, scale=True),
    PathHead(name="lead_prob", output_size=3, mlp=True, scale=False),
    PathHead(name="action", output_size=ACTION_LEN * 2, mlp=True, scale=True),
]
TEMPORAL_META_HEADS = [
    PathHead(name="desire_state", output_size=DESIRE_LEN, mlp=True, scale=False),
]
META_HEADS = [
    PathHead(name="lane_lines", output_size=4 * (2 * LL_SIZE), mlp=False, scale=False),
    PathHead(name="lane_lines_prob", output_size=8, mlp=False, scale=False),
    PathHead(name="road_edges", output_size=2 * (2 * RE_SIZE), mlp=False, scale=False),
    PathHead(name="meta", output_size=META_LEN, mlp=False, scale=False),
    PathHead(name="desire_pred", output_size=DESIRE_LEN * 4, mlp=False, scale=False),
    PathHead(name="road_transform", output_size=POSE_OUTSIZE * 2, mlp=False, scale=False),
    PathHead(name="wide_from_device_euler", output_size=3 * 2, mlp=False, scale=False),
]
POSE_HEADS = [
    PathHead(name="pose", output_size=POSE_OUTSIZE * 2, mlp=True, scale=True),
]
