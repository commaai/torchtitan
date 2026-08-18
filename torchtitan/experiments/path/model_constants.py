# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Path model input, output, and timing contract.

This is the canonical implementation for both TorchTitan PATH and the legacy
``xx.ml_tools.constants.model`` import path. Keep it free of xx and openpilot
dependencies: these values determine checkpoint shapes and the ONNX interface.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, TypedDict

import numpy as np


# These camera/model dimensions are part of the model contract. Keeping them
# here avoids importing openpilot hardware and transformation modules.
CAMERA_FPS = 20
MEDMODEL_INPUT_SIZE = (512, 256)
BIGMODEL_INPUT_SIZE = (1024, 512)
SBIGMODEL_INPUT_SIZE = (512, 256)


def index_function(idx: int, max_val: float = 192) -> float:
    return (max_val / 1024) * (idx**2)


IDX_N = 33
T_IDXS = np.array([index_function(idx, max_val=10.0) for idx in range(IDX_N)], dtype=np.float64)
X_IDXS = np.array([index_function(idx, max_val=192.0) for idx in range(IDX_N)], dtype=np.float64)
D_IDXS = np.array([index_function(idx, max_val=512.0) for idx in range(IDX_N)], dtype=np.float64)

W, H = MEDMODEL_INPUT_SIZE
BIG_W, BIG_H = BIGMODEL_INPUT_SIZE
SBIG_W, SBIG_H = SBIGMODEL_INPUT_SIZE

POSE_OUTSIZE = 6
META_LEN = 55
DESIRE_LEN = 8
ACTION_LEN = 2

LEAD_PRED_DIM = 4
LEAD_TRAJECTORY_DIM = 6 * 4
PLAN_WIDTH = 15
PLAN_SIZE = PLAN_WIDTH * IDX_N
LATERAL_PLANNER_SOLUTION_WIDTH = 4
LATERAL_PLANNER_SOLUTION_SIZE = LATERAL_PLANNER_SOLUTION_WIDTH * IDX_N
LL_SIZE = 2 * IDX_N
RE_SIZE = 2 * IDX_N

LEAD_MHP_N = 2
PLAN_MHP_N = 5


class VisionFrameType(Enum):
    YUV = "YUV"
    RGB = "RGB"


class ModelInputs:
    IMG = "img"
    BIG_IMG = "big_img"
    FEATURES = "features"
    DESIRE = "desire_pulse"
    TRAFFIC = "traffic_convention"
    ACTION_T = "action_t"
    ANTI_CHEATING_SAMPLES = "anti_cheating_samples"
    FEATURE_QUEUE = "feature_queue"


SUPERCOMBO_FPS = 5
N_FRAMES = 2
FRAME_TYPE = VisionFrameType.YUV
INPUT_FRAMES_NAMES = [ModelInputs.IMG, ModelInputs.BIG_IMG]

INPUT_ALIASES = {
    getattr(ModelInputs, name): [getattr(ModelInputs, name)] for name in dir(ModelInputs) if not name.startswith("__")
}
INPUT_ALIASES[ModelInputs.IMG] += ["input_imgs", "map"]
INPUT_ALIASES[ModelInputs.BIG_IMG] += ["big_input_imgs"]
INPUT_ALIASES[ModelInputs.FEATURES] += ["vision_features"]
INPUT_ALIASES[ModelInputs.DESIRE] += ["desire_pulse"]
INPUT_ALIASES[ModelInputs.TRAFFIC] += ["traffic_convention"]
INPUT_ALIASES[ModelInputs.ACTION_T] += ["action_t"]
INPUT_ALIASES[ModelInputs.FEATURE_QUEUE] += ["features_buffer"]

ALIASES_TO_CANONICAL_NAMES = {alias: name for name, aliases in INPUT_ALIASES.items() for alias in aliases}

VISION_INPUTS_RGB = {
    ModelInputs.IMG: (3, H, W),
    ModelInputs.BIG_IMG: (3, SBIG_H, SBIG_W),
}

VISION_INPUTS_YUV = {
    ModelInputs.IMG: (6, H // 2, W // 2),
    ModelInputs.BIG_IMG: (6, SBIG_H // 2, SBIG_W // 2),
}

VISION_INPUTS = {
    VisionFrameType.RGB: VISION_INPUTS_RGB,
    VisionFrameType.YUV: VISION_INPUTS_YUV,
}

TEMPORAL_INPUTS = {
    ModelInputs.FEATURES: (512,),
    ModelInputs.DESIRE: (DESIRE_LEN,),
    ModelInputs.TRAFFIC: (2,),
    ModelInputs.ACTION_T: (ACTION_LEN,),
}


def to_canonical(data: dict[str, Any]) -> dict[str, Any]:
    return {ALIASES_TO_CANONICAL_NAMES[name]: value for name, value in data.items()}


def to_aliases(data: dict[str, Any], aliases: list[str]) -> dict[str, Any]:
    canonical_names_to_aliases = to_canonical(dict(zip(aliases, aliases)))
    return {canonical_names_to_aliases[name]: value for name, value in data.items()}


class FrameConstants(TypedDict):
    fps: int
    frame_skip: int
    n_frames: int
    frame_shapes: dict[str, tuple[int, int, int]]
    dt: float
    desire_window_len: int
    temporal_len: int
    history_len: int
    history_idxs: np.ndarray


def frame_constants_from_fps(
    fps: int = SUPERCOMBO_FPS,
    n_frames: int = N_FRAMES,
    frame_type: VisionFrameType = FRAME_TYPE,
) -> FrameConstants:
    desire_window_len_seconds = 5
    history_fps = 4
    history_len_seconds = 2
    history_skip = fps // history_fps
    history_idxs = -np.arange(1, history_len_seconds * fps, history_skip).astype(np.int32)[::-1]
    desire_window_len = fps * desire_window_len_seconds
    temporal_len = desire_window_len + int(history_idxs[-1] - history_idxs[0])
    return {
        "fps": fps,
        "frame_skip": CAMERA_FPS // fps,
        "n_frames": n_frames,
        "frame_shapes": {name: (shape[0] * n_frames, *shape[1:]) for name, shape in VISION_INPUTS[frame_type].items()},
        "dt": 1 / fps,
        "desire_window_len": desire_window_len,
        "temporal_len": temporal_len,
        "history_len": fps * history_len_seconds,
        "history_idxs": history_idxs,
    }
