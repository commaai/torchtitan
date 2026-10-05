# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
import posixpath
from typing import cast
from xx.training.lib.torchtitan.onnx_checkpoint import OnnxCheckpointManager
from xx.training.path.model_constants import VISION_INPUTS_YUV
from xx.training.path.onnx_checkpoint import _TemporalPolicyOnnxModel, _VisionOnnxModel, PathOnnxCheckpointManager

import torch
import torch.nn as nn

from torchtitan.components.checkpoint import OPTIMIZER
from torchtitan.components import fs

from .model import ACTION_HEAD_NAME, RLDrivingModel
from .lora import merge_lora


class _TargetActorOnnxModel(nn.Module):
    def __init__(self, model: RLDrivingModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.model.target_forward(inputs)


class RLDrivingOnnxCheckpointManager(OnnxCheckpointManager):
    @dataclass(kw_only=True, slots=True)
    class Config(OnnxCheckpointManager.Config):
        checkpoint_base_folder: str = ""

    def __init__(self, config: Config, **kwargs) -> None:
        if config.checkpoint_base_folder:
            kwargs["base_folder"] = config.checkpoint_base_folder
        super().__init__(config, **kwargs)
        if self.enable:
            self.states.pop(OPTIMIZER)

    def _export_onnx(self, model: nn.Module, path: str) -> None:
        model = cast(RLDrivingModel, model)
        merge_lora(model)
        inputs = dict(zip(self.input_names, self._build_onnx_inputs()))
        if model.target_vision is not None:
            # This is the unsharded CPU export copy. Reuse PATH's rollout
            # interfaces, exporting EMA vision alongside the EMA action policy.
            model.vision = model.target_vision
            model.temporal_policy = model.off_policy
            vision_inputs = {
                name: torch.zeros((1, channels * 2, height, width), dtype=torch.float32)
                for name, (channels, height, width) in VISION_INPUTS_YUV.items()
            }
            folder = posixpath.dirname(path)
            self._export_one(
                _VisionOnnxModel(model).eval(), vision_inputs,
                fs.join_path(folder, "vision.onnx"), external_data=True,
            )
            self._export_one(
                _TemporalPolicyOnnxModel(model).eval(), inputs,
                fs.join_path(folder, "temporal_policy.onnx"),
            )
        # Publish model.onnx last: rollout workers use it as the ready signal.
        self._export_one(
            _TargetActorOnnxModel(model).eval(),
            inputs,
            path,
            output_names=[ACTION_HEAD_NAME],
        )

    def _post_export_hook(self, onnx_data: bytes) -> bytes:
        return PathOnnxCheckpointManager._post_export_hook(self, onnx_data)
