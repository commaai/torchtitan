# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn

from torchtitan.components.checkpoint import OPTIMIZER
from xx.training.lib.torchtitan.onnx_checkpoint import OnnxCheckpointManager

from .model import ACTION_HEAD_NAME, RLDrivingModel


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
        inputs = dict(zip(self.input_names, self._build_onnx_inputs()))
        self._export_one(
            _TargetActorOnnxModel(model).eval(),
            inputs,
            path,
            output_names=[ACTION_HEAD_NAME],
        )
