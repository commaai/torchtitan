# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from xx.ml_tools.constants.model import ModelInputs

from torchtitan.experiments.path.model_config import model_config

from .model import actor_config


VISION_OUTPUT_ORDER = tuple(
    "lane_lines lane_lines_prob road_edges meta desire_pred pose wide_from_device_euler road_transform".split()
)
OFF_POLICY_OUTPUT_ORDER = ("plan", "lead", "lead_prob", "desire_state")
ON_POLICY_OUTPUT_ORDER = ("action",)
OUTPUT_ORDER = (*VISION_OUTPUT_ORDER, *OFF_POLICY_OUTPUT_ORDER, *ON_POLICY_OUTPUT_ORDER, "hidden_state")


class Supercombo(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        config = model_config()
        self.vision = config.vision.build()
        self.point_policy = config.point_policy.build()
        self.off_policy = config.temporal_policy.build()
        self.on_policy = actor_config().build()

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        current = self.vision(inputs)
        features = torch.cat((inputs["features_buffer"], current[:, None]), dim=1)
        outputs = self.point_policy(current)
        for policy, names in ((self.off_policy, OFF_POLICY_OUTPUT_ORDER), (self.on_policy, ON_POLICY_OUTPUT_ORDER)):
            policy_outputs = policy(
                features,
                inputs[ModelInputs.DESIRE],
                inputs[ModelInputs.TRAFFIC][:, None],
                inputs[ModelInputs.ACTION_T][:, None],
            )
            for name in names:
                value = policy_outputs[name]
                outputs[name] = value[:, -1] if value.ndim == 3 else value
        outputs["hidden_state"] = current.detach()
        output = torch.cat([outputs[name] for name in OUTPUT_ORDER], dim=1)
        return torch.nn.functional.pad(output, (0, -output.shape[1] % 4))
