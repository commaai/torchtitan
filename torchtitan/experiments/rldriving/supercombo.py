# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import MethodType

import torch
from xx.ml_tools.constants.model import ModelInputs
from torchtitan.experiments.path.model import PathSelfAttention
from torchtitan.experiments.path.model_config import model_config
from .model import actor_config


VISION_OUTPUT_ORDER = tuple(
    "lane_lines lane_lines_prob road_edges meta desire_pred pose wide_from_device_euler road_transform".split()
)
OFF_POLICY_OUTPUT_ORDER = ("plan", "lead", "lead_prob", "desire_state")
ON_POLICY_OUTPUT_ORDER = ("action",)
OUTPUT_ORDER = (*VISION_OUTPUT_ORDER, *OFF_POLICY_OUTPUT_ORDER, *ON_POLICY_OUTPUT_ORDER, "hidden_state")


def _naive_attention(self: PathSelfAttention, x: torch.Tensor) -> torch.Tensor:
    b, t, _ = x.shape
    qkv = self.c_attn(self.norm(x)).view(b, t, 3, self.n_head, self.head_dim)
    q, k, v = (value.squeeze(0) for value in qkv.permute(2, 0, 3, 1, 4).split(1))
    q, k = self.q_norm(q), self.k_norm(k)
    scores = (q @ k.transpose(-2, -1)) * self.head_dim**-0.5
    x = (scores.masked_fill(~self._supercombo_mask, float("-inf")).softmax(-1) @ v).transpose(1, 2)
    return self.dropout(self.c_proj(x.reshape(b, t, self.n_head * self.head_dim)))


class Supercombo(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        config = model_config()
        self.vision = config.vision.build()
        self.point_policy = config.point_policy.build()
        self.off_policy = config.temporal_policy.build()
        self.on_policy = actor_config().build()
        for policy in (self.off_policy, self.on_policy):
            for layer in policy.temporal_summarizer.transformer.layers:
                attention = layer.attention
                mask = torch.ones(
                    1, 1, policy.temporal_summarizer.block_size, policy.temporal_summarizer.block_size, dtype=torch.bool
                )
                attention.register_buffer("_supercombo_mask", mask.tril(), persistent=False)
                attention.forward = MethodType(_naive_attention, attention)

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
