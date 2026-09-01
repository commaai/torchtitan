# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import MethodType

from xx.training.path.model import PathSelfAttention
from xx.training.path.model_config import model_config
from xx.training.path.model_constants import ModelInputs, SPATIAL_SIZE

import torch

from .model import actor_config


VISION_OUTPUT_ORDER = tuple(
    "lane_lines lane_lines_prob road_edges meta desire_pred pose wide_from_device_euler road_transform".split()
)
OFF_POLICY_OUTPUT_ORDER = ("plan", "lead", "lead_prob", "desire_state")
ON_POLICY_OUTPUT_ORDER = ("action",)
OUTPUT_ORDER = (*VISION_OUTPUT_ORDER, *OFF_POLICY_OUTPUT_ORDER, *ON_POLICY_OUTPUT_ORDER, "hidden_state")
TINYGRAD_ONNX_DOMAIN = "org.tinygrad"
TINYGRAD_CONTIGUOUS_OP = "Contiguous"


def _tinygrad_contiguous(x: torch.Tensor) -> torch.Tensor:
    if torch.onnx.is_in_onnx_export():
        return torch.onnx.ops.symbolic(
            f"{TINYGRAD_ONNX_DOMAIN}::{TINYGRAD_CONTIGUOUS_OP}",
            (x,),
            dtype=x.dtype,
            shape=x.shape,
            version=1,
        )
    return x.contiguous()


class _TinygradContiguous(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _tinygrad_contiguous(x)


def _naive_attention(self: PathSelfAttention, x: torch.Tensor) -> torch.Tensor:
    b, t, _ = x.shape
    qkv = self.c_attn(self.norm(x)).view(b, t, 3, self.n_head, self.head_dim)
    q, k, v = (value.squeeze(0) for value in qkv.permute(2, 0, 3, 1, 4).split(1))
    q, k = self.q_norm(q), self.k_norm(k)
    scores = (q @ k.transpose(-2, -1)) * self.head_dim**-0.5
    x = (scores.masked_fill(~self._supercombo_mask, float("-inf")).softmax(-1) @ v).transpose(1, 2)
    return self.dropout(self.c_proj(x.reshape(b, t, self.n_head * self.head_dim)))


# some micro optimizations can be made but not worth it for now
# there are some Unsqueeze -> Gather that can be bipassed (traffic_convention, action_t)
class Supercombo(torch.nn.Module):
    def __init__(self, vision_flavor: str = "convnext_xxlarge") -> None:
        super().__init__()
        config = model_config(vision_flavor)
        config.temporal_policy.temporal_summarizer.dense_training_outputs = False
        self.vision = config.vision.build()
        if not isinstance(self.vision.encoder.norm_pre, torch.nn.Identity):
            raise TypeError("expected ConvNeXt norm_pre to be Identity")
        # Match the ConvNeXt reassociation heuristic's realization boundary: after the
        # final residual block and before the reduction in the head LayerNorm.
        self.vision.encoder.norm_pre = _TinygradContiguous()
        self.point_policy = config.point_policy.build()
        self.off_policy = config.temporal_policy.build()
        self.on_policy = actor_config().build()
        output_size = SPATIAL_SIZE * self.vision.config.vision_features + sum(
            hydra.final_layer[name].out_features
            for hydra, names in (
                (self.point_policy.hydra, VISION_OUTPUT_ORDER),
                (self.off_policy.temporal_hydra, OFF_POLICY_OUTPUT_ORDER),
                (self.on_policy.temporal_hydra, ON_POLICY_OUTPUT_ORDER),
            )
            for name in names
        )
        self.register_buffer("pad", torch.zeros(1, -output_size % 4), persistent=False)
        for policy in (self.off_policy, self.on_policy):
            summarizer = policy.temporal_summarizer
            n_tokens = summarizer.temporal_size * summarizer.spatial_size
            for layer in policy.temporal_summarizer.transformer.layers:
                attention = layer.attention
                mask = torch.ones(1, 1, n_tokens, n_tokens, dtype=torch.bool)
                attention.register_buffer("_supercombo_mask", mask.tril(), persistent=False)
                attention.forward = MethodType(_naive_attention, attention)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        current = self.vision(inputs)
        features = torch.cat((inputs["features_buffer"], current[:, None]), dim=1)
        outputs = self.point_policy(current.mean(dim=1))
        for policy, names in ((self.off_policy, OFF_POLICY_OUTPUT_ORDER), (self.on_policy, ON_POLICY_OUTPUT_ORDER)):
            policy_outputs = policy(
                features,
                inputs[ModelInputs.DESIRE],
                inputs[ModelInputs.TRAFFIC][:, None],
                inputs[ModelInputs.ACTION_T][:, None],
            )
            outputs.update({name: policy_outputs[name] for name in names})
        outputs["hidden_state"] = current.detach().flatten(1)
        return torch.cat([outputs[name] for name in OUTPUT_ORDER] + [self.pad], dim=1)
