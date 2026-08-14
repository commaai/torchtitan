# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import re
import sys
from pathlib import Path

import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file

from torchtitan.experiments.path.config_registry import _model_config


def back_port_off_policy_model(input_path: Path, output_path: Path) -> None:
    hparams = json.loads((input_path / "hparams.json").read_text(encoding="utf-8"))
    state_dict = load_file(input_path / "model_state_dict.safetensors")

    model_config = _model_config(hparams["model"]["timm_backbone"])
    temporal_hparams = hparams["model"]["temporal_policy"]
    hidden_dim = int(temporal_hparams["n_embd"] * temporal_hparams["mlp_mult"])
    for layer in model_config.temporal_policy.temporal_summarizer.transformer.layers:
        layer.mlp.c_fc.out_features = hidden_dim
        layer.mlp.c_proj.in_features = hidden_dim

    temporal_policy = {}
    point_policy = {}
    vision = {}
    for name, value in state_dict.items():
        if name == "policy.temporal_summarizer.transformer.mask":
            continue
        if name.startswith("policy."):
            name = name.removeprefix("policy.")
            name = name.replace("_desire_encode.", "desire_encoder.net.")
            name = name.replace("_traffic_encode.", "traffic_encoder.net.")
            name = name.replace("_action_t_encode.", "action_t_encoder.net.")
            name = re.sub(r"transformer\.(\d+)\.attn\.", r"transformer.layers.\1.attention.", name)
            name = re.sub(r"transformer\.(\d+)\.mlp\.", r"transformer.layers.\1.mlp.", name)
            temporal_policy[name.replace(".layer_norm.", ".norm.")] = value
        elif name.startswith("point_policy."):
            name = name.removeprefix("point_policy.").replace(".layer_norm.", ".norm.")
            point_policy[name] = value
        elif name.startswith("vision."):
            name = name.removeprefix("vision.").replace("_en.", "encoder.", 1)
            vision[name] = value

    output_hparams = hparams | {
        "torchtitan": {
            "model": {
                "temporal_policy": model_config.temporal_policy.to_dict(),
                "point_policy": model_config.point_policy.to_dict(),
                "vision": model_config.vision.to_dict()
                | {
                    "act_layer_name": hparams["model"]["timm_kwargs"]["act_layer_name"],
                    "norm_layer_name": hparams["model"]["timm_kwargs"]["norm_layer_name"],
                    "norm_eps": 1e-3,
                    "norm_momentum": 1e-2,
                },
            },
        }
    }
    dcp.save(
        {
            "temporal_policy": temporal_policy,
            "point_policy": point_policy,
            "vision": vision,
        },
        checkpoint_id=output_path,
        no_dist=True,
    )
    (output_path / "hparams.json").write_text(json.dumps(output_hparams, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    back_port_off_policy_model(Path(sys.argv[1]), Path(sys.argv[2]))
