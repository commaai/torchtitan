# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .model_wan import WanModel


class WanStateDictAdapter(StateDictAdapter):
    """Load the official Wan safetensors whose FQNs match the local model."""

    def __init__(
        self,
        model_config: WanModel.Config,
        hf_assets_path: str | None,
    ) -> None:
        self.model_config = model_config
        self.hf_assets_path = hf_assets_path
        self.fqn_to_index_mapping = None

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        return state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        return hf_state_dict


__all__ = ["WanStateDictAdapter"]
