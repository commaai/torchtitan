# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torchtitan.components.optimizer import register_float8_precompute_scale_hook
from torchtitan.components.quantization import Float8LinearConverter
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .model import parallelize_wan, wan_debug_config, wan_ti2v_5b_config
from .self_forcing import (
    parallelize_wan_self_forcing,
    self_forcing_config,
)
from .state_dict_adapter import WanStateDictAdapter


WAN_FLOAT8_FILTER_FQNS = [
    "text_embedding",
    "time_embedding",
    "time_projection",
    "head",
]


def model_registry(
    flavor: str,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    configs = {
        "wan_ti2v_5b": wan_ti2v_5b_config,
        "wan_debug": wan_debug_config,
    }
    try:
        config = configs[flavor]()
    except KeyError as exc:
        raise ValueError(f"unsupported Wan worldmodel flavor {flavor!r}; choose from {sorted(configs)}") from exc
    if converters is not None:
        validate_converter_order(converters)
        for converter in converters:
            converter.build().convert(config)
    return ModelSpec(
        name="wan",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_wan,
        pipelining_fn=None,
        post_optimizer_build_fn=register_float8_precompute_scale_hook,
        state_dict_adapter=WanStateDictAdapter,
    )


def self_forcing_model_registry(
    flavor: str,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    configs = {
        "wan_ti2v_5b": wan_ti2v_5b_config,
        "wan_debug": wan_debug_config,
    }
    try:
        config = self_forcing_config(configs[flavor]())
    except KeyError as exc:
        raise ValueError(f"unsupported Wan Self-Forcing flavor {flavor!r}; choose from {sorted(configs)}") from exc
    if converters is not None:
        validate_converter_order(converters)
        for converter in converters:
            converter.build().convert(config)
    return ModelSpec(
        name="wan_self_forcing",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_wan_self_forcing,
        pipelining_fn=None,
        post_optimizer_build_fn=register_float8_precompute_scale_hook,
        state_dict_adapter=WanStateDictAdapter,
    )


def _wan_blocks_only_float8(
    *,
    model_compile_enabled: bool,
    emulate: bool = False,
) -> Float8LinearConverter.Config:
    return Float8LinearConverter.Config(
        recipe_name="tensorwise",
        filter_fqns=WAN_FLOAT8_FILTER_FQNS,
        emulate=emulate,
        enable_fsdp_float8_all_gather=True,
        precompute_float8_dynamic_scale_for_fsdp=True,
        model_compile_enabled=model_compile_enabled,
    )
