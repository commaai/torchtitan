# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchAO-backed FP8 compute retained for tensorwise training references."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from torchtitan.models.common.linear import Linear
from torchtitan.observability import structured_logger as sl


_float8_import_error: ImportError | None = None
try:
    from torchao.float8.float8_linear import (
        Float8Linear as TorchAOFloat8Linear,
        matmul_with_hp_or_float8_args,
    )

    class Float8Linear(Linear, TorchAOFloat8Linear):
        """Keep TorchAO compute inside upstream Linear's remat/TP boundaries."""

        @dataclass(kw_only=True, slots=True)
        class Config(Linear.Config):
            _torchao_config: Any = None
            inplace_wgrad_accum: bool = False
            precompute_float8_dynamic_scale_for_fsdp: bool = False

            def __post_init__(self) -> None:
                if self.num_linears != 1:
                    raise ValueError(
                        "Legacy Float8Linear does not support stacked projections."
                    )
                if self.inplace_wgrad_accum:
                    raise ValueError(
                        "Legacy Float8Linear does not support inplace_wgrad_accum."
                    )

        def __init__(self, config: Config):
            # Calling TorchAO directly skips Linear.__init__, whose signature
            # accepts a TorchTitan config rather than nn.Linear arguments.
            TorchAOFloat8Linear.__init__(
                self,
                config.in_features,
                config.out_features,
                bias=config.bias,
                config=config._torchao_config,
            )
            self.num_linears = config.num_linears
            self.inplace_wgrad_accum = config.inplace_wgrad_accum
            self.precompute_float8_dynamic_scale_for_fsdp = (
                config.precompute_float8_dynamic_scale_for_fsdp
            )

        def _linear(
            self,
            input: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor | None,
        ) -> torch.Tensor:
            # Match TorchAOFloat8Linear.forward, but use the explicit operands
            # supplied by Linear's remat region and tensor-parallel boundaries.
            if torch.is_autocast_enabled():
                input = input.to(torch.get_autocast_dtype("cuda"))
            output = matmul_with_hp_or_float8_args.apply(
                input, weight.t(), self.linear_mm_config, self.config
            )
            if bias is not None:
                output = output + bias.to(output.dtype)
            return output

except ImportError as error:
    _float8_import_error = error
    Float8Linear = None


def register_float8_precompute_scale_hook(
    optimizers: Optimizer,
    model_parts: list[nn.Module],
) -> None:
    """Precompute opted-in TorchAO weight scales after all optimizer updates."""
    models_to_precompute = [
        model
        for model in model_parts
        if any(
            getattr(module, "precompute_float8_dynamic_scale_for_fsdp", False)
            for module in model.modules()
        )
    ]
    if not models_to_precompute:
        return

    from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

    def _precompute_scales_hook(
        _optimizer: Optimizer,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        with sl.log_trace_span("float8_precompute_scales"):
            for model in models_to_precompute:
                precompute_float8_dynamic_scale_for_fsdp(model)

    optimizers.register_step_post_hook(_precompute_scales_hook)
