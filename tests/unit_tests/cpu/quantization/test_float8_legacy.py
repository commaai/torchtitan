# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from unittest import mock

import pytest
import torch
import torch_remat as remat

pytest.importorskip("torchao.float8")
from torchao.float8 import Float8LinearConfig
from torchao.float8.config import ScalingGranularity, ScalingType
from torchao.float8.float8_linear import Float8Linear as TorchAOFloat8Linear

from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.models.base.module import Module
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.linear import (
    ColumnParallelLinear,
    Linear,
    RowParallelLinear,
)
from torchtitan.quantization.float8_legacy import Float8Linear
from torchtitan.quantization.utils import get_quantized_linear, has_quantization


@pytest.mark.parametrize("owner", [Linear, ColumnParallelLinear, RowParallelLinear])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("save_regions", [None, (), ("*",)])
def test_emulated_compute_matches_torchao(owner, bias, save_regions):
    torch.manual_seed(42)
    config = Float8LinearConfig(emulate=True)
    module = (
        get_quantized_linear(Float8Linear, owner)
        .Config(in_features=32, out_features=16, bias=bias, _torchao_config=config)
        .build()
    )
    reference = TorchAOFloat8Linear(32, 16, bias=bias, config=config)
    reference.load_state_dict(module.state_dict())
    input = torch.randn(2, 16, 32, requires_grad=True)
    reference_input = input.detach().clone().requires_grad_()
    if save_regions is None:
        output = module(input)
    else:
        module.configure_remat_regions(save_regions)
        output = remat.checkpoint()(module)(input)
    expected = reference(reference_input)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    grad = torch.randn_like(output)
    output.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(input.grad, reference_input.grad, rtol=0, atol=0)
    for param, reference_param in zip(module.parameters(), reference.parameters()):
        torch.testing.assert_close(param.grad, reference_param.grad, rtol=0, atol=0)


def test_meta_initialization_and_quantization_detection():
    config = Float8Linear.Config(
        in_features=16,
        out_features=16,
        _torchao_config=Float8LinearConfig(emulate=True),
        param_init={"weight": torch.nn.init.ones_},
        sharding_config=ShardingConfig(),
    )
    with torch.device("meta"):
        module = config.build()
    module.to_empty(device="cpu")
    module.init_states()
    assert module._sharding_config is config.sharding_config
    assert has_quantization(config)
    torch.testing.assert_close(module.weight, torch.ones_like(module.weight))


def test_unsupported_compute_options_rejected():
    for kwargs in ({"num_linears": 3}, {"inplace_wgrad_accum": True}):
        with pytest.raises(ValueError, match="does not support"):
            Float8Linear.Config(in_features=16, out_features=16, **kwargs)


def test_optimizer_hook_runs_once_after_all_updates():
    models = [torch.nn.Linear(16, 16) for _ in range(2)]
    models[0].precompute_float8_dynamic_scale_for_fsdp = True
    initial = [model.weight.detach().clone() for model in models]
    for model in models:
        for param in model.parameters():
            param.grad = torch.ones_like(param)

    def precompute(model):
        assert model is models[0]
        for part, before in zip(models, initial):
            assert not torch.equal(part.weight, before)

    with mock.patch(
        "torchao.float8.precompute_float8_dynamic_scale_for_fsdp",
        side_effect=precompute,
    ) as hook:
        optimizers = OptimizersContainer.Config(
            optimizers=[AdamW.Config(pattern=".*", fused=False)]
        ).build(model_parts=models)
        optimizers.step()
        hook.assert_called_once_with(models[0])


def test_optimizer_without_opt_in_does_not_precompute():
    with mock.patch("torchao.float8.precompute_float8_dynamic_scale_for_fsdp") as hook:
        optimizers = OptimizersContainer.Config(
            optimizers=[AdamW.Config(pattern=".*", fused=False)]
        ).build(model_parts=[torch.nn.Linear(16, 16)])
        optimizers.step()
        hook.assert_not_called()


class _LinearTree(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        linears: list[Linear.Config]


@pytest.mark.parametrize("owner", [Linear, ColumnParallelLinear, RowParallelLinear])
def test_converter_root_preserves_fields_and_tensorwise_recipe(owner):
    from torchtitan.config.transform import Float8LinearConverter

    source = owner.Config(
        in_features=16,
        out_features=32,
        bias=True,
        param_init={"weight": torch.nn.init.ones_},
        sharding_config=ShardingConfig(),
    )
    converter = Float8LinearConverter.Config(
        emulate=True,
        enable_fsdp_float8_all_gather=True,
        precompute_float8_dynamic_scale_for_fsdp=True,
    ).build()
    converted = converter.convert(source)
    assert isinstance(converted, Float8Linear.Config)
    assert issubclass(converted._owner, owner)
    assert converted.param_init is source.param_init
    assert converted.sharding_config is source.sharding_config
    assert converted.bias
    assert not converted.inplace_wgrad_accum
    assert converted.precompute_float8_dynamic_scale_for_fsdp
    assert converted._torchao_config.enable_fsdp_float8_all_gather
    for cast in (
        converted._torchao_config.cast_config_input,
        converted._torchao_config.cast_config_weight,
        converted._torchao_config.cast_config_grad_output,
    ):
        assert cast.scaling_type is ScalingType.DYNAMIC
        assert cast.scaling_granularity is ScalingGranularity.TENSORWISE


def test_converter_filters_nested_linears():
    from torchtitan.config.transform import Float8LinearConverter

    source = _LinearTree.Config(
        linears=[
            Linear.Config(in_features=16, out_features=16),
            Linear.Config(in_features=16, out_features=16),
            Linear.Config(in_features=15, out_features=16),
        ]
    )
    original = list(source.linears)
    converted = (
        Float8LinearConverter.Config(emulate=True, filter_fqns=["linears.1"])
        .build()
        .convert(source)
    )
    assert converted is source
    assert isinstance(source.linears[0], Float8Linear.Config)
    assert source.linears[1:] == original[1:]
