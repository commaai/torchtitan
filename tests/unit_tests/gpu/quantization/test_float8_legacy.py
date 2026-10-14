# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace
from unittest import mock

import pytest
import torch

pytest.importorskip("torchao.float8")
from torchao.float8 import Float8LinearConfig
from torchao.float8.float8_linear import Float8Linear as TorchAOFloat8Linear

from torchtitan.quantization.float8_legacy import Float8Linear
from torchtitan.tools.utils import has_cuda_capability


@pytest.mark.skipif(not has_cuda_capability(8, 9), reason="Requires SM89+ FP8 GEMMs")
@pytest.mark.parametrize("bias", [False, True])
def test_tensorwise_fp8_matches_pre_rebase_torchao_compute(bias):
    torch.manual_seed(42)
    config = replace(
        Float8LinearConfig.from_recipe_name("tensorwise"),
        enable_fsdp_float8_all_gather=True,
    )
    with torch.device("cuda"):
        module = (
            Float8Linear.Config(
                in_features=128, out_features=64, bias=bias, _torchao_config=config
            )
            .build()
            .to(torch.bfloat16)
        )
        reference = TorchAOFloat8Linear(128, 64, bias=bias, config=config).to(
            torch.bfloat16
        )
        reference.load_state_dict(module.state_dict())
        input = torch.randn(2, 16, 128, dtype=torch.bfloat16, requires_grad=True)
        reference_input = input.detach().clone().requires_grad_()
        with mock.patch("torch._scaled_mm", wraps=torch._scaled_mm) as scaled_mm:
            output = module(input)
            output.sum().backward()
            assert scaled_mm.call_count >= 3
        expected = reference(reference_input)
        expected.sum().backward()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(input.grad, reference_input.grad, rtol=0, atol=0)
    for param, reference_param in zip(module.parameters(), reference.parameters()):
        torch.testing.assert_close(param.grad, reference_param.grad, rtol=0, atol=0)
