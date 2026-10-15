# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantized module and tensor building blocks."""

# Fork export restored after upstream removed Float8; implementation provenance:
# https://github.com/commaai/torchtitan/commit/122d0220d9576054f25bcbd05389d4c1298647d2
from .float8_legacy import Float8Linear
from .mxfp8 import MXFP8Linear
from .nvfp4 import NVFP4Linear


__all__ = [
    "Float8Linear",
    "MXFP8Linear",
    "NVFP4Linear",
]
