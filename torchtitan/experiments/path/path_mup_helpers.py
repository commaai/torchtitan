# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

BASE_WIDTH = 256


def hidden_std(fan_in: int) -> float:
    return fan_in**-0.5


def scale_dims(
    dims_base: tuple[int, ...], width: int, base_width: int = BASE_WIDTH
) -> tuple[int, ...]:
    return tuple(d * width // base_width for d in dims_base)
