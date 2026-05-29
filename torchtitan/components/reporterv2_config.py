# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os


DEFAULT_REPORTERV2_HOST = "http://data-gen.comma.life:3080/reporterv2"


def get_reporterv2_host() -> str:
    host = os.getenv("REPORTERV2_HOST")
    if not host:
        raise ValueError("REPORTERV2_HOST must be set when using ReporterV2.")
    return host


def get_reporterv2_training_id() -> str:
    training_id = os.getenv("REPORTERV2_TRAINING_ID")
    if not training_id:
        raise ValueError("REPORTERV2_TRAINING_ID must be set when using ReporterV2.")
    return training_id
