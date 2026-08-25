# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchtitan.observability import structured_logger as sl
from torchtitan.trainer import Trainer


class Comma1MPathTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        pass

    @sl.log_trace_span("post_dataloading_process")
    def post_dataloading_process(
        self,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict]:
        self.ntokens_seen += labels.shape[0]
        return input_dict, labels, {}

    def close(self) -> None:
        self.dataloader.close()
        super().close()
