# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch

from torchtitan.distributed import utils as dist_utils
from torchtitan.observability import structured_logger as sl
from torchtitan.trainer import Trainer

from .dataset import COMMA1M_IMGS_TARGET
from .loss import PathMSELoss


class Comma1MPathTrainer(Trainer):
    path_loss: PathMSELoss

    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if not isinstance(self.loss_fn, PathMSELoss):
            raise TypeError(f"Comma1MPathTrainer requires PathMSELoss, got {type(self.loss_fn).__name__}")
        self.path_loss = self.loss_fn

        base_log = self.metrics_processor.log

        def log_with_loss_components(
            step: int,
            global_avg_loss: float,
            global_max_loss: float,
            grad_norm: float,
            extra_metrics: dict[str, Any] | None = None,
        ) -> None:
            metrics = dict(extra_metrics or {})
            loss_mesh = self.parallel_dims.get_optional_mesh("loss")
            metrics.update(
                {
                    name: dist_utils.dist_sum(value, loss_mesh)
                    for name, value in self.path_loss.get_component_metrics().items()
                }
            )
            base_log(
                step,
                global_avg_loss,
                global_max_loss,
                grad_norm,
                extra_metrics=metrics,
            )

        self.metrics_processor.log = log_with_loss_components

    def train_step(self, data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]) -> None:
        self.path_loss.reset_component_metrics()
        super().train_step(data_iterator)

    @sl.log_trace_span("post_dataloading_process")
    def post_dataloading_process(
        self,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict]:
        inputs = {name: value for name, value in input_dict.items() if name != COMMA1M_IMGS_TARGET}
        targets = {
            "plan": labels,
            "imgs": input_dict[COMMA1M_IMGS_TARGET],
        }
        self.ntokens_seen += labels.shape[0]
        return inputs, targets, {}

    def close(self) -> None:
        self.dataloader.close()
        super().close()
