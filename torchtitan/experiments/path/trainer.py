# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from xx.training.lib.trainer import BaseTrainer

from .loss import PathLoss
from .onnx_checkpoint import PathOnnxCheckpointManager
from .validate import PathValidator, segment_names_and_fidxs_from_info


class PathTrainer(BaseTrainer):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseTrainer.Config):
        loss: PathLoss.Config
        validator: PathValidator.Config
        checkpoint: PathOnnxCheckpointManager.Config
        miniray: dict[str, Any] = field(default_factory=dict)
        fps: int

        def __post_init__(self) -> None:
            BaseTrainer.Config.__post_init__(self)
            if self.codedir:
                self.miniray = {**self.miniray, "codedir": self.codedir}
                self.validator.miniray = {
                    **self.validator.miniray,
                    "codedir": self.codedir,
                }

    def __init__(self, config: Config):
        super().__init__(config)
        self.loss_fn.to(self.device)

    def unique_ids(self, batch: Any) -> Iterable[str]:
        inputs, _ = batch
        info = inputs.get("info")
        if info is None:
            return ()
        return (name for name, _ in segment_names_and_fidxs_from_info(info))

    def close(self) -> None:
        self.dataloader.close()
        if self.config.validator.enable:
            self.validator.close()
        super().close()

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        validator_unique_segment_counter = getattr(getattr(self, "validator", None), "unique_segment_counter", None)
        if validator_unique_segment_counter is not None:
            state["validation_unique_segment_counter"] = validator_unique_segment_counter.state_dict()
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        validator_unique_segment_counter = getattr(getattr(self, "validator", None), "unique_segment_counter", None)
        if validator_unique_segment_counter is not None and "validation_unique_segment_counter" in state_dict:
            validator_unique_segment_counter.load_state_dict(state_dict["validation_unique_segment_counter"])
