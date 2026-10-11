# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn

from torchtitan.components.checkpoint import OPTIMIZER
from torchtitan.tools.logging import logger
from xx.training.lib.checkpoint import MODELS_HOST
from xx.training.lib.torchtitan import dav
from xx.training.lib.torchtitan.onnx_checkpoint import _check_onnx_size, _rank, OnnxCheckpointManager

from .model import ACTION_HEAD_NAME, RLDrivingModel


class _TargetActorOnnxModel(nn.Module):
    def __init__(self, model: RLDrivingModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.model.target_forward(inputs)


class RLDrivingOnnxCheckpointManager(OnnxCheckpointManager):
    @dataclass(kw_only=True, slots=True)
    class Config(OnnxCheckpointManager.Config):
        checkpoint_base_folder: str = ""

        enable_checkpoint_dav: bool = False

        checkpoint_dav_interval: int = 1

    def __init__(self, config: Config, **kwargs) -> None:
        if config.checkpoint_base_folder:
            kwargs["base_folder"] = config.checkpoint_base_folder
        super().__init__(config, **kwargs)
        if self.enable:
            self.states.pop(OPTIMIZER)

        self.checkpoint_dav_url: str | None = None
        self.checkpoint_dav_id: str | None = None
        self.checkpoint_dav_interval = 0
        if config.enable and not config.load_only and config.enable_checkpoint_dav:
            cluster = os.environ["SLURM_CLUSTER_NAME"]
            job = os.environ["SLURM_JOB_ID"]
            training_id = os.environ["REPORTERV2_TRAINING_ID"]
            self.checkpoint_dav_interval = config.checkpoint_dav_interval
            self.checkpoint_dav_id = f"dav@{cluster}:{job}:{training_id}/-1"
            if _rank() == 0:
                self.checkpoint_dav_url = f"{MODELS_HOST}/dav/{cluster}/{job}/{training_id}/"
                training_args = self.states["train_state"].config.to_dict()
                training_args["training_id"] = training_id
                training_args.setdefault("trainer", training_args["model_spec"]["name"])
                metadata = {
                    "training_args": training_args,
                    "last_epoch": -1,
                    "checkpoint_keys": [self.onnx_file],
                }
                dav.write_file(self.checkpoint_dav_url + "metaexperiment.json", json.dumps(metadata, default=str).encode())
            logger.info("Rollout checkpoint: %s", self.checkpoint_dav_id)

    def save(self, curr_step: int, last_step: bool = False) -> bool:
        saved = super().save(curr_step, last_step)
        self.checkpoint_dav_save(curr_step, force=last_step)
        return saved

    @torch.no_grad()
    def checkpoint_dav_save(self, curr_step: int, *, force: bool = False) -> bool:
        if not self.checkpoint_dav_interval:
            return False
        if (not force and curr_step % self.checkpoint_dav_interval):
            return False
        model = self._gather_cpu_model()
        if model is not None and self.checkpoint_dav_url is not None:
            path = self.checkpoint_dav_url + f"checkpoint/-1/{self.onnx_file}"
            data = self._export_onnx(model)
            _check_onnx_size(path, len(data))
            dav.write_file(path, data, mtime=curr_step)
        self._wait_for_rank0()
        return True

    def close(self):
        super().close()
        url = getattr(self, "checkpoint_dav_url", None)
        self.checkpoint_dav_url = None
        if url is not None:
            try:
                dav.delete(url, missing_ok=True)
            except OSError:
                logger.warning("Could not delete DAV checkpoint %s; Slurm epilog will retry", url, exc_info=True)

    def _export_onnx(self, model: nn.Module) -> bytes:
        model = cast(RLDrivingModel, model)
        inputs = dict(zip(self.input_names, self._build_onnx_inputs()))
        return self._export_one(
            _TargetActorOnnxModel(model).eval(), inputs, output_names=[ACTION_HEAD_NAME],
        )
