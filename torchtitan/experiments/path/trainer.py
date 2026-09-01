# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

import torch

from torchtitan.trainer import Trainer
from xx.training.lib.torchtitan.base_trainer import BaseTrainer

from .loss import PathLoss
from .onnx_checkpoint import PathOnnxCheckpointManager
from .plan_vae import prepare_plan_latent_batch, prepare_plan_vae_batch
from .validate import PathValidator


class PathTrainer(BaseTrainer):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseTrainer.Config):
        loss: PathLoss.Config
        validator: PathValidator.Config
        checkpoint: PathOnnxCheckpointManager.Config
        training_stage: Literal["plan_vae", "policy"]
        vae_kl_warmup_steps: int = 0

        def __post_init__(self) -> None:
            Trainer.Config.__post_init__(self)
            if self.parallelism.pipeline_parallel_degree > 1:
                raise ValueError("PathTrainer does not support pipeline parallelism")
            if self.codedir:
                self.miniray = {**self.miniray, "codedir": self.codedir}
                self.validator.miniray = {
                    **self.validator.miniray,
                    "codedir": self.codedir,
                }

    def __init__(self, config: Config):
        self.training_stage = config.training_stage
        self.vae_kl_warmup_steps = config.vae_kl_warmup_steps
        super().__init__(config)
        self.loss_fn.to(self.device)
        self._plan_vae_verified = self.training_stage == "plan_vae"
        if self.training_stage == "plan_vae":
            assert self.model_parts[0].temporal_policy.plan_vae is not None
            self.model_parts[0].temporal_policy.plan_vae.mark_pretrained()

    def _prepare_batch(
        self,
        input_dict: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if self.training_stage == "plan_vae":
            kl_weight_scale = min(self.step / self.vae_kl_warmup_steps, 1.0) if self.vae_kl_warmup_steps > 0 else 1.0
            plan_vae = self.model_parts[0].temporal_policy.plan_vae
            assert plan_vae is not None
            return prepare_plan_vae_batch(
                input_dict,
                labels,
                kl_weight_scale=kl_weight_scale,
                normalization=plan_vae.normalization,
            )
        if not self._plan_vae_verified:
            plan_vae = self.model_parts[0].temporal_policy.plan_vae
            assert plan_vae is not None
            if not plan_vae.is_pretrained():
                raise RuntimeError(
                    "Policy training requires a pretrained plan VAE checkpoint; "
                    "set checkpoint.initial_load_path to a completed plan_vae checkpoint"
                )
            self._plan_vae_verified = True
        if self.model_parts[0].config.plan_loss in ("latent_mse", "latent_nll"):
            plan_vae = self.model_parts[0].temporal_policy.plan_vae
            assert plan_vae is not None
            labels = prepare_plan_latent_batch(
                labels,
                plan_vae.encode_mean,
                normalization=plan_vae.normalization,
            )
        return input_dict, labels

    def batch_generator(
        self,
        data_iterable: Iterable[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]],
    ) -> Iterator[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]:
        for input_dict, targets in super().batch_generator(data_iterable):
            if self.training_stage == "plan_vae":
                yield {}, {"plan": targets["plan"]}
            else:
                yield input_dict, targets

    def sample_count(
        self,
        batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
    ) -> int:
        input_dict, targets = batch
        tensors = input_dict if input_dict else targets
        return next(iter(tensors.values())).shape[0]

    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
        local_samples: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        input_dict, labels = self._prepare_batch(input_dict, labels)
        loss, metrics = super().forward_backward_step(
            input_dict=input_dict,
            labels=labels,
            local_samples=local_samples,
        )
        return loss, {f"path/{name}": value for name, value in metrics.items() if name != "loss"}

    # TODO do we really need to close those?
    def close(self) -> None:
        self.dataloader.close()
        if self.config.validator.enable:
            self.validator.close()
        super().close()

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.pop("unique_id_counter", None)
        state["unique_segment_counter"] = self.unique_id_counter.state_dict()
        validator_unique_segment_counter = getattr(getattr(self, "validator", None), "unique_segment_counter", None)
        if validator_unique_segment_counter is not None:
            state["validation_unique_segment_counter"] = validator_unique_segment_counter.state_dict()
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        if "unique_segment_counter" in state_dict:
            self.unique_id_counter.load_state_dict(state_dict["unique_segment_counter"])
        validator_unique_segment_counter = getattr(getattr(self, "validator", None), "unique_segment_counter", None)
        if validator_unique_segment_counter is not None and "validation_unique_segment_counter" in state_dict:
            validator_unique_segment_counter.load_state_dict(state_dict["validation_unique_segment_counter"])
