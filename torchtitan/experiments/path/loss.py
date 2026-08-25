# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from torchtitan.components.loss import BaseLoss

from torchtitan.config import CompileConfig
from torchtitan.tools.logging import logger


def path_mse_components(
    pred: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_plan = targets["plan"]
    plan = pred["plan"][..., : target_plan.shape[-1]]
    plan_mse = torch.nn.functional.mse_loss(
        plan.float(),
        target_plan.float().detach(),
        reduction="sum",
    )

    pred_imgs = (pred["imgs"].float() / 127.5 - 1.0).clamp(-1.0, 1.0)
    target_imgs = (targets["imgs"].permute(0, 3, 1, 2).float() / 127.5 - 1.0).clamp(-1.0, 1.0)
    imgs_mse = torch.nn.functional.mse_loss(
        pred_imgs,
        target_imgs.detach(),
        reduction="sum",
    )
    imgs_mse = imgs_mse * (target_plan.numel() / target_imgs.numel())
    return plan_mse + 100.0 * imgs_mse, plan_mse, imgs_mse


def path_mse(
    pred: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    loss, _, _ = path_mse_components(pred, targets)
    return loss


class PathMSELoss(BaseLoss):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        pass

    def __init__(
        self,
        config: Config,
        *,
        compile_config: CompileConfig | None = None,
    ) -> None:
        del config
        self.fn = path_mse
        self.component_fn: Callable[
            [dict[str, torch.Tensor], dict[str, torch.Tensor]],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = path_mse_components
        self._component_metrics: dict[str, torch.Tensor] = {}
        if compile_config is not None and compile_config.enable and "loss" in compile_config.components:
            logger.info("Compiling the loss function with torch.compile")
            self.component_fn = torch.compile(self.component_fn, backend=compile_config.backend)

    def reset_component_metrics(self) -> None:
        self._component_metrics.clear()

    def get_component_metrics(self) -> dict[str, torch.Tensor]:
        return self._component_metrics.copy()

    def __call__(
        self,
        pred: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        global_valid_tokens: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss, plan_mse, imgs_mse = self.component_fn(pred, targets)
        if global_valid_tokens is not None:
            loss = loss / global_valid_tokens
            plan_mse = plan_mse / global_valid_tokens
            imgs_mse = imgs_mse / global_valid_tokens

        for name, value in (
            ("loss_metrics/plan_mse", plan_mse),
            ("loss_metrics/imgs_mse", imgs_mse),
        ):
            value = value.detach()
            accumulated = self._component_metrics.get(name)
            self._component_metrics[name] = value if accumulated is None else accumulated + value
        return loss


class PathLoss(BaseLoss):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        pass

    def __init__(self, config: Config, *, compile_config: CompileConfig | None = None):
        from xx.training.lib.driving import DrivingLoss, DrivingMetric

        del config
        self.fn = None
        self.loss_fn = DrivingLoss()
        self.metric_fn = DrivingMetric()
        if compile_config is not None and compile_config.enable and "loss" in compile_config.components:
            logger.info("Compiling the path loss and metric functions with torch.compile")
            self.loss_fn = torch.compile(self.loss_fn, backend=compile_config.backend)
            self.metric_fn = torch.compile(self.metric_fn, backend=compile_config.backend)

    def to(self, device: torch.device) -> PathLoss:
        self.loss_fn.to(device)
        self.metric_fn.to(device)
        return self

    def __call__(
        self,
        pred: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        global_valid_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del global_valid_tokens
        pred = {k: v.float() if v.is_floating_point() else v for k, v in pred.items()}
        loss, losses = self.loss_fn(pred, targets)
        metrics = losses | self.metric_fn(pred, targets)
        return loss, {f"path/{name}": value for name, value in metrics.items() if name != "loss"}
