# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Reconstruction losses for the Wan VAE experiment."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch

from torchtitan.components.loss import BaseLoss
from torchtitan.config import CompileConfig


def _normalize_target(labels: torch.Tensor) -> torch.Tensor:
    if labels.is_floating_point():
        return labels.float()
    return labels.float().div(127.5).sub(1.0)


def pixel_reconstruction_loss(
    pred: torch.Tensor,
    labels: torch.Tensor,
    *,
    mae_weight: float,
    mse_weight: float,
) -> torch.Tensor:
    target = _normalize_target(labels).detach()
    error = pred.float() - target
    return mae_weight * error.abs().mean() + mse_weight * error.square().mean()


class WanReconstructionLoss(BaseLoss):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        mae_weight: float = 1.0
        mse_weight: float = 1.0
        lpips_weight: float = 0.0
        lpips_frame_stride: int = 4

        def __post_init__(self) -> None:
            if min(self.mae_weight, self.mse_weight, self.lpips_weight) < 0:
                raise ValueError("loss weights must be non-negative")
            if self.mae_weight + self.mse_weight + self.lpips_weight == 0:
                raise ValueError("at least one reconstruction loss must be enabled")
            if self.lpips_frame_stride <= 0:
                raise ValueError("lpips_frame_stride must be positive")

    def __init__(
        self,
        config: Config,
        *,
        compile_config: CompileConfig | None = None,
    ) -> None:
        self.config = config
        self.fn = partial(
            pixel_reconstruction_loss,
            mae_weight=config.mae_weight,
            mse_weight=config.mse_weight,
        )
        self.lpips = None
        if config.lpips_weight:
            from xx.training.lib.lpips import LPIPS

            with torch.device("meta"):
                self.lpips = LPIPS()
            self.lpips.load_from_pretrained(strict=True, assign=True)
            self.lpips.eval().requires_grad_(False)
        elif compile_config is not None:
            self._maybe_compile(compile_config)

    @staticmethod
    def _flatten_video_frames(video: torch.Tensor, *, frame_stride: int = 1) -> torch.Tensor:
        if video.ndim == 6:
            # [B, V, C, T, H, W] -> [B*V*T, C, H, W]
            video = video[:, :, :, ::frame_stride]
            return video.permute(0, 1, 3, 2, 4, 5).flatten(0, 2)
        if video.ndim == 5:
            video = video[:, :, ::frame_stride]
            return video.permute(0, 2, 1, 3, 4).flatten(0, 1)
        raise ValueError(f"expected a 5-D or 6-D video tensor, got {video.shape}")

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: float | None = None,
    ) -> torch.Tensor:
        del global_valid_tokens  # losses below already use mean reductions
        loss = self.fn(pred, labels)
        if self.lpips is None:
            return loss

        target = _normalize_target(labels).detach()
        pred_frames = self._flatten_video_frames(
            pred,
            frame_stride=self.config.lpips_frame_stride,
        )
        target_frames = self._flatten_video_frames(
            target,
            frame_stride=self.config.lpips_frame_stride,
        )
        first_parameter = next(self.lpips.parameters())
        if first_parameter.device != pred.device:
            self.lpips.to(device=pred.device)
        lpips = self.lpips(pred_frames.float(), target_frames.float()).mean()
        return loss + self.config.lpips_weight * lpips
