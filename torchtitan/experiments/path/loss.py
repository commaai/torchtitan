# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from torchtitan.components.loss import BaseLoss

from torchtitan.config import CompileConfig
from torchtitan.tools.logging import logger

from .model_constants import IDX_N, PLAN_WIDTH, T_IDXS
from .plan_vae import (
    PLAN_CHANNEL_STD,
    PLAN_LATENT,
    PLAN_LATENT_LOGVAR,
    PLAN_LATENT_MASK,
    PLAN_LATENT_TARGET,
    PLAN_VAE_KL_WEIGHT_SCALE,
    PLAN_VAE_LOGVAR,
    PLAN_VAE_MASK,
    PLAN_VAE_MEAN,
    PLAN_VAE_PRIOR_RECONSTRUCTION,
    PLAN_VAE_RECONSTRUCTION,
    PLAN_VAE_SAMPLED_RECONSTRUCTION,
    PLAN_VAE_TARGET,
    PlanNormalization,
    unnormalize_plan,
)


def _masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mse = F.mse_loss(pred.float(), target.float(), reduction="none")
    flat_mse = mse.flatten(start_dim=1)
    flat_mask = mask.flatten(start_dim=1).float()
    return (flat_mse * flat_mask).sum(dim=-1) / flat_mask.sum(dim=-1).clamp_min(1.0)


def _masked_gaussian_nll(
    mean: torch.Tensor,
    target: torch.Tensor,
    log_var: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    log_var = log_var.float().clamp(min=-15.0)
    nll = 0.5 * (log_var + (target - mean).square() * torch.exp(-log_var))
    flat_nll = nll.flatten(start_dim=1)
    flat_mask = mask.flatten(start_dim=1).float()
    return (flat_nll * flat_mask).sum(dim=-1) / flat_mask.sum(dim=-1).clamp_min(1.0)


def _masked_channel_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    channel_weights: torch.Tensor,
) -> torch.Tensor:
    mse = F.mse_loss(pred.float(), target.float(), reduction="none").unflatten(-1, (IDX_N, PLAN_WIDTH))
    shaped_mask = mask.unflatten(-1, (IDX_N, PLAN_WIDTH)).float()
    weights = channel_weights.view(*([1] * (mse.ndim - 1)), PLAN_WIDTH)
    weighted_mask = shaped_mask * weights
    return _masked_mean(mse, weighted_mask)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    flat_value = value.flatten(start_dim=1)
    flat_mask = mask.flatten(start_dim=1).float()
    return (flat_value * flat_mask).sum(dim=-1) / flat_mask.sum(dim=-1).clamp_min(1.0)


def _physical_reconstruction_losses(
    normalized_prediction_BTP: torch.Tensor,
    normalized_target_BTP: torch.Tensor,
    plan_mask_BTP: torch.Tensor,
    channel_weights: torch.Tensor,
    *,
    normalization: PlanNormalization,
    smooth_l1_beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction_BTHW = unnormalize_plan(
        normalized_prediction_BTP,
        normalization=normalization,
    ).unflatten(-1, (IDX_N, PLAN_WIDTH))
    target_BTHW = unnormalize_plan(
        normalized_target_BTP,
        normalization=normalization,
    ).unflatten(-1, (IDX_N, PLAN_WIDTH))
    mask_BTHW = plan_mask_BTP.unflatten(-1, (IDX_N, PLAN_WIDTH))
    weights = channel_weights.view(*([1] * (prediction_BTHW.ndim - 1)), PLAN_WIDTH)
    weighted_mask = mask_BTHW * weights
    absolute_error = (prediction_BTHW - target_BTHW).abs()
    smooth_l1 = F.smooth_l1_loss(
        prediction_BTHW,
        target_BTHW,
        reduction="none",
        beta=smooth_l1_beta,
    )
    return (
        _masked_mean(absolute_error, mask_BTHW),
        _masked_mean(absolute_error, weighted_mask),
        _masked_mean(smooth_l1, mask_BTHW),
        _masked_mean(smooth_l1, weighted_mask),
    )


def _plan_consistency_loss(
    normalized_plan_BTP: torch.Tensor,
    plan_mask_BTP: torch.Tensor,
    normalized_target_BTP: torch.Tensor | None = None,
    *,
    normalization: PlanNormalization = "pooled",
) -> torch.Tensor:
    plan_BTHW = unnormalize_plan(
        normalized_plan_BTP,
        normalization=normalization,
    ).unflatten(-1, (IDX_N, PLAN_WIDTH))
    target_BTHW = (
        unnormalize_plan(
            normalized_target_BTP,
            normalization=normalization,
        ).unflatten(-1, (IDX_N, PLAN_WIDTH))
        if normalized_target_BTP is not None
        else None
    )
    mask_BTHW = plan_mask_BTP.unflatten(-1, (IDX_N, PLAN_WIDTH))
    dt = plan_BTHW.new_tensor(T_IDXS[1:] - T_IDXS[:-1]).view(1, 1, IDX_N - 1, 1)
    channel_std = plan_BTHW.new_tensor(PLAN_CHANNEL_STD)

    def integrated_derivative_loss(value: slice, derivative: slice) -> torch.Tensor:
        delta = plan_BTHW[..., 1:, value] - plan_BTHW[..., :-1, value]
        midpoint_derivative = 0.5 * (plan_BTHW[..., 1:, derivative] + plan_BTHW[..., :-1, derivative])
        residual = delta - midpoint_derivative * dt
        if target_BTHW is not None:
            target_delta = target_BTHW[..., 1:, value] - target_BTHW[..., :-1, value]
            target_midpoint_derivative = 0.5 * (target_BTHW[..., 1:, derivative] + target_BTHW[..., :-1, derivative])
            residual = residual - (target_delta - target_midpoint_derivative * dt)
        scale = channel_std[value].view(1, 1, 1, -1) / (IDX_N - 1)
        scale = scale + channel_std[derivative].view(1, 1, 1, -1) * dt
        valid = (
            mask_BTHW[..., 1:, value]
            & mask_BTHW[..., :-1, value]
            & mask_BTHW[..., 1:, derivative]
            & mask_BTHW[..., :-1, derivative]
        )
        return _masked_mean((residual / scale).square(), valid)

    return torch.stack(
        (
            integrated_derivative_loss(slice(0, 3), slice(3, 6)),
            integrated_derivative_loss(slice(3, 6), slice(6, 9)),
            integrated_derivative_loss(slice(9, 12), slice(12, 15)),
        )
    ).mean(dim=0)


def _vae_position_l2(
    normalized_prediction_BTP: torch.Tensor,
    normalized_target_BTP: torch.Tensor,
    plan_mask_BTP: torch.Tensor,
    *,
    normalization: PlanNormalization = "pooled",
) -> torch.Tensor:
    prediction_BTHW = unnormalize_plan(
        normalized_prediction_BTP,
        normalization=normalization,
    ).unflatten(-1, (IDX_N, PLAN_WIDTH))
    target_BTHW = unnormalize_plan(
        normalized_target_BTP,
        normalization=normalization,
    ).unflatten(-1, (IDX_N, PLAN_WIDTH))
    valid_BTH = plan_mask_BTP.unflatten(-1, (IDX_N, PLAN_WIDTH))[..., :3].all(dim=-1)
    return _masked_mean(torch.linalg.vector_norm(prediction_BTHW[..., :3] - target_BTHW[..., :3], dim=-1), valid_BTH)


def _plan_backward_loss(
    normalized_plan_BTP: torch.Tensor,
    plan_mask_BTP: torch.Tensor,
    *,
    normalization: PlanNormalization = "pooled",
) -> torch.Tensor:
    plan_BTHW = unnormalize_plan(
        normalized_plan_BTP,
        normalization=normalization,
    ).unflatten(-1, (IDX_N, PLAN_WIDTH))
    mask_BTHW = plan_mask_BTP.unflatten(-1, (IDX_N, PLAN_WIDTH))
    delta_x = plan_BTHW[..., 1:, 0] - plan_BTHW[..., :-1, 0]
    valid = mask_BTHW[..., 1:, 0] & mask_BTHW[..., :-1, 0]
    scale = plan_BTHW.new_tensor(PLAN_CHANNEL_STD[0] / (IDX_N - 1))
    return _masked_mean((F.relu(-delta_x) / scale).square(), valid)


def _mean_nonbatch(value: torch.Tensor) -> torch.Tensor:
    return value.flatten(start_dim=1).mean(dim=-1)


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
        plan_loss_weight: float = 5.0
        vae_reconstruction_weight: float = 1.0
        vae_sampled_reconstruction_weight: float = 0.25
        vae_kl_weight: float = 1e-2
        vae_kl_free_bits: float = 0.05
        vae_consistency_weight: float = 0.1
        vae_backward_weight: float = 0.05
        vae_prior_consistency_weight: float = 0.01
        vae_prior_backward_weight: float = 0.05
        vae_position_weight: float = 10.0
        vae_reconstruction_loss: Literal["normalized_mse", "physical_smooth_l1"] = "normalized_mse"
        vae_smooth_l1_beta: float = 1e-3
        plan_normalization: PlanNormalization = "pooled"

    def __init__(self, config: Config, *, compile_config: CompileConfig | None = None):
        from xx.training.lib.driving import DrivingLoss, DrivingMetric

        self.plan_loss_weight = config.plan_loss_weight
        self.vae_reconstruction_weight = config.vae_reconstruction_weight
        self.vae_sampled_reconstruction_weight = config.vae_sampled_reconstruction_weight
        self.vae_kl_weight = config.vae_kl_weight
        self.vae_kl_free_bits = config.vae_kl_free_bits
        self.vae_consistency_weight = config.vae_consistency_weight
        self.vae_backward_weight = config.vae_backward_weight
        self.vae_prior_consistency_weight = config.vae_prior_consistency_weight
        self.vae_prior_backward_weight = config.vae_prior_backward_weight
        self.vae_position_weight = config.vae_position_weight
        self.vae_reconstruction_loss = config.vae_reconstruction_loss
        self.vae_smooth_l1_beta = config.vae_smooth_l1_beta
        self.plan_normalization = config.plan_normalization
        if self.vae_reconstruction_loss not in ("normalized_mse", "physical_smooth_l1"):
            raise ValueError(f"Unknown VAE reconstruction loss: {self.vae_reconstruction_loss}")
        if self.vae_smooth_l1_beta < 0:
            raise ValueError("vae_smooth_l1_beta must be non-negative")
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
        if PLAN_LATENT_TARGET in targets:
            # latent plan loss: the head predicts the frozen-encoder latent of the GT plan
            if PLAN_LATENT_LOGVAR in pred:
                latent_loss = _masked_gaussian_nll(
                    pred[PLAN_LATENT],
                    targets[PLAN_LATENT_TARGET],
                    pred[PLAN_LATENT_LOGVAR],
                    targets[PLAN_LATENT_MASK],
                )
                latent_metric_name = "plan_latent_nll"
            else:
                latent_loss = _masked_mse(
                    pred[PLAN_LATENT],
                    targets[PLAN_LATENT_TARGET],
                    targets[PLAN_LATENT_MASK],
                )
                latent_metric_name = "plan_latent_mse"

            plan_mask = torch.isfinite(targets["plan"])
            plan_mse = _masked_mse(
                pred["plan"],
                torch.nan_to_num(targets["plan"], nan=0.0),
                plan_mask,
            )
            base_pred = {
                name: value for name, value in pred.items() if name not in (PLAN_LATENT, PLAN_LATENT_LOGVAR, "plan")
            }
            base_targets = {name: value for name, value in targets.items() if name != "plan"}
            loss, losses = self.loss_fn(base_pred, base_targets)
            metrics = self.metric_fn(base_pred, base_targets)
            loss = loss + self.plan_loss_weight * latent_loss
            return loss, {f"path/{name}": value for name, value in (losses | metrics).items() if name != "loss"} | {
                latent_metric_name: latent_loss.detach(),
                "plan_mse": plan_mse.detach(),
            }
        if PLAN_VAE_RECONSTRUCTION in pred:
            reconstruction = _masked_mse(
                pred[PLAN_VAE_RECONSTRUCTION],
                targets[PLAN_VAE_TARGET],
                targets[PLAN_VAE_MASK],
            )
            channel_weights = pred[PLAN_VAE_RECONSTRUCTION].new_ones(PLAN_WIDTH)
            channel_weights[:3] = self.vae_position_weight
            weighted_reconstruction = _masked_channel_weighted_mse(
                pred[PLAN_VAE_RECONSTRUCTION],
                targets[PLAN_VAE_TARGET],
                targets[PLAN_VAE_MASK],
                channel_weights,
            )
            (
                physical_mae,
                weighted_physical_mae,
                physical_smooth_l1,
                weighted_physical_smooth_l1,
            ) = _physical_reconstruction_losses(
                pred[PLAN_VAE_RECONSTRUCTION],
                targets[PLAN_VAE_TARGET],
                targets[PLAN_VAE_MASK],
                channel_weights,
                normalization=self.plan_normalization,
                smooth_l1_beta=self.vae_smooth_l1_beta,
            )
            reconstruction_objective = (
                weighted_physical_smooth_l1
                if self.vae_reconstruction_loss == "physical_smooth_l1"
                else weighted_reconstruction
            )
            sampled_reconstruction = torch.zeros_like(reconstruction)
            sampled_weighted_reconstruction = torch.zeros_like(weighted_reconstruction)
            sampled_objective = torch.zeros_like(reconstruction_objective)
            if PLAN_VAE_SAMPLED_RECONSTRUCTION in pred:
                sampled_reconstruction = _masked_mse(
                    pred[PLAN_VAE_SAMPLED_RECONSTRUCTION],
                    targets[PLAN_VAE_TARGET],
                    targets[PLAN_VAE_MASK],
                )
                sampled_weighted_reconstruction = _masked_channel_weighted_mse(
                    pred[PLAN_VAE_SAMPLED_RECONSTRUCTION],
                    targets[PLAN_VAE_TARGET],
                    targets[PLAN_VAE_MASK],
                    channel_weights,
                )
                if self.vae_reconstruction_loss == "physical_smooth_l1":
                    _, _, _, sampled_objective = _physical_reconstruction_losses(
                        pred[PLAN_VAE_SAMPLED_RECONSTRUCTION],
                        targets[PLAN_VAE_TARGET],
                        targets[PLAN_VAE_MASK],
                        channel_weights,
                        normalization=self.plan_normalization,
                        smooth_l1_beta=self.vae_smooth_l1_beta,
                    )
                else:
                    sampled_objective = sampled_weighted_reconstruction
            mean = pred[PLAN_VAE_MEAN]
            logvar = pred[PLAN_VAE_LOGVAR]
            kl_per_dimension = -0.5 * (1.0 + logvar - mean.square() - logvar.exp())
            kl_nats = _mean_nonbatch(kl_per_dimension.sum(dim=-1))
            kl_penalty = _mean_nonbatch(F.relu(kl_per_dimension - self.vae_kl_free_bits))
            kl_weight_scale = targets.get(PLAN_VAE_KL_WEIGHT_SCALE)
            if kl_weight_scale is None:
                kl_weight_scale = torch.ones_like(kl_nats)
            consistency = torch.zeros_like(reconstruction)
            if self.vae_consistency_weight:
                consistency = _plan_consistency_loss(
                    pred[PLAN_VAE_RECONSTRUCTION],
                    targets[PLAN_VAE_MASK],
                    targets[PLAN_VAE_TARGET],
                    normalization=self.plan_normalization,
                )
            backward = torch.zeros_like(reconstruction)
            if self.vae_backward_weight:
                backward = _plan_backward_loss(
                    pred[PLAN_VAE_RECONSTRUCTION],
                    targets[PLAN_VAE_MASK],
                    normalization=self.plan_normalization,
                )
            prior_consistency = torch.zeros_like(consistency)
            prior_backward = torch.zeros_like(backward)
            if PLAN_VAE_PRIOR_RECONSTRUCTION in pred:
                prior_mask = torch.ones_like(pred[PLAN_VAE_PRIOR_RECONSTRUCTION], dtype=torch.bool)
                if self.vae_prior_consistency_weight:
                    prior_consistency = _plan_consistency_loss(
                        pred[PLAN_VAE_PRIOR_RECONSTRUCTION],
                        prior_mask,
                        normalization=self.plan_normalization,
                    )
                if self.vae_prior_backward_weight:
                    prior_backward = _plan_backward_loss(
                        pred[PLAN_VAE_PRIOR_RECONSTRUCTION],
                        prior_mask,
                        normalization=self.plan_normalization,
                    )
            loss = (
                self.vae_reconstruction_weight * reconstruction_objective
                + self.vae_sampled_reconstruction_weight * sampled_objective
                + self.vae_kl_weight * kl_weight_scale * kl_penalty
                + self.vae_consistency_weight * consistency
                + self.vae_backward_weight * backward
                + self.vae_prior_consistency_weight * prior_consistency
                + self.vae_prior_backward_weight * prior_backward
            )
            mean_kl_per_dimension = kl_per_dimension.mean(dim=tuple(range(1, kl_per_dimension.ndim - 1)))
            metrics = {
                "plan_vae_reconstruction": reconstruction.detach(),
                "plan_vae_weighted_reconstruction": weighted_reconstruction.detach(),
                "plan_vae_reconstruction_objective": reconstruction_objective.detach(),
                "plan_vae_physical_mae": physical_mae.detach(),
                "plan_vae_weighted_physical_mae": weighted_physical_mae.detach(),
                "plan_vae_physical_smooth_l1": physical_smooth_l1.detach(),
                "plan_vae_weighted_physical_smooth_l1": weighted_physical_smooth_l1.detach(),
                "plan_vae_sampled_reconstruction": sampled_reconstruction.detach(),
                "plan_vae_sampled_weighted_reconstruction": sampled_weighted_reconstruction.detach(),
                "plan_vae_kl": kl_nats.detach(),
                "plan_vae_kl_bits": (kl_nats / 0.6931471805599453).detach(),
                "plan_vae_kl_per_dimension": _mean_nonbatch(kl_per_dimension).detach(),
                "plan_vae_kl_penalty": kl_penalty.detach(),
                "plan_vae_kl_weight_scale": kl_weight_scale.detach(),
                "plan_vae_active_latents": (mean_kl_per_dimension > self.vae_kl_free_bits).sum(dim=-1).detach(),
                "plan_vae_consistency": consistency.detach(),
                "plan_vae_backward": backward.detach(),
                "plan_vae_prior_consistency": prior_consistency.detach(),
                "plan_vae_prior_backward": prior_backward.detach(),
                "plan_vae_position_l2": _vae_position_l2(
                    pred[PLAN_VAE_RECONSTRUCTION],
                    targets[PLAN_VAE_TARGET],
                    targets[PLAN_VAE_MASK],
                    normalization=self.plan_normalization,
                ).detach(),
                "plan_vae_latent_mean_rms": _mean_nonbatch(mean.square()).sqrt().detach(),
                "plan_vae_posterior_std": _mean_nonbatch(torch.exp(0.5 * logvar)).detach(),
            }
            return loss, metrics

        loss, losses = self.loss_fn(pred, targets)
        metrics = losses | self.metric_fn(pred, targets)
        return loss, {f"path/{name}": value for name, value in metrics.items() if name != "loss"}
