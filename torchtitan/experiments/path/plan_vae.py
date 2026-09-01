# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch

from .model_constants import IDX_N, ModelInputs, PLAN_WIDTH
from .plan_statistics import PLAN_HORIZON_MEAN, PLAN_HORIZON_STD


# B: batch, T: temporal steps, H: plan horizon, P: flattened plan values, W: plan channels.
PLAN_VAE_MASK = "plan_vae_mask"
PLAN_VAE_TARGET = "plan_vae_target"
PLAN_VAE_RECONSTRUCTION = "plan_vae_reconstruction"
PLAN_VAE_SAMPLED_RECONSTRUCTION = "plan_vae_sampled_reconstruction"
PLAN_VAE_PRIOR_RECONSTRUCTION = "plan_vae_prior_reconstruction"
PLAN_VAE_MEAN = "plan_vae_mean"
PLAN_VAE_LOGVAR = "plan_vae_logvar"
PLAN_VAE_KL_WEIGHT_SCALE = "plan_vae_kl_weight_scale"
PLAN_LATENT = "plan_latent"
PLAN_LATENT_TARGET = "plan_latent_target"
PLAN_LATENT_MASK = "plan_latent_mask"
PLAN_HORIZON_MIN_STD = 1e-4
PlanNormalization = Literal["pooled", "per_horizon"]
PlanLoss = Literal["decoded_laplacian", "latent_mse", "latent_nll"]
PLAN_LATENT_LOGVAR = "plan_latent_logvar"


# PlanTargets statistics pooled over the 33-step plan horizon.
PLAN_CHANNEL_MEAN = (
    58.004444,
    0.064027,
    -0.253495,
    17.318117,
    -0.007723,
    -0.063853,
    -0.003761,
    0.008714,
    -0.003468,
    0.000167,
    0.004313,
    0.001905,
    -0.000007,
    0.000230,
    0.000679,
)

PLAN_CHANNEL_STD = (
    67.975006,
    8.744651,
    1.828071,
    9.362529,
    0.273544,
    0.338191,
    0.605918,
    0.622219,
    0.212066,
    0.016435,
    0.022022,
    0.216342,
    0.013488,
    0.012930,
    0.064485,
)


def _plan_normalization_stats(
    plan_BTP: torch.Tensor,
    normalization: PlanNormalization,
) -> tuple[torch.Tensor, torch.Tensor]:
    if normalization == "pooled":
        return plan_BTP.new_tensor(PLAN_CHANNEL_MEAN), plan_BTP.new_tensor(PLAN_CHANNEL_STD)
    if normalization == "per_horizon":
        mean_HW = plan_BTP.new_tensor(PLAN_HORIZON_MEAN)
        std_HW = plan_BTP.new_tensor(PLAN_HORIZON_STD).clamp_min(PLAN_HORIZON_MIN_STD)
        return mean_HW, std_HW
    raise ValueError(f"Unknown plan normalization: {normalization}")


def normalize_plan(
    plan_BTP: torch.Tensor,
    *,
    normalization: PlanNormalization = "pooled",
) -> torch.Tensor:
    plan_BTHW = plan_BTP.unflatten(-1, (IDX_N, PLAN_WIDTH))
    mean, std = _plan_normalization_stats(plan_BTP, normalization)
    return ((plan_BTHW - mean) / std).flatten(-2)


def unnormalize_plan(
    plan_BTP: torch.Tensor,
    *,
    normalization: PlanNormalization = "pooled",
) -> torch.Tensor:
    plan_BTHW = plan_BTP.unflatten(-1, (IDX_N, PLAN_WIDTH))
    mean, std = _plan_normalization_stats(plan_BTP, normalization)
    return (plan_BTHW * std + mean).flatten(-2)


def _normalized_plan_and_mask(
    targets: dict[str, torch.Tensor],
    normalization: PlanNormalization,
) -> tuple[torch.Tensor, torch.Tensor]:
    plan_BTP = targets["plan"].float()
    plan_mask_BTP = torch.isfinite(plan_BTP)
    clean_plan_BTP = plan_BTP.masked_fill(~plan_mask_BTP, 0)
    normalized_plan_BTP = normalize_plan(clean_plan_BTP, normalization=normalization).masked_fill(~plan_mask_BTP, 0)
    return normalized_plan_BTP, plan_mask_BTP


@torch.no_grad()
def prepare_plan_vae_batch(
    input_dict: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    kl_weight_scale: float = 1.0,
    normalization: PlanNormalization = "pooled",
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    normalized_plan_BTP, plan_mask_BTP = _normalized_plan_and_mask(targets, normalization)
    return (
        {**input_dict, ModelInputs.PLAN_VAE: normalized_plan_BTP},
        {
            **targets,
            PLAN_VAE_TARGET: normalized_plan_BTP,
            PLAN_VAE_MASK: plan_mask_BTP,
            PLAN_VAE_KL_WEIGHT_SCALE: torch.full(
                (normalized_plan_BTP.shape[0],),
                kl_weight_scale,
                dtype=torch.float32,
                device=normalized_plan_BTP.device,
            ),
        },
    )


@torch.no_grad()
def prepare_plan_latent_batch(
    targets: dict[str, torch.Tensor],
    encode_plan: Callable[[torch.Tensor], torch.Tensor],
    *,
    normalization: PlanNormalization = "pooled",
) -> dict[str, torch.Tensor]:
    """Add the frozen-encoder latent of the ground-truth plan as the MSE target."""
    normalized_plan_BTP, plan_mask_BTP = _normalized_plan_and_mask(targets, normalization)
    latent_BTZ = encode_plan(normalized_plan_BTP).float()
    latent_mask_BTZ = plan_mask_BTP.all(dim=-1, keepdim=True).expand_as(latent_BTZ)
    return {
        **targets,
        PLAN_LATENT_TARGET: latent_BTZ,
        PLAN_LATENT_MASK: latent_mask_BTZ,
    }
