from typing import Any

import math
import torch
import torch.nn.functional as F

from torchtitan.experiments.worldmodel.compressor import images_to_latents
from torchtitan.experiments.worldmodel.model import WorldModel


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def _floating_model_dtype(model: torch.nn.Module) -> torch.dtype:
    for param in model.parameters():
        if param.is_floating_point():
            return param.dtype
    return torch.float32


def get_pose_dropout_mask(*, batch_size: int, num_frames: int, pose_dropout: float, device: torch.device, train: bool) -> torch.Tensor:
    drop_prob = pose_dropout if train else 0.0
    return torch.rand((batch_size, num_frames), device=device) < drop_prob


def laplacian_density_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    *,
    std_clamp: float = 1e-3,
    loss_clamp: float = 1000.0,
) -> torch.Tensor:
    n_values = y_pred.shape[-1] // 2
    mu_true = y_true[..., :n_values]
    mu_pred = y_pred[..., :n_values]
    mask = ~torch.isnan(mu_true)
    mu_true = mu_true.masked_fill(~mask, 0).detach()
    log_sigma_raw = y_pred[..., n_values:]
    err = torch.abs(mu_true - mu_pred)
    log_sigma_min = torch.clamp(log_sigma_raw, min=math.log(std_clamp))
    log_sigma = torch.max(log_sigma_raw, torch.log(1e-6 + err / loss_clamp))
    return mask * (err * torch.exp(-log_sigma) + log_sigma_min)


def prepare_worldmodel_batch(
    model: WorldModel,
    input_dict: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    device: torch.device,
    scheduler: torch.nn.Module,
    discrete_timesteps: torch.Tensor,
    compressor_encoder: torch.nn.Module,
    pose_dropout: float,
    train: bool,
    dtype: torch.dtype | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    model_dtype = dtype or _floating_model_dtype(model) # TODO might not be needed
    latents = images_to_latents(compressor_encoder, input_dict["imgs"], input_dict["big_imgs"], device=device, dtype=model_dtype)
    augments_pos_ref_augment = input_dict["augments_pos_ref_augment"].to(device=device, dtype=model_dtype)
    ref_augment_from_augments_euler = input_dict["ref_augment_from_augments_euler"].to(device=device, dtype=model_dtype)
    fidxs = input_dict["fidxs"].to(device=device, dtype=torch.int64)
    targets = move_to_device(targets, device)
    batch_size, num_frames = latents.shape[:2]

    with torch.no_grad():
        latents = model.scale_latents(latents)
        noise = torch.randn_like(latents)
        if train:
            timesteps = scheduler.sample_timestep((batch_size, num_frames))
        else:
            indexes = torch.randint(0, discrete_timesteps.numel(), (batch_size,), device=device)
            timesteps = discrete_timesteps[indexes][:, None].expand(-1, num_frames).clone()

        pose_mask = get_pose_dropout_mask(batch_size=batch_size, num_frames=num_frames, pose_dropout=pose_dropout, device=device, train=train)
        augments_pos_ref_augment[pose_mask] = 0
        ref_augment_from_augments_euler[pose_mask] = 0
        pose_mask = pose_mask.to(dtype=torch.int64)
        mask = torch.ones_like(latents, device=device, dtype=torch.bool)
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        targets = {**targets, "v": latents - noise, "mask": mask}

    return {
        "x": noisy_latents,
        "t": timesteps,
        "augments_pos_ref_augment": augments_pos_ref_augment,
        "ref_augment_from_augments_euler": ref_augment_from_augments_euler,
        "pose_mask": pose_mask,
        "fidx": fidxs,
    }, targets


def compute_worldmodel_losses(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    batch_size: int,
    plan_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = torch.zeros((batch_size,), device=next(iter(outputs.values())).device)
    terms: dict[str, torch.Tensor] = {}

    if "sample" in outputs:
        pred = outputs["sample"]
        target = targets["v"].to(dtype=pred.dtype)
        mse = F.mse_loss(pred.float(), target.float(), reduction="none").flatten(1)
        flat_mask = targets["mask"].flatten(1).float()
        diffusion_loss = (mse * flat_mask).sum(dim=1) / flat_mask.sum(dim=1).clamp_min(1.0)
        loss = loss + diffusion_loss
        terms["diffusion_loss"] = diffusion_loss.detach()

    if "plan" in outputs and "plan" in targets:
        plan_pred = outputs["plan"]
        plan_target = targets["plan"].to(device=plan_pred.device, dtype=plan_pred.dtype)
        plan_values = plan_pred.shape[-1] // 2
        plan_loss = laplacian_density_loss(plan_target.float(), plan_pred.float()).flatten(1).mean(dim=1)
        plan_squared_error = F.mse_loss(plan_pred[..., :plan_values].float(), plan_target[..., :plan_values].float(), reduction="none").flatten(1)
        plan_mse = torch.nanmean(plan_squared_error, dim=1)
        loss = loss + (plan_loss_weight if "sample" in outputs else 1.0) * plan_loss
        terms["plan_loss"] = plan_loss.detach()
        terms["plan_mse"] = plan_mse.detach()

    if not terms:
        raise RuntimeError("worldmodel produced no trainable loss outputs")

    terms["loss"] = loss.detach()
    return loss, terms
