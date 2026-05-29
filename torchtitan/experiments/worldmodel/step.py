from typing import Any

import torch

from torchtitan.experiments.worldmodel.compressor import images_to_latents
from torchtitan.experiments.worldmodel.loss import compute_worldmodel_losses, laplacian_density_loss  # noqa: F401
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
