# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Minimal two-chunk Self-Forcing experiment for causal Wan."""

from __future__ import annotations

import math
from copy import copy
from dataclasses import dataclass, field, fields
from typing import Any

import torch
import torch.nn as nn

from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.experiments.worldmodel.schedulers import RFScheduler

from .model import WanLayerNorm, WanModel, WanRMSNorm, parallelize_wan


# Shape suffixes follow wanworldmodel.model: B=batch, F=latent frames,
# C=channels, H/W=latent spatial dimensions, and L=text sequence length.


def _wan_config_values(config: WanModel.Config) -> dict[str, Any]:
    return {item.name: getattr(config, item.name) for item in fields(WanModel.Config) if item.init}


def bidirectional_score_config(config: WanModel.Config) -> WanModel.Config:
    """Clone a Wan architecture while selecting bidirectional attention."""
    values = _wan_config_values(config)
    values["attention_mask"] = "NONE"
    return WanModel.Config(**values)


class WanSelfForcingModel(WanModel):
    """Causal generator with frozen-real and trainable-fake score models."""

    @dataclass(kw_only=True, slots=True)
    class Config(WanModel.Config):
        real_score: WanModel.Config = field(default_factory=WanModel.Config)
        fake_score: WanModel.Config = field(default_factory=WanModel.Config)

        def __post_init__(self) -> None:
            WanModel.Config.__post_init__(self)
            generator_values = _wan_config_values(self)
            generator_values.pop("attention_mask")
            for name, score_config in (
                ("real_score", self.real_score),
                ("fake_score", self.fake_score),
            ):
                score_values = _wan_config_values(score_config)
                score_values.pop("attention_mask")
                if score_values != generator_values:
                    raise ValueError(f"Self-Forcing {name} architecture must match the generator")
                if score_config.attention_mask != "NONE":
                    raise ValueError(f"Self-Forcing {name} must use bidirectional attention")
            if self.attention_mask != "BLOCKWISE_LOWER_TRIANGLE":
                raise ValueError("Self-Forcing generator must use blockwise causal attention")

        def update_from_config(self, *, config: Any, **kwargs: Any) -> None:
            WanModel.Config.update_from_config(self, config=config, **kwargs)
            self.real_score.update_from_config(config=config, **kwargs)
            self.fake_score.update_from_config(config=config, **kwargs)

        def build(self, **kwargs: Any) -> "WanSelfForcingModel":
            if kwargs:
                raise ValueError("WanSelfForcingModel.Config.build does not accept kwargs")
            if self._owner is None:
                raise NotImplementedError("WanSelfForcingModel.Config has no owner class")
            self._sync_derived_fields()
            self.real_score._sync_derived_fields()
            self.fake_score._sync_derived_fields()
            return self._owner(config=copy(self))

        def generator_config(self) -> WanModel.Config:
            """Return the causal generator-only config used for packaging."""
            return WanModel.Config(**_wan_config_values(self))

    def __init__(self, config: Config):
        super().__init__(config)
        self.real_score = config.real_score.build()
        self.real_score.requires_grad_(False)
        self.fake_score = config.fake_score.build()
        # This buffer distinguishes a causal-only initial checkpoint from a
        # resumed Self-Forcing checkpoint that already contains a learned fake
        # score model.
        self.register_buffer(
            "_fake_score_initialized",
            torch.tensor(False),
            persistent=True,
        )

    def reset_parameters(self) -> None:
        """Initialize only the inherited generator, never the score children."""
        self.patch_embedding.reset_parameters()
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))

        root_modules = (
            self.patch_embedding,
            self.text_embedding,
            self.time_embedding,
            self.time_projection,
            self.blocks,
            self.head,
        )
        visited: set[int] = set()
        for root_module in root_modules:
            for module in root_module.modules():
                if id(module) in visited:
                    continue
                visited.add(id(module))
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, WanRMSNorm):
                    module.reset_parameters()
                elif isinstance(module, WanLayerNorm) and module.elementwise_affine:
                    module.reset_parameters()

        for module in self.text_embedding.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
        for module in self.time_embedding.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
        for block in self.blocks:
            nn.init.normal_(block.modulation, std=self.dim**-0.5)
        nn.init.normal_(self.head.modulation, std=self.dim**-0.5)
        nn.init.zeros_(self.head.head.weight)

    def _init_self_buffers(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        device = buffer_device
        if device is None:
            device = self.patch_embedding.weight.device
        self._fake_score_initialized = torch.zeros(
            (),
            dtype=torch.bool,
            device=device,
        )

    def state_dict(
        self,
        destination: Any = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, Any]:
        """Checkpoint the generator and fake score, but omit the frozen teacher."""
        state = WanModel.state_dict(
            self,
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
        )
        teacher_prefix = f"{prefix}real_score."
        for key in tuple(state):
            if key.startswith(teacher_prefix):
                del state[key]
        return state

    def generator_state_dict(self) -> dict[str, Any]:
        state = WanModel.state_dict(self)
        excluded_prefixes = ("real_score.", "fake_score.")
        return {
            key: value
            for key, value in state.items()
            if not key.startswith(excluded_prefixes) and key != "_fake_score_initialized"
        }

    @torch.no_grad()
    def initialize_score_models_from_generator(self) -> None:
        """Debug-only initialization when official HF weights are unavailable."""
        generator_state = self.generator_state_dict()
        self.real_score.load_state_dict(generator_state, strict=True)
        self.fake_score.load_state_dict(generator_state, strict=True)
        self._fake_score_initialized.fill_(True)

    @torch.no_grad()
    def initialize_fake_score_from_real(self) -> None:
        self.fake_score.load_state_dict(
            self.real_score.state_dict(),
            strict=True,
        )
        self._fake_score_initialized.fill_(True)

    def fake_score_is_initialized(self) -> bool:
        return bool(self._fake_score_initialized.item())


def self_forcing_config(config: WanModel.Config) -> WanSelfForcingModel.Config:
    """Wrap a causal Wan architecture in the minimal Self-Forcing model."""
    score_config = bidirectional_score_config(config)
    return WanSelfForcingModel.Config(
        **_wan_config_values(config),
        real_score=score_config,
        fake_score=bidirectional_score_config(config),
    )


def parallelize_wan_self_forcing(
    model: WanSelfForcingModel,
    *,
    parallel_dims: Any,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: Any,
    dump_folder: str,
) -> WanSelfForcingModel:
    """Shard each Wan independently, then shard their composite root."""
    parallelize_wan(
        model.real_score,
        parallel_dims=parallel_dims,
        training=training,
        parallelism=parallelism,
        compile_config=compile_config,
        ac_config=None,
        dump_folder=dump_folder,
    )
    parallelize_wan(
        model.fake_score,
        parallel_dims=parallel_dims,
        training=training,
        parallelism=parallelism,
        compile_config=compile_config,
        ac_config=ac_config,
        dump_folder=dump_folder,
    )
    parallelize_wan(
        model,
        parallel_dims=parallel_dims,
        training=training,
        parallelism=parallelism,
        compile_config=compile_config,
        ac_config=ac_config,
        dump_folder=dump_folder,
    )
    return model


def shifted_rf_scheduler(
    *,
    steps: int,
    shift: float,
    device: torch.device,
) -> RFScheduler:
    if steps <= 0:
        raise ValueError(f"Self-Forcing rollout_steps must be positive, got {steps}")
    if shift <= 0:
        raise ValueError(f"Self-Forcing rollout_shift must be positive, got {shift}")
    scheduler = RFScheduler(steps=steps, inference_schedule="linear").to(device=device)
    if shift != 1.0:
        shifted = shift * scheduler.timesteps / (1 + (shift - 1) * scheduler.timesteps)
        scheduler.timesteps.copy_(shifted)
        scheduler.dt.copy_(-torch.diff(shifted))
    return scheduler


def _seq_len(model: WanModel, latents_BFCHW: torch.Tensor) -> int:
    _, frames, _, height, width = latents_BFCHW.shape
    latent_shape = (frames, height, width)
    if any(size % patch for size, patch in zip(latent_shape, model.patch_size)):
        raise ValueError(f"Wan latent shape {latent_shape} must be divisible by patch size {model.patch_size}")
    return math.prod(size // patch for size, patch in zip(latent_shape, model.patch_size))


def _forward_BFCHW(
    model: WanModel,
    latents_BFCHW: torch.Tensor,
    timesteps_BF: torch.Tensor,
    text_context_BLC: torch.Tensor,
) -> torch.Tensor:
    latents_BCFHW = latents_BFCHW.permute(0, 2, 1, 3, 4)
    outputs = model(
        list(latents_BCFHW.unbind(0)),
        timesteps_BF * 1000.0,
        list(text_context_BLC.unbind(0)),
        _seq_len(model, latents_BFCHW),
    )
    if not isinstance(outputs, list) or len(outputs) != latents_BFCHW.size(0):
        raise TypeError("Wan Self-Forcing forward must return one tensor per sample")
    return torch.stack(outputs).permute(0, 2, 1, 3, 4)


def generate_next_latent(
    *,
    model: WanModel,
    context_BFCHW: torch.Tensor,
    text_context_BLC: torch.Tensor,
    scheduler: RFScheduler,
    retain_final_step_graph: bool,
) -> torch.Tensor:
    """Run the exact autoregressive one-latent RF solver used at inference."""
    batch_size = context_BFCHW.size(0)
    candidate_B1CHW = torch.randn(
        (batch_size, 1, *context_BFCHW.shape[2:]),
        device=context_BFCHW.device,
        dtype=context_BFCHW.dtype,
    )
    for step_index, timestep in enumerate(scheduler.timesteps[:-1]):
        keep_graph = retain_final_step_graph and step_index == scheduler.timesteps.numel() - 2
        with torch.set_grad_enabled(keep_graph):
            model_input_BFCHW = torch.cat(
                (context_BFCHW, candidate_B1CHW),
                dim=1,
            )
            timesteps_BF = torch.zeros(
                model_input_BFCHW.shape[:2],
                device=model_input_BFCHW.device,
                dtype=torch.float32,
            )
            timesteps_BF[:, -1] = timestep
            velocity_BFCHW = _forward_BFCHW(
                model,
                model_input_BFCHW,
                timesteps_BF,
                text_context_BLC,
            )
            candidate_B1CHW = scheduler.step(
                -velocity_BFCHW[:, -1:],
                step_index,
                candidate_B1CHW,
            ).to(dtype=context_BFCHW.dtype)
    return candidate_B1CHW


def slide_self_forcing_context(
    context_BFCHW: torch.Tensor,
    generated_B1CHW: torch.Tensor,
) -> torch.Tensor:
    """Retain z0, evict the oldest recent latent, and append one generation."""
    if context_BFCHW.ndim != 5 or context_BFCHW.size(1) != 10:
        raise ValueError("Self-Forcing context must contain z0 plus nine recent latents")
    expected_shape = (
        context_BFCHW.size(0),
        1,
        *context_BFCHW.shape[2:],
    )
    if generated_B1CHW.shape != expected_shape:
        raise ValueError(f"generated latent must have shape {expected_shape}, got {tuple(generated_B1CHW.shape)}")
    return torch.cat(
        (
            context_BFCHW[:, :1],
            context_BFCHW[:, 2:],
            generated_B1CHW,
        ),
        dim=1,
    )


@dataclass(slots=True)
class SelfForcingLosses:
    loss_B: torch.Tensor
    dmd_loss_B: torch.Tensor
    fake_score_loss_B: torch.Tensor
    dmd_gradient_norm_B: torch.Tensor


def self_forcing_losses(
    *,
    model: WanSelfForcingModel,
    latents_BFCHW: torch.Tensor,
    text_context_BLC: torch.Tensor,
    rollout_scheduler: RFScheduler,
    score_timesteps_B: torch.Tensor,
    dmd_normalizer_min: float,
    fake_score_loss_weight: float,
) -> SelfForcingLosses:
    """Generate two chunks and compute DMD plus fake-score RF losses."""
    if latents_BFCHW.ndim != 5:
        raise ValueError(f"Self-Forcing latents must have shape [B, F, C, H, W], got {tuple(latents_BFCHW.shape)}")
    if latents_BFCHW.size(1) != 11:
        raise ValueError(
            f"Minimal Self-Forcing requires [z0, nine history latents, target] (11 total), got {latents_BFCHW.size(1)}"
        )
    if score_timesteps_B.shape != (latents_BFCHW.size(0),):
        raise ValueError("Self-Forcing score timesteps must contain one value per sample")
    if dmd_normalizer_min <= 0:
        raise ValueError("Self-Forcing dmd_normalizer_min must be positive")

    initial_context_BFCHW = latents_BFCHW[:, :10]
    generated_1_B1CHW = generate_next_latent(
        model=model,
        context_BFCHW=initial_context_BFCHW,
        text_context_BLC=text_context_BLC,
        scheduler=rollout_scheduler,
        retain_final_step_graph=False,
    ).detach()
    rolling_context_BFCHW = slide_self_forcing_context(
        initial_context_BFCHW,
        generated_1_B1CHW,
    )
    generated_2_B1CHW = generate_next_latent(
        model=model,
        context_BFCHW=rolling_context_BFCHW,
        text_context_BLC=text_context_BLC,
        scheduler=rollout_scheduler,
        retain_final_step_graph=True,
    )
    generated_window_BFCHW = torch.cat(
        (rolling_context_BFCHW, generated_2_B1CHW),
        dim=1,
    )

    detached_window_BFCHW = generated_window_BFCHW.detach()
    score_noise_BFCHW = torch.randn_like(detached_window_BFCHW)
    score_timesteps_BF = score_timesteps_B[:, None].expand(detached_window_BFCHW.shape[:2]).clone()
    # z0 remains the exact clean attention sink under every training path.
    score_timesteps_BF[:, 0] = 0.0
    noisy_window_BFCHW = (1 - score_timesteps_BF[:, :, None, None, None]) * detached_window_BFCHW + score_timesteps_BF[
        :, :, None, None, None
    ] * score_noise_BFCHW

    with torch.no_grad():
        real_velocity_BFCHW = _forward_BFCHW(
            model.real_score,
            noisy_window_BFCHW,
            score_timesteps_BF,
            text_context_BLC,
        )
    fake_velocity_BFCHW = _forward_BFCHW(
        model.fake_score,
        noisy_window_BFCHW,
        score_timesteps_BF,
        text_context_BLC,
    )

    score_t_BF111 = score_timesteps_BF[:, :, None, None, None]
    real_x0_BFCHW = noisy_window_BFCHW - score_t_BF111 * real_velocity_BFCHW
    fake_x0_BFCHW = noisy_window_BFCHW - score_t_BF111 * fake_velocity_BFCHW
    raw_dmd_gradient_B1CHW = (fake_x0_BFCHW[:, -1:] - real_x0_BFCHW[:, -1:]).detach()
    normalizer_B = raw_dmd_gradient_B1CHW.float().abs().mean(dim=(1, 2, 3, 4)).clamp_min(dmd_normalizer_min)
    dmd_gradient_B1CHW = raw_dmd_gradient_B1CHW / normalizer_B[:, None, None, None, None]
    dmd_target_B1CHW = generated_2_B1CHW.detach() - dmd_gradient_B1CHW
    dmd_loss_B = 0.5 * (generated_2_B1CHW.float() - dmd_target_B1CHW.float()).square().mean(dim=(1, 2, 3, 4))

    fake_target_BFCHW = score_noise_BFCHW - detached_window_BFCHW
    fake_score_loss_B = 0.5 * (fake_velocity_BFCHW[:, -2:].float() - fake_target_BFCHW[:, -2:].float()).square().mean(
        dim=(1, 2, 3, 4)
    )
    loss_B = dmd_loss_B + fake_score_loss_weight * fake_score_loss_B
    return SelfForcingLosses(
        loss_B=loss_B,
        dmd_loss_B=dmd_loss_B.detach(),
        fake_score_loss_B=fake_score_loss_B.detach(),
        dmd_gradient_norm_B=raw_dmd_gradient_B1CHW.float().square().mean(dim=(1, 2, 3, 4)).sqrt(),
    )


__all__ = [
    "SelfForcingLosses",
    "WanSelfForcingModel",
    "bidirectional_score_config",
    "generate_next_latent",
    "parallelize_wan_self_forcing",
    "self_forcing_config",
    "self_forcing_losses",
    "shifted_rf_scheduler",
    "slide_self_forcing_context",
]
