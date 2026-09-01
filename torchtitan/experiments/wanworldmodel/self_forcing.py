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
import torch.distributed as dist
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
    frame_indices_BF: torch.Tensor,
) -> torch.Tensor:
    latents_BCFHW = latents_BFCHW.permute(0, 2, 1, 3, 4)
    outputs = model(
        list(latents_BCFHW.unbind(0)),
        timesteps_BF * 1000.0,
        list(text_context_BLC.unbind(0)),
        _seq_len(model, latents_BFCHW),
        frame_indices=frame_indices_BF,
    )
    if not isinstance(outputs, list) or len(outputs) != latents_BFCHW.size(0):
        raise TypeError("Wan Self-Forcing forward must return one tensor per sample")
    return torch.stack(outputs).permute(0, 2, 1, 3, 4)


def generate_next_latent(
    *,
    model: WanModel,
    context_BFCHW: torch.Tensor,
    context_frame_indices_BF: torch.Tensor,
    target_frame_indices_B1: torch.Tensor,
    text_context_BLC: torch.Tensor,
    scheduler: RFScheduler,
    exit_step_index: int,
    retain_exit_step_graph: bool,
) -> torch.Tensor:
    """Predict one clean latent with the standard random-exit SDE estimator."""
    batch_size = context_BFCHW.size(0)
    if context_frame_indices_BF.shape != context_BFCHW.shape[:2]:
        raise ValueError(
            "context frame indices must have shape "
            f"{tuple(context_BFCHW.shape[:2])}, got "
            f"{tuple(context_frame_indices_BF.shape)}"
        )
    if target_frame_indices_B1.shape != (batch_size, 1):
        raise ValueError(
            f"target frame indices must have shape [{batch_size}, 1], got {tuple(target_frame_indices_B1.shape)}"
        )
    model_frame_indices_BF = torch.cat(
        (context_frame_indices_BF, target_frame_indices_B1),
        dim=1,
    )
    num_denoising_steps = scheduler.timesteps.numel() - 1
    if not 0 <= exit_step_index < num_denoising_steps:
        raise ValueError(f"exit_step_index={exit_step_index} must be in [0, {num_denoising_steps - 1}]")
    candidate_B1CHW = torch.randn(
        (batch_size, 1, *context_BFCHW.shape[2:]),
        device=context_BFCHW.device,
        dtype=context_BFCHW.dtype,
    )
    for step_index, timestep in enumerate(scheduler.timesteps[: exit_step_index + 1]):
        is_exit_step = step_index == exit_step_index
        with torch.set_grad_enabled(retain_exit_step_graph and is_exit_step):
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
                model_frame_indices_BF,
            )
            predicted_x0_B1CHW = (candidate_B1CHW - timestep * velocity_BFCHW[:, -1:]).to(dtype=context_BFCHW.dtype)
        if is_exit_step:
            return predicted_x0_B1CHW

        # Standard Self-Forcing uses an SDE transition between student
        # evaluations. The preceding prediction is detached by the no-grad
        # context, while the next exit prediction is the only differentiable
        # model call.
        next_timestep = scheduler.timesteps[step_index + 1]
        candidate_B1CHW = (
            (1 - next_timestep) * predicted_x0_B1CHW + next_timestep * torch.randn_like(predicted_x0_B1CHW)
        ).to(dtype=context_BFCHW.dtype)

    raise AssertionError("Self-Forcing rollout did not reach its exit step")


def sample_self_forcing_exit_step(
    scheduler: RFScheduler,
    *,
    device: torch.device,
) -> int:
    """Sample one exit shared by all blocks, batches, and distributed ranks."""
    num_denoising_steps = scheduler.timesteps.numel() - 1
    if num_denoising_steps <= 0:
        raise ValueError("Self-Forcing scheduler must contain a denoising step")

    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        exit_step = torch.randint(
            num_denoising_steps,
            (1,),
            device=device,
            dtype=torch.int64,
        )
    else:
        exit_step = torch.empty((1,), device=device, dtype=torch.int64)
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(exit_step, src=0)
    return int(exit_step.item())


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


def slide_self_forcing_frame_indices(
    context_frame_indices_BF: torch.Tensor,
    generated_frame_indices_B1: torch.Tensor,
) -> torch.Tensor:
    """Apply the latent context eviction rule to temporal RoPE coordinates."""
    if context_frame_indices_BF.ndim != 2 or context_frame_indices_BF.size(1) != 10:
        raise ValueError("Self-Forcing frame indices must contain z0 plus nine recent positions")
    expected_shape = (context_frame_indices_BF.size(0), 1)
    if generated_frame_indices_B1.shape != expected_shape:
        raise ValueError(
            f"generated frame indices must have shape {expected_shape}, got {tuple(generated_frame_indices_B1.shape)}"
        )
    return torch.cat(
        (
            context_frame_indices_BF[:, :1],
            context_frame_indices_BF[:, 2:],
            generated_frame_indices_B1,
        ),
        dim=1,
    )


@dataclass(slots=True)
class SelfForcingGeneratorLosses:
    dmd_loss_B: torch.Tensor
    dmd_gradient_norm_B: torch.Tensor
    dmd_normalizer_B: torch.Tensor
    rollout_exit_timestep_B: torch.Tensor


@dataclass(slots=True)
class SelfForcingCriticLosses:
    fake_score_loss_B: torch.Tensor
    rollout_exit_timestep_B: torch.Tensor


@dataclass(slots=True)
class _SelfForcingRollout:
    generated_window_BFCHW: torch.Tensor
    generated_window_frame_indices_BF: torch.Tensor
    generated_2_B1CHW: torch.Tensor


def _normalized_dmd_gradient(
    *,
    generated_B1CHW: torch.Tensor,
    fake_x0_B1CHW: torch.Tensor,
    real_x0_B1CHW: torch.Tensor,
    normalizer_min: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the DMD gradient normalized by generator-to-teacher error."""
    raw_gradient_B1CHW = (fake_x0_B1CHW - real_x0_B1CHW).detach()
    normalizer_B = (
        (generated_B1CHW.detach() - real_x0_B1CHW).float().abs().mean(dim=(1, 2, 3, 4)).clamp_min(normalizer_min)
    )
    gradient_B1CHW = torch.nan_to_num(raw_gradient_B1CHW / normalizer_B[:, None, None, None, None])
    raw_gradient_norm_B = raw_gradient_B1CHW.float().square().mean(dim=(1, 2, 3, 4)).sqrt()
    return gradient_B1CHW, raw_gradient_norm_B, normalizer_B


def _self_forcing_rollout(
    *,
    model: WanSelfForcingModel,
    latents_BFCHW: torch.Tensor,
    frame_indices_BF: torch.Tensor,
    text_context_BLC: torch.Tensor,
    rollout_scheduler: RFScheduler,
    rollout_exit_step_index: int,
    retain_generator_graph: bool,
) -> _SelfForcingRollout:
    """Generate two autoregressive chunks from one real context window."""
    if latents_BFCHW.ndim != 5:
        raise ValueError(f"Self-Forcing latents must have shape [B, F, C, H, W], got {tuple(latents_BFCHW.shape)}")
    if latents_BFCHW.size(1) != 11:
        raise ValueError(
            f"Minimal Self-Forcing requires [z0, nine history latents, target] (11 total), got {latents_BFCHW.size(1)}"
        )
    if frame_indices_BF.shape != latents_BFCHW.shape[:2]:
        raise ValueError(
            "Self-Forcing frame indices must have shape "
            f"{tuple(latents_BFCHW.shape[:2])}, got "
            f"{tuple(frame_indices_BF.shape)}"
        )
    initial_context_BFCHW = latents_BFCHW[:, :10]
    initial_context_frame_indices_BF = frame_indices_BF[:, :10]
    generated_1_frame_indices_B1 = frame_indices_BF[:, 10:]
    generated_1_B1CHW = generate_next_latent(
        model=model,
        context_BFCHW=initial_context_BFCHW,
        context_frame_indices_BF=initial_context_frame_indices_BF,
        target_frame_indices_B1=generated_1_frame_indices_B1,
        text_context_BLC=text_context_BLC,
        scheduler=rollout_scheduler,
        exit_step_index=rollout_exit_step_index,
        retain_exit_step_graph=False,
    ).detach()
    rolling_context_BFCHW = slide_self_forcing_context(
        initial_context_BFCHW,
        generated_1_B1CHW,
    )
    rolling_context_frame_indices_BF = slide_self_forcing_frame_indices(
        initial_context_frame_indices_BF,
        generated_1_frame_indices_B1,
    )
    generated_2_frame_indices_B1 = generated_1_frame_indices_B1 + 1
    generated_2_B1CHW = generate_next_latent(
        model=model,
        context_BFCHW=rolling_context_BFCHW,
        context_frame_indices_BF=rolling_context_frame_indices_BF,
        target_frame_indices_B1=generated_2_frame_indices_B1,
        text_context_BLC=text_context_BLC,
        scheduler=rollout_scheduler,
        exit_step_index=rollout_exit_step_index,
        retain_exit_step_graph=retain_generator_graph,
    )
    generated_window_BFCHW = torch.cat(
        (rolling_context_BFCHW, generated_2_B1CHW),
        dim=1,
    )
    generated_window_frame_indices_BF = torch.cat(
        (rolling_context_frame_indices_BF, generated_2_frame_indices_B1),
        dim=1,
    )
    return _SelfForcingRollout(
        generated_window_BFCHW=generated_window_BFCHW,
        generated_window_frame_indices_BF=generated_window_frame_indices_BF,
        generated_2_B1CHW=generated_2_B1CHW,
    )


def _noisy_score_window(
    rollout: _SelfForcingRollout,
    score_timesteps_B: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    detached_window_BFCHW = rollout.generated_window_BFCHW.detach()
    if detached_window_BFCHW.size(1) != 11:
        raise ValueError(
            "Self-Forcing score window must contain nine clean conditioning "
            f"latents followed by two generated latents, got {detached_window_BFCHW.size(1)}"
        )
    if score_timesteps_B.shape != (detached_window_BFCHW.size(0),):
        raise ValueError("Self-Forcing score timesteps must contain one value per sample")

    score_noise_BFCHW = torch.randn_like(detached_window_BFCHW)
    score_timesteps_BF = score_timesteps_B[:, None].expand(detached_window_BFCHW.shape[:2]).clone()
    # The score problem is conditional: [z0, eight real history latents] stay
    # exactly clean, matching rollout inference, while only [g1, g2] are noised.
    score_timesteps_BF[:, :-2] = 0.0
    noisy_window_BFCHW = (1 - score_timesteps_BF[:, :, None, None, None]) * detached_window_BFCHW + score_timesteps_BF[
        :, :, None, None, None
    ] * score_noise_BFCHW
    return noisy_window_BFCHW, score_noise_BFCHW, score_timesteps_BF


def self_forcing_generator_losses(
    *,
    model: WanSelfForcingModel,
    latents_BFCHW: torch.Tensor,
    frame_indices_BF: torch.Tensor,
    text_context_BLC: torch.Tensor,
    rollout_scheduler: RFScheduler,
    rollout_exit_step_index: int,
    score_timesteps_B: torch.Tensor,
    dmd_normalizer_min: float,
) -> SelfForcingGeneratorLosses:
    """Compute DMD on a differentiable second generated chunk."""
    if dmd_normalizer_min <= 0:
        raise ValueError("Self-Forcing dmd_normalizer_min must be positive")
    rollout = _self_forcing_rollout(
        model=model,
        latents_BFCHW=latents_BFCHW,
        frame_indices_BF=frame_indices_BF,
        text_context_BLC=text_context_BLC,
        rollout_scheduler=rollout_scheduler,
        rollout_exit_step_index=rollout_exit_step_index,
        retain_generator_graph=True,
    )
    (
        noisy_window_BFCHW,
        _score_noise_BFCHW,
        score_timesteps_BF,
    ) = _noisy_score_window(rollout, score_timesteps_B)

    with torch.no_grad():
        fake_velocity_BFCHW = _forward_BFCHW(
            model.fake_score,
            noisy_window_BFCHW,
            score_timesteps_BF,
            text_context_BLC,
            rollout.generated_window_frame_indices_BF,
        )
        real_velocity_BFCHW = _forward_BFCHW(
            model.real_score,
            noisy_window_BFCHW,
            score_timesteps_BF,
            text_context_BLC,
            rollout.generated_window_frame_indices_BF,
        )
    score_t_BF111 = score_timesteps_BF[:, :, None, None, None]
    real_x0_B1CHW = (noisy_window_BFCHW - score_t_BF111 * real_velocity_BFCHW)[:, -1:]
    fake_x0_B1CHW = (noisy_window_BFCHW - score_t_BF111 * fake_velocity_BFCHW)[:, -1:]
    (
        dmd_gradient_B1CHW,
        dmd_gradient_norm_B,
        dmd_normalizer_B,
    ) = _normalized_dmd_gradient(
        generated_B1CHW=rollout.generated_2_B1CHW,
        fake_x0_B1CHW=fake_x0_B1CHW,
        real_x0_B1CHW=real_x0_B1CHW,
        normalizer_min=dmd_normalizer_min,
    )
    dmd_target_B1CHW = rollout.generated_2_B1CHW.detach() - dmd_gradient_B1CHW
    dmd_loss_B = 0.5 * (rollout.generated_2_B1CHW.float() - dmd_target_B1CHW.float()).square().mean(dim=(1, 2, 3, 4))
    return SelfForcingGeneratorLosses(
        dmd_loss_B=dmd_loss_B,
        dmd_gradient_norm_B=dmd_gradient_norm_B,
        dmd_normalizer_B=dmd_normalizer_B,
        rollout_exit_timestep_B=torch.full_like(
            dmd_loss_B,
            float(rollout_scheduler.timesteps[rollout_exit_step_index]),
        ),
    )


def self_forcing_critic_losses(
    *,
    model: WanSelfForcingModel,
    latents_BFCHW: torch.Tensor,
    frame_indices_BF: torch.Tensor,
    text_context_BLC: torch.Tensor,
    rollout_scheduler: RFScheduler,
    rollout_exit_step_index: int,
    score_timesteps_B: torch.Tensor,
) -> SelfForcingCriticLosses:
    """Fit the fake score on a fresh, fully detached student rollout."""
    with torch.no_grad():
        rollout = _self_forcing_rollout(
            model=model,
            latents_BFCHW=latents_BFCHW,
            frame_indices_BF=frame_indices_BF,
            text_context_BLC=text_context_BLC,
            rollout_scheduler=rollout_scheduler,
            rollout_exit_step_index=rollout_exit_step_index,
            retain_generator_graph=False,
        )
        (
            noisy_window_BFCHW,
            score_noise_BFCHW,
            score_timesteps_BF,
        ) = _noisy_score_window(rollout, score_timesteps_B)

    fake_velocity_BFCHW = _forward_BFCHW(
        model.fake_score,
        noisy_window_BFCHW,
        score_timesteps_BF,
        text_context_BLC,
        rollout.generated_window_frame_indices_BF,
    )

    fake_target_BFCHW = score_noise_BFCHW - rollout.generated_window_BFCHW.detach()
    fake_score_loss_B = (fake_velocity_BFCHW[:, -2:].float() - fake_target_BFCHW[:, -2:].float()).square().mean(
        dim=(1, 2, 3, 4)
    )
    return SelfForcingCriticLosses(
        fake_score_loss_B=fake_score_loss_B,
        rollout_exit_timestep_B=torch.full_like(
            fake_score_loss_B,
            float(rollout_scheduler.timesteps[rollout_exit_step_index]),
        ),
    )


__all__ = [
    "SelfForcingCriticLosses",
    "SelfForcingGeneratorLosses",
    "WanSelfForcingModel",
    "bidirectional_score_config",
    "generate_next_latent",
    "parallelize_wan_self_forcing",
    "sample_self_forcing_exit_step",
    "self_forcing_config",
    "self_forcing_critic_losses",
    "self_forcing_generator_losses",
    "shifted_rf_scheduler",
    "slide_self_forcing_context",
    "slide_self_forcing_frame_indices",
]
