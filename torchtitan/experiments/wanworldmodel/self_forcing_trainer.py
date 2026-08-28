# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer for the isolated Wan Self-Forcing experiment."""

from __future__ import annotations

import gc
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import cast

import torch
import torch.distributed.checkpoint as dcp

from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.experiments.wan_vae.tokenizer import WanVAETokenizer
from torchtitan.observability import structured_logger as sl
from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer

from .self_forcing import (
    WanSelfForcingModel,
    self_forcing_losses,
    shifted_rf_scheduler,
)
from .state_dict_adapter import WanStateDictAdapter
from .torchpackage_checkpoint import WanTorchPackageConfig
from .trainer import (
    WanWorldModelTrainer,
    _iter_wan_sink_window_batches,
    _model_batch_size,
    _validate_wanworldmodel_config,
)


def _iter_self_forcing_window_batches(
    latents_BFCHW: torch.Tensor,
    *,
    inference_prefill_frames: int,
) -> Iterator[torch.Tensor]:
    """Drain every temporal start while retaining paired fcam/ecam samples."""
    source_batch_size = latents_BFCHW.size(0)
    for packed_BFCHW in _iter_wan_sink_window_batches(
        latents_BFCHW,
        inference_prefill_frames=inference_prefill_frames,
    ):
        if packed_BFCHW.size(0) % source_batch_size:
            raise ValueError("packed Wan windows must be divisible by their source batch")
        windows_per_source = packed_BFCHW.size(0) // source_batch_size
        grouped_BKFCHW = packed_BFCHW.unflatten(
            0,
            (source_batch_size, windows_per_source),
        )
        for window_index in range(windows_per_source):
            yield grouped_BKFCHW[:, window_index].contiguous()


class WanSelfForcingTrainer(WanWorldModelTrainer):
    """Train a causal generator against online bidirectional score models."""

    @dataclass(kw_only=True, slots=True)
    class Config(WanWorldModelTrainer.Config):
        teacher_checkpoint_path: str
        load_teacher_from_hf: bool = True
        rollout_steps: int = 15
        rollout_shift: float = 1.0
        dmd_min_timestep: float = 0.02
        dmd_max_timestep: float = 0.98
        dmd_normalizer_min: float = 1e-5
        fake_score_loss_weight: float = 1.0

        def __post_init__(self) -> None:
            Trainer.Config.__post_init__(self)
            _validate_wan_self_forcing_config(self)

    def __init__(self, config: Config):
        super().__init__(config)
        self.rollout_scheduler = shifted_rf_scheduler(
            steps=config.rollout_steps,
            shift=config.rollout_shift,
            device=self.device,
        )
        self._score_models_ready = False

    @staticmethod
    def _apply_float8(
        config: Config,
        model_config: WanSelfForcingModel.Config,
    ) -> None:
        from .model_config import _wan_blocks_only_float8

        model_compile_enabled = config.compile.enable and "model" in config.compile.components
        converter = _wan_blocks_only_float8(
            model_compile_enabled=model_compile_enabled,
            emulate=config.float8.emulate,
        )
        # Traversing the composite config converts the generator and both score
        # networks while retaining their distinct attention-mask choices.
        converter.build().convert(model_config)

    def _build_package_config(
        self,
        model_config: WanSelfForcingModel.Config,
    ) -> WanTorchPackageConfig:
        generator_config = model_config.generator_config()
        tokenizer = cast(WanVAETokenizer, self.tokenizer)
        text_context_BLC = tokenizer.fixed_text_context(
            1,
            expected_dim=generator_config.text_dim,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        return WanTorchPackageConfig(
            model_config=generator_config,
            text_context_LC=(text_context_BLC[0].clone() if text_context_BLC is not None else None),
            text_prompt=tokenizer.config.text_prompt,
        )

    def batch_generator(
        self,
        data_iterable: Iterable[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]],
    ) -> Iterator[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]:
        """Encode once and emit one paired-view temporal start per step."""
        data_iterator = iter(data_iterable)
        tokenizer = cast(WanVAETokenizer, self.tokenizer)
        inference_prefill_frames = self.config.dataloader.inference_prefill_frames

        while True:
            data_load_start = time.perf_counter()
            try:
                input_dict, targets = next(data_iterator)
            except StopIteration as ex:
                raise DataloaderExhaustedError() from ex
            source_data_loading_time = time.perf_counter() - data_load_start
            if targets:
                raise ValueError("Wan Self-Forcing does not use dataset targets")

            with sl.log_trace_span("wan_encode_continuous_stream"):
                encoded_BFCHW = tokenizer.encode(
                    input_dict,
                    device=self.device,
                    dtype=self.dtype,
                )
                source_latents_BFCHW = encoded_BFCHW.detach().to(device="cpu")
            del encoded_BFCHW, input_dict, targets

            for windowed_latents_BFCHW in _iter_self_forcing_window_batches(
                source_latents_BFCHW,
                inference_prefill_frames=inference_prefill_frames,
            ):
                self.metrics_processor.data_loading_times.append(source_data_loading_time)
                source_data_loading_time = 0.0
                windowed_inputs = {"latents": windowed_latents_BFCHW}
                self.metrics_processor.ntokens_since_last_log += (
                    _model_batch_size(windowed_inputs) * self.config.training.seq_len
                )
                yield windowed_inputs, {}

    @torch.no_grad()
    def _load_real_score_from_hf(self, model: WanSelfForcingModel) -> None:
        checkpoint_path = self.config.teacher_checkpoint_path
        logger.info(
            "Loading frozen Self-Forcing real-score model from %s",
            checkpoint_path,
        )
        begin = time.monotonic()
        state = model.real_score.state_dict()
        adapter = WanStateDictAdapter(
            model.real_score.config,
            checkpoint_path,
        )
        dcp.load(
            state,
            storage_reader=adapter.get_hf_storage_reader(checkpoint_path),
        )
        model.real_score.load_state_dict(state, strict=True)
        del state
        gc.collect()
        logger.info(
            "Loaded frozen Self-Forcing real-score model in %.2f seconds",
            time.monotonic() - begin,
        )

    @torch.no_grad()
    def _ensure_score_models_initialized(
        self,
        model: WanSelfForcingModel,
    ) -> None:
        if self._score_models_ready:
            model.real_score.eval()
            return

        if self.config.load_teacher_from_hf:
            self._load_real_score_from_hf(model)
            if not model.fake_score_is_initialized():
                model.initialize_fake_score_from_real()
        else:
            model.initialize_score_models_from_generator()

        model.real_score.requires_grad_(False)
        model.real_score.eval()
        self._score_models_ready = True

    @sl.log_trace_span("fwd_bwd")
    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        local_samples: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if targets:
            raise ValueError("Wan Self-Forcing does not use dataset targets")
        model = cast(WanSelfForcingModel, self.model_parts[0])
        self._ensure_score_models_initialized(model)

        with sl.log_trace_span("wan_self_forcing_prepare_batch"):
            latents_BFCHW = cast(WanVAETokenizer, self.tokenizer).encode(
                input_dict,
                device=self.device,
                dtype=self.dtype,
            )
            expected_frames = self.config.dataloader.inference_prefill_frames + 1
            if latents_BFCHW.size(1) != expected_frames:
                raise ValueError(
                    "Wan Self-Forcing trainer requires preselected sink windows "
                    f"with {expected_frames} latents, got "
                    f"{latents_BFCHW.size(1)}"
                )
            if latents_BFCHW.size(2) != model.in_dim:
                raise ValueError(f"Wan model expects {model.in_dim} latent channels, got {latents_BFCHW.size(2)}")
            batch_size = latents_BFCHW.size(0)
            text_context_BLC = cast(
                WanVAETokenizer,
                self.tokenizer,
            ).fixed_text_context(
                batch_size,
                expected_dim=model.text_dim,
                device=self.device,
                dtype=self.dtype,
            )
            if text_context_BLC is None:
                text_context_BLC = model.get_null_text_embedding(
                    batch_size,
                    device=self.device,
                    dtype=self.dtype,
                )
            score_timesteps_B = self.train_noise_scheduler.sample_timestep((batch_size,)).clamp(
                min=self.config.dmd_min_timestep,
                max=self.config.dmd_max_timestep,
            )

        self.ntokens_seen += batch_size * self.config.training.seq_len
        with self.train_context():
            losses = self_forcing_losses(
                model=model,
                latents_BFCHW=latents_BFCHW,
                text_context_BLC=text_context_BLC,
                rollout_scheduler=self.rollout_scheduler,
                score_timesteps_B=score_timesteps_B,
                dmd_normalizer_min=self.config.dmd_normalizer_min,
                fake_score_loss_weight=self.config.fake_score_loss_weight,
            )
            loss = losses.loss_B.sum() / local_samples
            loss.backward()

        return loss, {
            "dmd_loss": losses.dmd_loss_B,
            "fake_score_loss": losses.fake_score_loss_B,
            "dmd_gradient_norm": losses.dmd_gradient_norm_B,
        }


def _validate_wan_self_forcing_config(
    config: WanSelfForcingTrainer.Config,
) -> None:
    _validate_wanworldmodel_config(config)
    model_config = config.model_spec.model
    if not isinstance(model_config, WanSelfForcingModel.Config):
        raise TypeError("Wan Self-Forcing model_spec must contain WanSelfForcingModel.Config")
    if config.dataloader.inference_prefill_frames != 10:
        raise ValueError("Minimal Wan Self-Forcing requires ten prefill latents (z0 plus nine recent latents)")
    if config.rollout_steps <= 0:
        raise ValueError("Self-Forcing rollout_steps must be positive")
    if config.rollout_shift <= 0:
        raise ValueError("Self-Forcing rollout_shift must be positive")
    if not 0 <= config.dmd_min_timestep < config.dmd_max_timestep <= 1:
        raise ValueError("Self-Forcing DMD timestep range must satisfy 0 <= min < max <= 1")
    if config.dmd_normalizer_min <= 0:
        raise ValueError("Self-Forcing dmd_normalizer_min must be positive")
    if config.fake_score_loss_weight < 0:
        raise ValueError("Self-Forcing fake_score_loss_weight cannot be negative")
    if config.load_teacher_from_hf and not config.teacher_checkpoint_path:
        raise ValueError("Self-Forcing teacher_checkpoint_path is required for HF loading")
    if config.validator.enable:
        raise ValueError("Minimal Self-Forcing does not define an offline validation loss")


__all__ = ["WanSelfForcingTrainer"]
