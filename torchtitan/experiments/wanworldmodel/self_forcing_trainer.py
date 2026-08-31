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
from typing import Any, cast

import torch
import torch.distributed.checkpoint as dcp

from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.wan_vae.tokenizer import WanVAETokenizer
from torchtitan.experiments.worldmodel.trainer import (
    _copy_microbatch,
    _segment_names_from_info,
)
from torchtitan.observability import structured_logger as sl
from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer

from .self_forcing import (
    WanSelfForcingModel,
    sample_self_forcing_exit_step,
    self_forcing_critic_losses,
    self_forcing_generator_losses,
    shifted_rf_scheduler,
)
from .self_forcing_optimizer import (
    WanSelfForcingLRSchedulers,
    WanSelfForcingOptimizers,
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
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Drain every temporal start while retaining paired fcam/ecam samples."""
    source_batch_size = latents_BFCHW.size(0)
    for packed_BFCHW, packed_frame_indices_BF in _iter_wan_sink_window_batches(
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
        grouped_frame_indices_BKF = packed_frame_indices_BF.unflatten(
            0,
            (source_batch_size, windows_per_source),
        )
        for window_index in range(windows_per_source):
            yield (
                grouped_BKFCHW[:, window_index].contiguous(),
                grouped_frame_indices_BKF[:, window_index].contiguous(),
            )


@dataclass(slots=True)
class _PreparedSelfForcingBatch:
    latents_BFCHW: torch.Tensor
    frame_indices_BF: torch.Tensor
    text_context_BLC: torch.Tensor


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
        generator_update_interval: int = 5

        def __post_init__(self) -> None:
            Trainer.Config.__post_init__(self)
            _validate_wan_self_forcing_config(self)

    def __init__(self, config: Config):
        super().__init__(config)
        if not isinstance(self.optimizers, WanSelfForcingOptimizers):
            raise TypeError("Wan Self-Forcing requires WanSelfForcingOptimizers")
        if not isinstance(self.lr_schedulers, WanSelfForcingLRSchedulers):
            raise TypeError("Wan Self-Forcing requires WanSelfForcingLRSchedulers")
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
            inference_sampler="sde",
            inference_steps=self.config.rollout_steps,
            inference_schedule="linear",
            inference_shift=self.config.rollout_shift,
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

            for windowed_latents_BFCHW, frame_indices_BF in _iter_self_forcing_window_batches(
                source_latents_BFCHW,
                inference_prefill_frames=inference_prefill_frames,
            ):
                self.metrics_processor.data_loading_times.append(source_data_loading_time)
                source_data_loading_time = 0.0
                windowed_inputs = {
                    "latents": windowed_latents_BFCHW,
                    "frame_indices": frame_indices_BF,
                }
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

    def _prepare_self_forcing_batch(
        self,
        *,
        model: WanSelfForcingModel,
        input_dict: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> _PreparedSelfForcingBatch:
        if targets:
            raise ValueError("Wan Self-Forcing does not use dataset targets")

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
            frame_indices_BF = input_dict["frame_indices"].to(
                device=self.device,
                dtype=torch.int64,
            )
            if frame_indices_BF.shape != (batch_size, expected_frames):
                raise ValueError(
                    "Wan Self-Forcing frame indices must have shape "
                    f"[{batch_size}, {expected_frames}], got "
                    f"{tuple(frame_indices_BF.shape)}"
                )
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

        return _PreparedSelfForcingBatch(
            latents_BFCHW=latents_BFCHW,
            frame_indices_BF=frame_indices_BF,
            text_context_BLC=text_context_BLC,
        )

    def _sample_score_timesteps(self, batch_size: int) -> torch.Tensor:
        return self.train_noise_scheduler.sample_timestep((batch_size,)).clamp(
            min=self.config.dmd_min_timestep,
            max=self.config.dmd_max_timestep,
        )

    def train_step(
        self,
        data_iterator: Iterator[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]],
    ) -> None:
        optimizers = cast(WanSelfForcingOptimizers, self.optimizers)
        lr_schedulers = cast(WanSelfForcingLRSchedulers, self.lr_schedulers)
        optimizers.zero_grad()
        lr_metrics = lr_schedulers.get_metrics()

        parallel_dims = self.parallel_dims
        batch_mesh = parallel_dims.get_optional_mesh("batch")
        microbatches: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]] = []
        step_segment_names: set[str] = set()
        local_samples = torch.tensor(0, dtype=torch.int64)
        for _ in range(self.gradient_accumulation_steps):
            with sl.log_trace_span("fetching_batch"):
                input_dict, targets = next(data_iterator)
            local_samples += self._model_batch_size(input_dict)
            if "info" in input_dict:
                step_segment_names.update(_segment_names_from_info(input_dict["info"]))
            if self.gradient_accumulation_steps > 1:
                input_dict, targets = _copy_microbatch(input_dict, targets)
            microbatches.append((input_dict, targets))
        sl.log_trace_scalar({"local_samples": int(local_samples)})

        self.ntokens_seen += int(local_samples) * self.config.training.seq_len
        local_samples = local_samples.to(self.device)
        global_samples = (
            dist_utils.dist_sum(local_samples, batch_mesh) if batch_mesh is not None else float(local_samples.item())
        )
        local_samples_float = local_samples.to(dtype=torch.float32)

        model = cast(WanSelfForcingModel, self.model_parts[0])
        self._ensure_score_models_initialized(model)
        update_generator = self.step % self.config.generator_update_interval == 0
        accumulated_losses: list[torch.Tensor] = []
        metric_sums: dict[str, torch.Tensor] = {}

        def accumulate_metric(name: str, value_B: torch.Tensor) -> None:
            metric_sums[name] = (
                metric_sums.get(name, torch.zeros((), device=self.device)) + value_B.detach().float().sum()
            )

        zero = torch.zeros((), device=self.device)
        generator_grad_norm = zero
        waited_for_staging = False

        if update_generator:
            generator_exit_step_index = sample_self_forcing_exit_step(
                self.rollout_scheduler,
                device=self.device,
            )
            with sl.log_trace_span("wan_self_forcing_generator_fwd_bwd"):
                for input_dict, targets in microbatches:
                    prepared = self._prepare_self_forcing_batch(
                        model=model,
                        input_dict=input_dict,
                        targets=targets,
                    )
                    score_timesteps_B = self._sample_score_timesteps(prepared.latents_BFCHW.size(0))
                    with self.train_context():
                        losses = self_forcing_generator_losses(
                            model=model,
                            latents_BFCHW=prepared.latents_BFCHW,
                            frame_indices_BF=prepared.frame_indices_BF,
                            text_context_BLC=prepared.text_context_BLC,
                            rollout_scheduler=self.rollout_scheduler,
                            rollout_exit_step_index=generator_exit_step_index,
                            score_timesteps_B=score_timesteps_B,
                            dmd_normalizer_min=self.config.dmd_normalizer_min,
                        )
                        loss = losses.dmd_loss_B.sum() / local_samples_float
                        loss.backward()
                    accumulated_losses.append(loss.detach())
                    accumulate_metric("dmd_loss", losses.dmd_loss_B)
                    accumulate_metric(
                        "dmd_gradient_norm",
                        losses.dmd_gradient_norm_B,
                    )
                    accumulate_metric("dmd_normalizer", losses.dmd_normalizer_B)
                    accumulate_metric(
                        "generator_rollout_exit_timestep",
                        losses.rollout_exit_timestep_B,
                    )
                    del prepared, score_timesteps_B, losses, loss

            with sl.log_trace_span("wan_self_forcing_generator_optim"):
                generator_grad_norm = dist_utils.clip_grad_norm_(
                    optimizers.parameters_for("generator"),
                    self.config.training.max_norm,
                    foreach=True,
                    pp_mesh=parallel_dims.get_optional_mesh("pp"),
                    ep_enabled=parallel_dims.ep_enabled,
                )
                self.checkpointer.maybe_wait_for_staging()
                waited_for_staging = True
                optimizers.step(role="generator")
                lr_schedulers.step("generator")
                optimizers.zero_grad(role="generator")
        else:
            for name in (
                "dmd_loss",
                "dmd_gradient_norm",
                "dmd_normalizer",
                "generator_rollout_exit_timestep",
            ):
                metric_sums[name] = zero

        metric_sums["generator_update"] = local_samples_float if update_generator else zero

        # The critic always trains on a new rollout. On generator-update steps,
        # this happens after the generator optimizer has already changed its
        # parameters, matching standard alternating DMD training semantics.
        critic_exit_step_index = sample_self_forcing_exit_step(
            self.rollout_scheduler,
            device=self.device,
        )
        with sl.log_trace_span("wan_self_forcing_critic_fwd_bwd"):
            for input_dict, targets in microbatches:
                prepared = self._prepare_self_forcing_batch(
                    model=model,
                    input_dict=input_dict,
                    targets=targets,
                )
                score_timesteps_B = self._sample_score_timesteps(prepared.latents_BFCHW.size(0))
                with self.train_context():
                    losses = self_forcing_critic_losses(
                        model=model,
                        latents_BFCHW=prepared.latents_BFCHW,
                        frame_indices_BF=prepared.frame_indices_BF,
                        text_context_BLC=prepared.text_context_BLC,
                        rollout_scheduler=self.rollout_scheduler,
                        rollout_exit_step_index=critic_exit_step_index,
                        score_timesteps_B=score_timesteps_B,
                    )
                    loss = self.config.fake_score_loss_weight * losses.fake_score_loss_B.sum() / local_samples_float
                    loss.backward()
                accumulated_losses.append(loss.detach())
                accumulate_metric("fake_score_loss", losses.fake_score_loss_B)
                accumulate_metric(
                    "critic_rollout_exit_timestep",
                    losses.rollout_exit_timestep_B,
                )
                del prepared, score_timesteps_B, losses, loss

        with sl.log_trace_span("wan_self_forcing_critic_optim"):
            fake_score_grad_norm = dist_utils.clip_grad_norm_(
                optimizers.parameters_for("fake_score"),
                self.config.training.max_norm,
                foreach=True,
                pp_mesh=parallel_dims.get_optional_mesh("pp"),
                ep_enabled=parallel_dims.ep_enabled,
            )
            if not waited_for_staging:
                self.checkpointer.maybe_wait_for_staging()
            optimizers.step(role="fake_score")
            lr_schedulers.step("fake_score")
            optimizers.zero_grad(role="fake_score")

        self.unique_segment_counter.update(step_segment_names)

        if not self.metrics_processor.should_log(self.step):
            return

        loss = torch.sum(torch.stack(accumulated_losses))
        if parallel_dims.dp_cp_enabled:
            loss_mesh = parallel_dims.get_optional_mesh("loss")
            global_avg_loss = dist_utils.dist_sum(loss * local_samples_float, loss_mesh) / global_samples
            global_max_loss = dist_utils.dist_max(loss.detach(), loss_mesh)
            global_ntokens_seen = dist_utils.dist_sum(
                torch.tensor(
                    self.ntokens_seen,
                    dtype=torch.int64,
                    device=self.device,
                ),
                loss_mesh,
            )
            metric_values = {
                name: dist_utils.dist_sum(value, loss_mesh) / global_samples for name, value in metric_sums.items()
            }
        else:
            global_avg_loss = global_max_loss = float(loss.detach().item())
            global_ntokens_seen = self.ntokens_seen
            metric_values = {name: float((value / global_samples).item()) for name, value in metric_sums.items()}

        combined_grad_norm = torch.sqrt(generator_grad_norm.float().square() + fake_score_grad_norm.float().square())
        extra_metrics: dict[str, Any] = {
            "n_tokens_seen": global_ntokens_seen,
            **lr_metrics,
            **{f"worldmodel/{name}": value for name, value in metric_values.items()},
            "worldmodel/generator_grad_norm": float(generator_grad_norm.item()),
            "worldmodel/fake_score_grad_norm": float(fake_score_grad_norm.item()),
            "dataset/unique_segments_seen": (
                self.unique_segment_counter.global_count(batch_mesh.get_group())
                if batch_mesh is not None
                else self.unique_segment_counter.local_count()
            ),
        }
        stats = self.dataloader.stats() if hasattr(self.dataloader, "stats") else None
        if stats is not None:
            extra_metrics.update(
                {
                    "dataloader/shuffle_full": stats.full,
                    "dataloader/shuffle_empty": stats.empty,
                    "dataloader/shuffle_in_flight": stats.in_flight,
                }
            )
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            float(combined_grad_norm.item()),
            extra_metrics=extra_metrics,
        )


def _validate_wan_self_forcing_config(
    config: WanSelfForcingTrainer.Config,
) -> None:
    _validate_wanworldmodel_config(config)
    model_config = config.model_spec.model
    if not isinstance(model_config, WanSelfForcingModel.Config):
        raise TypeError("Wan Self-Forcing model_spec must contain WanSelfForcingModel.Config")
    if not isinstance(config.optimizer, WanSelfForcingOptimizers.Config):
        raise TypeError("Wan Self-Forcing requires WanSelfForcingOptimizers.Config")
    if not isinstance(config.lr_scheduler, WanSelfForcingLRSchedulers.Config):
        raise TypeError("Wan Self-Forcing requires WanSelfForcingLRSchedulers.Config")
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
    if config.generator_update_interval <= 0:
        raise ValueError("Self-Forcing generator_update_interval must be positive")
    if config.load_teacher_from_hf and not config.teacher_checkpoint_path:
        raise ValueError("Self-Forcing teacher_checkpoint_path is required for HF loading")
    if config.validator.enable:
        raise ValueError("Minimal Self-Forcing does not define an offline validation loss")
    source_latent_frames = 1 + (config.dataloader.clip_frames - 1) // 4
    if source_latent_frames >= model_config.rope_max_seq_len:
        raise ValueError(
            "Wan Self-Forcing's second generated latent requires temporal RoPE "
            f"position {source_latent_frames}, but rope_max_seq_len is "
            f"{model_config.rope_max_seq_len}"
        )


__all__ = ["WanSelfForcingTrainer"]
