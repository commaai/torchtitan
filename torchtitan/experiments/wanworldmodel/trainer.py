# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, cast

import torch

from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.experiments.wan_vae.tokenizer import WanVAETokenizer
from torchtitan.experiments.worldmodel.schedulers import RFScheduler
from torchtitan.experiments.worldmodel.trainer import (
    WorldModelTrainer,
    WorldModelValidator,
)
from torchtitan.observability import structured_logger as sl
from torchtitan.trainer import Trainer

from .dataset import WanWorldModelDataLoader
from .model import WanModel
from .torchpackage_checkpoint import (
    WanTorchPackageConfig,
    WanTorchPackageCheckpointManager,
)


WAN_WINDOWS_PER_SOURCE_STEP = 2


def _batch_size(inputs: dict[str, torch.Tensor]) -> int:
    return next(iter(inputs.values())).shape[0]


def _model_batch_size(inputs: dict[str, torch.Tensor]) -> int:
    batch_size = _batch_size(inputs)
    if "imgs" in inputs and "big_imgs" in inputs:
        return WanVAETokenizer.NUM_VIEWS * batch_size
    return batch_size


def _select_wan_sink_window(
    latents_BFCHW: torch.Tensor,
    *,
    inference_prefill_frames: int,
) -> torch.Tensor:
    """Select the maximum-distance Wan window for validation and direct calls."""
    batch_size, source_frames = latents_BFCHW.shape[:2]
    if batch_size % WanVAETokenizer.NUM_VIEWS:
        raise ValueError("Wan sink-window selection requires paired fcam/ecam latent batches")
    if inference_prefill_frames <= 0:
        raise ValueError("Wan sink-window selection requires a positive prefill")

    continuation_frames = inference_prefill_frames
    max_recent_start = source_frames - continuation_frames
    if max_recent_start < 1:
        raise ValueError(
            f"continuous Wan stream has {source_frames} latents, but sink plus "
            f"{continuation_frames} continuation latents are required"
        )

    recent_start_B = torch.full(
        (batch_size,),
        max_recent_start,
        device=latents_BFCHW.device,
        dtype=torch.int64,
    )
    recent_indices_BF = recent_start_B[:, None] + torch.arange(
        continuation_frames,
        device=latents_BFCHW.device,
    )
    frame_indices_BF = torch.cat(
        (
            torch.zeros(
                (batch_size, 1),
                device=latents_BFCHW.device,
                dtype=torch.int64,
            ),
            recent_indices_BF,
        ),
        dim=1,
    )
    batch_indices_BF = torch.arange(
        batch_size,
        device=latents_BFCHW.device,
    )[:, None]
    return latents_BFCHW[batch_indices_BF, frame_indices_BF]


def _iter_wan_sink_window_batches(
    latents_BFCHW: torch.Tensor,
    *,
    inference_prefill_frames: int,
) -> Iterator[torch.Tensor]:
    """Yield every sink window once in fixed-size shuffled training batches."""
    if latents_BFCHW.ndim != 5:
        raise ValueError(f"Wan sink-window batching requires [B, F, C, H, W] latents, got {tuple(latents_BFCHW.shape)}")
    batch_size, source_frames = latents_BFCHW.shape[:2]
    if batch_size % WanVAETokenizer.NUM_VIEWS:
        raise ValueError("Wan sink-window batching requires paired fcam/ecam latent batches")
    if inference_prefill_frames <= 0:
        raise ValueError("Wan sink-window batching requires a positive prefill")

    continuation_frames = inference_prefill_frames
    max_recent_start = source_frames - continuation_frames
    if max_recent_start < 1:
        raise ValueError(
            f"continuous Wan stream has {source_frames} latents, but sink plus "
            f"{continuation_frames} continuation latents are required"
        )
    logical_batch_size = batch_size // WanVAETokenizer.NUM_VIEWS
    shuffled_starts_BS = torch.stack(
        [torch.randperm(max_recent_start, device=latents_BFCHW.device) + 1 for _ in range(logical_batch_size)]
    )
    # The tokenizer orders all fcam samples before all ecam samples. Reuse each
    # clip's window order for its paired view.
    shuffled_starts_BS = shuffled_starts_BS.repeat(
        WanVAETokenizer.NUM_VIEWS,
        1,
    )
    continuation_offsets_F = torch.arange(
        continuation_frames,
        device=latents_BFCHW.device,
    )
    batch_indices_B = torch.arange(batch_size, device=latents_BFCHW.device)

    for offset in range(0, max_recent_start, WAN_WINDOWS_PER_SOURCE_STEP):
        recent_starts_BK = shuffled_starts_BS[:, offset : offset + WAN_WINDOWS_PER_SOURCE_STEP]
        windows_per_source = recent_starts_BK.shape[1]
        recent_indices_BKF = recent_starts_BK[:, :, None] + continuation_offsets_F
        frame_indices_BKF = torch.cat(
            (
                torch.zeros(
                    (batch_size, windows_per_source, 1),
                    device=latents_BFCHW.device,
                    dtype=torch.int64,
                ),
                recent_indices_BKF,
            ),
            dim=2,
        )
        windows_BKFCHW = latents_BFCHW[
            batch_indices_B[:, None, None],
            frame_indices_BKF,
        ]
        yield windows_BKFCHW.flatten(0, 1).contiguous()


def _prepare_wan_batch(
    *,
    model: WanModel,
    tokenizer: WanVAETokenizer,
    input_dict: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    scheduler: RFScheduler,
    discrete_timesteps: torch.Tensor,
    inference_prefill_frames: int,
    future_size_frames: int,
    no_noise_prefill_frames_prob: float,
    fake_timesteps_prob: float,
    train: bool,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    del targets
    latents_BFCHW = tokenizer.encode(input_dict, device=device, dtype=dtype)
    if latents_BFCHW.ndim != 5:
        raise ValueError(f"Wan latents must have shape [B, F, C, H, W], got {tuple(latents_BFCHW.shape)}")
    latents_BFCHW = _select_wan_sink_window(
        latents_BFCHW,
        inference_prefill_frames=inference_prefill_frames,
    )
    batch_size, num_frames, channels, height, width = latents_BFCHW.shape
    if not 0 < inference_prefill_frames < num_frames:
        raise ValueError(
            "Wan inference_prefill_frames must retain the sink and leave a target; "
            f"got {inference_prefill_frames} for {num_frames} model frames"
        )
    if channels != model.in_dim:
        raise ValueError(f"Wan model expects {model.in_dim} latent channels, got {channels}")

    latent_shape = (num_frames, height, width)
    if any(size % patch for size, patch in zip(latent_shape, model.patch_size)):
        raise ValueError(f"Wan latent shape {latent_shape} must be divisible by patch size {model.patch_size}")
    seq_len = 1
    for size, patch in zip(latent_shape, model.patch_size):
        seq_len *= size // patch

    with torch.no_grad():
        noise_BFCHW = torch.randn_like(latents_BFCHW)
        if train:
            timesteps_B = scheduler.sample_timestep((batch_size,))
        else:
            indexes_B = torch.randint(
                0,
                discrete_timesteps.numel(),
                (batch_size,),
                device=device,
            )
            timesteps_B = discrete_timesteps[indexes_B]
        timesteps_BF = timesteps_B[:, None].expand(batch_size, num_frames).clone()
        mask_BFCHW = torch.ones_like(
            latents_BFCHW,
            device=device,
            dtype=torch.bool,
        )
        noise_timesteps_BF = timesteps_BF.clone()
        if torch.rand((), device=device) < no_noise_prefill_frames_prob:
            end = min(inference_prefill_frames, num_frames)
            mask_BFCHW[:, :end] = False
            timesteps_BF[:, :end] = scheduler.no_noise_timestep
            noise_timesteps_BF[:, :end] = scheduler.no_noise_timestep
            start = min(future_size_frames, end)
            if start < end and torch.rand((), device=device) < fake_timesteps_prob:
                noise_timesteps_BF[:, start:end] = scheduler.sample_timestep((batch_size, end - start))
                mask_BFCHW[:, start:] = True

        # z0 is the persistent one-frame attention sink. It is always exact
        # clean conditioning and is never a direct flow-matching target.
        timesteps_BF[:, 0] = scheduler.no_noise_timestep
        noise_timesteps_BF[:, 0] = scheduler.no_noise_timestep
        mask_BFCHW[:, 0] = False
        noisy_latents_BFCHW = scheduler.add_noise(
            latents_BFCHW,
            noise_BFCHW,
            noise_timesteps_BF,
        )
        prepared_targets = {
            # Wan predicts epsilon-x_0 for x_t=(1-t)*x_0+t*epsilon.
            "v": noise_BFCHW - latents_BFCHW,
            "mask": mask_BFCHW,
        }

        text_context_BLC = tokenizer.fixed_text_context(
            batch_size,
            expected_dim=model.text_dim,
            device=device,
            dtype=dtype,
        )
        if text_context_BLC is None:
            text_context_BLC = model.get_null_text_embedding(
                batch_size,
                device=device,
                dtype=dtype,
            )

    noisy_latents_BCFHW = noisy_latents_BFCHW.permute(0, 2, 1, 3, 4)
    return {
        "x": list(noisy_latents_BCFHW.unbind(0)),
        # Wan conditions on [0, 1000]; the scheduler and target use [0, 1].
        "t": timesteps_BF * 1000.0,
        "context": list(text_context_BLC.unbind(0)),
        "seq_len": seq_len,
    }, prepared_targets


def _loss_outputs(outputs: Any) -> dict[str, torch.Tensor]:
    if not isinstance(outputs, list) or not outputs:
        raise TypeError(f"unsupported Wan output type {type(outputs).__name__}")
    if any(sample_CFH.ndim != 4 for sample_CFH in outputs):
        raise ValueError("Wan outputs must contain [C, F, H, W] tensors")
    sample_BCFHW = torch.stack(outputs)
    return {"sample": sample_BCFHW.permute(0, 2, 1, 3, 4)}


class WanWorldModelValidator(WorldModelValidator):
    @dataclass(kw_only=True, slots=True)
    class Config(WorldModelValidator.Config):
        dataloader: WanWorldModelDataLoader.Config

    def _model_batch_size(self, inputs: dict[str, torch.Tensor]) -> int:
        return _model_batch_size(inputs)

    def _prepare_batch(
        self,
        *,
        model: WanModel,
        input_dict: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        scheduler: RFScheduler,
        discrete_timesteps: torch.Tensor,
        train: bool,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        return _prepare_wan_batch(
            model=model,
            tokenizer=cast(WanVAETokenizer, self.tokenizer),
            input_dict=input_dict,
            targets=targets,
            device=device,
            dtype=dtype,
            scheduler=scheduler,
            discrete_timesteps=discrete_timesteps,
            inference_prefill_frames=self.config.dataloader.inference_prefill_frames,
            future_size_frames=self.config.dataloader.future_size_frames,
            no_noise_prefill_frames_prob=self.config.no_noise_prefill_frames_prob,
            fake_timesteps_prob=self.config.fake_timesteps_prob,
            train=train,
        )

    def _normalize_outputs(self, outputs: Any) -> dict[str, torch.Tensor]:
        return _loss_outputs(outputs)


class WanWorldModelTrainer(WorldModelTrainer):
    @dataclass(kw_only=True, slots=True)
    class Config(WorldModelTrainer.Config):
        dataloader: WanWorldModelDataLoader.Config
        tokenizer: WanVAETokenizer.Config
        validator: WanWorldModelValidator.Config
        checkpoint: WanTorchPackageCheckpointManager.Config

        def __post_init__(self) -> None:
            Trainer.Config.__post_init__(self)
            _validate_wanworldmodel_config(self)

    @staticmethod
    def _apply_float8(config: Config, model_config: WanModel.Config) -> None:
        from .model_config import _wan_blocks_only_float8

        model_compile_enabled = config.compile.enable and "model" in config.compile.components
        converter = _wan_blocks_only_float8(
            model_compile_enabled=model_compile_enabled,
            emulate=config.float8.emulate,
        )
        converter.build().convert(model_config)

    def _build_package_config(self, model_config: WanModel.Config) -> Any:
        tokenizer = cast(WanVAETokenizer, self.tokenizer)
        text_context_BLC = tokenizer.fixed_text_context(
            1,
            expected_dim=model_config.text_dim,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        return WanTorchPackageConfig(
            model_config=model_config,
            text_context_LC=(text_context_BLC[0].clone() if text_context_BLC is not None else None),
            text_prompt=tokenizer.config.text_prompt,
        )

    def _model_batch_size(self, inputs: dict[str, torch.Tensor]) -> int:
        return _model_batch_size(inputs)

    def _prepare_batch(
        self,
        *,
        model: WanModel,
        input_dict: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        train: bool,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        return _prepare_wan_batch(
            model=model,
            tokenizer=cast(WanVAETokenizer, self.tokenizer),
            input_dict=input_dict,
            targets=targets,
            device=self.device,
            dtype=self.dtype,
            scheduler=self.train_noise_scheduler,
            discrete_timesteps=self.discrete_timesteps,
            inference_prefill_frames=self.config.dataloader.inference_prefill_frames,
            future_size_frames=self.config.dataloader.future_size_frames,
            no_noise_prefill_frames_prob=self.config.no_noise_prefill_frames_prob,
            fake_timesteps_prob=self.config.fake_timesteps_prob,
            train=train,
        )

    def _normalize_outputs(self, outputs: Any) -> dict[str, torch.Tensor]:
        return _loss_outputs(outputs)

    def batch_generator(
        self,
        data_iterable: Iterable[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]],
    ) -> Iterator[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]:
        """Encode a continuous stream once, then drain every temporal window."""
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
                raise ValueError("Wan continuous-window training does not use dataset targets")

            with sl.log_trace_span("wan_encode_continuous_stream"):
                encoded_BFCHW = tokenizer.encode(
                    input_dict,
                    device=self.device,
                    dtype=self.dtype,
                )
                # Keep the continuous stream on CPU while its shuffled windows
                # are consumed so activation memory does not scale with it.
                source_latents_BFCHW = encoded_BFCHW.detach().to(device="cpu")
            del encoded_BFCHW, input_dict, targets

            for windowed_latents_BFCHW in _iter_wan_sink_window_batches(
                source_latents_BFCHW,
                inference_prefill_frames=inference_prefill_frames,
            ):
                self.metrics_processor.data_loading_times.append(source_data_loading_time)
                source_data_loading_time = 0.0
                windowed_inputs = {"latents": windowed_latents_BFCHW}
                self.metrics_processor.ntokens_since_last_log += (
                    self._model_batch_size(windowed_inputs) * self.config.training.seq_len
                )
                yield windowed_inputs, {}


def _validate_wanworldmodel_config(config: WanWorldModelTrainer.Config) -> None:
    if config.model_spec is None:
        raise ValueError("Wan worldmodel requires a model_spec")
    model_config = config.model_spec.model
    if not isinstance(model_config, WanModel.Config):
        raise TypeError("Wan worldmodel model_spec must contain WanModel.Config")
    if (
        config.parallelism.tensor_parallel_degree > 1
        or config.parallelism.pipeline_parallel_degree > 1
        or config.parallelism.context_parallel_degree > 1
        or config.parallelism.expert_parallel_degree > 1
    ):
        raise ValueError("Wan worldmodel supports FSDP/HSDP only")
    if not isinstance(config.dataloader, WanWorldModelDataLoader.Config):
        raise TypeError("Wan worldmodel requires WanWorldModelDataLoader.Config")
    if not isinstance(config.tokenizer, WanVAETokenizer.Config):
        raise TypeError("Wan worldmodel requires WanVAETokenizer.Config")
    if not config.dataloader.mock_latents and not config.tokenizer.compressor_model:
        raise ValueError("Wan image training requires a pretrained VAE checkpoint")
    if config.tokenizer.image_size != config.dataloader.image_size:
        raise ValueError("Wan tokenizer and dataloader image_size must match")
    if model_config.in_dim != config.dataloader.latent_channels:
        raise ValueError("Wan model in_dim must match dataloader latent_channels")
    if model_config.out_dim != config.dataloader.latent_channels:
        raise ValueError("Wan model out_dim must match dataloader latent_channels")
    if not 0.0 <= config.no_noise_prefill_frames_prob <= 1.0:
        raise ValueError("no_noise_prefill_frames_prob must be in [0, 1]")
    if not 0.0 <= config.fake_timesteps_prob <= 1.0:
        raise ValueError("fake_timesteps_prob must be in [0, 1]")

    model_rgb_frames = config.dataloader.context_size_frames + config.dataloader.future_size_frames
    model_latent_frames = 1 + (model_rgb_frames - 1) // 4
    latent_shape = (model_latent_frames, *config.dataloader.latent_size)
    if any(size % patch for size, patch in zip(latent_shape, model_config.patch_size)):
        raise ValueError(f"Wan latent shape {latent_shape} must be divisible by patch size {model_config.patch_size}")
    seq_len = 1
    for size, patch in zip(latent_shape, model_config.patch_size):
        seq_len *= size // patch
    model_config._sync_derived_fields()
    config.training.seq_len = seq_len
