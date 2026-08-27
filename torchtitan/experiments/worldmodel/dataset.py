# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.experiments.wan_vae.dataset import NATIVE_FPS, WanVAEDataLoader


PLAN_SIZE = 15 * 33 * 2


@dataclass(slots=True)
class _DiffusionConfig:
    base_dir: str
    feature_dir: str | None
    compressor_model: str
    context_size_frames: int
    future_size_frames: int
    max_future_frames: int
    inference_prefill_frames: int
    fps: int
    train_skip: int
    val_skip: int
    nan_engaged_plans: bool

    def skip(self, val: bool) -> int:
        return self.val_skip if val else self.train_skip


class _MockDataset:
    def __init__(self, config: WorldModelDataLoader.Config) -> None:
        self.config = config
        self.segments = list(range(1024))

    def __iter__(self) -> Iterator[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]:
        while True:
            for _ in self.segments:
                yield self._mock_batch()

    def _mock_batch(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        cfg = self.config
        batch = cfg.mock_segment_batch_size
        frames = cfg.context_size_frames + cfg.future_size_frames
        latent_h, latent_w = cfg.latent_size
        inputs: dict[str, np.ndarray]
        if cfg.feature_dir is None:
            height, width = cfg.image_size
            inputs = {
                "imgs": np.random.randint(0, 256, (batch, frames, height, width, 3), dtype=np.uint8),
                "big_imgs": np.random.randint(0, 256, (batch, frames, height, width, 3), dtype=np.uint8),
            }
        else:
            inputs = {
                "latents": np.random.randn(batch, frames, cfg.in_channels, latent_h, latent_w).astype(np.float32),
            }

        inputs.update(
            {
                "augments_pos_ref_augment": np.random.randn(batch, frames, 3).astype(np.float32),
                "ref_augment_from_augments_euler": np.random.randn(batch, frames, 3).astype(np.float32),
                "fidxs": np.tile(np.arange(frames, dtype=np.int64), (batch, 1)),
                "info": np.zeros((batch, 512), dtype=np.uint8),
            }
        )
        targets = {"plan": np.random.randn(batch, PLAN_SIZE).astype(np.float32)}
        return inputs, targets


class WorldModelDataLoader(BaseDataLoader):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        dataset: str
        dataset_path: str | None
        split: Literal["train", "val"]
        shuffle_size: int
        min_mixing: float
        num_writers: int
        num_readers: int
        fill_once: bool
        base_dir: str
        feature_dir: str | None
        compressor_model: str
        in_channels: int
        latent_size: tuple[int, int]
        image_size: tuple[int, int]
        context_size_frames: int
        future_size_frames: int
        max_future_frames: int
        inference_prefill_frames: int
        fps: int
        train_skip: int
        val_skip: int
        nan_engaged_plans: bool
        limit: int | None
        mock_data: bool
        mock_segment_batch_size: int

        def __post_init__(self) -> None:
            total_frames = self.context_size_frames + self.future_size_frames
            if total_frames <= 0:
                raise ValueError("context_size_frames + future_size_frames must be positive")
            if self.inference_prefill_frames > total_frames:
                raise ValueError("inference_prefill_frames must fit in total frames")
            if self.shuffle_size <= 0:
                raise ValueError("shuffle_size must be positive")

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        seq_len: int,
        local_batch_size: int,
        snapshot_every_n_steps: int | None = 1,
        validation_steps: int = 1,
        **kwargs: Any,
    ) -> None:
        del tokenizer, seq_len, snapshot_every_n_steps, kwargs
        from xx.training.lib.dataloader import DataLoader

        from gigashuffle import DataloaderConfig

        self.config = config
        self.local_batch_size = local_batch_size
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.local_rank = int(os.environ.get("LOCAL_RANK", dp_rank))
        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", dp_world_size))
        node_rank = int(os.environ.get("GROUP_RANK", dp_rank // max(1, self.local_world_size)))
        run_id = os.environ.get("REPORTERV2_TRAINING_ID") or "worldmodel"
        val = config.split == "val"
        shuffle_size = local_batch_size * validation_steps * self.local_world_size if val else config.shuffle_size

        self.dataset = self._build_dataset(config, val=val, global_rank=dp_rank, global_world_size=dp_world_size)
        self.loader = DataLoader(
            self.dataset,
            DataloaderConfig(
                bs=local_batch_size,
                shuffle_size=shuffle_size,
                min_mixing=1 if val else config.min_mixing,
                num_writers=1 if val else config.num_writers,
                num_readers=1 if val else config.num_readers,
                fill_once=config.fill_once or val,
                local_rank=self.local_rank,
                global_rank=dp_rank,
                local_world_size=self.local_world_size,
                global_world_size=dp_world_size,
                queue_name=f"{run_id}-{config.split}-node{node_rank}",
            ),
        )
        self._iterator: Any | None = None

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]:
        iterator = iter(self.loader)
        self._iterator = iterator
        try:
            for inputs, targets in iterator:
                yield inputs, targets
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
            if self._iterator is iterator:
                self._iterator = None

    def close(self) -> None:
        if self._iterator is not None:
            close = getattr(self._iterator, "close", None)
            if callable(close):
                close()
            self._iterator = None
        shutdown = getattr(self.loader, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()

    def state_dict(self) -> dict[str, int]:
        return {}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        return

    def stats(self) -> Any:
        return self.loader.stats()

    @staticmethod
    def _build_dataset(config: Config, *, val: bool, global_rank: int, global_world_size: int):
        if config.mock_data:
            return _MockDataset(config)

        from xx.common.training_helpers import train_and_test_targets_from_file
        from xx.training.diffusion.dataloader import get_data_from_seg, IGNORE_EXCEPTIONS
        from xx.training.lib.dataloader import GenericDataset

        train_segments, val_segments = train_and_test_targets_from_file(
            config.dataset_path or config.dataset, limit=config.limit
        )
        segments = val_segments if val else train_segments
        np.random.seed(42)
        np.random.shuffle(segments)
        return GenericDataset(
            segments=segments,
            get_data_from_seg=get_data_from_seg,
            config=_DiffusionConfig(
                base_dir=config.base_dir,
                feature_dir=config.feature_dir,
                compressor_model=config.compressor_model,
                context_size_frames=config.context_size_frames,
                future_size_frames=config.future_size_frames,
                max_future_frames=config.max_future_frames,
                inference_prefill_frames=config.inference_prefill_frames,
                fps=config.fps,
                train_skip=config.train_skip,
                val_skip=config.val_skip,
                nan_engaged_plans=config.nan_engaged_plans,
            ),
            val=val,
            local_rank=int(os.environ.get("LOCAL_RANK", "0")),
            ignore_exceptions=IGNORE_EXCEPTIONS,
            global_rank=global_rank,
            global_world_size=global_world_size,
        )


class _MockWanWorldModelDataset:
    def __init__(self, config: "WanWorldModelDataLoader.Config") -> None:
        self.config = config
        self.segments = list(range(1024))

    def __iter__(self) -> Iterator[tuple[dict[str, np.ndarray]]]:
        rng = np.random.default_rng(0)
        # Match the real path: produce the complete continuous VAE stream and
        # select the model-facing sink window after encoding.
        latent_frames = 1 + (self.config.clip_frames - 1) // 4
        latent_height, latent_width = self.config.latent_size
        shape = (
            2 * self.config.mock_segment_batch_size,
            latent_frames,
            self.config.latent_channels,
            latent_height,
            latent_width,
        )
        while True:
            for _ in self.segments:
                latents = rng.standard_normal(shape, dtype=np.float32)
                yield ({"latents": latents},)


class WanWorldModelDataLoader(WanVAEDataLoader):
    """Expose continuous native-rate Wan clips through the worldmodel batch ABI."""

    @dataclass(kw_only=True, slots=True)
    class Config(WanVAEDataLoader.Config):
        clip_frames: int = field(init=False, default=0)
        context_size_frames: int = 41
        future_size_frames: int = 0
        # Maximum source-frame distance from z0 to the causal endpoint of the
        # first selected continuation latent.
        max_sink_distance_frames: int = 10 * NATIVE_FPS
        # The model always retains latent zero as a one-frame sink, followed by
        # recent four-frame latent groups. Ten prefix latents leave one real
        # four-frame target in the default 41-frame supervised span.
        inference_prefill_frames: int = 10
        latent_channels: int = 48
        latent_size: tuple[int, int] = (16, 32)
        mock_latents: bool = False

        def __post_init__(self) -> None:
            model_rgb_frames = self.context_size_frames + self.future_size_frames
            if self.context_size_frames <= 0:
                raise ValueError("context_size_frames must be positive")
            if self.future_size_frames < 0:
                raise ValueError("future_size_frames must be non-negative")
            if self.max_sink_distance_frames < 4 or self.max_sink_distance_frames % 4:
                raise ValueError("max_sink_distance_frames must be a positive multiple of 4")
            if model_rgb_frames < 1 or (model_rgb_frames - 1) % 4:
                raise ValueError("context_size_frames + future_size_frames must equal 1 + 4*n")
            self.clip_frames = self.max_sink_distance_frames + model_rgb_frames - 4
            WanVAEDataLoader.Config.__post_init__(self)
            model_latent_frames = 1 + (model_rgb_frames - 1) // 4
            if self.inference_prefill_frames != model_latent_frames - 1:
                raise ValueError(
                    "inference_prefill_frames must retain the Wan sink and leave "
                    "exactly one latent target; "
                    f"got {self.inference_prefill_frames} for "
                    f"{model_latent_frames} model latent frames"
                )
            if self.latent_channels <= 0:
                raise ValueError("latent_channels must be positive")
            if any(size <= 0 for size in self.latent_size):
                raise ValueError("latent_size values must be positive")
            expected_latent_size = tuple(size // 16 for size in self.image_size)
            if not self.mock_latents and self.latent_size != expected_latent_size:
                raise ValueError(f"latent_size {self.latent_size} does not match image_size {self.image_size}")
            if self.mock_latents and not self.mock_data:
                raise ValueError("mock_latents requires mock_data")

    @staticmethod
    def _build_dataset(
        config: Config,
        *,
        val: bool,
        global_rank: int,
        global_world_size: int,
    ):
        if config.mock_latents:
            return _MockWanWorldModelDataset(config)
        return WanVAEDataLoader._build_dataset(
            config,
            val=val,
            global_rank=global_rank,
            global_world_size=global_world_size,
        )

    def __iter__(
        self,
    ) -> Iterator[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]:
        iterator = iter(self.loader)
        self._iterator = iterator
        try:
            for inputs in iterator:
                if "latents" in inputs:
                    latents_BVFCHW = inputs["latents"]
                    if latents_BVFCHW.ndim != 6 or latents_BVFCHW.shape[1] != 2:
                        raise ValueError(
                            "mock Wan latents must have shape [B, 2, F, C, H, W], "
                            f"got {tuple(latents_BVFCHW.shape)}"
                        )
                    latents = latents_BVFCHW.permute(1, 0, 2, 3, 4, 5).flatten(0, 1)
                    yield {"latents": latents}, {}
                    continue

                videos_BVCTHW = inputs["input"]
                if videos_BVCTHW.ndim != 6 or videos_BVCTHW.shape[1:3] != (2, 3):
                    raise ValueError(
                        f"Wan worldmodel videos must have shape [B, 2, 3, T, H, W], got {tuple(videos_BVCTHW.shape)}"
                    )
                imgs_BTHWC = videos_BVCTHW[:, 0].permute(0, 2, 3, 4, 1)
                big_imgs_BTHWC = videos_BVCTHW[:, 1].permute(0, 2, 3, 4, 1)
                yield (
                    {
                        "imgs": imgs_BTHWC,
                        "big_imgs": big_imgs_BTHWC,
                    },
                    {},
                )
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
            if self._iterator is iterator:
                self._iterator = None


def main() -> None:
    from xx.comma_data.constants import DEFAULT_TRAIN_LIST

    from torchtitan.experiments.worldmodel.dataset_config import _dataloader_config

    config = _dataloader_config(split="train", dataset=DEFAULT_TRAIN_LIST)
    dataset = WorldModelDataLoader._build_dataset(config, val=False, global_rank=0, global_world_size=1)
    inputs, targets = next(iter(dataset))
    print(
        {
            "dataset": config.dataset_path or config.dataset,
            "inputs": {key: (tuple(value.shape), str(value.dtype)) for key, value in inputs.items()},
            "targets": {key: (tuple(value.shape), str(value.dtype)) for key, value in targets.items()},
        }
    )


if __name__ == "__main__":
    main()
