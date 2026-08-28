# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import torch

from torchtitan.experiments.wan_vae.dataset import NATIVE_FPS, WanVAEDataLoader


class _MockWanWorldModelDataset:
    def __init__(self, config: "WanWorldModelDataLoader.Config") -> None:
        self.config = config
        self.segments = list(range(1024))

    def __iter__(self) -> Iterator[tuple[dict[str, np.ndarray]]]:
        rng = np.random.default_rng(0)
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
    """Expose continuous native-rate Wan clips through the trainer batch ABI."""

    @dataclass(kw_only=True, slots=True)
    class Config(WanVAEDataLoader.Config):
        clip_frames: int = field(init=False, default=0)
        context_size_frames: int = 41
        future_size_frames: int = 0
        # Maximum source-frame distance from z0 to the causal endpoint of the
        # first selected continuation latent.
        max_sink_distance_frames: int = 10 * NATIVE_FPS
        # Retain z0 and nine recent latents; the eleventh latent is the target.
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
                            f"mock Wan latents must have shape [B, 2, F, C, H, W], got {tuple(latents_BVFCHW.shape)}"
                        )
                    latents_BFCHW = latents_BVFCHW.permute(1, 0, 2, 3, 4, 5).flatten(0, 1)
                    yield {"latents": latents_BFCHW}, {}
                    continue

                videos_BVCTHW = inputs["input"]
                if videos_BVCTHW.ndim != 6 or videos_BVCTHW.shape[1:3] != (2, 3):
                    raise ValueError(
                        f"Wan worldmodel videos must have shape [B, 2, 3, T, H, W], got {tuple(videos_BVCTHW.shape)}"
                    )
                imgs_BTHWC = videos_BVCTHW[:, 0].permute(0, 2, 3, 4, 1)
                big_imgs_BTHWC = videos_BVCTHW[:, 1].permute(0, 2, 3, 4, 1)
                yield {"imgs": imgs_BTHWC, "big_imgs": big_imgs_BTHWC}, {}
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
            if self._iterator is iterator:
                self._iterator = None
