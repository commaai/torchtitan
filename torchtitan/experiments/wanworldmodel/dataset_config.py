# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from xx.comma_data.constants import BASE_DIR_GT, DEFAULT_TRAIN_LIST

from torchtitan.experiments.wan_vae.dataset import DEFAULT_IMAGE_SIZE, NATIVE_FPS

from .dataset import WanWorldModelDataLoader


def _dataloader_config(
    *,
    split: str,
    dataset: str = DEFAULT_TRAIN_LIST,
    dataset_path: str | None = None,
    pipeline_dir: str = BASE_DIR_GT,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    latent_channels: int = 48,
    latent_size: tuple[int, int] = (16, 32),
    context_size_frames: int = 41,
    future_size_frames: int = 0,
    max_sink_distance_frames: int = 10 * NATIVE_FPS,
    inference_prefill_frames: int = 10,
    limit: int | None = None,
    mock_data: bool = False,
    mock_segment_batch_size: int = 1,
    mock_latents: bool = False,
) -> WanWorldModelDataLoader.Config:
    return WanWorldModelDataLoader.Config(
        dataset=dataset,
        dataset_path=dataset_path,
        split=split,
        pipeline_dir=pipeline_dir,
        image_size=image_size,
        context_size_frames=context_size_frames,
        future_size_frames=future_size_frames,
        max_sink_distance_frames=max_sink_distance_frames,
        inference_prefill_frames=inference_prefill_frames,
        fps=NATIVE_FPS,
        clips_per_segment=1,
        limit=limit,
        mock_data=mock_data,
        mock_segment_batch_size=mock_segment_batch_size,
        latent_channels=latent_channels,
        latent_size=latent_size,
        mock_latents=mock_latents,
    )
