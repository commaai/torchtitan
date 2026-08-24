# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from xx.comma_data.constants import BASE_DIR_GT, DEFAULT_TRAIN_LIST

from torchtitan.experiments.wan_vae.dataset import DEFAULT_IMAGE_SIZE, NATIVE_FPS

from .dataset import WanWorldModelDataLoader, WorldModelDataLoader
from .model_config import COMPRESSOR_MODEL, LATENT_CHANNELS, LATENT_SIZE

IMAGE_SIZE = (128, 256)


def _dataloader_config(
    *,
    split: str,
    dataset: str = DEFAULT_TRAIN_LIST,
    dataset_path: str | None = None,
    shuffle_size: int = 50_000,
    min_mixing: float = 0.5,
    num_writers: int = 2,
    num_readers: int = 4,
    fill_once: bool = False,
    base_dir: str = BASE_DIR_GT,
    feature_dir: str | None = None,
    compressor_model: str = COMPRESSOR_MODEL,
    in_channels: int = LATENT_CHANNELS,
    latent_size: tuple[int, int] = LATENT_SIZE,
    image_size: tuple[int, int] = IMAGE_SIZE,
    context_size_frames: int = 10,
    future_size_frames: int = 5,
    max_future_frames: int = 50,
    inference_prefill_frames: int = 14,
    fps: int = 5,
    train_skip: int = 40,
    val_skip: int = 800,
    nan_engaged_plans: bool = False,
    limit: int | None = None,
    mock_data: bool = False,
    mock_segment_batch_size: int = 8,
) -> WorldModelDataLoader.Config:
    return WorldModelDataLoader.Config(
        dataset=dataset,
        dataset_path=dataset_path,
        split=split,
        shuffle_size=shuffle_size,
        min_mixing=min_mixing,
        num_writers=num_writers,
        num_readers=num_readers,
        fill_once=fill_once,
        base_dir=base_dir,
        feature_dir=feature_dir,
        compressor_model=compressor_model,
        in_channels=in_channels,
        latent_size=latent_size,
        image_size=image_size,
        context_size_frames=context_size_frames,
        future_size_frames=future_size_frames,
        max_future_frames=max_future_frames,
        inference_prefill_frames=inference_prefill_frames,
        fps=fps,
        train_skip=train_skip,
        val_skip=val_skip,
        nan_engaged_plans=nan_engaged_plans,
        limit=limit,
        mock_data=mock_data,
        mock_segment_batch_size=mock_segment_batch_size,
    )


def _wan_dataloader_config(
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
    shuffle_size: int = 512,
    min_mixing: float = 0.5,
    num_writers: int = 2,
    num_readers: int = 2,
    fill_once: bool = False,
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
        clip_frames=context_size_frames + future_size_frames,
        context_size_frames=context_size_frames,
        future_size_frames=future_size_frames,
        fps=NATIVE_FPS,
        clips_per_segment=1,
        shuffle_size=shuffle_size,
        min_mixing=min_mixing,
        num_writers=num_writers,
        num_readers=num_readers,
        fill_once=fill_once,
        limit=limit,
        mock_data=mock_data,
        mock_segment_batch_size=mock_segment_batch_size,
        latent_channels=latent_channels,
        latent_size=latent_size,
        mock_latents=mock_latents,
    )
