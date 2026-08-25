# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Native-20-fps, synchronized fcam/ecam clips for Wan VAE training."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.tokenizer import BaseTokenizer


CAMERA_NAMES = ("fcam", "ecam")
NATIVE_FPS = 20
DEFAULT_CLIP_FRAMES = 57
DEFAULT_IMAGE_SIZE = (256, 512)


def _full_frame_camera_perspectives(
    image_size: tuple[int, int],
    device_type: Any,
):
    from xx.common.camera_helpers import get_all_camera_attributes_by_device_type, get_scaled_intrinsics
    from xx.common.frame_helpers import PerspectiveParams

    height, width = image_size
    output_size = (width, height)
    camera_attributes = get_all_camera_attributes_by_device_type(device_type)

    def scale(camera_name: str):
        camera = camera_attributes[camera_name]
        return PerspectiveParams(
            to_intr=get_scaled_intrinsics(
                camera["K"],
                camera["SIZE"],
                output_size,
            ),
            to_size=output_size,
        )

    return (scale("narrow_road"), scale("wide_road"))


def _clip_starts(
    *,
    num_available_frames: int,
    clip_frames: int,
    clips_per_segment: int,
    val: bool,
) -> np.ndarray:
    max_start = num_available_frames - clip_frames
    if max_start < 0:
        raise ValueError(f"segment has {num_available_frames} frames, fewer than the required {clip_frames}")
    if clips_per_segment == 1:
        return np.array(
            [max_start // 2 if val else np.random.randint(max_start + 1)],
            dtype=np.int64,
        )
    if val:
        return np.linspace(0, max_start, clips_per_segment, dtype=np.int64)
    return np.random.randint(0, max_start + 1, size=clips_per_segment)


def decode_synchronized_clip(
    target: str,
    *,
    pipeline_dir: str,
    start_fidx: int,
    clip_frames: int = DEFAULT_CLIP_FRAMES,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    local_rank: int = 0,
) -> np.ndarray:
    """Decode one clip as ``[V, C, T, H, W]`` uint8."""

    from xx.common.column_store import ColumnStoreReader
    from xx.common.nv_frame_helpers import NvSegmentFrameIterator

    frame_info_path = os.path.join(pipeline_dir, "FrameInfo", target)
    with ColumnStoreReader(frame_info_path) as frame_info:
        synchronized_frames = list(
            NvSegmentFrameIterator(
                target,
                frame_info=frame_info,
                calib=np.zeros(3, dtype=np.float32),
                output_perspectives=list(_full_frame_camera_perspectives(image_size, frame_info["device_type"])),
                gpuID=local_rank,
                pipeline_dir=pipeline_dir,
                start_fidx=start_fidx,
                end_fidx=start_fidx + clip_frames,
                frame_skip=1,
            )
        )
    if len(synchronized_frames) != clip_frames:
        raise ValueError(f"expected {clip_frames} synchronized frames from {target}, got {len(synchronized_frames)}")
    clip = np.stack(synchronized_frames, axis=0).transpose(1, 4, 0, 2, 3)
    expected = (len(CAMERA_NAMES), 3, clip_frames, *image_size)
    if clip.shape != expected:
        raise ValueError(f"decoded clip from {target} has shape {clip.shape}, expected {expected}")
    return np.ascontiguousarray(clip, dtype=np.uint8)


def get_data_from_segment(
    target: str,
    config: "WanVAEDataLoader.Config",
    val: bool,
    local_rank: int = 0,
) -> tuple[dict[str, np.ndarray]]:
    """Decode synchronized camera clips as ``[N, V, C, T, H, W]`` uint8.

    ``NvSegmentFrameIterator`` zips full-frame narrow and wide camera streams
    at the same source frame index. Camera-native intrinsics scaled to the Wan
    input size and zero calibration make the warp a whole-frame resize with no
    crop. ``frame_skip=1`` is deliberate: Wan's temporal compression consumes
    native 20-fps frames. The returned source clip is always temporally
    continuous; world-model sink-window selection happens only after VAE
    encoding.
    """

    from xx.common.column_store import ColumnStoreReader

    frame_info_path = os.path.join(config.pipeline_dir, "FrameInfo", target)
    with ColumnStoreReader(frame_info_path) as frame_info:
        if "fcamera" not in frame_info or "ecamera" not in frame_info:
            raise ValueError(f"{target} does not contain both fcamera and ecamera")
        num_available_frames = min(
            len(frame_info["fcamera"]["t"]),
            len(frame_info["ecamera"]["t"]),
        )

    starts = _clip_starts(
        num_available_frames=num_available_frames,
        clip_frames=config.clip_frames,
        clips_per_segment=config.clips_per_segment,
        val=val,
    )
    clips = [
        decode_synchronized_clip(
            target,
            pipeline_dir=config.pipeline_dir,
            start_fidx=start,
            clip_frames=config.clip_frames,
            image_size=config.image_size,
            local_rank=local_rank,
        )
        for start in starts.tolist()
    ]

    videos = np.stack(clips, axis=0)
    # Store the clip only once in Gigashuffle. WanVAEDataLoader exposes the same
    # tensor as the reconstruction target after reading the batch.
    return ({"input": videos},)


class _MockWanClipDataset:
    def __init__(self, config: "WanVAEDataLoader.Config") -> None:
        self.config = config
        self.segments = list(range(1024))

    def __iter__(self) -> Iterator[tuple[dict[str, np.ndarray]]]:
        rng = np.random.default_rng(0)
        while True:
            for _ in self.segments:
                shape = (
                    self.config.mock_segment_batch_size,
                    len(CAMERA_NAMES),
                    3,
                    self.config.clip_frames,
                    *self.config.image_size,
                )
                videos = rng.integers(0, 256, size=shape, dtype=np.uint8)
                yield ({"input": videos},)


class WanVAEDataLoader(BaseDataLoader):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        dataset: str
        split: Literal["train", "val"]
        pipeline_dir: str
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE
        clip_frames: int = DEFAULT_CLIP_FRAMES
        fps: int = NATIVE_FPS
        clips_per_segment: int = 1
        shuffle_size: int = 512
        min_mixing: float = 0.5
        num_writers: int = 2
        num_readers: int = 2
        fill_once: bool = False
        limit: int | None = None
        mock_data: bool = False
        mock_segment_batch_size: int = 1

        def __post_init__(self) -> None:
            if self.fps != NATIVE_FPS:
                raise ValueError(f"Wan VAE requires native {NATIVE_FPS}-fps video, got {self.fps}")
            if self.clip_frames < 1 or (self.clip_frames - 1) % 4:
                raise ValueError("clip_frames must equal 1 + 4*n")
            if any(size <= 0 or size % 16 for size in self.image_size):
                raise ValueError("image dimensions must be positive multiples of 16")
            if self.clips_per_segment <= 0:
                raise ValueError("clips_per_segment must be positive")
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
        from gigashuffle import DataloaderConfig, MultiprocessShuffledDataloader

        self.config = config
        self.local_batch_size = local_batch_size
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.local_rank = int(os.environ.get("LOCAL_RANK", dp_rank))
        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", dp_world_size))
        node_rank = int(os.environ.get("GROUP_RANK", dp_rank // max(1, self.local_world_size)))
        run_id = os.environ.get("REPORTERV2_TRAINING_ID") or "wan-vae"
        val = config.split == "val"
        shuffle_size = local_batch_size * validation_steps * self.local_world_size if val else config.shuffle_size
        self.dataset = self._build_dataset(
            config,
            val=val,
            global_rank=dp_rank,
            global_world_size=dp_world_size,
        )
        self.loader = MultiprocessShuffledDataloader(
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

    @staticmethod
    def _build_dataset(
        config: Config,
        *,
        val: bool,
        global_rank: int,
        global_world_size: int,
    ):
        if config.mock_data:
            return _MockWanClipDataset(config)

        from urllib3.exceptions import HTTPError

        from xx.common.column_store import ColumnStoreException
        from xx.common.training_helpers import train_and_test_targets_from_file
        from xx.pipeline.exceptions import DataBadError, DataMissingError
        from xx.training.lib.dataloader import GenericDataset

        train_segments, val_segments = train_and_test_targets_from_file(
            config.dataset_path or config.dataset,
            limit=config.limit,
        )
        segments = val_segments if val else train_segments
        return GenericDataset(
            segments=segments,
            get_data_from_seg=get_data_from_segment,
            config=config,
            val=val,
            local_rank=int(os.environ.get("LOCAL_RANK", "0")),
            ignore_exceptions=(
                AssertionError,
                ColumnStoreException,
                DataMissingError,
                DataBadError,
                FileNotFoundError,
                HTTPError,
                StopIteration,
                ValueError,
            ),
            global_rank=global_rank,
            global_world_size=global_world_size,
        )

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        iterator = iter(self.loader)
        self._iterator = iterator
        try:
            for (inputs,) in iterator:
                yield inputs, inputs["input"]
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
        del state_dict
