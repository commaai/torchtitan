from collections.abc import Iterator
from dataclasses import dataclass
import os
import random
from typing import Any

import numpy as np
from openpilot.tools.lib.url_file import URLFileException
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset, get_worker_info

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from xx.common.column_store import ColumnStoreException
from xx.pipeline.exceptions import DataBadError, DataMissingError


PLAN_SIZE = 15 * 33 * 2
IGNORE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AssertionError,
    ColumnStoreException,
    DataMissingError,
    DataBadError,
    URLFileException,
    StopIteration,
    ValueError,
)
Sample = tuple[dict[str, Any], dict[str, Any]]


def _get_strides(array, skip):
    for i in range(len(array.strides) - 1):
        assert array.strides[i] == array.strides[i + 1] * array.shape[i + 1]
    return (array.strides[0] * skip, *array.strides)


def get_temporal_slices(data, idxs, temporal_len, skip):
    if data.ndim <= 1:
        data = np.expand_dims(data, -1)
    output_shape = (len(idxs), temporal_len, *data.shape[1:])
    start_idx = idxs[0] - temporal_len + 1
    return np.lib.stride_tricks.as_strided(data[start_idx:], output_shape, _get_strides(data, skip))


def _load_segments_from_file(path: str, *, val: bool) -> list[str]:
    from xx.common.training_helpers import train_and_test_targets_from_file
    train_targets, test_targets = train_and_test_targets_from_file(path)
    return test_targets if val else train_targets


def _dump_info(info: dict[str, Any]) -> np.ndarray:
    from xx.common.helpers import dump_info
    return dump_info(info)

def get_frame_to_frame_pos_euler(states, calib_from_device):
    from openpilot.common.transformations.orientation import euler_from_rot, rot_from_quat
    from xx.stages.lib.ekf_models.loc_kf import States

    augments_pos_ref_augment = np.zeros((len(states), 3), dtype=np.float64)
    ref_augment_from_augments_euler = np.zeros((len(states), 3), dtype=np.float64)

    if len(states) <= 1:
      return augments_pos_ref_augment, ref_augment_from_augments_euler

    ecef_from_devices = rot_from_quat(states[:, States.ECEF_ORIENTATION])
    devices_from_ecef = ecef_from_devices.transpose(0, 2, 1)
    augments_from_ecef = np.einsum("ad,fde->fae", calib_from_device, devices_from_ecef)

    ref_augments_from_ecef = augments_from_ecef[:-1]
    target_augments_from_ecef = augments_from_ecef[1:]
    ecef_from_target_augments = target_augments_from_ecef.transpose(0, 2, 1)

    target_pos_ecef = states[1:, States.ECEF_POS]
    ref_pos_ecef = states[:-1, States.ECEF_POS]
    augments_pos_ref_augment[1:] = np.einsum("fae,fe->fa", ref_augments_from_ecef, target_pos_ecef - ref_pos_ecef)
    ref_augment_from_target_augment = np.einsum("fra,fat->frt", ref_augments_from_ecef, ecef_from_target_augments)
    ref_augment_from_augments_euler[1:] = euler_from_rot(ref_augment_from_target_augment)
    return augments_pos_ref_augment, ref_augment_from_augments_euler


def get_data_from_seg(target: str, config, val: bool, local_rank: int):
    from openpilot.common.transformations.orientation import rot_from_euler
    from openpilot.system.loggerd.config import CAMERA_FPS
    from xx.common.column_store import ColumnStoreReader
    from xx.common.frame_helpers import Perspective
    from xx.common.numpy_helpers import deep_interp_np
    from xx.common.nv_frame_helpers import NvSegmentFrameIterator

    msh3d_path = os.path.join(config.base_dir, "Mesh3D", target)
    pt_path = os.path.join(config.base_dir, "PlanTargets", target)

    frame_skip = CAMERA_FPS // config.fps
    skip = config.val_skip if val else config.train_skip
    skip = skip // frame_skip
    start_fidx = np.random.randint(0, frame_skip)

    inputs: dict[str, np.ndarray] = {}
    with ColumnStoreReader(msh3d_path) as mesh:
        camera_frame_times = mesh["frame_times"]
        num_frames = len(camera_frame_times)
        max_offset = num_frames // frame_skip - config.max_future_frames - 1
        offset = min(np.random.randint(0, skip), max_offset)
        states = deep_interp_np(camera_frame_times, mesh["t"], mesh["states"])
        states_sim_times = np.ascontiguousarray(states[start_fidx::frame_skip][offset:])

    with ColumnStoreReader(pt_path) as pt:
        calib_from_device = rot_from_euler(pt["calib"][-1]).T
        plan_sim_times = pt["plan"].astype(np.float32)[start_fidx::frame_skip][offset:]

    imgs, big_imgs = zip(
        *NvSegmentFrameIterator(
            target,
            output_perspectives=[Perspective.wmedmodel, Perspective.wbigmodel],
            start_fidx=start_fidx,
            frame_skip=frame_skip,
            pipeline_dir=config.base_dir,
            gpuID=local_rank,
        )
    )
    imgs = np.ascontiguousarray(np.stack(imgs)[offset:])
    big_imgs = np.ascontiguousarray(np.stack(big_imgs)[offset:])
    assert imgs.shape[0] == plan_sim_times.shape[0], "weird segment"
    idxs = np.arange(config.max_future_frames, imgs.shape[0], skip)
    inputs["imgs"] = get_temporal_slices(imgs, idxs, config.max_future_frames, skip)
    inputs["big_imgs"] = get_temporal_slices(big_imgs, idxs, config.max_future_frames, skip)

    fidxs = np.tile(np.arange(config.max_future_frames, dtype=np.int64), (len(idxs), 1))
    states_sim_times = get_temporal_slices(states_sim_times, idxs, config.max_future_frames, skip)
    in_segment_fidxs = np.arange(0, num_frames, dtype=np.int64)[start_fidx::frame_skip][offset:]
    in_segment_fidxs = get_temporal_slices(in_segment_fidxs, idxs, config.max_future_frames, skip).squeeze(-1)

    batch, timesteps = states_sim_times.shape[:2]
    augments_pos_ref_augment = np.zeros((batch, timesteps, 3))
    ref_augment_from_augments_euler = np.zeros((batch, timesteps, 3))

    for idx in range(len(states_sim_times)):
        pose = get_frame_to_frame_pos_euler(states_sim_times[idx], calib_from_device=calib_from_device)
        augments_pos_ref_augment[idx] = pose[0]
        ref_augment_from_augments_euler[idx] = pose[1]

    plan_sim_times = plan_sim_times[idxs - config.max_future_frames + config.context_size_frames]
    nan_filter = np.isnan(augments_pos_ref_augment).any(axis=(1, 2))

    inputs["augments_pos_ref_augment"] = augments_pos_ref_augment.astype(np.float32)
    inputs["ref_augment_from_augments_euler"] = ref_augment_from_augments_euler.astype(np.float32)
    inputs["fidxs"] = fidxs
    inputs["in_segment_fidxs"] = in_segment_fidxs
    outputs = {"plan": plan_sim_times.reshape(len(idxs), -1).astype(np.float32)}

    future_start = np.random.randint(
        config.context_size_frames,
        config.max_future_frames - config.future_size_frames + 1,
    )
    for key in inputs:
        context = inputs[key][:, : config.context_size_frames]
        future = inputs[key][:, future_start : future_start + config.future_size_frames]
        inputs[key] = np.concatenate([future, context], axis=1)

    infos = [{"name": target, "start_fidx": start_fidx, "offset": offset} | {"future_start": future_start} for _ in range(len(idxs))]
    for idx, info in enumerate(infos):
        info["in_segment_fidxs"] = inputs["in_segment_fidxs"][idx].tolist()
    inputs["info"] = np.stack([_dump_info(info) for info in infos])

    for key in inputs:
        inputs[key] = inputs[key][~nan_filter]
    for key in outputs:
        outputs[key] = outputs[key][~nan_filter]
    return inputs, outputs


class WorldModelDataset(IterableDataset, Stateful):
    def __init__(self, config: "WorldModelDataLoader.Config", *, dp_rank: int, dp_world_size: int, local_rank: int = 0, val: bool = False):
        self.config = config
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.local_rank = local_rank
        self.val = val
        self._sample_idx = 0
        self._epoch = 0

        if config.mock_data:
            self.segments = [f"mock-{idx}" for idx in range(1024)]
        else:
            self.segments = _load_segments_from_file(config.dataset, val=val)

        self.segments = self.segments[dp_rank::dp_world_size]

    def __iter__(self) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        while True:
            order = list(range(len(self.segments)))
            random.Random(self._epoch + self.dp_rank).shuffle(order)
            yield from self._iter_epoch_samples(order[worker_id::num_workers])
            self._epoch += 1
            if not self.config.infinite:
                return

    def _iter_epoch_samples(self, order: list[int]) -> Iterator[Sample]:
        for idx in order:
            try:
                batch = (
                    self._mock_batch() if self.config.mock_data
                    else
                    (get_data_from_seg(self.segments[idx],config=self.config, val=self.val, local_rank=self.local_rank))
                )
                yield from self._flatten_batch(batch)
            except IGNORE_EXCEPTIONS:
                continue

    def _mock_batch(self):
        total_frames = self.config.context_size_frames + self.config.future_size_frames
        height, width = self.config.image_size
        batch = self.config.mock_segment_batch_size
        inputs = {
            "imgs": np.random.randint(0, 256, (batch, total_frames, height, width, 3), dtype=np.uint8),
            "big_imgs": np.random.randint(0, 256, (batch, total_frames, height, width, 3), dtype=np.uint8),
            "augments_pos_ref_augment": np.random.randn(batch, total_frames, 3).astype(np.float32),
            "ref_augment_from_augments_euler": np.random.randn(batch, total_frames, 3).astype(np.float32),
            "fidxs": np.tile(np.arange(total_frames, dtype=np.int64), (batch, 1)),
            "info": np.zeros((batch, 512), dtype=np.uint8),
        }
        outputs = {
            "plan": np.random.randn(batch, PLAN_SIZE).astype(np.float32),
        }
        return inputs, outputs

    @staticmethod
    def _slice_mapping(mapping: dict[str, Any], idx: int):
        return {key: value[idx] for key, value in mapping.items()}

    def _flatten_batch(self, batch):
        inputs, outputs = batch
        if not inputs:
            return
        first_key = next(iter(inputs))
        batch_size = len(inputs[first_key])
        for idx in range(batch_size):
            self._sample_idx += 1
            yield (
                self._slice_mapping(inputs, idx),
                self._slice_mapping(outputs, idx),
            )

    def state_dict(self):
        return {"sample_idx": self._sample_idx, "epoch": self._epoch}

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict.get("sample_idx", 0)
        self._epoch = state_dict.get("epoch", 0)


class WorldModelDataLoader(ParallelAwareDataloader):
    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        dataset: str = "/home/batman/xx/datasets/lists/train_500k_20250717.txt"
        infinite: bool = True
        mock_data: bool = False
        mock_segment_batch_size: int = 8

        base_dir: str = "http://data-ssd.comma.life/runner/training_2025_07"
        compressor_model: str = "4672da0d-19f5-44f8-a5fb-2215981c9c0e"
        compressor_encoder_path: str = ""
        in_channels: int = 32
        latent_size: tuple[int, int] = (16, 32)
        image_size: tuple[int, int] = (128, 256)

        context_size_frames: int = 10
        future_size_frames: int = 5
        max_future_frames: int = 50
        inference_conditioning_frames: int = 14
        fps: int = 5
        train_skip: int = 40
        val_skip: int = 800

        def __post_init__(self):
            if self.context_size_frames + self.future_size_frames <= 0:
                raise ValueError("context + future frames must be positive")
            if self.inference_conditioning_frames > (self.context_size_frames + self.future_size_frames):
                raise ValueError("inference_conditioning_frames must fit within total frames")

    def __init__(self, config: Config, *, dp_world_size: int, dp_rank: int, local_batch_size: int, tokenizer: BaseTokenizer | None = None, val: bool = False, **kwargs):
        del tokenizer, kwargs
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        ds = WorldModelDataset(config, dp_rank=dp_rank, dp_world_size=dp_world_size, local_rank=local_rank, val=val)
        dataloader_kwargs = {
            "num_workers": config.num_workers,
            "persistent_workers": config.persistent_workers,
            "pin_memory": config.pin_memory,
            "prefetch_factor": config.prefetch_factor,
            "multiprocessing_context": config.multiprocessing_context,
            "in_order": config.in_order,
            "batch_size": local_batch_size,
        }
        super().__init__(ds, dp_rank=dp_rank, dp_world_size=dp_world_size, **dataloader_kwargs)
