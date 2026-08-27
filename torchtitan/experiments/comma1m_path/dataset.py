# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

import cv2
import fsspec
import numpy as np
import PyNvVideoCodec
import reverse_geocoder

from openpilot.cereal import log
from openpilot.common.transformations.camera import denormalize, DEVICE_CAMERAS, view_frame_from_device_frame
from openpilot.common.transformations.coordinates import ecef2geodetic
from openpilot.common.transformations.model import (
    BIGMODEL_INPUT_SIZE,
    bigmodel_intrinsics,
    MEDMODEL_INPUT_SIZE,
    medmodel_intrinsics,
    SBIGMODEL_INPUT_SIZE,
    sbigmodel_intrinsics,
)
from openpilot.common.transformations.orientation import euler_from_rot, rot_from_euler, rot_from_quat
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan, get_curvature_from_plan, MIN_SPEED
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan
from openpilot.system.loggerd.config import CAMERA_FPS
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import get_worker_info

from torchtitan.observability import structured_logger as sl

COMMA1M_REPO_ID = "commaai/comma1M"

__all__ = ["COMMA1M_REPO_ID"]

T_IDXS = np.asarray(ModelConstants.T_IDXS)
W, H = MEDMODEL_INPUT_SIZE
BIG_W, BIG_H = BIGMODEL_INPUT_SIZE
SBIG_W, SBIG_H = SBIGMODEL_INPUT_SIZE
BIG_MODEL_CORNERS = np.asarray([[0, 0, 1], [BIG_W, 0, 1], [0, BIG_H, 1], [BIG_W, BIG_H, 1]])
CAMERA_BY_DEVICE_TYPE = {
    log.InitData.DeviceType.neo: ("neo", "unknown"),
    log.InitData.DeviceType.tici: ("tici", "ar0231"),
    log.InitData.DeviceType.tizi: ("tizi", "ar0231"),
    log.InitData.DeviceType.mici: ("mici", "os04c10"),
}
ROAD_POINTS = np.asarray(
    [[-1.0, 1.22, 100.0], [1.0, 1.22, 100.0], [-1.0, 1.22, 200.0], [1.0, 1.22, 200.0]],
    dtype=np.float32,
)
LHT_COUNTRIES = frozenset(
    "AG AI AU BB BD BM BN BS BT BW CC CK CX CY DM FJ FK GB GD GG GY HK ID IE IM IN JE JM JP KE KI KN KY LC "
    "LK LS MO MS MT MU MV MW MY MZ NA NF NP NR NU NZ PG PK PN SB SC SG SH SR SZ TC TH TK TL TO TT TV TZ UG "
    "VC VG VI WS ZA ZM ZW".split()
)


class Comma1MDataset:
    """Iterable over public comma1M segments."""

    def __init__(
        self,
        config: Any,
        val: bool,
        local_rank: int,
        global_rank: int,
        global_world_size: int,
    ) -> None:
        assert config.plan_only
        assert not config.rgb
        assert config.dataset_path is not None
        assert not config.deterministic_fidxs

        self.config = config
        self.val = val
        self.local_rank = local_rank
        self.global_world_size = global_world_size
        self.fs, root = fsspec.url_to_fs(config.dataset_path)
        self.data_dir = f"{root.rstrip('/')}/data"
        segments = sorted(path.rsplit("/", 2)[-2] for path in self.fs.glob(f"{self.data_dir}/*/fcamera.hevc"))[
            : config.limit
        ]
        segments = [segment for segment in segments if (hash(int(segment, 16)) % 10 == 0) == val]
        self.segments = segments[global_rank::global_world_size]
        sl.log_trace_scalar(
            {
                "comma1m_segments": len(segments),
                "local_comma1m_segments": len(self.segments),
            }
        )

    def __iter__(self):
        worker = get_worker_info()
        segments = self.segments
        if worker is not None:
            writer_count = worker.num_workers // self.global_world_size
            segments = segments[worker.id % writer_count :: writer_count]
        if not segments:
            raise ValueError("comma1M split is empty; increase limit or reduce ranks/writers")

        while True:
            order = segments.copy()
            random.shuffle(order)
            for segment in order:
                yield _load_segment(self.fs, f"{self.data_dir}/{segment}", self.config, self.val, self.local_rank)


def _decode(fs, path: str, index: np.ndarray, wanted: set[int], gpu_id: int) -> dict[int, np.ndarray]:
    if not wanted:
        return {}

    iframe_indexes = np.flatnonzero(index[:-1, 0] == 2)
    next_iframe = np.searchsorted(iframe_indexes, max(wanted), side="right")
    read_end = int(iframe_indexes[next_iframe]) if next_iframe < len(iframe_indexes) else len(index) - 1
    payload = fs.cat_file(path, end=int(index[read_end, 1]))

    decoder = PyNvVideoCodec.CreateDecoder(
        gpuid=gpu_id,
        codec=PyNvVideoCodec.cudaVideoCodec.HEVC,
        usedevicememory=False,
        outputColorType=PyNvVideoCodec.OutputColorType.RGB,
    )
    decoded = {}
    frame_idx = 0
    for packet in (payload, b""):
        packet_array = np.frombuffer(packet, dtype=np.uint8)
        packet_data = PyNvVideoCodec.PacketData()
        packet_data.bsl = len(packet)
        packet_data.bsl_data = packet_array.ctypes.data
        for frame in decoder.Decode(packet_data):
            if frame_idx in wanted:
                decoded[frame_idx] = np.from_dlpack(frame).copy()
            frame_idx += 1
    return decoded


def _transform_matrix(from_intrinsics: np.ndarray, to_intrinsics: np.ndarray, eulers: np.ndarray) -> np.ndarray:
    after = ROAD_POINTS @ rot_from_euler(eulers)
    before_pixels = denormalize(ROAD_POINTS[:, :2] / ROAD_POINTS[:, 2, None], intrinsics=from_intrinsics)
    after_pixels = denormalize(after[:, :2] / after[:, 2, None], intrinsics=to_intrinsics)
    return cv2.getPerspectiveTransform(before_pixels.astype(np.float32), after_pixels.astype(np.float32))


def _pack_image(rgb: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_I420)
    y = yuv[:height].reshape(height // 2, 2, width // 2, 2).transpose(3, 1, 0, 2).reshape(4, height // 2, width // 2)
    uv = yuv[height:].reshape(2, height // 2, width // 2)
    return np.concatenate((y, uv))


def _rgb_from_tensor(tensor: np.ndarray, size: tuple[int, int] = (128, 256)) -> np.ndarray:
    height = tensor.shape[2] * 2
    width = tensor.shape[3] * 2
    frames = np.zeros((tensor.shape[0], height * 3 // 2, width), dtype=np.uint8)
    frames[:, 0:height:2, 0::2] = tensor[:, 0]
    frames[:, 1:height:2, 0::2] = tensor[:, 1]
    frames[:, 0:height:2, 1::2] = tensor[:, 2]
    frames[:, 1:height:2, 1::2] = tensor[:, 3]
    frames[:, height : height + height // 4] = tensor[:, 4].reshape(-1, height // 4, width)
    frames[:, height + height // 4 : height + height // 2] = tensor[:, 5].reshape(-1, height // 4, width)

    output_height, output_width = size
    output = np.zeros((tensor.shape[0], output_height, output_width, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_I420)
        output[index] = cv2.resize(rgb, (output_width, output_height))
    return output


def _calibration_augmentations(rpy: np.ndarray, num_samples: int, val: bool) -> tuple[np.ndarray, np.ndarray]:
    if val:
        multipliers = np.zeros(num_samples)
    else:
        multipliers = np.asarray(
            [float(np.random.uniform()) if np.random.uniform() > 0.8 else 0.0 for _ in range(num_samples)]
        )
    augment_from_calib = rot_from_euler(rpy * multipliers[:, None])
    augment_from_device = np.einsum("tij,jk->tik", augment_from_calib, rot_from_euler(rpy).T)
    eulers_view = np.einsum(
        "ij,tj->ti", view_frame_from_device_frame, euler_from_rot(augment_from_device.swapaxes(1, 2))
    )
    return augment_from_calib, eulers_view


def _load_images(
    fs,
    segment_dir: str,
    frame_info: dict[str, np.ndarray],
    fidxs: np.ndarray,
    temporal_fidxs: np.ndarray,
    eulers_view: np.ndarray,
    frame_skip: int,
    n_frames: int,
    local_rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    big_camera = "ecamera" if "ecamera/index" in frame_info else "fcamera"
    device_type = int(frame_info["device_type"].item())
    cameras = DEVICE_CAMERAS[CAMERA_BY_DEVICE_TYPE[device_type]]

    fcamera_intrinsics = cameras.narrow_road.intrinsics
    big_intrinsics = (cameras.wide_road if big_camera == "ecamera" else cameras.narrow_road).intrinsics
    med_matrices = [_transform_matrix(fcamera_intrinsics, medmodel_intrinsics, eulers) for eulers in eulers_view]
    big_matrices = [_transform_matrix(big_intrinsics, sbigmodel_intrinsics, eulers) for eulers in eulers_view]

    source_width = int(frame_info[f"{big_camera}/width"].item())
    source_height = int(frame_info[f"{big_camera}/height"].item())
    valid = np.ones(len(fidxs), dtype=bool)
    for sample_idx, eulers in enumerate(eulers_view):
        inverse = np.linalg.inv(_transform_matrix(big_intrinsics, bigmodel_intrinsics, eulers))
        source = BIG_MODEL_CORNERS @ inverse.T
        source = source[:, :2] / source[:, 2, None]
        valid[sample_idx] = np.all(
            (source[:, 0] > 0) & (source[:, 0] < source_width) & (source[:, 1] > 0) & (source[:, 1] < source_height)
        )

    requests: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for (sample_idx, temporal_idx), temporal_fidx in np.ndenumerate(temporal_fidxs):
        if valid[sample_idx]:
            for frame_number in range(n_frames):
                source_fidx = int(temporal_fidx + (frame_number + 1 - n_frames) * frame_skip)
                requests[source_fidx].append((sample_idx, temporal_idx, frame_number))

    wanted = set(requests)
    fcamera_frames = _decode(fs, f"{segment_dir}/fcamera.hevc", frame_info["fcamera/index"], wanted, local_rank)
    big_frames = (
        fcamera_frames
        if big_camera == "fcamera"
        else _decode(fs, f"{segment_dir}/ecamera.hevc", frame_info["ecamera/index"], wanted, local_rank)
    )
    images = np.zeros((len(fidxs), temporal_fidxs.shape[1], 6 * n_frames, H // 2, W // 2), dtype=np.uint8)
    big_images = np.zeros((len(fidxs), temporal_fidxs.shape[1], 6 * n_frames, SBIG_H // 2, SBIG_W // 2), dtype=np.uint8)
    for source_fidx, destinations in requests.items():
        for sample_idx, temporal_idx, frame_number in destinations:
            channel_slice = slice(6 * frame_number, 6 * (frame_number + 1))
            image = cv2.warpPerspective(
                fcamera_frames[source_fidx], med_matrices[sample_idx], (W, H), borderMode=cv2.BORDER_REPLICATE
            )
            big_image = cv2.warpPerspective(
                big_frames[source_fidx], big_matrices[sample_idx], (SBIG_W, SBIG_H), borderMode=cv2.BORDER_REPLICATE
            )
            images[sample_idx, temporal_idx, channel_slice] = _pack_image(image)
            big_images[sample_idx, temporal_idx, channel_slice] = _pack_image(big_image)
    return images, big_images, valid


def _load_targets(
    localizer: dict[str, np.ndarray],
    temporal_fidxs: np.ndarray,
    augment_from_calib: np.ndarray,
    action_t: np.ndarray,
) -> dict[str, np.ndarray]:
    frame_t = localizer["frame_t"]
    states = localizer["frame_states"].copy()
    states[:, 10:13] = gaussian_filter1d(states[:, 10:13], 2, axis=0, radius=5, mode="nearest")
    states[:, 19:22] = gaussian_filter1d(states[:, 19:22], 2, axis=0, radius=5, mode="nearest")
    calib_from_device = rot_from_euler(localizer["rpy"]).T
    plan = np.empty((*temporal_fidxs.shape, len(T_IDXS), 15), dtype=np.float32)

    for (sample_idx, temporal_idx), fidx in np.ndenumerate(temporal_fidxs):
        query_t = T_IDXS + frame_t[fidx]
        indexes = np.clip(np.searchsorted(frame_t, query_t) - 1, 0, len(frame_t) - 2)
        distance = (query_t - frame_t[indexes]) / (frame_t[indexes + 1] - frame_t[indexes])
        future = (states[indexes].T * (1 - distance)).T + (states[indexes + 1].T * distance).T
        device_from_ecef = rot_from_quat(future[:, 3:7]).swapaxes(1, 2)
        calib_from_ecef = np.einsum("ij,tjk->tik", calib_from_device, device_from_ecef)

        value = np.empty((len(T_IDXS), 15), dtype=np.float64)
        value[:, Plan.POSITION] = np.einsum("ij,tj->ti", calib_from_ecef[0], future[:, 0:3] - future[0, 0:3])
        value[:, Plan.VELOCITY] = np.einsum("tij,tj->ti", calib_from_ecef, future[:, 7:10])
        value[:, Plan.ACCELERATION] = np.einsum("ij,tj->ti", calib_from_device, future[:, 19:22])
        value[:, Plan.T_FROM_CURRENT_EULER] = euler_from_rot(
            np.einsum("ij,tjk->tik", calib_from_ecef[0], calib_from_ecef.swapaxes(1, 2))
        )
        value[:, Plan.ORIENTATION_RATE] = np.einsum("ij,tj->ti", calib_from_device, future[:, 10:13])
        value = value.astype(np.float32)

        augment = augment_from_calib[sample_idx]
        for value_slice in (Plan.POSITION, Plan.VELOCITY, Plan.ACCELERATION, Plan.ORIENTATION_RATE):
            value[:, value_slice] = np.einsum("ij,tj->ti", augment, value[:, value_slice])
        value[:, Plan.T_FROM_CURRENT_EULER] = euler_from_rot(
            np.einsum("ij,tjk->tik", augment, rot_from_euler(value[:, Plan.T_FROM_CURRENT_EULER]))
        )
        plan[sample_idx, temporal_idx] = value

    flat_plan = plan.reshape(-1, len(T_IDXS), 15)
    flat_action_t = np.broadcast_to(action_t[:, None], (*plan.shape[:2], 2)).reshape(-1, 2)
    action = np.zeros((len(flat_plan), 2), dtype=np.float32)
    for idx, (flat_value, times) in enumerate(zip(flat_plan, flat_action_t, strict=True)):
        velocity = max(float(flat_value[0, Plan.VELOCITY.start]), MIN_SPEED)
        curvature = get_curvature_from_plan(
            flat_value[:, Plan.T_FROM_CURRENT_EULER.start + 2],
            flat_value[:, Plan.ORIENTATION_RATE.start + 2],
            T_IDXS,
            velocity,
            float(times[0]),
        )
        action[idx, 0] = curvature * velocity**2
        action[idx, 1] = get_accel_from_plan(
            flat_value[:, Plan.VELOCITY.start],
            flat_value[:, Plan.ACCELERATION.start],
            T_IDXS,
            action_t=float(times[1]),
        )
    return {"plan": plan, "action": action.reshape(*plan.shape[:2], 2)}


def _load_segment(
    fs, segment_dir: str, config: Any, val: bool, local_rank: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    from safetensors.numpy import load

    frame_info = load(fs.cat_file(f"{segment_dir}/frame_info.safetensors"))
    localizer = load(fs.cat_file(f"{segment_dir}/localizer.safetensors"))
    frame_skip = CAMERA_FPS // config.fps
    history_idxs = -np.arange(1, 2 * config.fps, config.fps // 4, dtype=np.int32)[::-1]
    temporal_len = 5 * config.fps + int(history_idxs[-1] - history_idxs[0])
    start = frame_skip * (temporal_len + 1) + int(np.random.randint(0, 5))
    candidates = np.arange(start, len(frame_info["fcamera/t"]) - 200, 40)
    sample_count = min(len(candidates), 8)
    fidxs = np.random.choice(candidates, sample_count, replace=False)

    temporal_fidxs = np.asarray(
        [[fidx + (history_idx + 1) * frame_skip for history_idx in history_idxs] for fidx in fidxs]
    )
    augment_from_calib, eulers_view = _calibration_augmentations(localizer["rpy"], len(fidxs), val)

    images, big_images, valid = _load_images(
        fs,
        segment_dir,
        frame_info,
        fidxs,
        temporal_fidxs,
        eulers_view,
        frame_skip,
        config.n_frames,
        local_rank,
    )
    action_t = np.random.uniform(0.0, 1.0, size=(len(fidxs), 2)).astype(np.float32)
    targets = _load_targets(localizer, temporal_fidxs, augment_from_calib, action_t)

    traffic = np.zeros((len(fidxs), temporal_len, 2), dtype=np.float32)
    latitude, longitude, _ = ecef2geodetic(localizer["frame_states"][-1, 0:3])
    country = reverse_geocoder.search((latitude, longitude), mode=1, verbose=False)[0]["cc"]
    traffic[:, :, int(country in LHT_COUNTRIES)] = 1
    inputs = {
        "img": images,
        "big_img": big_images,
        "desire_pulse": np.zeros((len(fidxs), temporal_len, 8), dtype=np.float32),  # TODO: get from log
        "traffic_convention": traffic,
        "action_t": np.broadcast_to(action_t[:, None], (len(fidxs), temporal_len, 2)).copy(),
    }

    inputs = {name: np.ascontiguousarray(value[valid]) for name, value in inputs.items()}
    targets = {
        name: np.ascontiguousarray(value[valid]).reshape(valid.sum(), temporal_fidxs.shape[1], -1)
        for name, value in targets.items()
    }
    targets["imgs"] = np.concatenate(
        [
            _rgb_from_tensor(inputs["img"][:, -1, -6:]),
            _rgb_from_tensor(inputs["big_img"][:, -1, -6:]),
        ],
        axis=3,
    )
    return inputs, targets
