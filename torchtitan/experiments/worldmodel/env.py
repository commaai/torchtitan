#!/usr/bin/env python3
"""Minimal fixed-future world-model rollout on one comma1M segment."""

import argparse
import io
from pathlib import Path

import av
import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from openpilot.common.transformations.camera import DEVICE_CAMERAS, view_frame_from_device_frame
from openpilot.common.transformations.orientation import euler_from_rot, rot_from_euler, rot_from_quat
from safetensors.numpy import load_file
from torch.package import PackageImporter


SEGMENT = "2333e255f1de1fe83ea5975b24a3cc67"
VAE = "commaai/vit-ae-2x-f8c32"
FPS = 5
FRAME_SKIP = 4  # comma1M road cameras run at 20 Hz.
CONTEXT_FRAMES = 10  # Nine known frames and one frame to predict.
FUTURE_FRAMES = 5
ROLLOUT_FRAMES = 50
FUTURE_START = ROLLOUT_FRAMES - FUTURE_FRAMES

# This fixed segment was recorded by a comma four (mici/os04c10).
CAMERAS = DEVICE_CAMERAS[("mici", "os04c10")]
NARROW_K = np.array([[455, 0, 128], [0, 455, 23.8], [0, 0, 1]], dtype=np.float64)
WIDE_K = np.array([[227.5, 0, 128], [0, 227.5, 75.9], [0, 0, 1]], dtype=np.float64)


def warp_matrix(source_k: np.ndarray, target_k: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    points = np.array([[-1, 1.22, 100], [1, 1.22, 100], [-1, 1.22, 200], [1, 1.22, 200]], dtype=np.float32)
    before = (rot_from_euler(view_frame_from_device_frame @ rpy) @ points.T).T
    before = np.column_stack((before[:, :2] / before[:, 2, None], np.ones(4))) @ source_k.T
    after = np.column_stack((points[:, :2] / points[:, 2, None], np.ones(4))) @ target_k.T
    return cv2.getPerspectiveTransform(before[:, :2].astype(np.float32), after[:, :2].astype(np.float32))


def frame_poses(states: np.ndarray, rpy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return each frame's translation and rotation relative to its predecessor."""
    positions = np.zeros((len(states), 3), dtype=np.float32)
    eulers = np.zeros_like(positions)
    ecef_from_devices = rot_from_quat(states[:, 3:7])
    augments_from_ecef = np.einsum("ij,fjk->fik", rot_from_euler(rpy).T, ecef_from_devices.transpose(0, 2, 1))
    positions[1:] = np.einsum("fij,fj->fi", augments_from_ecef[:-1], states[1:, :3] - states[:-1, :3])
    ref_from_targets = np.einsum("fij,fjk->fik", augments_from_ecef[:-1], augments_from_ecef[1:].transpose(0, 2, 1))
    eulers[1:] = euler_from_rot(ref_from_targets)
    return positions, eulers


def load_segment() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = {
        name: hf_hub_download("commaai/comma1M", f"data/{SEGMENT}/{name}", repo_type="dataset")
        for name in ("fcamera.hevc", "ecamera.hevc", "localizer.safetensors")
    }
    localizer = load_file(files["localizer.safetensors"])
    rpy = localizer["rpy"]

    views = []
    for name, source_k, target_k in (
        ("fcamera.hevc", CAMERAS.fcam.intrinsics, NARROW_K),
        ("ecamera.hevc", CAMERAS.ecam.intrinsics, WIDE_K),
    ):
        matrix = warp_matrix(source_k, target_k, rpy)
        frames = []
        with av.open(files[name], format="hevc") as video:
            for frame_index, frame in enumerate(video.decode(video=0)):
                if frame_index % FRAME_SKIP == 0:
                    frame = frame.to_ndarray(format="rgb24")
                    frames.append(cv2.warpPerspective(frame, matrix, (256, 128), borderMode=cv2.BORDER_REPLICATE))
                    if len(frames) == ROLLOUT_FRAMES:
                        break
        views.append(np.stack(frames))

    frames = np.concatenate(views, axis=-1)
    positions, eulers = frame_poses(localizer["frame_states"][::FRAME_SKIP][:ROLLOUT_FRAMES], rpy)
    return frames, positions, eulers


class WorldModelEnv:
    def __init__(self, model_path: Path, device: str, sampling_steps: int, cfg: float):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("the v0 environment requires CUDA")
        if sampling_steps <= 0:
            raise ValueError("sampling_steps must be positive")
        self.sampling_steps = sampling_steps
        self.cfg = cfg
        torch.manual_seed(0)

        importer = PackageImporter(str(model_path))
        model_io = importer.load_pickle("meta", "meta.pkl")["model_io"]
        model_shape = tuple(model_io["in_shape"]["latents"])
        expected_shape = (1, CONTEXT_FRAMES + FUTURE_FRAMES, 32, 16, 32)
        if model_shape != expected_shape:
            raise ValueError(f"world model expects {model_shape}; this environment needs {expected_shape}")
        self.model_dtype = model_io["in_dtype"]["latents"]
        self.vae_dtype = torch.bfloat16

        frames, positions, eulers = load_segment()
        encoder = torch.export.load(hf_hub_download(VAE, "encoder.pt2")).module().to(self.device, self.vae_dtype)
        self.decoder = torch.export.load(hf_hub_download(VAE, "decoder.pt2")).module().to(self.device, self.vae_dtype)
        pixels = torch.from_numpy(frames).permute(0, 3, 1, 2).to(self.device, self.vae_dtype)
        with torch.inference_mode():
            self.latents = encoder(pixels.div(127.5).sub(1)).to(self.model_dtype)
        del encoder, pixels
        self.source_frames = self.decode(self.latents)

        self.positions = torch.from_numpy(positions).to(self.device, self.model_dtype)
        self.eulers = torch.from_numpy(eulers).to(self.device, self.model_dtype)
        self.model = importer.load_pickle("model", "model.pkl")
        state = torch.load(
            io.BytesIO(importer.load_binary("assets", "state_dict.pt")),
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(state, strict=True, assign=True)
        self.model.eval()
        del state
        self.reset()

    def reset(self) -> None:
        self.target = CONTEXT_FRAMES - 1
        self.context = self.latents[: self.target].clone()
        self.future = self.latents[FUTURE_START:ROLLOUT_FRAMES].clone()
        self.future_fidxs = torch.arange(FUTURE_START, ROLLOUT_FRAMES, device=self.device)

    @property
    def done(self) -> bool:
        return self.target == FUTURE_START

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor) -> np.ndarray:
        frames = self.decoder(latents.to(self.vae_dtype)).float().add(1).mul(127.5).clamp(0, 255).byte()
        return frames.cpu().numpy()

    @torch.inference_mode()
    def step(self) -> np.ndarray:
        if self.done:
            raise StopIteration

        total_frames = CONTEXT_FRAMES + FUTURE_FRAMES
        position = torch.zeros((1, total_frames, 3), device=self.device, dtype=self.model_dtype)
        euler = torch.zeros_like(position)
        pose_mask = torch.ones((1, total_frames), device=self.device, dtype=torch.int64)
        position[0, -1] = self.positions[self.target]
        euler[0, -1] = self.eulers[self.target]
        pose_mask[0, -1] = 0

        inputs = {
            "latents": torch.cat((self.future, self.context, torch.randn_like(self.context[:1])))[None],
            "augments_pos_ref_augment": position,
            "ref_augment_from_augments_euler": euler,
            "pose_mask": pose_mask,
            "fidxs": torch.cat((self.future_fidxs, torch.arange(CONTEXT_FRAMES, device=self.device)))[None],
        }
        output = self.model.generate(
            **{name: value.clone() for name, value in inputs.items()},
            steps=self.sampling_steps,
            num_conditioning_frames=total_frames - 1,
            dtype=self.model_dtype,
            inference_schedule="linear",
            cfg=self.cfg,
        )
        latent = output["latents"][0, 0]
        self.context = torch.cat((self.context[1:], latent[None]))
        self.future_fidxs -= 1
        self.target += 1

        return self.decode(latent[None])[0]


def stacked_views(frame: np.ndarray) -> np.ndarray:
    return np.concatenate(np.split(frame, 2, axis=-1), axis=0)


def write_video(path: Path, frames: list[np.ndarray]) -> None:
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=FPS)
        stream.width, stream.height = frames[0].shape[1], frames[0].shape[0]
        stream.pix_fmt = "yuv420p"
        for image in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="local world-model .torchpackage")
    parser.add_argument("-o", "--output", type=Path, default=Path("worldmodel.mp4"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sampling-steps", type=int, default=15)
    parser.add_argument("--cfg", type=float, default=2.0)
    args = parser.parse_args()

    env = WorldModelEnv(args.model, args.device, args.sampling_steps, args.cfg)
    frames = [stacked_views(frame) for frame in env.source_frames[: CONTEXT_FRAMES - 1]]
    while not env.done:
        print(f"predicting frame {env.target}/{FUTURE_START - 1}")
        frames.append(stacked_views(env.step()))
    frames.extend(stacked_views(frame) for frame in env.source_frames[FUTURE_START:ROLLOUT_FRAMES])
    write_video(args.output, frames)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
