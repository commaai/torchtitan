# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Reconstruct one paired-camera clip and compare with the legacy compressor."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import (
    CAMERA_NAMES,
    DEFAULT_CLIP_FRAMES,
    DEFAULT_IMAGE_SIZE,
    NATIVE_FPS,
    decode_synchronized_clip,
)
from .model import WanVAE


def _autocast(device: torch.device, dtype: torch.dtype):
    if dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _to_tanh_video(video: torch.Tensor) -> torch.Tensor:
    if video.dtype == torch.uint8:
        return video.float().div(127.5).sub(1.0)
    return video.float().clamp(-1, 1)


def _validate_clip(clip: torch.Tensor) -> torch.Tensor:
    if clip.ndim == 6 and clip.shape[0] == 1:
        clip = clip[0]
    if clip.ndim != 5 or clip.shape[:2] != (2, 3):
        raise ValueError(f"clip must have shape [2, 3, T, H, W] in fcam/ecam order, got {tuple(clip.shape)}")
    if (clip.shape[2] - 1) % 4:
        raise ValueError("clip length must equal 1 + 4*n")
    return clip.contiguous()


def _to_uint8_video(video: torch.Tensor) -> torch.Tensor:
    return _to_tanh_video(video).add(1).mul(127.5).round().clamp(0, 255).to(torch.uint8).cpu()


def comparison_video_frames(
    videos: dict[str, torch.Tensor],
    *,
    draw_labels: bool = True,
) -> np.ndarray:
    """Build ``[T, H, W, 3]`` RGB frames with models as rows and views as columns."""

    if not videos:
        raise ValueError("at least one video is required")
    uint8_videos = {name: _to_uint8_video(_validate_clip(video)) for name, video in videos.items()}
    shapes = {name: tuple(video.shape) for name, video in uint8_videos.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"comparison videos must have identical shapes, got {shapes}")

    _, _, num_frames, height, width = next(iter(uint8_videos.values())).shape
    label_height = 32 if draw_labels else 0
    output_frames: list[np.ndarray] = []
    for frame_index in range(num_frames):
        rows: list[np.ndarray] = []
        for model_name, video in uint8_videos.items():
            # [V, C, H, W] -> [H, V*W, C]
            tiles = video[:, :, frame_index].permute(0, 2, 3, 1).numpy()
            image_row = np.concatenate(list(tiles), axis=1)
            if draw_labels:
                import cv2

                label_row = np.zeros((label_height, image_row.shape[1], 3), dtype=np.uint8)
                for view_index, camera_name in enumerate(CAMERA_NAMES):
                    cv2.putText(
                        label_row,
                        f"{model_name} / {camera_name}",
                        (view_index * width + 8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                image_row = np.concatenate((label_row, image_row), axis=0)
            rows.append(image_row)
        output_frames.append(np.concatenate(rows, axis=0))
    return np.stack(output_frames)


def write_comparison_mp4(
    videos: dict[str, torch.Tensor],
    output: Path,
    *,
    fps: int = NATIVE_FPS,
    crf: int = 20,
) -> Path:
    if output.suffix.lower() != ".mp4":
        raise ValueError(f"video output must use the .mp4 extension, got {output}")
    if fps <= 0:
        raise ValueError("video fps must be positive")
    if not 0 <= crf <= 51:
        raise ValueError("video CRF must be between 0 and 51")

    from xx.common.video_helpers import write_mp4

    output.parent.mkdir(parents=True, exist_ok=True)
    frames = comparison_video_frames(videos)
    write_mp4(frames, str(output), fps=fps, crf=crf)
    if not output.is_file():
        raise RuntimeError(f"video writer did not create {output}")
    return output


@torch.inference_mode()
def reconstruct_wan(
    model: WanVAE,
    clip: torch.Tensor,
    *,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """Reconstruct views sequentially to bound activation memory."""

    views = []
    for view in clip.unbind(0):
        with _autocast(view.device, compute_dtype):
            latents = model.encode(view.unsqueeze(0), output_dtype=None)
            reconstruction = model.decode(
                latents,
                clamp=True,
                output_dtype=torch.float32,
            )
        views.append(reconstruction[0])
    return torch.stack(views)


@torch.inference_mode()
def reconstruct_legacy(
    clip: torch.Tensor,
    *,
    device: torch.device,
    model_id: str = "commaai/vit-ae-2x-f8c32",
    batch_size: int = 8,
) -> torch.Tensor:
    """Run the existing paired-view ViT autoencoder exported on Hugging Face."""

    from huggingface_hub import hf_hub_download

    encoder = torch.export.load(hf_hub_download(model_id, "encoder.pt2")).module()
    decoder = torch.export.load(hf_hub_download(model_id, "decoder.pt2")).module()
    # ExportedProgram modules already carry their frozen inference graph and
    # intentionally reject nn.Module.eval(). Device movement is supported.
    encoder.to(device)
    decoder.to(device)

    target_height, target_width = clip.shape[-2:]
    # [V, C, T, H, W] -> [T, V*C, 128, 256], preserving paired views.
    pairs = clip.permute(2, 0, 1, 3, 4).flatten(1, 2)
    pairs = F.interpolate(pairs, size=(128, 256), mode="bilinear", align_corners=False)
    outputs = []
    for start in range(0, pairs.shape[0], batch_size):
        batch = pairs[start : start + batch_size].to(device)
        decoded = decoder(encoder(batch))
        if decoded.ndim != 4:
            raise ValueError(f"legacy decoder returned invalid shape {decoded.shape}")
        if decoded.shape[1] == 6:
            decoded_nchw = decoded
        elif decoded.shape[-1] == 6:
            decoded_nchw = decoded.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"legacy decoder must return six paired RGB channels, got {decoded.shape}")
        outputs.append(decoded_nchw.float().cpu())
    decoded = torch.cat(outputs)
    decoded = decoded.unflatten(1, (2, 3)).flatten(0, 1)
    decoded = F.interpolate(
        decoded,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )
    return decoded.unflatten(0, (clip.shape[2], 2)).permute(1, 2, 0, 3, 4)


def _load_lpips(device: torch.device):
    from xx.training.lib.lpips import LPIPS

    with torch.device("meta"):
        model = LPIPS()
    model.load_from_pretrained(strict=True, assign=True)
    return model.to(device).eval().requires_grad_(False)


@torch.inference_mode()
def reconstruction_metrics(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    *,
    lpips_model=None,
    lpips_frame_stride: int = 4,
) -> dict[str, object]:
    target = _to_tanh_video(target)
    reconstruction = _to_tanh_video(reconstruction)
    error = reconstruction - target
    per_view_mse = error.square().mean(dim=(1, 2, 3, 4))
    metrics: dict[str, object] = {
        "mae": float(error.abs().mean()),
        "mse": float(error.square().mean()),
        "psnr_db": float(10 * torch.log10(4.0 / error.square().mean())),
        "per_view": {
            "fcam": {
                "mae": float(error[0].abs().mean()),
                "mse": float(per_view_mse[0]),
            },
            "ecam": {
                "mae": float(error[1].abs().mean()),
                "mse": float(per_view_mse[1]),
            },
        },
    }
    if lpips_model is not None:
        target_frames = target[:, :, ::lpips_frame_stride].permute(0, 2, 1, 3, 4).flatten(0, 1)
        reconstruction_frames = reconstruction[:, :, ::lpips_frame_stride].permute(0, 2, 1, 3, 4).flatten(0, 1)
        device = next(lpips_model.parameters()).device
        values = []
        for start in range(0, target_frames.shape[0], 8):
            values.append(
                lpips_model(
                    reconstruction_frames[start : start + 8].to(device),
                    target_frames[start : start + 8].to(device),
                ).cpu()
            )
        metrics["lpips"] = float(torch.cat(values).mean())
    return metrics


def _load_clip(path: Path) -> torch.Tensor:
    if path.suffix == ".npy":
        return torch.from_numpy(np.load(path))
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        for key in ("video", "clip", "input"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, torch.Tensor):
        raise TypeError(f"{path} did not contain a video tensor")
    return payload


def main() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--clip", type=Path, help=".pt or .npy [2,3,T,H,W] clip")
    source.add_argument("--segment", help="comma segment identifier")
    parser.add_argument("--pipeline-dir")
    parser.add_argument("--start-fidx", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root / "weights/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=("float32", "bfloat16"),
        default="bfloat16" if torch.cuda.is_available() else "float32",
    )
    parser.add_argument("--compare-legacy", action="store_true")
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--output", type=Path, help="optional .pt reconstruction payload")
    parser.add_argument("--output-video", type=Path, help="optional labeled MP4 comparison grid")
    parser.add_argument("--video-fps", type=int, default=NATIVE_FPS)
    parser.add_argument("--video-crf", type=int, default=20)
    args = parser.parse_args()

    if args.clip is not None:
        clip = _load_clip(args.clip)
    else:
        if args.pipeline_dir is None:
            from xx.comma_data.constants import BASE_DIR_GT

            args.pipeline_dir = BASE_DIR_GT
        clip = torch.from_numpy(
            decode_synchronized_clip(
                args.segment,
                pipeline_dir=args.pipeline_dir,
                start_fidx=args.start_fidx,
                clip_frames=DEFAULT_CLIP_FRAMES,
                image_size=DEFAULT_IMAGE_SIZE,
            )
        )
    clip = _validate_clip(clip)
    target = _to_tanh_video(clip)
    device = torch.device(args.device)
    compute_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[args.compute_dtype]
    model = WanVAE.from_pretrained(args.checkpoint, device=device)
    wan_reconstruction = reconstruct_wan(
        model,
        target.to(device),
        compute_dtype=compute_dtype,
    ).cpu()
    lpips_model = _load_lpips(device) if args.lpips else None
    metrics = {
        "wan2.2": reconstruction_metrics(
            target,
            wan_reconstruction,
            lpips_model=lpips_model,
        )
    }
    payload = {
        "target": target,
        "wan2.2": wan_reconstruction,
    }

    if args.compare_legacy:
        legacy_reconstruction = reconstruct_legacy(
            target,
            device=device,
        )
        payload["vit-ae-2x-f8c32"] = legacy_reconstruction
        metrics["vit-ae-2x-f8c32"] = reconstruction_metrics(
            target,
            legacy_reconstruction,
            lpips_model=lpips_model,
        )

    if args.output_video is not None:
        video_rows = {
            "target": payload["target"],
            "Wan 2.2": payload["wan2.2"],
        }
        if "vit-ae-2x-f8c32" in payload:
            video_rows["vit-ae-2x-f8c32"] = payload["vit-ae-2x-f8c32"]
        write_comparison_mp4(
            video_rows,
            args.output_video,
            fps=args.video_fps,
            crf=args.video_crf,
        )

    print(json.dumps(metrics, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({**payload, "metrics": metrics}, args.output)


if __name__ == "__main__":
    main()
