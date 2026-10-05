"""Reconstruct the paired YUV420 inputs used by DrivingModelRunner."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from xx.common.compressor_helpers import COMPRESSOR_STATS
from xx.training.path.model_constants import ModelInputs, VISION_INPUTS_YUV
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.experiments.worldmodel.tokenizer import WorldModelTokenizer


COMPRESSOR_MODEL = "c04337f8-b83f-4e34-b07a-5f7396978d67"


def rgb_to_yuv420(rgb: torch.Tensor) -> torch.Tensor:
    """NHWC uint8 RGB -> N,6,H/2,W/2, matching OpenCV I420 + frames_to_tensor.

    OpenCV samples chroma at the top left pixel of each 2x2 block. The luma
    channel order is (even/even, odd/even, even/odd, odd/odd), not pixel_unshuffle.
    """
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.shape[1] % 2 or rgb.shape[2] % 2:
        raise ValueError(f"Expected even-sized N,H,W,3 RGB images, got {tuple(rgb.shape)}")
    r, g, b = rgb.to(torch.int32).unbind(-1)
    y = (269484 * r + 528482 * g + 102760 * b + (16 << 20) + (1 << 19)) >> 20
    r, g, b = (channel[:, ::2, ::2] for channel in (r, g, b))
    u = (-155188 * r - 305135 * g + 460324 * b + (128 << 20) + (1 << 19)) >> 20
    v = (460324 * r - 385875 * g - 74448 * b + (128 << 20) + (1 << 19)) >> 20
    planes = (y[:, ::2, ::2], y[:, 1::2, ::2], y[:, ::2, 1::2], y[:, 1::2, 1::2], u, v)
    return torch.stack(planes, dim=1).clamp(0, 255).to(torch.uint8)


class RLDrivingTokenizer(WorldModelTokenizer):
    @dataclass(kw_only=True, slots=True)
    class Config(WorldModelTokenizer.Config):
        compressor_model: str = COMPRESSOR_MODEL
        compressor_in_channels: int = 6
        latent_mean: float = COMPRESSOR_STATS[COMPRESSOR_MODEL]["mean"]
        latent_std: float = COMPRESSOR_STATS[COMPRESSOR_MODEL]["std"]
        latent_max: float = 10.0
        decoder_dtype: str = "bfloat16"

    def __init__(self, config: Config, **kwargs):
        super().__init__(config, **kwargs)
        if config.latent_std <= 0 or config.latent_max <= 0:
            raise ValueError("Latent standard deviation and quantization range must be positive")
        if config.decoder_dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("Unsupported decoder_dtype")

    @torch.no_grad()
    def reconstruct(
        self,
        inputs: dict[str, torch.Tensor],
        *,
        history_idxs: tuple[int, ...],
        temporal_len: int,
        device: torch.device,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Decode each required frame once, then pair [previous, current].

        The current latent window includes one predecessor before the policy
        window. Future latents follow it without a gap; all indices are at the
        rollout FPS, independently of train_skip (which skips training samples).
        """
        current = inputs["quantized_latents"]
        future = inputs["next_quantized_latents"]
        if current.dtype != torch.uint8 or future.dtype != torch.uint8:
            raise ValueError("Expected uint8 quantized_latents from the rollout dataloader")
        if current.shape[1] != temporal_len + 1:
            raise ValueError("Full-model LoRA needs temporal_len + 1 latents, including the preceding image")
        n_step = future.shape[1]
        if n_step < 1 or not history_idxs or min(history_idxs) < -temporal_len or max(history_idxs) >= 0:
            raise ValueError("Invalid latent history or bootstrap window")
        pairs = [
            [(temporal_len + offset + index, temporal_len + offset + index + 1) for index in history_idxs]
            for offset in (0, 1, n_step)
        ]
        frame_idxs = sorted({index for window in pairs for pair in window for index in pair})
        latents = torch.cat((current, future), dim=1)[:, frame_idxs].to(device=device, dtype=torch.float32)
        latents = (latents / 255.0 - 0.5) * (2 * self.config.latent_max)
        mean = inputs.get("compressor_mean", self.config.latent_mean)
        std = inputs.get("compressor_std", self.config.latent_std)
        if isinstance(mean, torch.Tensor):
            mean = mean.to(device).reshape(-1, 1, 1, 1, 1)
        if isinstance(std, torch.Tensor):
            std = std.to(device).reshape(-1, 1, 1, 1, 1)
        cameras = self.decode(latents * std + mean, device=device, dtype=TORCH_DTYPE_MAP[self.config.decoder_dtype])
        windows = ({}, {}, {})
        lookup = {frame: index for index, frame in enumerate(frame_idxs)}
        for name, rgb in zip((ModelInputs.IMG, ModelInputs.BIG_IMG), cameras, strict=True):
            batch = rgb.shape[0]
            height, width = VISION_INPUTS_YUV[name][-2:]
            rgb = rgb.flatten(0, 1)
            # VAE outputs are quantized before resize, as in VAEModelRunner.
            if rgb.shape[1:3] != (2 * height, 2 * width):
                rgb = F.interpolate(
                    rgb.permute(0, 3, 1, 2).float(), size=(2 * height, 2 * width), mode="bilinear", align_corners=False
                )
                rgb = rgb.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
            packed = rgb_to_yuv420(rgb).unflatten(0, (batch, len(frame_idxs)))
            for window, frame_pairs in zip(windows, pairs, strict=True):
                indices = [lookup[frame] for pair in frame_pairs for frame in pair]
                window[name] = packed[:, indices].reshape(batch, len(history_idxs), 12, height, width)
        return windows
