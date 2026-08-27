# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""World-model video tokenizer backed by the Wan 2.2 VAE."""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass

import einops
import torch
from safetensors import safe_open

from torchtitan.components.tokenizer import BaseTokenizer

from .dataset import DEFAULT_IMAGE_SIZE
from .model import WanVAE


DEFAULT_WAN_TEXT_PROMPT = "realistic driving video from dashcam point of view"
DEFAULT_WAN_TEXT_CONTEXT_FILENAME = "umt5_realistic_driving_dashcam_context.safetensors"


class WanVAETokenizer(BaseTokenizer):
    """Encode paired fcam/ecam clips with one shared Wan VAE.

    The world-model-facing representation stacks two independent Wan latents
    in view-major batch order (all fcam samples, then all ecam samples)::

        [2*B, T_latent, 48, H/16, W/16]

    Wan's causal VAE consumes native ``1 + 4*n`` RGB-frame clips without
    synthesizing or dropping frames.
    """

    NUM_VIEWS = 2

    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenizer.Config):
        compressor_model: str = ""
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE
        text_context_path: str = ""
        text_prompt: str = ""

        def __post_init__(self) -> None:
            if any(size <= 0 or size % 16 for size in self.image_size):
                raise ValueError("image dimensions must be positive multiples of 16")
            if bool(self.text_context_path) != bool(self.text_prompt):
                raise ValueError("text_context_path and text_prompt must be configured together")

    def __init__(
        self,
        config: Config,
        *,
        tokenizer_path: str | None = None,
    ) -> None:
        del tokenizer_path
        super().__init__()
        self.config = config
        self._model: WanVAE | None = None
        self._model_key: tuple[torch.device, torch.dtype] | None = None
        self._model_operation: str | None = None
        self._text_context_cpu: torch.Tensor | None = None

    def encode(
        self,
        inputs: dict[str, torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if "latents" in inputs:
            return inputs["latents"].to(device=device, dtype=dtype)

        model = self._encoder_on(device=device, dtype=dtype)
        imgs = inputs["imgs"]
        big_imgs = inputs["big_imgs"]
        self._validate_image_pair(imgs, big_imgs)

        encoded: list[torch.Tensor] = []
        with torch.inference_mode():
            # Encode one (sample, view) video at a time. The complete temporal
            # sequence remains causal and continuous, while only one full-size
            # video and its patchified copy reside on CUDA at once.
            for view_batch in (imgs, big_imgs):
                for video_THWC in view_batch.unbind(0):
                    video_BCTHW = einops.rearrange(
                        video_THWC,
                        "t h w c -> 1 c t h w",
                    ).to(device=device, dtype=dtype, copy=True)
                    video_BCTHW.div_(255.0).mul_(2).sub_(1).clamp_(-1, 1)
                    encoded.append(
                        model.encode(
                            video_BCTHW,
                            chunk_size=4,
                            output_dtype=None,
                        )
                    )

        # Keep the world-model tokenizer ABI at [B, T, C, H, W], stacking all
        # fcam samples before all ecam samples in the batch dimension.
        return torch.cat(encoded, dim=0).permute(0, 2, 1, 3, 4).to(dtype=dtype)

    def decode(
        self,
        latents: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model = self._decoder_on(device=device, dtype=dtype)
        if latents.ndim != 5:
            raise ValueError(f"latents must have shape [2*B, T, C, H, W], got {tuple(latents.shape)}")
        if latents.shape[0] % self.NUM_VIEWS:
            raise ValueError(f"paired Wan latents require an even batch size, got {latents.shape[0]}")
        if latents.shape[1] < 1:
            raise ValueError("Wan latents must contain at least one frame")
        if latents.shape[2] != model.z_dim:
            raise ValueError(f"Wan latents require {model.z_dim} channels, got {latents.shape[2]}")

        view_latents = einops.rearrange(
            latents.to(device=device, dtype=dtype),
            "(v b) t c h w -> b v c t h w",
            v=self.NUM_VIEWS,
        )
        with torch.inference_mode():
            images = model.decode_views(
                view_latents,
                clamp=True,
                output_dtype=torch.float32,
            )

        images = images.add(1).mul(127.5).round().clamp(0, 255).to(torch.uint8)
        images = einops.rearrange(images, "b v c t h w -> v b t h w c")
        return images[0], images[1]

    def get_vocab_size(self) -> int:
        return 0

    def compressor_input_size(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[int, int]:
        del device, dtype
        return self.config.image_size

    def fixed_text_context(
        self,
        batch_size: int,
        *,
        expected_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Load and repeat a cached raw UMT5 context without loading UMT5."""
        if not self.config.text_context_path:
            return None
        if self._text_context_cpu is None:
            path = os.path.expanduser(self.config.text_context_path)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"cached Wan text context does not exist: {path}; "
                    "create it with torchtitan.experiments.wan_vae.cache_text_context"
                )
            with safe_open(path, framework="pt", device="cpu") as context_file:
                metadata = context_file.metadata() or {}
                cached_prompt = metadata.get("prompt")
                if cached_prompt != self.config.text_prompt:
                    raise ValueError(
                        f"cached Wan text prompt {cached_prompt!r} does not match "
                        f"configured prompt {self.config.text_prompt!r}"
                    )
                if "context" not in context_file.keys():
                    raise ValueError(f"cached Wan text context has no 'context' tensor: {path}")
                context_LC = context_file.get_tensor("context")
            if context_LC.ndim != 2 or context_LC.shape[1] != expected_dim:
                raise ValueError(
                    f"cached Wan text context must have shape [L, {expected_dim}], "
                    f"got {tuple(context_LC.shape)}"
                )
            if not torch.isfinite(context_LC).all():
                raise ValueError("cached Wan text context contains non-finite values")
            self._text_context_cpu = context_LC.contiguous()

        return self._text_context_cpu.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)

    def _encoder_on(self, *, device: torch.device, dtype: torch.dtype) -> WanVAE:
        return self._model_on(device=device, dtype=dtype, operation="encode")

    def _decoder_on(self, *, device: torch.device, dtype: torch.dtype) -> WanVAE:
        return self._model_on(device=device, dtype=dtype, operation="decode")

    def _model_on(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        operation: str,
    ) -> WanVAE:
        if operation not in {"encode", "decode"}:
            raise ValueError(f"unsupported Wan VAE operation {operation!r}")
        if not self.config.compressor_model:
            raise ValueError(f"cannot {operation} without tokenizer.compressor_model")
        device = torch.device(device)
        key = (device, dtype)
        if self._model is not None and self._model_operation not in (None, operation):
            previous_device = self._model_key[0] if self._model_key is not None else None
            self._model = None
            self._model_key = None
            gc.collect()
            if previous_device is not None and previous_device.type == "cuda":
                torch.cuda.empty_cache()
        if self._model is None:
            self._model = WanVAE.from_pretrained(
                self._checkpoint_path(),
                device=device,
                dtype=dtype,
                component="encoder" if operation == "encode" else "decoder",
            )
            self._model_key = key
        elif self._model_key != key:
            self._model = self._model.to(device=device, dtype=dtype)
            self._model_key = key
        self._model_operation = operation
        return self._model

    def _checkpoint_path(self) -> str:
        model = self.config.compressor_model
        if os.path.isdir(model):
            checkpoint = os.path.join(model, "Wan2.2_VAE.pth")
            if os.path.isfile(checkpoint):
                return checkpoint
            raise FileNotFoundError(checkpoint)
        if os.path.isfile(model):
            return model
        if "/" in model:
            from huggingface_hub import hf_hub_download

            return hf_hub_download(model, "Wan2.2_VAE.pth")
        raise FileNotFoundError(model)

    def _validate_image_pair(
        self,
        imgs: torch.Tensor,
        big_imgs: torch.Tensor,
    ) -> None:
        if imgs.shape != big_imgs.shape:
            raise ValueError(
                f"imgs and big_imgs must have identical shapes, got {tuple(imgs.shape)} and {tuple(big_imgs.shape)}"
            )
        if imgs.ndim != 5 or imgs.shape[-1] != 3:
            raise ValueError(f"images must have shape [B, T, H, W, 3], got {tuple(imgs.shape)}")
        if imgs.shape[1] < 1:
            raise ValueError("Wan clips must contain at least one input frame")
        if tuple(imgs.shape[2:4]) != self.config.image_size:
            raise ValueError(f"images must have size {self.config.image_size}, got {tuple(imgs.shape[2:4])}")


__all__ = [
    "DEFAULT_WAN_TEXT_CONTEXT_FILENAME",
    "DEFAULT_WAN_TEXT_PROMPT",
    "WanVAETokenizer",
]
