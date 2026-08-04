# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Literal

import einops
import torch

from torchtitan.components.tokenizer import BaseTokenizer


class WorldModelTokenizer(BaseTokenizer):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenizer.Config):
        compressor_model: str = ""
        compressor_in_channels: Literal[3, 6, "auto"] = "auto"

    def __init__(
        self,
        config: Config,
        *,
        tokenizer_path: str | None = None,
    ) -> None:
        del tokenizer_path
        super().__init__()
        self.config = config
        self._encoder: torch.nn.Module | None = None
        self._encoder_key: tuple[torch.device, torch.dtype] | None = None
        self._decoder: torch.nn.Module | None = None
        self._decoder_key: tuple[torch.device, torch.dtype] | None = None

    def encode(
        self,
        inputs: dict[str, torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if "latents" in inputs:
            return inputs["latents"].to(device=device, dtype=dtype)

        encoder = self._encoder_on(device=device, dtype=dtype)
        imgs = inputs["imgs"]
        big_imgs = inputs["big_imgs"]
        batch, timesteps = imgs.shape[:2]
        in_channels = self._compressor_in_channels(encoder)
        if in_channels == 3:
            rearrange_spec = "nc b t h w c -> (nc b t) c h w"
            inverse_spec = "(nc b t) c h w -> b t (nc c) h w"
        elif in_channels == 6:
            rearrange_spec = "nc b t h w c -> (b t) (nc c) h w"
            inverse_spec = "(b t) (nc c) h w -> b t (nc c) h w"
        else:
            raise ValueError(f"unsupported compressor input channels: {in_channels}")

        with torch.inference_mode():
            x = einops.rearrange(
                [imgs, big_imgs],
                rearrange_spec,
                nc=2,
                b=batch,
                t=timesteps,
            ).to(device=device, dtype=dtype)
            x = x.div(255.0).mul(2).sub(1).clamp(-1, 1)
            latents = encoder(x)
            if isinstance(latents, tuple):
                latents = latents[0]
            return einops.rearrange(
                latents,
                inverse_spec,
                nc=2,
                b=batch,
                t=timesteps,
            )

    def decode(
        self,
        latents: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        decoder = self._decoder_on(device=device, dtype=dtype)
        batch, timesteps = latents.shape[:2]
        latents = einops.rearrange(latents, "b t c h w -> (b t) c h w")
        with torch.inference_mode():
            images = decoder(latents.to(device=device, dtype=dtype))
            if isinstance(images, tuple):
                images = images[0]

        images = images.float().add(1).div(2).mul(255).clamp(0, 255).to(torch.uint8)
        images = einops.rearrange(
            images,
            "(b t) h w (nc c) -> nc b t h w c",
            b=batch,
            t=timesteps,
            nc=2,
        )
        return images[0], images[1]

    def get_vocab_size(self) -> int:
        return 0

    def compressor_input_size(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[int, int]:
        encoder = self._encoder_on(device=device, dtype=dtype)
        shape = encoder.get_buffer("example_shapes").tolist()[0]
        return int(shape[-2]), int(shape[-1])

    def _encoder_on(self, *, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
        if not self.config.compressor_model:
            raise ValueError("inputs contain images, but tokenizer.compressor_model is empty")
        key = (device, dtype)
        if self._encoder is None:
            self._encoder = self._load_encoder()
        if self._encoder_key != key:
            self._encoder = self._encoder.to(device=device, dtype=dtype)
            self._encoder_key = key
        return self._encoder

    def _decoder_on(self, *, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
        if not self.config.compressor_model:
            raise ValueError("cannot decode without tokenizer.compressor_model")
        key = (device, dtype)
        if self._decoder is None:
            self._decoder = self._load_compressor_component("decoder.pt2")
        if self._decoder_key != key:
            self._decoder = self._decoder.to(device=device, dtype=dtype)
            self._decoder_key = key
        return self._decoder

    def _load_encoder(self) -> torch.nn.Module:
        return self._load_compressor_component("encoder.pt2")

    def _load_compressor_component(self, filename: str) -> torch.nn.Module:
        model = self.config.compressor_model
        if os.path.isdir(model):
            model = os.path.join(model, filename)
        elif os.path.isfile(model) and filename != "encoder.pt2":
            model = os.path.join(os.path.dirname(model), filename)
        if os.path.exists(model):
            return torch.export.load(model).module()
        if "/" in model:
            from huggingface_hub import hf_hub_download

            return torch.export.load(hf_hub_download(model, filename)).module()

        from xx.training.lib.checkpoint import Checkpoint

        return torch.export.load(io.BytesIO(Checkpoint(model)[filename])).module()

    def _compressor_in_channels(self, encoder: torch.nn.Module) -> int:
        configured = self.config.compressor_in_channels
        if configured != "auto":
            return configured
        try:
            return int(encoder.get_buffer("example_shapes").tolist()[0][1])
        except Exception:
            return 6
