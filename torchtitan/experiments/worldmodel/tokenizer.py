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
        decode_batch_size: int = 8
        decoder_layout: Literal["nhwc", "nchw"] = "nhwc"
        decoder_output_scale: Literal["tanh", "uint8"] = "tanh"
        compile_decoder: bool = False

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
        if config.decode_batch_size < 1:
            raise ValueError("tokenizer.decode_batch_size must be positive")

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

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode unnormalized B,T,C,H,W latents into two B,T,H,W,3 uint8 cameras.

        Use no_grad, not inference_mode: the resulting images may be saved by a
        trainable vision encoder for backward. The VAE itself stays frozen.
        """
        if latents.ndim != 5:
            raise ValueError(f"Expected B,T,C,H,W latents, got {tuple(latents.shape)}")
        decoder = self._decoder_on(device=device, dtype=dtype)
        batch, timesteps, channels = latents.shape[:3]
        in_channels = self.config.compressor_in_channels
        if in_channels == "auto":
            if hasattr(decoder, "example_shapes"):
                decoder_channels = int(decoder.get_buffer("example_shapes")[0, 1])
                if channels not in (decoder_channels, 2 * decoder_channels):
                    raise ValueError("Latent channels do not match the VAE decoder")
                in_channels = 6 if channels == decoder_channels else 3
            elif self._encoder is not None:
                in_channels = self._compressor_in_channels(self._encoder)
            else:
                raise ValueError("Set compressor_in_channels for decoders without example_shapes")
        if in_channels == 6:
            flat = latents.flatten(0, 1)
        elif in_channels == 3:
            flat = einops.rearrange(latents, "b t (nc c) h w -> (nc b t) c h w", nc=2)
        else:
            raise ValueError(f"Unsupported compressor input channels: {in_channels}")

        decoded = []
        for chunk in flat.split(self.config.decode_batch_size):
            output = decoder(chunk.to(device=device, dtype=dtype))
            if isinstance(output, dict):
                output = output["imgs_out"]
            elif isinstance(output, tuple):
                output = output[0]
            if self.config.decoder_layout == "nchw":
                output = output.permute(0, 2, 3, 1)
            if output.ndim != 4 or output.shape[-1] != in_channels:
                raise ValueError(f"Unexpected VAE decoder output shape: {tuple(output.shape)}")
            output = output.float()
            if self.config.decoder_output_scale == "tanh":
                output = (output + 1) * 127.5
            decoded.append(output.clamp(0, 255).to(torch.uint8))
        images = torch.cat(decoded)
        if in_channels == 6:
            images = images.unflatten(0, (batch, timesteps))
            return images[..., :3], images[..., 3:]
        images = images.unflatten(0, (2, batch, timesteps))
        return images[0], images[1]

    def _decoder_on(self, *, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
        if not self.config.compressor_model:
            raise ValueError("Decoding latents requires tokenizer.compressor_model")
        key = (device, dtype)
        if self._decoder is None:
            self._decoder = self._load_compressor_component("decoder.pt2").requires_grad_(False)
        if self._decoder_key != key:
            self._decoder = self._decoder.to(device=device, dtype=dtype)
            self._decoder_key = key
            if self.config.compile_decoder:
                self._decoder.compile()
        return self._decoder

    def get_vocab_size(self) -> int:
        return 0

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

    def _load_encoder(self) -> torch.nn.Module:
        return self._load_compressor_component("encoder.pt2")

    def compressor_input_size(self, *, device: torch.device, dtype: torch.dtype) -> tuple[int, int]:
        encoder = self._encoder_on(device=device, dtype=dtype)
        shape = encoder.get_buffer("example_shapes")[0].tolist()
        return int(shape[-2]), int(shape[-1])

    def _load_compressor_component(self, filename: str) -> torch.nn.Module:
        model = self.config.compressor_model
        if os.path.isdir(model):
            model = os.path.join(model, filename)
        elif os.path.isfile(model) and os.path.basename(model) in ("encoder.pt2", "decoder.pt2"):
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
