# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange

from torchtitan.experiments.worldmodel.tokenizer import WorldModelTokenizer


# B: batch, T: packed time steps, F: images per time step, Q: unpacked images.
# H/W: RGB size, h/w: packed YUV420 size.


def yuv420_to_rgb(images_N6hw: torch.Tensor) -> torch.Tensor:
    images_N6hw = images_N6hw.float()
    luma_N4hw = images_N6hw[:, [0, 2, 1, 3]]
    luma_N1HW = F.pixel_shuffle(luma_N4hw, 2)
    chroma_N2HW = F.interpolate(images_N6hw[:, 4:6], scale_factor=2, mode="nearest")

    y_NHW = luma_N1HW[:, 0] - 16
    u_NHW = chroma_N2HW[:, 0] - 128
    v_NHW = chroma_N2HW[:, 1] - 128
    rgb_N3HW = torch.stack(
        (
            1.16438356 * y_NHW + 1.59602678 * v_NHW,
            1.16438356 * y_NHW - 0.39176229 * u_NHW - 0.81296764 * v_NHW,
            1.16438356 * y_NHW + 2.01723214 * u_NHW,
        ),
        dim=1,
    )
    return rgb_N3HW.clamp(0, 255)


def rgb_to_yuv420(images_N3HW: torch.Tensor) -> torch.Tensor:
    red_NHW, green_NHW, blue_NHW = images_N3HW.float().unbind(dim=1)
    luma_N1HW = (
        16
        + 0.25678824 * red_NHW
        + 0.50412941 * green_NHW
        + 0.09790588 * blue_NHW
    ).unsqueeze(1)
    chroma_N2HW = torch.stack(
        (
            128
            - 0.14822290 * red_NHW
            - 0.29099279 * green_NHW
            + 0.43921569 * blue_NHW,
            128
            + 0.43921569 * red_NHW
            - 0.36778831 * green_NHW
            - 0.07142737 * blue_NHW,
        ),
        dim=1,
    )
    chroma_N2hw = chroma_N2HW[:, :, 0::2, 0::2]
    luma_N4hw = torch.stack(
        (
            luma_N1HW[:, 0, 0::2, 0::2],
            luma_N1HW[:, 0, 1::2, 0::2],
            luma_N1HW[:, 0, 0::2, 1::2],
            luma_N1HW[:, 0, 1::2, 1::2],
        ),
        dim=1,
    )
    return (
        torch.cat((luma_N4hw, chroma_N2hw), dim=1)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
    )


def unpack_images(
    images_BTChw: torch.Tensor,
    *,
    image_size: tuple[int, int],
) -> torch.Tensor:
    batch, timesteps, channels, _, _ = images_BTChw.shape
    images_per_timestep = channels // 6
    images_N6hw = rearrange(
        images_BTChw,
        "b t (f c) h w -> (b t f) c h w",
        c=6,
    )
    images_N3HW = yuv420_to_rgb(images_N6hw)
    images_N3HW = F.interpolate(
        images_N3HW,
        size=image_size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return rearrange(
        images_N3HW,
        "(b t f) c h w -> b (t f) h w c",
        b=batch,
        t=timesteps,
        f=images_per_timestep,
    )


def pack_images(
    images_BQHWC: torch.Tensor,
    *,
    timesteps: int,
    packed_size: tuple[int, int],
) -> torch.Tensor:
    batch, num_images = images_BQHWC.shape[:2]
    images_per_timestep = num_images // timesteps
    images_N3HW = rearrange(images_BQHWC, "b q h w c -> (b q) c h w")
    images_N3HW = F.interpolate(
        images_N3HW.float(),
        size=(packed_size[0] * 2, packed_size[1] * 2),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    images_N6hw = rgb_to_yuv420(images_N3HW)
    return rearrange(
        images_N6hw,
        "(b t f) c h w -> b t (f c) h w",
        b=batch,
        t=timesteps,
        f=images_per_timestep,
    )


class PathTokenizer(WorldModelTokenizer):
    @dataclass(kw_only=True, slots=True)
    class Config(WorldModelTokenizer.Config):
        pass

    def reconstruct(
        self,
        inputs: dict[str, torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        image_size = self.compressor_input_size(device=device, dtype=dtype)
        imgs_BQHWC = unpack_images(inputs["img"], image_size=image_size)
        big_imgs_BQHWC = unpack_images(inputs["big_img"], image_size=image_size)
        latents_BQChw = self.encode(
            {"imgs": imgs_BQHWC, "big_imgs": big_imgs_BQHWC},
            device=device,
            dtype=dtype,
        )
        imgs_BQHWC, big_imgs_BQHWC = self.decode(
            latents_BQChw,
            device=device,
            dtype=dtype,
        )

        timesteps = inputs["img"].shape[1]
        packed_size = inputs["img"].shape[-2:]
        return {
            **inputs,
            "img": pack_images(
                imgs_BQHWC,
                timesteps=timesteps,
                packed_size=packed_size,
            ),
            "big_img": pack_images(
                big_imgs_BQHWC,
                timesteps=timesteps,
                packed_size=packed_size,
            ),
        }
