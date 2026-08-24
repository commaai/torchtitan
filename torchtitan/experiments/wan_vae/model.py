# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-2025 The Alibaba Wan Team Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""TorchTitan-native implementation of the Wan 2.2 video VAE.

The module hierarchy intentionally matches ``wan/modules/vae2_2.py`` from the
official Wan 2.2 repository.  That makes ``Wan2.2_VAE.pth`` load strictly,
without a key-renaming adapter, while giving TorchTitan a normal ``BaseModel``
with explicit batched-view helpers.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from torchtitan.protocols.model import BaseModel


CACHE_T = 2
TEMPORAL_COMPRESSION = 4
SPATIAL_COMPRESSION = 16

# Statistics published with Wan2.2-TI2V-5B.  The public Wan wrapper returns
# ``(posterior_mean - mean) / std`` rather than sampling the posterior.
WAN_VAE_MEAN = (
    -0.2289,
    -0.0052,
    -0.1323,
    -0.2339,
    -0.2799,
    0.0174,
    0.1838,
    0.1557,
    -0.1382,
    0.0542,
    0.2813,
    0.0891,
    0.1570,
    -0.0098,
    0.0375,
    -0.1825,
    -0.2246,
    -0.1207,
    -0.0698,
    0.5109,
    0.2665,
    -0.2108,
    -0.2158,
    0.2502,
    -0.2055,
    -0.0322,
    0.1109,
    0.1567,
    -0.0729,
    0.0899,
    -0.2799,
    -0.1230,
    -0.0313,
    -0.1649,
    0.0117,
    0.0723,
    -0.2839,
    -0.2083,
    -0.0520,
    0.3748,
    0.0152,
    0.1957,
    0.1433,
    -0.2944,
    0.3573,
    -0.0548,
    -0.1681,
    -0.0667,
)

WAN_VAE_STD = (
    0.4765,
    1.0364,
    0.4514,
    1.1677,
    0.5313,
    0.4990,
    0.4818,
    0.5013,
    0.8158,
    1.0344,
    0.5894,
    1.0901,
    0.6885,
    0.6165,
    0.8454,
    0.4978,
    0.5759,
    0.3523,
    0.7135,
    0.6804,
    0.5833,
    1.4146,
    0.8986,
    0.5659,
    0.7069,
    0.5338,
    0.4889,
    0.4917,
    0.4069,
    0.4999,
    0.6866,
    0.4093,
    0.5709,
    0.6065,
    0.6415,
    0.4944,
    0.5726,
    1.2042,
    0.5458,
    1.6887,
    0.3971,
    1.0600,
    0.3943,
    0.5537,
    0.5444,
    0.4089,
    0.7468,
    0.7744,
)


class CausalConv3d(nn.Conv3d):
    """3-D convolution padded only toward earlier timesteps."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Streaming convolutions receive a stable cache slot after the encoder
        # or decoder has been assembled.  A fixed integer is important for
        # torch.compile: mutating the upstream ``feat_idx=[0]`` cursor inside a
        # compiled graph aliases every cache write to slot zero.
        self.cache_slot: int | None = None
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x: torch.Tensor, cache_x: torch.Tensor | None = None) -> torch.Tensor:
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            x = torch.cat((cache_x.to(x.device), x), dim=2)
            padding[4] -= cache_x.shape[2]
        return super().forward(F.pad(x, padding))


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        channel_first: bool = True,
        images: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def reset_parameters(self) -> None:
        nn.init.ones_(self.gamma)
        if isinstance(self.bias, nn.Parameter):
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = 1 if self.channel_first else -1
        return F.normalize(x, dim=dim) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):
    """Nearest-neighbor interpolation with a bfloat16-safe implementation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type_as(x)


_FIRST_UPSAMPLE_CHUNK = object()
CacheEntry = torch.Tensor | object | None


class Resample(nn.Module):
    def __init__(self, dim: int, mode: str) -> None:
        if mode not in {
            "none",
            "upsample2d",
            "upsample3d",
            "downsample2d",
            "downsample3d",
        }:
            raise ValueError(f"unsupported resample mode {mode!r}")
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode in {"upsample2d", "upsample3d"}:
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
            if mode == "upsample3d":
                self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode in {"downsample2d", "downsample3d"}:
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
            if mode == "downsample3d":
                self.time_conv = CausalConv3d(
                    dim,
                    dim,
                    (3, 1, 1),
                    stride=(2, 1, 1),
                    padding=(0, 0, 0),
                )
        else:
            self.resample = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[CacheEntry] | None = None,
    ) -> torch.Tensor:
        b, c, t, h, w = x.shape
        if self.mode == "upsample3d" and feat_cache is not None:
            idx = _cache_slot(self.time_conv)
            previous = feat_cache[idx]
            if previous is None:
                feat_cache[idx] = _FIRST_UPSAMPLE_CHUNK
            else:
                cache_x = x[:, :, -CACHE_T:].clone()
                if cache_x.shape[2] < CACHE_T:
                    if previous is _FIRST_UPSAMPLE_CHUNK:
                        cache_x = torch.cat((torch.zeros_like(cache_x), cache_x), dim=2)
                    else:
                        assert isinstance(previous, torch.Tensor)
                        cache_x = torch.cat((previous[:, :, -1:].to(cache_x.device), cache_x), dim=2)
                if previous is _FIRST_UPSAMPLE_CHUNK:
                    x = self.time_conv(x)
                else:
                    assert isinstance(previous, torch.Tensor)
                    x = self.time_conv(x, previous)
                feat_cache[idx] = cache_x
                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0], x[:, 1]), dim=3)
                x = x.reshape(b, c, t * 2, h, w)

        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)

        if self.mode == "downsample3d" and feat_cache is not None:
            idx = _cache_slot(self.time_conv)
            previous = feat_cache[idx]
            if previous is None:
                feat_cache[idx] = x.clone()
            else:
                assert isinstance(previous, torch.Tensor)
                cache_x = x[:, :, -1:].clone()
                x = self.time_conv(torch.cat((previous[:, :, -1:].to(x.device), x), dim=2))
                feat_cache[idx] = cache_x
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual = nn.Sequential(
            RMSNorm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[CacheEntry] | None = None,
    ) -> torch.Tensor:
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = _cache_slot(layer)
                previous = feat_cache[idx]
                assert previous is None or isinstance(previous, torch.Tensor)
                cache_x = x[:, :, -CACHE_T:].clone()
                if cache_x.shape[2] < CACHE_T and previous is not None:
                    cache_x = torch.cat((previous[:, :, -1:].to(cache_x.device), cache_x), dim=2)
                x = layer(x, previous)
                feat_cache[idx] = cache_x
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """Per-frame, single-head spatial self-attention."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, t, h, w = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        return rearrange(x, "(b t) c h w -> b c t h w", t=t) + identity


def patchify(x: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    if patch_size == 1:
        return x
    if x.ndim == 4:
        return rearrange(
            x,
            "b c (h q) (w r) -> b (c r q) h w",
            q=patch_size,
            r=patch_size,
        )
    if x.ndim == 5:
        return rearrange(
            x,
            "b c f (h q) (w r) -> b (c r q) f h w",
            q=patch_size,
            r=patch_size,
        )
    raise ValueError(f"patchify expects a 4-D or 5-D tensor, got {tuple(x.shape)}")


def unpatchify(x: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    if patch_size == 1:
        return x
    if x.ndim == 4:
        return rearrange(
            x,
            "b (c r q) h w -> b c (h q) (w r)",
            q=patch_size,
            r=patch_size,
        )
    if x.ndim == 5:
        return rearrange(
            x,
            "b (c r q) f h w -> b c f (h q) (w r)",
            q=patch_size,
            r=patch_size,
        )
    raise ValueError(f"unpatchify expects a 4-D or 5-D tensor, got {tuple(x.shape)}")


class AvgDown3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t: int,
        factor_s: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        if in_channels * self.factor % out_channels:
            raise ValueError("AvgDown3D channel ratio must be integral")
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        b, c, t, h, w = x.shape
        x = x.view(
            b,
            c,
            t // self.factor_t,
            self.factor_t,
            h // self.factor_s,
            self.factor_s,
            w // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(
            b,
            c * self.factor,
            t // self.factor_t,
            h // self.factor_s,
            w // self.factor_s,
        )
        x = x.view(
            b,
            self.out_channels,
            self.group_size,
            t // self.factor_t,
            h // self.factor_s,
            w // self.factor_s,
        )
        return x.mean(dim=2)


class DupUp3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t: int,
        factor_s: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        if out_channels * self.factor % in_channels:
            raise ValueError("DupUp3D channel ratio must be integral")
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: torch.Tensor, first_chunk: bool = False) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.size(0),
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            x.size(2),
            x.size(3),
            x.size(4),
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(
            x.size(0),
            self.out_channels,
            x.size(2) * self.factor_t,
            x.size(4) * self.factor_s,
            x.size(6) * self.factor_s,
        )
        return x[:, :, self.factor_t - 1 :] if first_chunk else x


class DownResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        mult: int,
        temporal_downsample: bool = False,
        down_flag: bool = False,
    ) -> None:
        super().__init__()
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temporal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )
        downsamples: list[nn.Module] = []
        for _ in range(mult):
            downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        if down_flag:
            mode = "downsample3d" if temporal_downsample else "downsample2d"
            downsamples.append(Resample(out_dim, mode=mode))
        self.downsamples = nn.Sequential(*downsamples)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[CacheEntry] | None = None,
    ) -> torch.Tensor:
        x_copy = x.clone()
        for module in self.downsamples:
            x = module(x, feat_cache)
        return x + self.avg_shortcut(x_copy)


class UpResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        mult: int,
        temporal_upsample: bool = False,
        up_flag: bool = False,
    ) -> None:
        super().__init__()
        self.avg_shortcut = (
            DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temporal_upsample else 1,
                factor_s=2,
            )
            if up_flag
            else None
        )
        upsamples: list[nn.Module] = []
        for _ in range(mult):
            upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        if up_flag:
            mode = "upsample3d" if temporal_upsample else "upsample2d"
            upsamples.append(Resample(out_dim, mode=mode))
        self.upsamples = nn.Sequential(*upsamples)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[CacheEntry] | None = None,
        first_chunk: bool = False,
    ) -> torch.Tensor:
        x_main = x.clone()
        for module in self.upsamples:
            x_main = module(x_main, feat_cache)
        if self.avg_shortcut is None:
            return x_main
        return x_main + self.avg_shortcut(x, first_chunk)


class Encoder3d(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales: Sequence[float] = (),
        temporal_downsample: Sequence[bool] = (True, True, False),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = tuple(dim_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_scales = tuple(attn_scales)
        self.temporal_downsample = tuple(temporal_downsample)

        dims = [dim * u for u in (1, *dim_mult)]
        self.conv1 = CausalConv3d(12, dims[0], 3, padding=1)
        downsamples: list[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            temporal = temporal_downsample[i] if i < len(temporal_downsample) else False
            downsamples.append(
                DownResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks,
                    temporal_downsample=temporal,
                    down_flag=i != len(dim_mult) - 1,
                )
            )
        self.downsamples = nn.Sequential(*downsamples)
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )
        self.head = nn.Sequential(
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[CacheEntry] | None = None,
    ) -> torch.Tensor:
        if feat_cache is not None:
            idx = _cache_slot(self.conv1)
            previous = feat_cache[idx]
            assert previous is None or isinstance(previous, torch.Tensor)
            cache_x = x[:, :, -CACHE_T:].clone()
            if cache_x.shape[2] < CACHE_T and previous is not None:
                cache_x = torch.cat((previous[:, :, -1:].to(cache_x.device), cache_x), dim=2)
            x = self.conv1(x, previous)
            feat_cache[idx] = cache_x
        else:
            x = self.conv1(x)

        for layer in self.downsamples:
            x = layer(x, feat_cache)
        for layer in self.middle:
            if isinstance(layer, ResidualBlock):
                x = layer(x, feat_cache)
            else:
                x = layer(x)
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = _cache_slot(layer)
                previous = feat_cache[idx]
                assert previous is None or isinstance(previous, torch.Tensor)
                cache_x = x[:, :, -CACHE_T:].clone()
                if cache_x.shape[2] < CACHE_T and previous is not None:
                    cache_x = torch.cat((previous[:, :, -1:].to(cache_x.device), cache_x), dim=2)
                x = layer(x, previous)
                feat_cache[idx] = cache_x
            else:
                x = layer(x)
        return x


class Decoder3d(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales: Sequence[float] = (),
        temporal_upsample: Sequence[bool] = (False, True, True),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = tuple(dim_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_scales = tuple(attn_scales)
        self.temporal_upsample = tuple(temporal_upsample)

        dims = [dim * u for u in (dim_mult[-1], *reversed(dim_mult))]
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )
        upsamples: list[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            temporal = temporal_upsample[i] if i < len(temporal_upsample) else False
            upsamples.append(
                UpResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks + 1,
                    temporal_upsample=temporal,
                    up_flag=i != len(dim_mult) - 1,
                )
            )
        self.upsamples = nn.Sequential(*upsamples)
        self.head = nn.Sequential(
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, 12, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[CacheEntry] | None = None,
        first_chunk: bool = False,
    ) -> torch.Tensor:
        if feat_cache is not None:
            idx = _cache_slot(self.conv1)
            previous = feat_cache[idx]
            assert previous is None or isinstance(previous, torch.Tensor)
            cache_x = x[:, :, -CACHE_T:].clone()
            if cache_x.shape[2] < CACHE_T and previous is not None:
                cache_x = torch.cat((previous[:, :, -1:].to(cache_x.device), cache_x), dim=2)
            x = self.conv1(x, previous)
            feat_cache[idx] = cache_x
        else:
            x = self.conv1(x)

        for layer in self.middle:
            if isinstance(layer, ResidualBlock):
                x = layer(x, feat_cache)
            else:
                x = layer(x)
        for layer in self.upsamples:
            x = layer(x, feat_cache, first_chunk)
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = _cache_slot(layer)
                previous = feat_cache[idx]
                assert previous is None or isinstance(previous, torch.Tensor)
                cache_x = x[:, :, -CACHE_T:].clone()
                if cache_x.shape[2] < CACHE_T and previous is not None:
                    cache_x = torch.cat((previous[:, :, -1:].to(cache_x.device), cache_x), dim=2)
                x = layer(x, previous)
                feat_cache[idx] = cache_x
            else:
                x = layer(x)
        return x


def count_causal_conv3d(model: nn.Module) -> int:
    return sum(isinstance(module, CausalConv3d) for module in model.modules())


def _assign_cache_slots(model: nn.Module) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, CausalConv3d):
            module.cache_slot = count
            count += 1
    return count


def _cache_slot(module: CausalConv3d) -> int:
    if module.cache_slot is None:
        raise RuntimeError("streaming CausalConv3d does not have an assigned cache slot")
    return module.cache_slot


def _stream_chunk_lengths(num_frames: int, chunk_size: int | None) -> tuple[int, ...]:
    if num_frames < 1 or (num_frames - 1) % TEMPORAL_COMPRESSION:
        raise ValueError(f"Wan 2.2 clips must contain 1 + 4*n frames; received {num_frames}")
    remaining = num_frames - 1
    if remaining == 0:
        return (1,)
    if chunk_size is None:
        return (1, remaining)
    if chunk_size <= 0 or chunk_size % TEMPORAL_COMPRESSION:
        raise ValueError("chunk_size must be a positive multiple of 4 or None")
    return (1, *(min(chunk_size, remaining - offset) for offset in range(0, remaining, chunk_size)))


class WanVAE(BaseModel):
    """Wan 2.2 VAE core with explicit ``[B, V, C, T, H, W]`` helpers."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        dim: int = 160
        decoder_dim: int = 256
        latent_channels: int = 48
        dim_mult: tuple[int, ...] = (1, 2, 4, 4)
        num_res_blocks: int = 2
        attn_scales: tuple[float, ...] = ()
        temporal_downsample: tuple[bool, ...] = (False, True, True)
        dropout: float = 0.0
        latent_mean: tuple[float, ...] = WAN_VAE_MEAN
        latent_std: tuple[float, ...] = WAN_VAE_STD
        training_stage: Literal["frozen", "decoder", "full"] = "frozen"

        def __post_init__(self) -> None:
            if len(self.dim_mult) < 2:
                raise ValueError("dim_mult must contain at least two stages")
            if len(self.temporal_downsample) < len(self.dim_mult) - 1:
                raise ValueError("temporal_downsample needs one entry per resampling stage")
            if len(self.latent_mean) != self.latent_channels:
                raise ValueError("latent_mean length must equal latent_channels")
            if len(self.latent_std) != self.latent_channels:
                raise ValueError("latent_std length must equal latent_channels")
            if any(value <= 0 for value in self.latent_std):
                raise ValueError("latent_std entries must be positive")

        def update_from_config(self, *, config: Any, **kwargs: Any) -> None:
            del kwargs
            unsupported = {
                "tensor parallel": config.parallelism.tensor_parallel_degree,
                "context parallel": config.parallelism.context_parallel_degree,
                "pipeline parallel": config.parallelism.pipeline_parallel_degree,
                "expert parallel": config.parallelism.expert_parallel_degree,
            }
            for name, degree in unsupported.items():
                if degree > 1:
                    raise ValueError(f"Wan VAE supports FSDP/HSDP only, not {name}")

        def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
            del seq_len
            nparams = sum(parameter.numel() for parameter in model.parameters())
            return nparams, 6 * nparams

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.z_dim = config.latent_channels
        self.dim_mult = config.dim_mult
        self.num_res_blocks = config.num_res_blocks
        self.attn_scales = config.attn_scales
        self.temperal_downsample = config.temporal_downsample
        self.temperal_upsample = tuple(reversed(config.temporal_downsample))

        # Keep these names and nesting aligned with the official checkpoint.
        self.encoder = Encoder3d(
            config.dim,
            config.latent_channels * 2,
            config.dim_mult,
            config.num_res_blocks,
            config.attn_scales,
            config.temporal_downsample,
            config.dropout,
        )
        self.conv1 = CausalConv3d(config.latent_channels * 2, config.latent_channels * 2, 1)
        self.conv2 = CausalConv3d(config.latent_channels, config.latent_channels, 1)
        self.decoder = Decoder3d(
            config.decoder_dim,
            config.latent_channels,
            config.dim_mult,
            config.num_res_blocks,
            config.attn_scales,
            self.temperal_upsample,
            config.dropout,
        )
        self._encoder_cache_slots = _assign_cache_slots(self.encoder)
        self._decoder_cache_slots = _assign_cache_slots(self.decoder)
        self.register_buffer("_latent_mean", torch.tensor(config.latent_mean), persistent=False)
        self.register_buffer("_latent_std", torch.tensor(config.latent_std), persistent=False)
        self._apply_training_stage()

    def verify_module_protocol(self) -> None:
        # Raw nn.Module containers are intentional: their FQNs are the checkpoint ABI.
        pass

    def _apply_training_stage(self) -> None:
        self.requires_grad_(self.config.training_stage == "full")
        if self.config.training_stage == "decoder":
            self.decoder.requires_grad_(True)
            self.conv2.requires_grad_(True)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        device = buffer_device or self._latent_mean.device
        self._latent_mean = torch.tensor(self.config.latent_mean, dtype=torch.float32, device=device)
        self._latent_std = torch.tensor(self.config.latent_std, dtype=torch.float32, device=device)

    def init_states(self, *, buffer_device: torch.device | None = None) -> None:
        """Initialize a meta-built model before training from scratch."""

        def reset(module: nn.Module) -> None:
            module = getattr(module, "_checkpoint_wrapped_module", module)
            reset_parameters = getattr(module, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()

        self.apply(reset)
        for module in self.modules():
            module = getattr(module, "_checkpoint_wrapped_module", module)
            if isinstance(module, AttentionBlock):
                nn.init.zeros_(module.proj.weight)
        self._init_self_buffers(buffer_device=buffer_device)
        self._apply_training_stage()

    @staticmethod
    def _validate_video(video: torch.Tensor) -> None:
        if video.ndim != 5:
            raise ValueError(f"video must have shape [B, C, T, H, W], got {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"video must have three RGB channels, got {video.shape[1]}")
        if video.shape[-2] % 16 or video.shape[-1] % 16:
            raise ValueError("video height and width must be divisible by 16")
        _stream_chunk_lengths(video.shape[2], 4)

    def _scale_latents(self, latents: torch.Tensor) -> torch.Tensor:
        stats_dtype = (
            torch.get_autocast_dtype(latents.device.type)
            if torch.is_autocast_enabled(latents.device.type)
            else latents.dtype
        )
        mean = self._latent_mean.to(device=latents.device, dtype=stats_dtype)
        std = self._latent_std.to(device=latents.device, dtype=stats_dtype)
        # Match the official wrapper, which computes reciprocal statistics
        # before entering autocast. CUDA autocast otherwise promotes reciprocal
        # to FP32 and changes the published BF16 scaling constants.
        with torch.autocast(device_type=latents.device.type, enabled=False):
            inv_std = std.reciprocal()
        return (latents - mean.view(1, self.z_dim, 1, 1, 1)) * inv_std.view(1, self.z_dim, 1, 1, 1)

    def _unscale_latents(self, latents: torch.Tensor) -> torch.Tensor:
        stats_dtype = (
            torch.get_autocast_dtype(latents.device.type)
            if torch.is_autocast_enabled(latents.device.type)
            else latents.dtype
        )
        mean = self._latent_mean.to(device=latents.device, dtype=stats_dtype)
        std = self._latent_std.to(device=latents.device, dtype=stats_dtype)
        with torch.autocast(device_type=latents.device.type, enabled=False):
            inv_std = std.reciprocal()
        return latents / inv_std.view(1, self.z_dim, 1, 1, 1) + mean.view(1, self.z_dim, 1, 1, 1)

    def _encode_impl(self, video: torch.Tensor, *, chunk_size: int | None = 4) -> torch.Tensor:
        self._validate_video(video)
        x = patchify(video, patch_size=2)
        feat_cache: list[CacheEntry] = [None] * self._encoder_cache_slots
        outputs: list[torch.Tensor] = []
        offset = 0
        for length in _stream_chunk_lengths(x.shape[2], chunk_size):
            outputs.append(
                self.encoder(
                    x[:, :, offset : offset + length],
                    feat_cache=feat_cache,
                )
            )
            offset += length
        posterior = self.conv1(torch.cat(outputs, dim=2))
        mu, _log_var = posterior.chunk(2, dim=1)
        return self._scale_latents(mu)

    def encode(
        self,
        video: torch.Tensor,
        *,
        chunk_size: int | None = 4,
        output_dtype: torch.dtype | None = torch.float32,
    ) -> torch.Tensor:
        """Encode a full clip using official or coalesced causal chunks.

        ``chunk_size=4`` is the official path (one frame, then four at a time).
        ``chunk_size=None`` sends all frames after the first as one causal chunk;
        the two modes are expected to match within numerical tolerance.
        """

        latents = self._encode_impl(video, chunk_size=chunk_size)
        return latents.to(output_dtype) if output_dtype is not None else latents

    def encode_views(
        self,
        videos: torch.Tensor,
        *,
        chunk_size: int | None = 4,
        output_dtype: torch.dtype | None = torch.float32,
    ) -> torch.Tensor:
        if videos.ndim != 6:
            raise ValueError(f"videos must have shape [B, V, C, T, H, W], got {tuple(videos.shape)}")
        b, views, channels, frames, height, width = videos.shape
        flat = videos.reshape(b * views, channels, frames, height, width)
        latents = self.encode(flat, chunk_size=chunk_size, output_dtype=output_dtype)
        return latents.unflatten(0, (b, views))

    def _decode_impl(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 5 or latents.shape[1] != self.z_dim:
            raise ValueError(f"latents must have shape [B, {self.z_dim}, T, H, W], got {tuple(latents.shape)}")
        z = self._unscale_latents(latents)
        x = self.conv2(z)
        feat_cache: list[CacheEntry] = [None] * self._decoder_cache_slots
        outputs: list[torch.Tensor] = []
        for index in range(x.shape[2]):
            outputs.append(
                self.decoder(
                    x[:, :, index : index + 1],
                    feat_cache=feat_cache,
                    first_chunk=index == 0,
                )
            )
        return unpatchify(torch.cat(outputs, dim=2), patch_size=2)

    def decode(
        self,
        latents: torch.Tensor,
        *,
        clamp: bool = True,
        output_dtype: torch.dtype | None = torch.float32,
    ) -> torch.Tensor:
        video = self._decode_impl(latents)
        if clamp:
            video = video.clamp(-1, 1)
        return video.to(output_dtype) if output_dtype is not None else video

    def decode_views(
        self,
        latents: torch.Tensor,
        *,
        clamp: bool = True,
        output_dtype: torch.dtype | None = torch.float32,
    ) -> torch.Tensor:
        if latents.ndim != 6:
            raise ValueError(f"latents must have shape [B, V, C, T, H, W], got {tuple(latents.shape)}")
        b, views, channels, frames, height, width = latents.shape
        flat = latents.reshape(b * views, channels, frames, height, width)
        video = self.decode(flat, clamp=clamp, output_dtype=output_dtype)
        return video.unflatten(0, (b, views))

    def forward(
        self,
        video: torch.Tensor,
        *,
        chunk_size: int | None = 4,
        clamp: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        if not video.is_floating_point():
            video = video.float().div(127.5).sub(1.0)
        # FSDP cannot mixed-precision cast an integer input before forward.
        # Normalization happens inside forward, so explicitly align the newly
        # floating tensor with the root-owned input convolution.
        video = video.to(dtype=self.encoder.conv1.weight.dtype)
        if video.ndim == 6:
            b, views, channels, frames, height, width = video.shape
            flat = video.reshape(b * views, channels, frames, height, width)
            reconstruction = self._decode_impl(self._encode_impl(flat, chunk_size=chunk_size))
            if clamp:
                reconstruction = reconstruction.clamp(-1, 1)
            # FSDP2 attaches its pre-backward hook to the returned tensor. Do
            # not return the view produced by unflatten, since an in-place loss
            # operation on that view could otherwise discard the hook.
            return reconstruction.unflatten(0, (b, views)).clone()
        reconstruction = self._decode_impl(self._encode_impl(video, chunk_size=chunk_size))
        return reconstruction.clamp(-1, 1) if clamp else reconstruction

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | pathlib.Path,
        *,
        config: Config | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
        strict: bool = True,
    ) -> "WanVAE":
        """Load the official ``Wan2.2_VAE.pth`` with mmap-backed CPU storage."""

        path = pathlib.Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        config = config or cls.Config()
        with torch.device("meta"):
            model = cls(config)
        state_dict = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        incompatible = model.load_state_dict(state_dict, strict=strict, assign=True)
        if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
            raise RuntimeError(
                "official checkpoint did not load strictly: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        model._init_self_buffers(buffer_device=torch.device("cpu"))
        model.to(device=torch.device(device), dtype=dtype)
        model._apply_training_stage()
        return model.eval()
