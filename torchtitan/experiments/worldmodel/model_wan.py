# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Portions Copyright 2024-2025 The Alibaba Wan Team Authors and licensed
# under the Apache License, Version 2.0. This file is a modified, PyTorch-native
# local port of Wan2.2's wan/modules/model.py. It keeps the upstream module and
# parameter names so official Wan2.2 checkpoints load without conversion.

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import pathlib
import sys
import types
from collections.abc import Mapping, Sequence
from copy import copy
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, fields
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.config import CompileConfig, ParallelismConfig, TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.config.configurable import Configurable
from torchtitan.models.common.nn_modules import Linear
from torchtitan.protocols.model import BaseModel
from torchtitan.tools.logging import logger


# Shape suffix legend for this file:
# B: batch, S: padded video-patch sequence, L: text sequence, N: attention
# heads, D: head dimension, C: channel/embedding dimension, F/H/W: video
# frame/height/width dimensions, and M: a flattened position dimension.

WAN_TI2V_5B_REPO_ID = "Wan-AI/Wan2.2-TI2V-5B"
WAN_TI2V_5B_MODEL_TYPE = "ti2v"
WAN_TI2V_5B_VAE_STRIDE = (4, 16, 16)


@dataclass(kw_only=True, slots=True)
class WanAttentionLinearsConfig(Configurable.Config):
    q: Linear.Config
    k: Linear.Config
    v: Linear.Config
    o: Linear.Config


@dataclass(kw_only=True, slots=True)
class WanAttentionBlockLinearsConfig(Configurable.Config):
    self_attn: WanAttentionLinearsConfig
    cross_attn: WanAttentionLinearsConfig
    # Keep the list indices aligned with nn.Sequential so converter FQNs match
    # official checkpoint names (blocks.0.ffn.0 and blocks.0.ffn.2).
    ffn: list[Linear.Config | None]


@dataclass(kw_only=True, slots=True)
class WanHeadLinearsConfig(Configurable.Config):
    head: Linear.Config


def _linear_config(
    in_features: int,
    out_features: int,
    *,
    bias: bool = True,
    current: Linear.Config | None = None,
) -> Linear.Config:
    """Preserve converter-produced Linear configs when dimensions still match."""
    if (
        current is not None
        and current.in_features == in_features
        and current.out_features == out_features
        and current.bias == bias
    ):
        return current
    return Linear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
    )


def _linear_at(
    configs: Sequence[Linear.Config | None] | None,
    index: int,
) -> Linear.Config | None:
    if configs is None or index >= len(configs):
        return None
    return configs[index]


def _attention_linears_config(
    dim: int,
    current: WanAttentionLinearsConfig | None = None,
) -> WanAttentionLinearsConfig:
    return WanAttentionLinearsConfig(
        q=_linear_config(dim, dim, current=None if current is None else current.q),
        k=_linear_config(dim, dim, current=None if current is None else current.k),
        v=_linear_config(dim, dim, current=None if current is None else current.v),
        o=_linear_config(dim, dim, current=None if current is None else current.o),
    )


def _block_linears_config(
    dim: int,
    ffn_dim: int,
    current: WanAttentionBlockLinearsConfig | None = None,
) -> WanAttentionBlockLinearsConfig:
    current_ffn = None if current is None else current.ffn
    return WanAttentionBlockLinearsConfig(
        self_attn=_attention_linears_config(
            dim,
            None if current is None else current.self_attn,
        ),
        cross_attn=_attention_linears_config(
            dim,
            None if current is None else current.cross_attn,
        ),
        ffn=[
            _linear_config(
                dim,
                ffn_dim,
                current=_linear_at(current_ffn, 0),
            ),
            None,
            _linear_config(
                ffn_dim,
                dim,
                current=_linear_at(current_ffn, 2),
            ),
        ],
    )


def sinusoidal_embedding_1d(dim: int, position_M: torch.Tensor) -> torch.Tensor:
    """Build the timestep embedding used by the official Wan implementation."""
    if dim % 2 != 0:
        raise ValueError(f"sinusoidal embedding dim must be even, got {dim}")
    half = dim // 2
    position_M = position_M.to(torch.float64)
    frequencies_D = torch.pow(
        position_M.new_tensor(10000.0),
        -torch.arange(half, device=position_M.device, dtype=torch.float64) / half,
    )
    sinusoid_MD = torch.outer(position_M, frequencies_D)
    return torch.cat((torch.cos(sinusoid_MD), torch.sin(sinusoid_MD)), dim=1)


def rope_params(
    max_seq_len: int,
    dim: int,
    theta: float = 10000.0,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return complex rotary frequencies with Wan's float64 construction."""
    if dim % 2 != 0:
        raise ValueError(f"RoPE dim must be even, got {dim}")
    positions_M = torch.arange(max_seq_len, device=device, dtype=torch.float64)
    exponents_D = torch.arange(0, dim, 2, device=device, dtype=torch.float64) / dim
    angles_MD = torch.outer(positions_M, 1.0 / torch.pow(theta, exponents_D))
    return torch.polar(torch.ones_like(angles_MD), angles_MD)


def _build_rope_freqs(
    max_seq_len: int,
    head_dim: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    spatial_dim = 2 * (head_dim // 6)
    temporal_dim = head_dim - 2 * spatial_dim
    return torch.cat(
        (
            rope_params(max_seq_len, temporal_dim, device=device),
            rope_params(max_seq_len, spatial_dim, device=device),
            rope_params(max_seq_len, spatial_dim, device=device),
        ),
        dim=1,
    )


def rope_apply(
    x_BSND: torch.Tensor,
    grid_size: tuple[int, int, int],
    freqs_MD: torch.Tensor,
) -> torch.Tensor:
    """Apply Wan's factorized 3D RoPE to a full, uniform token block."""
    seq_len = x_BSND.size(1)
    frames, height, width = grid_size
    if seq_len != frames * height * width:
        raise ValueError(f"token length {seq_len} does not match RoPE grid {grid_size}")

    complex_dim = x_BSND.size(3) // 2
    temporal_freqs, height_freqs, width_freqs = freqs_MD.split(
        (complex_dim - 2 * (complex_dim // 3), complex_dim // 3, complex_dim // 3),
        dim=1,
    )
    position_freqs_SD = torch.cat(
        (
            temporal_freqs[:frames].view(frames, 1, 1, -1).expand(frames, height, width, -1),
            height_freqs[:height].view(1, height, 1, -1).expand(frames, height, width, -1),
            width_freqs[:width].view(1, 1, width, -1).expand(frames, height, width, -1),
        ),
        dim=-1,
    ).reshape(seq_len, 1, -1)
    x_BSND_complex = torch.view_as_complex(x_BSND.to(torch.float64).reshape(*x_BSND.shape[:-1], -1, 2))
    return torch.view_as_real(x_BSND_complex * position_freqs_SD.unsqueeze(0)).flatten(3).float()


def _autocast_disabled(tensor: torch.Tensor) -> AbstractContextManager:
    return torch.autocast(device_type=tensor.device.type, enabled=False)


def _attention_mask(
    *,
    query_len: int,
    key_len: int,
    key_lens_B: torch.Tensor | None,
    window_size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor | None:
    global_window = window_size == (-1, -1)
    if global_window and key_lens_B is None:
        return None
    # Avoid a data-dependent bool in compiled blocks. Eager uniform-shape calls
    # retain the mask-free SDPA fast path; compiled calls use an all-true mask.
    if (
        global_window
        and not torch.compiler.is_compiling()
        and key_lens_B is not None
        and bool(torch.all(key_lens_B == key_len))
    ):
        return None

    allowed_BQK = torch.ones(
        (1 if key_lens_B is None else key_lens_B.numel(), query_len, key_len),
        dtype=torch.bool,
        device=device,
    )
    if key_lens_B is not None:
        key_positions_K = torch.arange(key_len, device=device)
        allowed_BQK &= key_positions_K.view(1, 1, key_len) < key_lens_B.to(device=device).view(-1, 1, 1)

    left, right = window_size
    if not global_window:
        query_positions_Q = torch.arange(query_len, device=device).view(1, query_len, 1)
        key_positions_K = torch.arange(key_len, device=device).view(1, 1, key_len)
        if left >= 0:
            allowed_BQK &= key_positions_K >= query_positions_Q - left
        if right >= 0:
            allowed_BQK &= key_positions_K <= query_positions_Q + right
    return allowed_BQK.unsqueeze(1)


def _scaled_dot_product_attention(
    q_BSND: torch.Tensor,
    k_BLND: torch.Tensor,
    v_BLND: torch.Tensor,
    *,
    key_lens_B: torch.Tensor | None = None,
    window_size: tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    # Upstream FlashAttention casts normalized float32 queries and keys to the
    # value dtype before dispatching the kernel.
    q_BSND = q_BSND.to(v_BLND.dtype)
    k_BLND = k_BLND.to(v_BLND.dtype)
    q_BNSD = q_BSND.transpose(1, 2)
    k_BNLD = k_BLND.transpose(1, 2)
    v_BNLD = v_BLND.transpose(1, 2)
    attention_mask_B1SL = _attention_mask(
        query_len=q_BNSD.size(2),
        key_len=k_BNLD.size(2),
        key_lens_B=key_lens_B,
        window_size=window_size,
        device=q_BNSD.device,
    )
    output_BNSD = F.scaled_dot_product_attention(
        q_BNSD,
        k_BNLD,
        v_BNLD,
        attn_mask=attention_mask_B1SL,
        dropout_p=0.0,
        is_causal=False,
    )
    return output_BNSD.transpose(1, 2).contiguous()


class WanRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x_BSC: torch.Tensor) -> torch.Tensor:
        normalized_BSC = x_BSC.float()
        normalized_BSC = normalized_BSC * torch.rsqrt(normalized_BSC.square().mean(dim=-1, keepdim=True) + self.eps)
        return normalized_BSC.to(x_BSC.dtype) * self.weight

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim: int, eps: float = 1e-6, *, elementwise_affine: bool = False):
        super().__init__(dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x_BSC: torch.Tensor) -> torch.Tensor:
        weight_C = self.weight.float() if self.weight is not None else None
        bias_C = self.bias.float() if self.bias is not None else None
        normalized_BSC = F.layer_norm(x_BSC.float(), self.normalized_shape, weight_C, bias_C, self.eps)
        return normalized_BSC.to(x_BSC.dtype)


class WanSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6,
        linears: WanAttentionLinearsConfig | None = None,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        linears = linears or _attention_linears_config(dim)
        self.q = linears.q.build()
        self.k = linears.k.build()
        self.v = linears.v.build()
        self.o = linears.o.build()
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x_BSC: torch.Tensor,
        grid_size: tuple[int, int, int],
        freqs_MD: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = x_BSC.shape[:2]
        q_BSND = self.norm_q(self.q(x_BSC)).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k_BSND = self.norm_k(self.k(x_BSC)).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v_BSND = self.v(x_BSC).view(batch_size, seq_len, self.num_heads, self.head_dim)

        q_BSND = rope_apply(q_BSND, grid_size, freqs_MD).to(v_BSND.dtype)
        k_BSND = rope_apply(k_BSND, grid_size, freqs_MD).to(v_BSND.dtype)
        output_BSND = _scaled_dot_product_attention(
            q_BSND,
            k_BSND,
            v_BSND,
            window_size=self.window_size,
        )
        return self.o(output_BSND.flatten(2))


class WanCrossAttention(WanSelfAttention):
    def forward(
        self,
        x_BSC: torch.Tensor,
        context_BLC: torch.Tensor,
        context_lens_B: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = x_BSC.size(0)
        query_len = x_BSC.size(1)
        context_len = context_BLC.size(1)
        q_BSND = self.norm_q(self.q(x_BSC)).view(batch_size, query_len, self.num_heads, self.head_dim)
        k_BLND = self.norm_k(self.k(context_BLC)).view(batch_size, context_len, self.num_heads, self.head_dim)
        v_BLND = self.v(context_BLC).view(batch_size, context_len, self.num_heads, self.head_dim)
        output_BSND = _scaled_dot_product_attention(
            q_BSND,
            k_BLND,
            v_BLND,
            key_lens_B=context_lens_B,
        )
        return self.o(output_BSND.flatten(2))


class WanAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        window_size: tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        linears: WanAttentionBlockLinearsConfig | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        linears = linears or _block_linears_config(dim, ffn_dim)

        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(
            dim,
            num_heads,
            window_size,
            qk_norm,
            eps,
            linears.self_attn,
        )
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(
            dim,
            num_heads,
            (-1, -1),
            qk_norm,
            eps,
            linears.cross_attn,
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            linears.ffn[0].build(),
            nn.GELU(approximate="tanh"),
            linears.ffn[2].build(),
        )
        self.modulation = nn.Parameter(torch.empty(1, 6, dim))

    def forward(
        self,
        x_BSC: torch.Tensor,
        e_BS6C: torch.Tensor,
        grid_size: tuple[int, int, int],
        freqs_MD: torch.Tensor,
        context_BLC: torch.Tensor,
        context_lens_B: torch.Tensor | None,
    ) -> torch.Tensor:
        if e_BS6C.dtype != torch.float32:
            raise ValueError(f"Wan modulation must be float32, got {e_BS6C.dtype}")
        with _autocast_disabled(x_BSC):
            modulation_BS6C = self.modulation.float().unsqueeze(0) + e_BS6C
            shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = modulation_BS6C.chunk(6, dim=2)

        attention_input_BSC = self.norm1(x_BSC).float() * (1 + scale_msa.squeeze(2)) + shift_msa.squeeze(2)
        attention_output_BSC = self.self_attn(
            attention_input_BSC.to(self.self_attn.q.weight.dtype),
            grid_size,
            freqs_MD,
        )
        with _autocast_disabled(x_BSC):
            x_BSC = x_BSC.float() + attention_output_BSC.float() * gate_msa.squeeze(2)

        cross_attention_dtype = self.cross_attn.q.weight.dtype
        cross_attention_output_BSC = self.cross_attn(
            self.norm3(x_BSC).to(cross_attention_dtype),
            context_BLC.to(cross_attention_dtype),
            context_lens_B,
        )
        with _autocast_disabled(x_BSC):
            x_BSC = x_BSC.float() + cross_attention_output_BSC.float()
        ffn_input_BSC = self.norm2(x_BSC).float() * (1 + scale_ffn.squeeze(2)) + shift_ffn.squeeze(2)
        ffn_output_BSC = self.ffn(ffn_input_BSC.to(self.ffn[0].weight.dtype))
        with _autocast_disabled(x_BSC):
            return x_BSC.float() + ffn_output_BSC.float() * gate_ffn.squeeze(2)


class Head(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: int,
        patch_size: tuple[int, int, int],
        eps: float = 1e-6,
        linears: WanHeadLinearsConfig | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps
        self.norm = WanLayerNorm(dim, eps)
        linears = linears or WanHeadLinearsConfig(head=_linear_config(dim, math.prod(patch_size) * out_dim))
        self.head = linears.head.build()
        self.modulation = nn.Parameter(torch.empty(1, 2, dim))

    def forward(self, x_BSC: torch.Tensor, e_BSC: torch.Tensor) -> torch.Tensor:
        if e_BSC.dtype != torch.float32:
            raise ValueError(f"Wan head modulation must be float32, got {e_BSC.dtype}")
        with _autocast_disabled(x_BSC):
            shift_BS1C, scale_BS1C = (self.modulation.float().unsqueeze(0) + e_BSC.unsqueeze(2)).chunk(2, dim=2)
            normalized_BSC = self.norm(x_BSC).float()
            bias_C = self.head.bias.float() if self.head.bias is not None else None
            return F.linear(
                normalized_BSC * (1 + scale_BS1C.squeeze(2)) + shift_BS1C.squeeze(2),
                self.head.weight.float(),
                bias_C,
            )


class WanModel(BaseModel):
    """Wan2.2 transformer with official checkpoint and forward compatibility."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        model_type: str = WAN_TI2V_5B_MODEL_TYPE
        patch_size: tuple[int, int, int] = (1, 2, 2)
        text_len: int = 512
        in_dim: int = 48
        dim: int = 3072
        ffn_dim: int = 14336
        freq_dim: int = 256
        text_dim: int = 4096
        out_dim: int = 48
        num_heads: int = 24
        num_layers: int = 30
        window_size: tuple[int, int] = (-1, -1)
        qk_norm: bool = True
        cross_attn_norm: bool = True
        eps: float = 1e-6
        rope_max_seq_len: int = 1024
        # Derived configurable Linear trees let TorchTitan's training-time FP8
        # converter swap implementations without changing official module FQNs.
        text_embedding: list[Linear.Config | None] = field(init=False)
        time_embedding: list[Linear.Config | None] = field(init=False)
        time_projection: list[Linear.Config | None] = field(init=False)
        blocks: list[WanAttentionBlockLinearsConfig] = field(init=False)
        head: WanHeadLinearsConfig = field(init=False)

        def __post_init__(self) -> None:
            if self.model_type not in {"t2v", "i2v", "ti2v", "s2v"}:
                raise ValueError(f"unsupported Wan model_type {self.model_type!r}")
            if len(self.patch_size) != 3 or any(size <= 0 for size in self.patch_size):
                raise ValueError(f"patch_size must contain three positive values, got {self.patch_size}")
            if self.dim % self.num_heads != 0:
                raise ValueError(f"dim {self.dim} must be divisible by num_heads {self.num_heads}")
            head_dim = self.dim // self.num_heads
            if head_dim % 2 != 0:
                raise ValueError(f"attention head dim must be even, got {head_dim}")
            if self.freq_dim % 2 != 0:
                raise ValueError(f"freq_dim must be even, got {self.freq_dim}")
            if self.text_len <= 0 or self.num_layers <= 0 or self.rope_max_seq_len <= 0:
                raise ValueError("text_len, num_layers, and rope_max_seq_len must be positive")
            self._sync_derived_fields()

        def _sync_derived_fields(self) -> None:
            current_text = getattr(self, "text_embedding", None)
            current_time = getattr(self, "time_embedding", None)
            current_projection = getattr(self, "time_projection", None)
            current_blocks = getattr(self, "blocks", [])
            current_head = getattr(self, "head", None)
            self.text_embedding = [
                _linear_config(
                    self.text_dim,
                    self.dim,
                    current=_linear_at(current_text, 0),
                ),
                None,
                _linear_config(
                    self.dim,
                    self.dim,
                    current=_linear_at(current_text, 2),
                ),
            ]
            self.time_embedding = [
                _linear_config(
                    self.freq_dim,
                    self.dim,
                    current=_linear_at(current_time, 0),
                ),
                None,
                _linear_config(
                    self.dim,
                    self.dim,
                    current=_linear_at(current_time, 2),
                ),
            ]
            self.time_projection = [
                None,
                _linear_config(
                    self.dim,
                    self.dim * 6,
                    current=_linear_at(current_projection, 1),
                ),
            ]
            self.blocks = [
                _block_linears_config(
                    self.dim,
                    self.ffn_dim,
                    current_blocks[layer_idx] if layer_idx < len(current_blocks) else None,
                )
                for layer_idx in range(self.num_layers)
            ]
            self.head = WanHeadLinearsConfig(
                head=_linear_config(
                    self.dim,
                    math.prod(self.patch_size) * self.out_dim,
                    current=None if current_head is None else current_head.head,
                )
            )

        @classmethod
        def from_dict(cls, values: Mapping[str, Any]) -> "WanModel.Config":
            config_fields = {item.name for item in fields(cls)}
            init_fields = {item.name for item in fields(cls) if item.init}
            unknown = sorted(key for key in values if not key.startswith("_") and key not in config_fields)
            if unknown:
                raise ValueError(f"unsupported Wan config fields: {unknown}")
            kwargs = {key: value for key, value in values.items() if not key.startswith("_") and key in init_fields}
            for name in ("patch_size", "window_size"):
                if name in kwargs:
                    kwargs[name] = tuple(kwargs[name])
            return cls(**kwargs)

        def update_from_config(self, *, config: Any, **kwargs: Any) -> None:
            del kwargs
            if config.parallelism.spmd_backend == "full_dtensor":
                raise ValueError("Wan supports FSDP/HSDP only, not full DTensor")
            unsupported = {
                "tensor parallel": config.parallelism.tensor_parallel_degree,
                "context parallel": config.parallelism.context_parallel_degree,
                "pipeline parallel": config.parallelism.pipeline_parallel_degree,
                "expert parallel": config.parallelism.expert_parallel_degree,
            }
            for name, degree in unsupported.items():
                if degree > 1:
                    raise ValueError(f"Wan supports FSDP/HSDP only, not {name}")
            self._sync_derived_fields()

        def build(self, **kwargs: Any) -> "WanModel":
            if kwargs:
                raise ValueError("WanModel.Config.build does not accept kwargs")
            if self._owner is None:
                raise NotImplementedError("WanModel.Config has no owner class")
            self._sync_derived_fields()
            # Shallow-copy derived configs so converter-produced Float8Linear
            # configs survive construction (dataclasses.replace drops init=False).
            return self._owner(config=copy(self))

        def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
            num_parameters = sum(parameter.numel() for parameter in model.parameters())
            attention_flops = 12 * self.num_layers * self.dim * seq_len
            return num_parameters, 6 * num_parameters + attention_flops

    def __init__(self, config: Config):
        super().__init__()
        config._sync_derived_fields()
        self.config = config
        self.model_type = config.model_type
        self.patch_size = config.patch_size
        self.text_len = config.text_len
        self.in_dim = config.in_dim
        self.dim = config.dim
        self.ffn_dim = config.ffn_dim
        self.freq_dim = config.freq_dim
        self.text_dim = config.text_dim
        self.out_dim = config.out_dim
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.window_size = config.window_size
        self.qk_norm = config.qk_norm
        self.cross_attn_norm = config.cross_attn_norm
        self.eps = config.eps

        self.patch_embedding = nn.Conv3d(
            config.in_dim,
            config.dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.text_embedding = nn.Sequential(
            config.text_embedding[0].build(),
            nn.GELU(approximate="tanh"),
            config.text_embedding[2].build(),
        )
        self.time_embedding = nn.Sequential(
            config.time_embedding[0].build(),
            nn.SiLU(),
            config.time_embedding[2].build(),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            config.time_projection[1].build(),
        )
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    config.dim,
                    config.ffn_dim,
                    config.num_heads,
                    config.window_size,
                    config.qk_norm,
                    config.cross_attn_norm,
                    config.eps,
                    config.blocks[layer_idx],
                )
                for layer_idx in range(config.num_layers)
            ]
        )
        self.head = Head(
            config.dim,
            config.out_dim,
            config.patch_size,
            config.eps,
            config.head,
        )

        head_dim = config.dim // config.num_heads
        # Keep this as a plain tensor, matching upstream. Registering it as a
        # buffer would cast its complex128 values when model.to(dtype=...) runs.
        self.freqs = _build_rope_freqs(
            config.rope_max_seq_len,
            head_dim,
            device=torch.device("cpu"),
        )
        self.reset_parameters()

    def verify_module_protocol(self) -> None:
        # Parameter FQNs must mirror upstream nn.Module containers exactly.
        pass

    def reset_parameters(self) -> None:
        self.patch_embedding.reset_parameters()
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, WanRMSNorm):
                module.reset_parameters()
            elif isinstance(module, WanLayerNorm) and module.elementwise_affine:
                module.reset_parameters()

        for module in self.text_embedding.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
        for module in self.time_embedding.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
        for block in self.blocks:
            nn.init.normal_(block.modulation, std=self.dim**-0.5)
        nn.init.normal_(self.head.modulation, std=self.dim**-0.5)
        nn.init.zeros_(self.head.head.weight)

    def _get_rope_freqs(self, device: torch.device) -> torch.Tensor:
        if self.freqs.device.type == "meta":
            self.freqs = _build_rope_freqs(
                self.config.rope_max_seq_len,
                self.dim // self.num_heads,
                device=device,
            )
        elif self.freqs.device != device:
            self.freqs = self.freqs.to(device=device)
        return self.freqs

    def get_null_text_embedding(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the fixed raw context used when no text encoder is present."""
        context_B1C = torch.zeros(
            (batch_size, 1, self.text_dim),
            device=device,
            dtype=dtype,
        )
        context_B1C[..., 0] = 1
        return context_B1C

    def forward(
        self,
        x: Sequence[torch.Tensor],
        t: torch.Tensor,
        context: Sequence[torch.Tensor],
        seq_len: int,
        y: Sequence[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        """Run the upstream-compatible list-based Wan transformer forward."""
        batch_size = len(x)
        if batch_size == 0:
            raise ValueError("x must contain at least one video latent")
        if len(context) != batch_size:
            raise ValueError(f"context batch {len(context)} does not match x batch {batch_size}")
        if y is not None and len(y) != batch_size:
            raise ValueError(f"y batch {len(y)} does not match x batch {batch_size}")
        if self.model_type == "i2v" and y is None:
            raise ValueError("Wan i2v requires conditional video inputs in y")

        if y is not None:
            x = [torch.cat((sample_CFH, condition_CFH), dim=0) for sample_CFH, condition_CFH in zip(x, y)]

        latent_shape = tuple(x[0].shape)
        for sample_CFH in x:
            if sample_CFH.ndim != 4:
                raise ValueError(f"each x sample must have shape [C, F, H, W], got {sample_CFH.shape}")
            if sample_CFH.size(0) != self.in_dim:
                raise ValueError(f"patched input has {sample_CFH.size(0)} channels, expected {self.in_dim}")
            if tuple(sample_CFH.shape) != latent_shape:
                raise ValueError(
                    "all Wan latent samples must have the same shape; "
                    f"expected {latent_shape}, got {tuple(sample_CFH.shape)}"
                )

        device = self.patch_embedding.weight.device
        freqs_MD = self._get_rope_freqs(device)
        patches_BCFHW = self.patch_embedding(torch.stack(tuple(x)))
        grid_size = tuple(patches_BCFHW.shape[2:])
        x_BSC = patches_BCFHW.flatten(2).transpose(1, 2)
        if x_BSC.size(1) != seq_len:
            raise ValueError(f"full patch sequence has length {x_BSC.size(1)}, but seq_len is {seq_len}")

        if t.ndim == 1:
            if t.numel() != batch_size:
                raise ValueError(f"t batch {t.numel()} does not match x batch {batch_size}")
            t_BS = t[:, None].expand(batch_size, seq_len)
        elif t.shape == (batch_size, seq_len):
            t_BS = t
        else:
            raise ValueError(f"t must have shape [{batch_size}] or [{batch_size}, {seq_len}], got {tuple(t.shape)}")

        with _autocast_disabled(t_BS):
            timestep_embedding_BSC = sinusoidal_embedding_1d(self.freq_dim, t_BS.flatten()).unflatten(
                0, (batch_size, seq_len)
            )
            time_dtype = self.time_embedding[0].weight.dtype
            e_BSC = self.time_embedding(timestep_embedding_BSC.to(time_dtype)).float()
            projection_dtype = self.time_projection[1].weight.dtype
            e_BS6C = self.time_projection(e_BSC.to(projection_dtype)).float().unflatten(2, (6, self.dim))

        padded_context = []
        for context_LC in context:
            if context_LC.ndim != 2 or context_LC.size(1) != self.text_dim:
                raise ValueError(f"each context must have shape [L, {self.text_dim}], got {tuple(context_LC.shape)}")
            if context_LC.size(0) > self.text_len:
                raise ValueError(f"context length {context_LC.size(0)} exceeds text_len {self.text_len}")
            padded_context.append(
                torch.cat(
                    (
                        context_LC,
                        context_LC.new_zeros(self.text_len - context_LC.size(0), self.text_dim),
                    ),
                    dim=0,
                )
            )
        context_BLC = self.text_embedding(torch.stack(padded_context))

        for block in self.blocks:
            x_BSC = block(
                x_BSC,
                e_BS6C,
                grid_size,
                freqs_MD,
                context_BLC,
                None,
            )

        output_BSC = self.head(x_BSC, e_BSC)
        return [sample_CFH.float() for sample_CFH in self.unpatchify(output_BSC, grid_size)]

    def unpatchify(
        self,
        x_BSC: torch.Tensor,
        grid_size: tuple[int, int, int],
    ) -> list[torch.Tensor]:
        x_BFHWPTPHWPC = x_BSC.view(x_BSC.size(0), *grid_size, *self.patch_size, self.out_dim)
        x_BCFPTHPHWPW = torch.einsum("bfhwpqrc->bcfphqwr", x_BFHWPTPHWPC)
        x_BCFHW = x_BCFPTHPHWPW.reshape(
            x_BSC.size(0),
            self.out_dim,
            *[grid_dim * patch_dim for grid_dim, patch_dim in zip(grid_size, self.patch_size)],
        )
        return list(x_BCFHW.unbind(0))

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str | pathlib.Path,
        *,
        config: Config | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
        strict: bool = True,
    ) -> "WanModel":
        """Load an official Wan checkpoint without materializing a second state dict."""
        checkpoint_path = pathlib.Path(checkpoint_dir)
        if not checkpoint_path.is_dir():
            raise ValueError(f"checkpoint_dir is not a directory: {checkpoint_path}")
        if config is None:
            config_path = checkpoint_path / "config.json"
            if not config_path.is_file():
                raise ValueError(f"Wan config.json was not found in {checkpoint_path}")
            config = cls.Config.from_dict(json.loads(config_path.read_text()))

        with torch.device("meta"):
            model = cls(config)
        if dtype is not None:
            model.to(dtype=dtype)
        model.to_empty(device=torch.device(device))
        _load_safetensors_into_model(model, checkpoint_path, strict=strict)
        return model.eval()


def parallelize_wan(
    model: WanModel,
    *,
    parallel_dims: Any,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: Any,
    dump_folder: str,
) -> WanModel:
    """Apply the DiT training stack supported by Wan: AC, compile, and FSDP/HSDP."""
    if parallelism.spmd_backend == "full_dtensor":
        raise ValueError("Wan supports FSDP/HSDP only, not full DTensor")
    if parallel_dims.tp_enabled or parallel_dims.pp_enabled or parallel_dims.cp_enabled or parallel_dims.ep_enabled:
        raise ValueError("Wan supports FSDP/HSDP only")

    if ac_config is not None:
        _apply_wan_activation_checkpointing(
            model,
            ac_config,
            dump_folder=dump_folder,
        )
    if compile_config.enable and "model" in compile_config.components:
        _apply_wan_compile(model, compile_config)

    dp_mesh = parallel_dims.get_activated_mesh(["dp_replicate", "fsdp"])
    if dp_mesh is None:
        dp_mesh = parallel_dims.get_mesh("fsdp")
    _apply_wan_fsdp(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )
    logger.info("Applied HSDP to Wan" if parallel_dims.dp_replicate_enabled else "Applied FSDP to Wan")
    return model


def _apply_wan_activation_checkpointing(
    model: WanModel,
    ac_config: Any,
    *,
    dump_folder: str,
) -> None:
    from torchtitan.distributed.activation_checkpoint import FullAC, MemoryBudgetAC

    ac_policy = ac_config.build(dump_folder=dump_folder)
    if isinstance(ac_policy, MemoryBudgetAC):
        raise ValueError("Wan does not support memory-budget activation checkpointing")

    for layer_id, block in model.blocks.named_children():
        model.blocks.register_module(
            layer_id,
            ac_policy._wrap_block(block, base_fqn=f"blocks.{layer_id}"),
        )
    mode = "full" if isinstance(ac_policy, FullAC) else "selective"
    logger.info(f"Applied {mode} activation checkpointing to Wan")


def _apply_wan_compile(model: WanModel, compile_config: CompileConfig) -> None:
    """Compile Wan's custom repeated blocks and output head for training."""
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
    for block in model.blocks:
        block.compile(backend=compile_config.backend, fullgraph=True)
    model.head.compile(backend=compile_config.backend, fullgraph=True)
    logger.info("Compiling Wan attention blocks and output head with torch.compile")


def _apply_wan_fsdp(
    model: WanModel,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool,
    reshard_after_forward_policy: str,
    enable_symm_mem: bool,
) -> None:
    from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

    from torchtitan.distributed.fsdp import enable_fsdp_symm_mem, get_fsdp_reshard_after_forward_policy

    cast_inputs_mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=True,
    )
    preserve_inputs_mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    cast_inputs_fsdp_config: dict[str, Any] = {
        "mesh": dp_mesh,
        "mp_policy": cast_inputs_mp_policy,
    }
    preserve_inputs_fsdp_config: dict[str, Any] = {
        "mesh": dp_mesh,
        "mp_policy": preserve_inputs_mp_policy,
    }
    if cpu_offload:
        offload_policy = CPUOffloadPolicy()
        cast_inputs_fsdp_config["offload_policy"] = offload_policy
        preserve_inputs_fsdp_config["offload_policy"] = offload_policy
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy,
        pp_enabled,
    )

    for module in (
        model.patch_embedding,
        model.text_embedding,
        model.time_embedding,
        model.time_projection,
    ):
        fully_shard(
            module,
            **cast_inputs_fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    for block in model.blocks:
        fully_shard(
            block,
            **preserve_inputs_fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    fully_shard(
        model.head,
        **preserve_inputs_fsdp_config,
        reshard_after_forward=reshard_after_forward,
    )
    # Wan deliberately promotes timestep modulation and residual math to
    # float32. Preserve those tensors at the separately sharded block/head
    # boundaries; casting them back to float32 inside the modules would have
    # already discarded their precision. The embedding modules above still
    # cast inputs to the parameter compute dtype for their linear/conv ops.
    fully_shard(model, **preserve_inputs_fsdp_config)
    if enable_symm_mem:
        enable_fsdp_symm_mem(model)


def wan_ti2v_5b_config() -> WanModel.Config:
    """Return the official Wan2.2 TI2V-5B transformer configuration."""
    return WanModel.Config()


def wan_debug_config() -> WanModel.Config:
    """Return a reduced Wan configuration for CPU training smoke tests."""
    return WanModel.Config(
        patch_size=(1, 2, 2),
        text_len=4,
        in_dim=4,
        dim=32,
        ffn_dim=64,
        freq_dim=16,
        text_dim=16,
        out_dim=4,
        num_heads=4,
        num_layers=1,
        rope_max_seq_len=16,
    )


def _safetensors_paths(checkpoint_dir: pathlib.Path) -> list[pathlib.Path]:
    index_path = checkpoint_dir / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        if "weight_map" not in index:
            raise ValueError(f"safetensors index has no weight_map: {index_path}")
        paths = [checkpoint_dir / name for name in dict.fromkeys(index["weight_map"].values())]
    else:
        candidates = (
            checkpoint_dir / "diffusion_pytorch_model.safetensors",
            checkpoint_dir / "model.safetensors",
        )
        paths = [path for path in candidates if path.is_file()]
        if not paths:
            paths = sorted(checkpoint_dir.glob("*.safetensors"))
    if not paths:
        raise ValueError(f"no safetensors weights were found in {checkpoint_dir}")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"safetensors index references missing files: {missing}")
    return paths


@torch.no_grad()
def _load_safetensors_into_model(
    model: nn.Module,
    checkpoint_dir: pathlib.Path,
    *,
    strict: bool,
) -> None:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("loading Wan weights requires the safetensors package") from error

    targets = dict(model.named_parameters()) | dict(model.named_buffers())
    loaded = set()
    unexpected = []
    for shard_path in _safetensors_paths(checkpoint_dir):
        with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
            for key in shard.keys():
                target = targets.get(key)
                if target is None:
                    unexpected.append(key)
                    continue
                source = shard.get_tensor(key)
                if source.shape != target.shape:
                    raise ValueError(
                        f"Wan checkpoint shape mismatch for {key}: {tuple(source.shape)} != {tuple(target.shape)}"
                    )
                target.copy_(source.to(device=target.device, dtype=target.dtype))
                loaded.add(key)

    missing = sorted(set(targets) - loaded)
    if strict and (missing or unexpected):
        details = []
        if missing:
            details.append(f"missing keys: {missing[:20]}")
        if unexpected:
            details.append(f"unexpected keys: {unexpected[:20]}")
        raise ValueError("Wan checkpoint is incompatible; " + "; ".join(details))


def _local_config_from_upstream(config: Mapping[str, Any]) -> WanModel.Config:
    values = {
        "model_type": config.get("model_type", WAN_TI2V_5B_MODEL_TYPE),
        "patch_size": tuple(config.get("patch_size", (1, 2, 2))),
        "text_len": config["text_len"],
        "in_dim": config.get("in_dim", 48),
        "dim": config["dim"],
        "ffn_dim": config["ffn_dim"],
        "freq_dim": config["freq_dim"],
        "text_dim": config.get("text_dim", 4096),
        "out_dim": config.get("out_dim", 48),
        "num_heads": config["num_heads"],
        "num_layers": config["num_layers"],
        "window_size": tuple(config.get("window_size", (-1, -1))),
        "qk_norm": config.get("qk_norm", True),
        "cross_attn_norm": config.get("cross_attn_norm", True),
        "eps": config.get("eps", 1e-6),
    }
    return WanModel.Config(**values)


def _import_upstream_wan(wan_repo: pathlib.Path) -> tuple[Any, Any, Any, Any]:
    repo = wan_repo.resolve()
    if not (repo / "wan" / "modules" / "model.py").is_file():
        raise ValueError(f"--wan-repo does not point to a Wan2.2 checkout: {repo}")
    loaded_wan = sys.modules.get("wan")
    if loaded_wan is not None:
        loaded_path = pathlib.Path(loaded_wan.__file__).resolve()
        if not loaded_path.is_relative_to(repo):
            raise RuntimeError(f"a different wan package is already imported from {loaded_path}")
    else:
        # Import only TI2V's modules. Wan's package __init__ eagerly imports
        # unrelated speech and animation stacks with additional dependencies.
        wan_package = types.ModuleType("wan")
        wan_package.__file__ = str(repo / "wan" / "__init__.py")
        wan_package.__path__ = [str(repo / "wan")]
        wan_package.__package__ = "wan"
        sys.modules["wan"] = wan_package
    pipeline_module = importlib.import_module("wan.textimage2video")
    model_module = importlib.import_module("wan.modules.model")
    configs = importlib.import_module("wan.configs")
    utilities = importlib.import_module("wan.utils.utils")
    return pipeline_module.WanTI2V, configs.WAN_CONFIGS, utilities.save_video, model_module


def _upstream_sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: torch.Tensor | None = None,
    k_lens: torch.Tensor | None = None,
    window_size: tuple[int, int] = (-1, -1),
    **kwargs: Any,
) -> torch.Tensor:
    """Run the upstream reference architecture with PyTorch SDPA for parity."""
    del kwargs
    if q_lens is not None and not bool(torch.all(q_lens == q.size(1))):
        raise ValueError("the parity adapter does not support padded queries")
    return _scaled_dot_product_attention(
        q,
        k,
        v,
        key_lens_B=k_lens,
        window_size=window_size,
    )


def _run_pretrained_demo(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the pretrained Wan parity and generation demo requires CUDA")
    if (args.num_frames - 1) % WAN_TI2V_5B_VAE_STRIDE[0] != 0:
        raise ValueError("--num-frames must have the form 4n+1")
    if args.width % 32 != 0 or args.height % 32 != 0:
        raise ValueError("--width and --height must be divisible by 32")

    pipeline_cls, upstream_configs, save_video, upstream_model_module = _import_upstream_wan(args.wan_repo)
    upstream_model_module.flash_attention = _upstream_sdpa_reference
    upstream_config = upstream_configs["ti2v-5B"]
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)

    pipeline = pipeline_cls(
        config=upstream_config,
        checkpoint_dir=str(args.checkpoint_dir),
        device_id=args.device,
        t5_cpu=args.t5_cpu,
        init_on_cpu=True,
        convert_model_dtype=False,
    )
    upstream_model = pipeline.model.eval().requires_grad_(False)
    local_config = _local_config_from_upstream(upstream_config)
    with torch.device("meta"):
        local_model = WanModel(local_config)
    incompatible = local_model.load_state_dict(upstream_model.state_dict(), strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected state-dict incompatibility: {incompatible}")

    latent_frames = (args.num_frames - 1) // WAN_TI2V_5B_VAE_STRIDE[0] + 1
    latent_height = args.height // WAN_TI2V_5B_VAE_STRIDE[1]
    latent_width = args.width // WAN_TI2V_5B_VAE_STRIDE[2]
    seq_len = latent_frames * latent_height * latent_width // math.prod(local_config.patch_size)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latent_CFH = torch.randn(
        local_config.in_dim,
        latent_frames,
        latent_height,
        latent_width,
        generator=generator,
        device=device,
    )
    context_LC = torch.randn(
        min(16, local_config.text_len),
        local_config.text_dim,
        generator=generator,
        device=device,
    )
    timestep_B = torch.tensor([500.0], device=device)

    upstream_model.to(device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        upstream_output_CFH = upstream_model([latent_CFH], timestep_B, [context_LC], seq_len)[0]

    pipeline.model = None
    del upstream_model
    gc.collect()
    torch.cuda.empty_cache()

    local_model.to(device).eval().requires_grad_(False)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        local_output_CFH = local_model([latent_CFH], timestep_B, [context_LC], seq_len)[0]
    torch.testing.assert_close(
        local_output_CFH,
        upstream_output_CFH,
        rtol=args.rtol,
        atol=args.atol,
    )
    absolute_error = (local_output_CFH - upstream_output_CFH).abs()
    print(
        {
            "parity": "passed",
            "max_abs_error": absolute_error.max().item(),
            "mean_abs_error": absolute_error.mean().item(),
            "rtol": args.rtol,
            "atol": args.atol,
        }
    )

    pipeline.model = local_model
    video_CFH = pipeline.generate(
        input_prompt=args.prompt,
        size=(args.width, args.height),
        frame_num=args.num_frames,
        shift=args.shift,
        sample_solver="unipc",
        sampling_steps=args.steps,
        guide_scale=args.guidance_scale,
        n_prompt=args.negative_prompt,
        seed=args.seed,
        offload_model=True,
    )
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(
        tensor=video_CFH[None],
        save_file=str(output_path),
        fps=args.fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    if not output_path.is_file():
        raise RuntimeError(f"Wan video writer did not create {output_path}")
    print({"video": str(output_path), "shape": tuple(video_CFH.shape)})


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the local Wan2.2 TI2V-5B transformer with the official implementation, "
            "then generate a short video with Wan's tokenizer, text encoder, scheduler, and VAE."
        )
    )
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, required=True)
    parser.add_argument("--wan-repo", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("wan_ti2v_short.mp4"))
    parser.add_argument(
        "--prompt",
        default="realistic driving video from dashcam point of view",
    )
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=9)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--t5-cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    _run_pretrained_demo(parser.parse_args())


if __name__ == "__main__":
    main()
