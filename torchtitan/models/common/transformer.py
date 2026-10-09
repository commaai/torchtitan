# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.models.common.attention import FlexAttention, ScaledDotProductAttention
from torchtitan.models.common.nn_modules import GELU, Identity, LayerNorm, Linear, RMSNorm, SiLU
from torchtitan.protocols.module import Module, Sequential


class MLP(Sequential):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        norm: LayerNorm.Config | RMSNorm.Config | Identity.Config
        c_fc: Linear.Config
        c_proj: Linear.Config
        act: GELU.Config | SiLU.Config
        dropout: float
        norm_name: str = "norm"

    def __init__(self, config: Config):
        super().__init__(
            OrderedDict(
                {
                    config.norm_name: config.norm.build(),
                    "c_fc": config.c_fc.build(),
                    "act": config.act.build(),
                    "c_proj": config.c_proj.build(),
                    "dropout": nn.Dropout(config.dropout),
                }
            )
        )


class SelfAttention(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        norm: LayerNorm.Config | RMSNorm.Config | Identity.Config
        q_norm: LayerNorm.Config | RMSNorm.Config | None
        k_norm: LayerNorm.Config | RMSNorm.Config | None
        c_attn: Linear.Config
        c_proj: Linear.Config
        inner_attention: ScaledDotProductAttention.Config | FlexAttention.Config
        n_head: int
        dropout: float
        is_causal: bool = True
        attn_dropout: float = 0.0
        cast_qk_to_autocast: bool = False
        norm_name: str = "norm"

    def __init__(self, config: Config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.c_proj.in_features // config.n_head
        self.is_causal = config.is_causal
        self.attn_dropout = config.attn_dropout
        self.cast_qk_to_autocast = config.cast_qk_to_autocast
        self.norm_name = config.norm_name
        self.add_module(config.norm_name, config.norm.build())
        self.q_norm = config.q_norm.build() if config.q_norm is not None else nn.Identity()
        self.k_norm = config.k_norm.build() if config.k_norm is not None else nn.Identity()
        self.c_attn = config.c_attn.build()
        self.c_proj = config.c_proj.build()
        self.inner_attention = config.inner_attention.build()
        self.dropout = nn.Dropout(config.dropout)

    def project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        qkv = self.c_attn(self.get_submodule(self.norm_name)(x)).view(batch, seq_len, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.cast_qk_to_autocast and torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype(x.device.type)
            q, k = q.to(dtype=dtype), k.to(dtype=dtype)
        return q, k, v

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor | BlockMask | None = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q, k, v = self.project_qkv(x)
        scale = 1.0 / math.sqrt(self.head_dim)
        if isinstance(self.inner_attention, FlexAttention):
            y = self.inner_attention(q, k, v, attention_masks=input_mask, scale=scale)
        elif input_mask is None:
            y = self.inner_attention(q, k, v, scale=scale, is_causal=self.is_causal)
        else:
            assert isinstance(input_mask, torch.Tensor)
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=input_mask,
                dropout_p=self.attn_dropout if self.training else 0.0,
                scale=scale,
            ).transpose(1, 2)
        return self.dropout(self.c_proj(y.reshape(batch, seq_len, self.n_head * self.head_dim)))
