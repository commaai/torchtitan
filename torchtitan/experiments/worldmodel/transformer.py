from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from torchtitan.config import Configurable
from torchtitan.models.common.attention import FlexAttention, create_attention_mask
from torchtitan.models.common.nn_modules import Embedding, GELU, Identity, LayerNorm, Linear, RMSNorm, SiLU


TensorOrMask = torch.Tensor | BlockMask


@dataclass(kw_only=True, slots=True)
class TransformerConfig:
    n_layer: int = 24
    n_embd: int = 1024
    n_head: int = 16
    act: str = "GELU"
    block_size: int = 135 * 15
    attn_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    biased_linears: bool = False
    prenorm: bool = True
    qk_norm: bool = False
    mlp_mult: float = 4
    mlp_multiple_of: int = 256
    attention_mask: str = "NONE"
    attention_mask_mini_block_size: int | None = None
    norm: str = "LayerNorm"
    attention_impl: str = "FLEX"


def linear_config(in_features: int, out_features: int, *, bias: bool = False, current: Linear.Config | None = None) -> Linear.Config:
    if (
        current is not None
        and current.in_features == in_features
        and current.out_features == out_features
        and current.bias == bias
    ):
        return current
    return Linear.Config(in_features=in_features, out_features=out_features, bias=bias)


def make_linear(config: Linear.Config) -> nn.Module:
    return config.build()


def make_embedding(num_embeddings: int, embedding_dim: int) -> Embedding:
    return Embedding(Embedding.Config(num_embeddings=num_embeddings, embedding_dim=embedding_dim))


def make_silu() -> SiLU:
    return SiLU(SiLU.Config())


def make_identity() -> Identity:
    return Identity(Identity.Config())


def make_activation(name: str) -> nn.Module:
    if name == "GELU":
        return GELU(GELU.Config(approximate="tanh"))
    raise ValueError(f"unknown activation {name}")


def make_norm(name: str, normalized_shape: int, *, elementwise_affine: bool = True) -> nn.Module:
    if name == "LayerNorm":
        return LayerNorm(LayerNorm.Config(normalized_shape=normalized_shape, elementwise_affine=elementwise_affine))
    if name == "RMSNorm":
        return RMSNorm(RMSNorm.Config(normalized_shape=normalized_shape, elementwise_affine=elementwise_affine))
    raise ValueError(f"unknown norm {name}")


@dataclass(kw_only=True, slots=True)
class SelfAttentionLinearsConfig(Configurable.Config):
    c_attn: Linear.Config = field(default_factory=lambda: linear_config(1, 1))
    c_proj: Linear.Config = field(default_factory=lambda: linear_config(1, 1))


@dataclass(kw_only=True, slots=True)
class MLPLinearsConfig(Configurable.Config):
    c_fc: Linear.Config = field(default_factory=lambda: linear_config(1, 1))
    c_proj: Linear.Config = field(default_factory=lambda: linear_config(1, 1))


FFNLinearsConfig = MLPLinearsConfig


def self_attention_linears_config(config: TransformerConfig, current: SelfAttentionLinearsConfig | None = None) -> SelfAttentionLinearsConfig:
    return SelfAttentionLinearsConfig(
        c_attn=linear_config(config.n_embd, 3 * config.n_embd, bias=config.biased_linears, current=None if current is None else current.c_attn),
        c_proj=linear_config(config.n_embd, config.n_embd, bias=config.biased_linears, current=None if current is None else current.c_proj),
    )


def ffn_linears_config(config: TransformerConfig, current: FFNLinearsConfig | None = None) -> FFNLinearsConfig:
    hidden_features = mlp_hidden_dim(config.n_embd, config.mlp_mult, config.mlp_multiple_of)
    return MLPLinearsConfig(
        c_fc=linear_config(config.n_embd, hidden_features, bias=config.biased_linears, current=None if current is None else current.c_fc),
        c_proj=linear_config(hidden_features, config.n_embd, bias=config.biased_linears, current=None if current is None else current.c_proj),
    )


def attn_flops(cfg: TransformerConfig) -> int:
    layers, heads = cfg.n_layer, cfg.n_head
    head_dim, seq_len = cfg.n_embd // cfg.n_head, cfg.block_size
    return 12 * layers * heads * head_dim * seq_len * seq_len


def _cast_if_autocast_enabled(tensor: torch.Tensor) -> torch.Tensor:
    if torch.is_autocast_enabled():
        assert tensor.device.type in ["cuda", "cpu"]
        dtype = torch.get_autocast_dtype(tensor.device.type)
        return tensor.to(dtype=dtype)
    return tensor


def _blockwise_lower_triangular_causal_mask(attention_mask_mini_block_size: int, b, h, q_idx, kv_idx):
    q_block_idx = q_idx // attention_mask_mini_block_size
    kv_block_idx = kv_idx // attention_mask_mini_block_size
    return q_block_idx >= kv_block_idx


def _last_frame_causal_mask(block_size, attention_mask_mini_block_size, b, h, q_idx, kv_idx):
    q_ok = q_idx > block_size-attention_mask_mini_block_size
    kv_ok = kv_idx < block_size-attention_mask_mini_block_size
    return ( q_ok | kv_ok )


def get_dense_mask(mask_fn: Callable, block_size_q: int, block_size_kv: int):
    idx_q = torch.arange(block_size_q)
    idx_kv = torch.arange(block_size_kv)
    ii, jj = torch.meshgrid(idx_q, idx_kv, indexing="ij")
    return mask_fn(0, 0, ii, jj)


def get_mask_fn(config: TransformerConfig):
    match config.attention_mask:
        case "NONE":
            return None
        case "BLOCKWISE_LOWER_TRIANGLE":
            if config.attention_mask_mini_block_size is None:
                raise ValueError("BLOCKWISE_LOWER_TRIANGLE requires mini block size")
            return partial(_blockwise_lower_triangular_causal_mask, config.attention_mask_mini_block_size)
        case "LAST_FRAME_CAUSAL":
            if config.attention_mask_mini_block_size is None:
                raise ValueError("LAST_FRAME_CAUSAL requires mini block size")
            return partial(_last_frame_causal_mask, config.block_size, config.attention_mask_mini_block_size)
        case _:
            raise ValueError(f"unknown attention_mask {config.attention_mask}")


class SelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig, linears: SelfAttentionLinearsConfig | None = None):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.head_dim = config.n_embd // config.n_head
        self.config = config
        linears = self_attention_linears_config(config, linears)

        self.layer_norm = make_norm(config.norm, config.n_embd) if config.prenorm else make_identity()
        self.q_norm = make_norm(config.norm, self.head_dim) if config.qk_norm else make_identity()
        self.k_norm = make_norm(config.norm, self.head_dim) if config.qk_norm else make_identity()
        self.c_attn = make_linear(linears.c_attn)
        self.c_proj = make_linear(linears.c_proj)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.init_attention_attrs()

    @torch.no_grad()
    def init_attention_attrs(self):
        if self.config.attention_impl == "FLEX":
            self._attn = FlexAttention.Config().build()

    def attn(self, q, k, v, mask):
        if self.config.attention_impl == "FLEX":
            if k.dtype == torch.float8_e4m3fn:
                k, v = k.to(q.dtype), v.to(q.dtype)
            output = self._attn(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attention_masks=mask,
                scale=1.0 / math.sqrt(self.head_dim),
            )
            return output.transpose(1, 2)
        if self.config.attention_impl == "NAIVE":
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            if mask is not None:
                att = att.masked_fill(~mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            if self.config.attn_pdrop > 0.0:
                att = F.dropout(att, p=self.config.attn_pdrop, training=self.training)
            return att @ v
        raise ValueError(f"unknown attention_impl {self.config.attention_impl}")

    def forward(self, x: torch.Tensor, input_mask: TensorOrMask | None):
        batch, seq_len, emb_dim = x.size()
        x = self.layer_norm(x)
        qkv = self.c_attn(x).view(batch, seq_len, 3, self.config.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = _cast_if_autocast_enabled(q), _cast_if_autocast_enabled(k)

        y = self.attn(q, k, v, input_mask)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, emb_dim)
        return self.resid_dropout(self.c_proj(y))


def mlp_hidden_dim(n_embd: int, mlp_mult: float, mlp_multiple_of: int) -> int:
    hidden_dim = int(n_embd * mlp_mult)
    return mlp_multiple_of * ((hidden_dim + mlp_multiple_of - 1) // mlp_multiple_of)


def MLP(config: TransformerConfig, linears: MLPLinearsConfig | None = None):
    linears = ffn_linears_config(config, linears)
    if not isinstance(linears, MLPLinearsConfig):
        raise TypeError("MLP requires MLPLinearsConfig")
    return nn.Sequential(
        OrderedDict(
            {
                "layer_norm": (make_norm(config.norm, config.n_embd) if config.prenorm else make_identity()),
                "c_fc": make_linear(linears.c_fc),
                "act": make_activation(config.act),
                "c_proj": make_linear(linears.c_proj),
                "dropout": nn.Dropout(config.resid_pdrop),
            }
        )
    )


def build_ffn(config: TransformerConfig, linears: FFNLinearsConfig | None = None):
    return MLP(config, linears)


def build_attention_mask(config: TransformerConfig, device: torch.device):
    if config.attention_mask == "NONE":
        return None

    mask_fn = get_mask_fn(config)
    assert mask_fn is not None
    if config.attention_impl == "FLEX":
        if config.attn_pdrop > 0.0:
            raise NotImplementedError("FLEX attention does not support dropout")
        create_mask = create_block_mask if device.type == "meta" else create_attention_mask
        mask = create_mask(
            mask_fn,
            B=None,
            H=None,
            Q_LEN=config.block_size,
            KV_LEN=config.block_size,
            device=device,
        )
        mask.device = device
        return mask

    if config.attention_impl == "NAIVE":
        return get_dense_mask(mask_fn, config.block_size, config.block_size)[None, None].clone().to(dtype=torch.bool, device=device)

    raise ValueError(f"unknown attention_impl {config.attention_impl}")
