from collections import OrderedDict
from dataclasses import dataclass, field
import itertools
import math

import einops
import numpy as np
import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from einops.layers.torch import Rearrange

from torchtitan.config import Configurable
from torchtitan.experiments.worldmodel.transformer import (
    FFNLinearsConfig,
    MLPLinearsConfig,
    SelfAttentionLinearsConfig,
    TensorOrMask,
    TransformerConfig,
    attn_flops,
    build_attention_mask,
    build_ffn,
    ffn_linears_config,
    linear_config,
    make_embedding,
    make_linear,
    make_norm,
    make_silu,
    self_attention_linears_config,
)
from torchtitan.models.common.nn_modules import Linear
from torchtitan.protocols import BaseModel


PLAN_SIZE = 15 * 33 * 2
PLAN_HEAD_INIT_STD = 1e-3
PLAN_HEAD_INIT_LOG_SIGMA_SCALE = 5.0


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _init_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> None:
    nn.init.normal_(_local_tensor(tensor), mean=mean, std=std)


def _init_constant_(tensor: torch.Tensor, value: float) -> None:
    nn.init.constant_(_local_tensor(tensor), value)


def _init_zeros_(tensor: torch.Tensor) -> None:
    nn.init.zeros_(_local_tensor(tensor))


def _init_xavier_uniform_(tensor: torch.Tensor) -> None:
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    with torch.no_grad():
        _local_tensor(tensor).uniform_(-bound, bound)


def _init_plan_bias_(bias: torch.Tensor) -> None:
    local = _local_tensor(bias)
    with torch.no_grad():
        local.zero_()
        split = PLAN_SIZE // 2
        if not isinstance(bias, DTensor):
            local[split:].fill_(math.log(PLAN_HEAD_INIT_LOG_SIGMA_SCALE))
            return

        offset = 0
        for mesh_dim, placement in enumerate(bias.placements):
            if placement.is_shard() and placement.dim == 0:
                _, offset = placement.local_shard_size_and_offset(
                    bias.shape[0],
                    bias.device_mesh.size(mesh_dim),
                    bias.device_mesh.get_local_rank(mesh_dim),
                )
                break
        start = max(0, split - offset)
        if start < local.shape[0]:
            local[start:].fill_(math.log(PLAN_HEAD_INIT_LOG_SIGMA_SCALE))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    x = einops.rearrange(x, "b (t n) c -> b t n c", t=shift.shape[1])
    x = x * (1 + scale) + shift
    return einops.rearrange(x, "b t n c -> b (t n) c")


def gate(x: torch.Tensor, gate_value: torch.Tensor):
    x = einops.rearrange(x, "b (t n) c -> b t n c", t=gate_value.shape[1])
    x = gate_value * x
    return einops.rearrange(x, "b t n c -> b (t n) c")


def is_linear_like(module: nn.Module) -> bool:
    weight = getattr(module, "weight", None)
    return (
        isinstance(weight, torch.Tensor)
        and weight.ndim == 2
        and hasattr(module, "in_features")
        and hasattr(module, "out_features")
    )


def init_mlp_weights(mlp: nn.Sequential, std=0.02):
    for module in mlp:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            _init_normal_(module.weight, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                _init_constant_(module.bias, 0)
        elif is_linear_like(module):
            _init_normal_(module.weight, std=std)
            if module.bias is not None:
                _init_constant_(module.bias, 0)


def init_transformer_linear_weights(linear: nn.Module):
    _init_xavier_uniform_(linear.weight)
    if linear.bias is not None:
        _init_zeros_(linear.bias)


@dataclass(kw_only=True, slots=True)
class PatchEmbedderLinearsConfig(Configurable.Config):
    linear: Linear.Config = field(default_factory=lambda: linear_config(1, 1))


@dataclass(kw_only=True, slots=True)
class ConditioningEmbedderLinearsConfig(Configurable.Config):
    mlp_in: Linear.Config = field(default_factory=lambda: linear_config(1, 1))
    mlp_out: Linear.Config = field(default_factory=lambda: linear_config(1, 1))
    to_t6: Linear.Config = field(default_factory=lambda: linear_config(1, 1))
    to_t2: Linear.Config = field(default_factory=lambda: linear_config(1, 1))


@dataclass(kw_only=True, slots=True)
class DiTBlockLinearsConfig(Configurable.Config):
    attn: SelfAttentionLinearsConfig = field(default_factory=SelfAttentionLinearsConfig)
    mlp: FFNLinearsConfig = field(default_factory=MLPLinearsConfig)


@dataclass(kw_only=True, slots=True)
class FinalLayerLinearsConfig(Configurable.Config):
    linear: Linear.Config = field(default_factory=lambda: linear_config(1, 1))


@dataclass(kw_only=True, slots=True)
class PlanHeadLinearsConfig(Configurable.Config):
    blocks: list[FFNLinearsConfig] = field(default_factory=list)
    head: Linear.Config = field(default_factory=lambda: linear_config(1, 1))


def conditioning_embedder_linears_config(input_size: int, hidden_size: int, current: ConditioningEmbedderLinearsConfig | None = None) -> ConditioningEmbedderLinearsConfig:
    return ConditioningEmbedderLinearsConfig(
        mlp_in=linear_config(input_size, hidden_size, bias=True, current=None if current is None else current.mlp_in),
        mlp_out=linear_config(hidden_size, hidden_size, bias=True, current=None if current is None else current.mlp_out),
        to_t6=linear_config(hidden_size, 6 * hidden_size, bias=True, current=None if current is None else current.to_t6),
        to_t2=linear_config(hidden_size, 2 * hidden_size, bias=True, current=None if current is None else current.to_t2),
    )


def dit_block_linears_config(config: TransformerConfig, current: DiTBlockLinearsConfig | None = None) -> DiTBlockLinearsConfig:
    return DiTBlockLinearsConfig(
        attn=self_attention_linears_config(config, None if current is None else current.attn),
        mlp=ffn_linears_config(config, None if current is None else current.mlp),
    )


def plan_head_linears_config(config: TransformerConfig, current: PlanHeadLinearsConfig | None = None) -> PlanHeadLinearsConfig:
    current_blocks = [] if current is None else current.blocks
    return PlanHeadLinearsConfig(
        blocks=[ffn_linears_config(config, current_blocks[i] if i < len(current_blocks) else None) for i in range(config.n_layer)],
        head=linear_config(config.n_embd, PLAN_SIZE, bias=config.biased_linears, current=None if current is None else current.head),
    )


class PatchEmbedder(nn.Sequential):
    def __init__(self, patch_size, n_embd, linears: PatchEmbedderLinearsConfig, norm="LayerNorm"):
        super().__init__(
            Rearrange(
                "b (t pt) c (h ph) (w pw) -> b (t h w) (c pt ph pw)",
                pt=patch_size[0],
                ph=patch_size[1],
                pw=patch_size[2],
            ),
            make_linear(linears.linear),
            make_norm(norm, n_embd),
        )
        self.init_weights()

    def init_weights(self):
        init_transformer_linear_weights(self[1])


class ContinuousEmbedder(nn.Module):
    def __init__(self, linears: ConditioningEmbedderLinearsConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            make_linear(linears.mlp_in),
            make_silu(),
            make_linear(linears.mlp_out),
        )
        self.to_t6 = nn.Sequential(make_silu(), make_linear(linears.to_t6))
        self.to_t2 = nn.Sequential(make_silu(), make_linear(linears.to_t2))
        self.init_weights()

    def forward(self, x):
        x = self.mlp(x)
        return self.to_t6(x), self.to_t2(x)

    def init_weights(self):
        init_mlp_weights(self.mlp, std=0.02)
        init_mlp_weights(self.to_t6, std=0.02)
        init_mlp_weights(self.to_t2, std=0.02)


class DiscreteEmbedder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, linears: ConditioningEmbedderLinearsConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            make_embedding(input_size, hidden_size),
            make_silu(),
            make_linear(linears.mlp_out),
        )
        self.to_t6 = nn.Sequential(make_silu(), make_linear(linears.to_t6))
        self.to_t2 = nn.Sequential(make_silu(), make_linear(linears.to_t2))
        self.init_weights()

    def forward(self, x):
        x = self.mlp(x)
        return self.to_t6(x), self.to_t2(x)

    def init_weights(self):
        init_mlp_weights(self.mlp, std=0.02)
        init_mlp_weights(self.to_t6, std=0.02)
        init_mlp_weights(self.to_t2, std=0.02)


class TimestepEmbedder(nn.Module):
    def __init__(self, linears: ConditioningEmbedderLinearsConfig, frequency_embedding_size: int = 256, max_period: int = 10000, time_factor: float = 1000.0):
        super().__init__()
        self.mlp = nn.Sequential(
            make_linear(linears.mlp_in),
            make_silu(),
            make_linear(linears.mlp_out),
        )
        self.to_t6 = nn.Sequential(make_silu(), make_linear(linears.to_t6))
        self.to_t2 = nn.Sequential(make_silu(), make_linear(linears.to_t2))
        self.frequency_embedding_size = frequency_embedding_size
        if frequency_embedding_size % 2 != 0:
            raise ValueError("frequency_embedding_size must be even")
        self.half_frequency_embedding_size = frequency_embedding_size // 2
        self.max_period = max_period
        self.time_factor = time_factor
        self.init_weights()

    def timestep_embedding(self, t):
        t = self.time_factor * t.float()
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=self.half_frequency_embedding_size, device=t.device, dtype=torch.float32) / self.half_frequency_embedding_size
        )
        args = t[..., None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t):
        t_freq = self.timestep_embedding(t).to(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return self.to_t6(t_emb), self.to_t2(t_emb)

    def init_weights(self):
        init_mlp_weights(self.mlp, std=0.02)
        init_mlp_weights(self.to_t6, std=0.02)
        init_mlp_weights(self.to_t2, std=0.02)


class ScaleLayer(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.scale = nn.Parameter(torch.empty((n_features)))
        self.init_weights()

    def forward(self, x):
        return x * self.scale

    def reset_parameters(self):
        _init_constant_(self.scale, 1.0)

    def init_weights(self):
        self.reset_parameters()


class ResidualSequential(nn.Sequential):
    def forward(self, input):  # noqa: A002
        return super().forward(input) + input


def residual_ffn(config: TransformerConfig, linears: FFNLinearsConfig):
    ffn = build_ffn(config, linears)
    if not isinstance(ffn, nn.Sequential):
        raise TypeError("plan head FFN must be an nn.Sequential")
    return ResidualSequential(OrderedDict(ffn.named_children()))


class PlanHead(nn.Module):
    def __init__(self, config: "WorldModel.Config", linears: PlanHeadLinearsConfig):
        super().__init__()
        self.mlps = nn.ModuleList(residual_ffn(config.plan_head, linears.blocks[i]) for i in range(config.plan_head.n_layer))
        self.head = make_linear(linears.head)
        self.scale_layer = ScaleLayer(PLAN_SIZE)
        self.init_weights()

    def forward(self, x):
        for mlp in self.mlps:
            x = mlp(x)
        return self.scale_layer(self.head(x))

    def init_weights(self):
        for module in self.mlps.modules():
            if is_linear_like(module):
                init_transformer_linear_weights(module)
        _init_normal_(self.head.weight, std=PLAN_HEAD_INIT_STD)
        if self.head.bias is not None:
            _init_plan_bias_(self.head.bias)
        self.scale_layer.init_weights()


class DiTBlock(nn.Module):
    def __init__(self, config: "WorldModel.Config", linears: DiTBlockLinearsConfig):
        super().__init__()
        self.norm1 = make_norm(config.transformer.norm, config.transformer.n_embd, elementwise_affine=False)
        from torchtitan.experiments.worldmodel.transformer import SelfAttention

        self.attn = SelfAttention(config.transformer, linears.attn)
        self.norm2 = make_norm(config.transformer.norm, config.transformer.n_embd, elementwise_affine=False)
        self.mlp = build_ffn(config.transformer, linears.mlp)
        self.scale_shift_table = nn.Parameter(torch.empty(1, config.num_temporal_patches, 6, config.transformer.n_embd))
        self.init_weights()

    def forward(self, x, t, input_mask=None):
        batch = x.shape[0]
        chunks = (self.scale_shift_table + t.reshape(batch, self.scale_shift_table.shape[1], 6, -1)).chunk(6, dim=2)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks
        x = x + gate(self.attn(modulate(self.norm1(x), shift_msa, scale_msa), input_mask=input_mask), gate_msa)
        x = x + gate(self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)), gate_mlp)
        return x

    def reset_parameters(self):
        _init_normal_(self.scale_shift_table, mean=0.0, std=self.scale_shift_table.shape[-1] ** -0.5)

    def init_weights(self):
        self.reset_parameters()
        for module in itertools.chain(self.attn.modules(), self.mlp.modules()):
            if is_linear_like(module):
                init_transformer_linear_weights(module)


class FinalLayer(nn.Module):
    def __init__(self, config: "WorldModel.Config", linears: FinalLayerLinearsConfig):
        super().__init__()
        self.norm_final = make_norm(config.transformer.norm, config.transformer.n_embd, elementwise_affine=False)
        self.linear = make_linear(linears.linear)
        self.scale_shift_table = nn.Parameter(torch.empty(1, config.num_temporal_patches, 2, config.transformer.n_embd))
        self.init_weights()

    def forward(self, x, t):
        batch = x.shape[0]
        shift, scale = (self.scale_shift_table + t.reshape(batch, self.scale_shift_table.shape[1], 2, -1)).chunk(2, dim=2)
        return self.linear(modulate(self.norm_final(x), shift, scale))

    def reset_parameters(self):
        _init_normal_(self.scale_shift_table, mean=0.0, std=self.scale_shift_table.shape[-1] ** -0.5)

    def init_weights(self):
        self.reset_parameters()
        _init_constant_(self.linear.weight, 0)
        _init_constant_(self.linear.bias, 0)


class WorldModel(BaseModel):
    pos_embed: torch.Tensor

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        input_size: tuple[int, int, int] = (15, 16, 32)
        patch_size: tuple[int, int, int] = (1, 2, 2)
        in_channels: int = 32
        out_channels: int = 32
        pose_size: int = 6
        time_factor: float = 1.0
        compressor_mean: float = 0.001418
        compressor_std: float = 0.112629

        transformer: TransformerConfig = field(
            default_factory=lambda: TransformerConfig(
                n_layer=48,
                n_head=25,
                n_embd=1600,
                act="GELU",
                resid_pdrop=0.0,
                attn_pdrop=0.0,
                biased_linears=True,
                qk_norm=True,
                prenorm=False,
                attention_mask="BLOCKWISE_LOWER_TRIANGLE",
                attention_impl="FLEX",
            )
        )
        plan_head: TransformerConfig = field(
            default_factory=lambda: TransformerConfig(
                n_layer=4,
                act="GELU",
                biased_linears=True,
                prenorm=True,
                mlp_mult=2,
                mlp_multiple_of=1,
            )
        )
        x_embedder: PatchEmbedderLinearsConfig = field(default_factory=PatchEmbedderLinearsConfig)
        augments_pos_ref_augment_embedder: ConditioningEmbedderLinearsConfig = field(default_factory=ConditioningEmbedderLinearsConfig)
        ref_augment_from_augments_euler_embedder: ConditioningEmbedderLinearsConfig = field(default_factory=ConditioningEmbedderLinearsConfig)
        pose_mask_embedder: ConditioningEmbedderLinearsConfig = field(default_factory=ConditioningEmbedderLinearsConfig)
        t_embedder: ConditioningEmbedderLinearsConfig = field(default_factory=ConditioningEmbedderLinearsConfig)
        fidx_embedder: ConditioningEmbedderLinearsConfig = field(default_factory=ConditioningEmbedderLinearsConfig)
        blocks: list[DiTBlockLinearsConfig] = field(default_factory=list)
        final_layer: FinalLayerLinearsConfig | None = None
        plan_head_linears: PlanHeadLinearsConfig | None = None

        @property
        def num_spatial_patches(self) -> int:
            return math.prod(self.input_size[1:]) // math.prod(self.patch_size[1:])

        @property
        def num_temporal_patches(self) -> int:
            return self.input_size[0] // self.patch_size[0]

        @property
        def num_patches(self) -> int:
            return self.num_spatial_patches * self.num_temporal_patches

        def __post_init__(self):
            self._sync_derived_fields()

        def _sync_derived_fields(self):
            self.transformer.block_size = self.num_patches
            self.transformer.attention_mask_mini_block_size = self.num_spatial_patches
            self.plan_head.n_embd = self.transformer.n_embd
            hidden_size = self.transformer.n_embd
            pose_half = self.pose_size // 2
            self.x_embedder = PatchEmbedderLinearsConfig(linear=linear_config(self.in_channels * math.prod(self.patch_size), hidden_size, current=self.x_embedder.linear))
            self.augments_pos_ref_augment_embedder = conditioning_embedder_linears_config(pose_half, hidden_size, self.augments_pos_ref_augment_embedder)
            self.ref_augment_from_augments_euler_embedder = conditioning_embedder_linears_config(pose_half, hidden_size, self.ref_augment_from_augments_euler_embedder)
            self.pose_mask_embedder = conditioning_embedder_linears_config(2, hidden_size, self.pose_mask_embedder)
            self.t_embedder = conditioning_embedder_linears_config(256, hidden_size, self.t_embedder)
            self.fidx_embedder = conditioning_embedder_linears_config(50, hidden_size, self.fidx_embedder)
            self.blocks = [
                dit_block_linears_config(self.transformer, self.blocks[i] if i < len(self.blocks) else None)
                for i in range(self.transformer.n_layer)
            ]
            self.final_layer = (
                FinalLayerLinearsConfig(
                    linear=linear_config(hidden_size, math.prod(self.patch_size) * self.out_channels, bias=True, current=None if self.final_layer is None else self.final_layer.linear)
                )
                if self.out_channels > 0
                else None
            )
            self.plan_head_linears = plan_head_linears_config(self.plan_head, self.plan_head_linears) if self.plan_head.n_layer >= 0 else None

        def update_from_config(self, *, trainer_config, **kwargs) -> None:
            self._sync_derived_fields()

        def get_nparams_and_flops(self, model: nn.Module, seq_len: int):
            nparams = sum(p.numel() for p in model.parameters())
            flops_per_token = 6 * nparams + attn_flops(self.transformer) // max(1, self.num_patches)
            return nparams, flops_per_token

    def __init__(self, config: Config):
        super().__init__()
        config._sync_derived_fields()
        self.config = config

        self.x_embedder = PatchEmbedder(config.patch_size, config.transformer.n_embd, config.x_embedder, config.transformer.norm)
        pose_half = self.config.pose_size // 2
        self.position_scale = ScaleLayer(pose_half)
        self.euler_scale = ScaleLayer(pose_half)
        self.augments_pos_ref_augment_embedder = ContinuousEmbedder(config.augments_pos_ref_augment_embedder)
        self.ref_augment_from_augments_euler_embedder = ContinuousEmbedder(config.ref_augment_from_augments_euler_embedder)
        self.pose_mask_embedder = DiscreteEmbedder(2, self.config.transformer.n_embd, config.pose_mask_embedder)
        self.t_embedder = TimestepEmbedder(config.t_embedder, time_factor=self.config.time_factor)
        self.fidx_embedder = DiscreteEmbedder(50, self.config.transformer.n_embd, config.fidx_embedder)

        self.blocks = nn.ModuleList(DiTBlock(config, config.blocks[i]) for i in range(config.transformer.n_layer))
        self.final_layer = FinalLayer(config, config.final_layer) if config.final_layer is not None else None
        self.plan_head = PlanHead(config, config.plan_head_linears) if config.plan_head_linears is not None else None

        self.register_buffer("pos_embed", torch.empty(1, self.config.num_patches, self.config.transformer.n_embd))

        self.mask: TensorOrMask | None = None
        self.init_weights(device=self.pos_embed.device)

    def verify_module_protocol(self) -> None:
        return

    @torch.no_grad()
    def setup_attention_attrs(self, device: torch.device):
        if self.config.transformer.attention_mask == "NONE":
            self.mask = None
            return
        if self.mask is not None and getattr(self.mask, "device", None) == device:
            return
        self.mask = build_attention_mask(self.config.transformer, device)

    def reset_parameters(self):
        spatial_grid = (
            self.config.input_size[1] // self.config.patch_size[1],
            self.config.input_size[2] // self.config.patch_size[2],
        )
        spatial_pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], spatial_grid)
        spatial_pos_embed = torch.from_numpy(spatial_pos_embed)
        spatial_pos_embed = spatial_pos_embed.to(dtype=self.pos_embed.dtype, device=self.pos_embed.device).unsqueeze(0)
        spatial_pos_embed = einops.repeat(spatial_pos_embed, "() n d -> () (t n) d", t=self.config.num_temporal_patches)

        temporal_pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.config.num_temporal_patches)
        temporal_pos_embed = torch.from_numpy(temporal_pos_embed)
        temporal_pos_embed = temporal_pos_embed.to(dtype=self.pos_embed.dtype, device=self.pos_embed.device).unsqueeze(0)
        temporal_pos_embed = einops.repeat(temporal_pos_embed, "() t d -> () (t n) d", n=self.config.num_spatial_patches)
        self.pos_embed[:] = spatial_pos_embed + temporal_pos_embed

    def init_weights(self, *, device=None, buffer_device=None, **kwargs):
        target_device = buffer_device or device or self.pos_embed.device
        if isinstance(target_device, str):
            target_device = torch.device(target_device)
        self.setup_attention_attrs(target_device)
        self.reset_parameters()

        def _init(module):
            module = getattr(module, "_checkpoint_wrapped_module", module)
            if hasattr(module, "init_weights"):
                module.init_weights()

        for module in self.children():
            if isinstance(module, nn.ModuleList):
                for child in module:
                    _init(child)
            else:
                _init(module)

    def unpatchify(self, x):
        return einops.rearrange(
            x,
            "b (t h w) (c pt ph pw) -> b (t pt) c (h ph) (w pw)",
            c=self.config.out_channels,
            pt=self.config.patch_size[0],
            ph=self.config.patch_size[1],
            pw=self.config.patch_size[2],
            h=self.config.input_size[1] // self.config.patch_size[1],
            w=self.config.input_size[2] // self.config.patch_size[2],
        )

    def scale_latents(self, latents):
        return (latents - self.config.compressor_mean) / self.config.compressor_std

    def forward(self, x, t, augments_pos_ref_augment, ref_augment_from_augments_euler, pose_mask, fidx):
        x = self.x_embedder(x) + self.pos_embed
        augments_pos_ref_augment = self.position_scale(augments_pos_ref_augment)
        ref_augment_from_augments_euler = self.euler_scale(ref_augment_from_augments_euler)

        t6, t2 = self.t_embedder(t)
        pos6, pos2 = self.augments_pos_ref_augment_embedder(augments_pos_ref_augment)
        euler6, euler2 = self.ref_augment_from_augments_euler_embedder(ref_augment_from_augments_euler)
        pose_mask6, pose_mask2 = self.pose_mask_embedder(pose_mask)
        fidx6, fidx2 = self.fidx_embedder(fidx)
        t6 = t6 + pos6 + euler6 + pose_mask6 + fidx6
        t2 = t2 + pos2 + euler2 + pose_mask2 + fidx2

        for block in self.blocks:
            x = block(x, t6, self.mask)

        outputs = {}
        if self.plan_head is not None:
            outputs["plan"] = self.plan_head(x[:, -1, :])
        if self.final_layer is not None:
            outputs["sample"] = self.unpatchify(self.final_layer(x, t2))
        return outputs

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, *grid_size])
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_1d_sincos_pos_embed(embed_dim, length):
    pos = np.arange(0, length)[..., None]
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)
