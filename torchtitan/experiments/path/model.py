# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import distribute_tensor, DTensor
from torch.utils.flop_counter import FlopCounterMode

from torchtitan.config import CompileConfig, ParallelismConfig, TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import (
    ActivationCheckpointing,
    ActivationCheckpointingConfig,
    FullAC,
    MemoryBudgetAC,
)
from torchtitan.distributed.fsdp import enable_fsdp_symm_mem, get_fsdp_reshard_after_forward_policy
from torchtitan.models.common import Embedding, LayerNorm, Linear, RMSNorm, SiLU
from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.module import Module, ModuleDict, ModuleList, Sequential
from torchtitan.tools.logging import logger

from . import convnext
from .model_constants import frame_constants_from_fps, ModelInputs, TEMPORAL_INPUTS, VisionFrameType


@dataclass(frozen=True)
class PathHead:
    name: str
    output_size: int
    mlp: bool
    scale: bool


class ScaleLayer(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        n_features: int

    def __init__(self, config: Config):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(config.n_features))

    def reset_parameters(self) -> None:
        nn.init.ones_(self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class PathMLP(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        norm: LayerNorm.Config | RMSNorm.Config
        c_fc: Linear.Config
        c_proj: Linear.Config
        act: str
        dropout: float

    def __init__(self, config: Config):
        super().__init__()
        self.norm = config.norm.build()
        self.c_fc = config.c_fc.build()
        self.act = nn.GELU(approximate="tanh") if config.act == "gelu_tanh" else nn.GELU()
        self.c_proj = config.c_proj.build()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.act(self.c_fc(self.norm(x)))))


class PathSelfAttention(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        norm: LayerNorm.Config | RMSNorm.Config
        q_norm: LayerNorm.Config | RMSNorm.Config | None
        k_norm: LayerNorm.Config | RMSNorm.Config | None
        c_attn: Linear.Config
        c_proj: Linear.Config
        inner_attention: ScaledDotProductAttention.Config
        n_head: int
        head_dim: int
        dropout: float
        is_causal: bool = True
        causal_block_size: int = 1

    def __init__(self, config: Config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.is_causal = config.is_causal
        self.causal_block_size = config.causal_block_size
        self.norm = config.norm.build()
        self.q_norm = config.q_norm.build() if config.q_norm is not None else nn.Identity()
        self.k_norm = config.k_norm.build() if config.k_norm is not None else nn.Identity()
        self.c_attn = config.c_attn.build()
        self.c_proj = config.c_proj.build()
        self.inner_attention = config.inner_attention.build()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.c_attn(self.norm(x)).view(b, t, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.is_causal and self.causal_block_size > 1:
            # Tokens are time-major. Let every spatial token attend within its frame
            # while retaining causal attention between frames.
            frame_idx = torch.arange(t, device=x.device) // self.causal_block_size
            attention_mask = frame_idx[:, None] >= frame_idx[None, :]
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask).transpose(1, 2)
        else:
            x = self.inner_attention(q, k, v, is_causal=self.is_causal)
        return self.dropout(self.c_proj(x.reshape(b, t, self.n_head * self.head_dim)))


class PathTransformerBlock(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        attention: PathSelfAttention.Config
        mlp: PathMLP.Config

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        self.mlp = config.mlp.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(x)
        return x + self.mlp(x)


class PathTransformer(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        layers: list[PathTransformerBlock.Config]

    def __init__(self, config: Config):
        super().__init__()
        self.layers = ModuleList([layer.build() for layer in config.layers])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def apply_activation_checkpointing(self, wrap, base_fqn: str) -> None:
        for layer_id, layer in enumerate(self.layers):
            self.layers[layer_id] = wrap(layer, f"{base_fqn}.layers.{layer_id}")

    def apply_fsdp(self, shard, reshard_after_forward: bool) -> None:
        for layer in self.layers:
            shard(layer, reshard_after_forward)


class SpatialUnvision(Module):
    """Training-only decoder from a spatial ConvNeXt token grid to two RGB views."""

    OUTPUT_SIZE = (128, 256)
    OUTPUT_CHANNELS = 6
    N_EMBD = 256
    N_HEAD = 8
    N_LAYER = 4

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_features: int
        grid_size: tuple[int, int]
        transformer: PathTransformer.Config

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        grid_h, grid_w = config.grid_size
        output_h, output_w = self.OUTPUT_SIZE
        self.patch_size = (output_h // grid_h, output_w // grid_w)
        dim = self.N_EMBD
        self.input_projection = Linear.Config(in_features=config.in_features, out_features=dim, bias=True).build()
        self.input_norm = LayerNorm.Config(normalized_shape=dim).build()
        self.transformer = config.transformer.build()
        self.output_norm = LayerNorm.Config(normalized_shape=dim).build()
        self.output_projection = Linear.Config(
            in_features=dim,
            out_features=self.OUTPUT_CHANNELS * (self.patch_size[0] * self.patch_size[1]),
            bias=True,
        ).build()
        self.pos_embedding = Embedding.Config(
            num_embeddings=grid_h * grid_w,
            embedding_dim=dim,
        ).build()

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        grid_h, grid_w = self.config.grid_size
        tokens = self.input_norm(self.input_projection(features))
        tokens = tokens + self.pos_embedding(torch.arange(grid_h * grid_w, device=features.device))
        tokens = self.output_projection(self.output_norm(self.transformer(tokens)))
        patch_h, patch_w = self.patch_size
        images = rearrange(
            tokens,
            "b (grid_h grid_w) (c patch_h patch_w) -> b c (grid_h patch_h) (grid_w patch_w)",
            grid_h=grid_h,
            grid_w=grid_w,
            patch_h=patch_h,
            patch_w=patch_w,
        )
        return {"imgs": ((images + 1.0) / 2.0) * 255.0}


class PointSummarizer(Module):
    """Collapse each spatial feature grid with a learned CLS token."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        mlp1: PathMLP.Config
        transformer: PathTransformer.Config
        pos_embedding: Embedding.Config
        spatial_size: int

    def __init__(self, config: Config):
        super().__init__()
        self.spatial_size = config.spatial_size
        self.n_features = config.pos_embedding.embedding_dim
        self.mlp1 = config.mlp1.build()
        self.transformer = config.transformer.build()
        self.pos_embedding = config.pos_embedding.build()
        self.cls_token = nn.Parameter(torch.empty(1, 1, self.n_features))

    def reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp1(x) + x
        pos = self.pos_embedding(torch.arange(self.spatial_size, device=x.device))
        x = x + pos
        leading_shape = x.shape[:-2]
        cls = self.cls_token.expand(*leading_shape, 1, -1)
        x = torch.cat((cls, x), dim=-2).reshape(-1, self.spatial_size + 1, self.n_features)
        x = self.transformer(x)
        return x.reshape(*leading_shape, self.spatial_size + 1, self.n_features)[..., 0, :]


class LinearEncoder(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_layer: Linear.Config
        out_layer: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.net = Sequential(config.in_layer.build(), SiLU.Config().build(), config.out_layer.build())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalSummarizer(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        mlp1: PathMLP.Config
        mlp2: PathMLP.Config
        desire_encoder: LinearEncoder.Config
        desire_window_len: int
        desire_window_starts: tuple[int, ...]
        traffic_encoder: LinearEncoder.Config
        action_t_encoder: LinearEncoder.Config
        transformer: PathTransformer.Config
        temporal_pos_embedding: Embedding.Config
        spatial_pos_embedding: Embedding.Config
        temporal_size: int
        spatial_size: int
        dense_training_outputs: bool

    def __init__(self, config: Config):
        super().__init__()
        self.temporal_size = config.temporal_size
        self.spatial_size = config.spatial_size
        self.dense_training_outputs = config.dense_training_outputs
        if len(config.desire_window_starts) != self.temporal_size:
            raise ValueError(f"Expected {self.temporal_size} desire window starts, got {len(config.desire_window_starts)}")
        self.desire_window_len = config.desire_window_len
        self.desire_window_starts = config.desire_window_starts
        self.register_buffer("desire_window_idxs", self._make_desire_window_idxs(), persistent=False)
        self.mlp1 = config.mlp1.build()
        self.mlp2 = config.mlp2.build()
        self.desire_encoder = config.desire_encoder.build()
        self.traffic_encoder = config.traffic_encoder.build()
        self.action_t_encoder = config.action_t_encoder.build()
        self.transformer = config.transformer.build()
        self.temporal_pos_embedding = config.temporal_pos_embedding.build()
        self.spatial_pos_embedding = config.spatial_pos_embedding.build()

    def _make_desire_window_idxs(self, device: torch.device | None = None) -> torch.Tensor:
        starts = torch.tensor(self.desire_window_starts, dtype=torch.long, device=device)
        offsets = torch.arange(self.desire_window_len, dtype=torch.long, device=device)
        return (starts[:, None] + offsets[None, :]).flatten()

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        device = buffer_device if buffer_device is not None else self.desire_window_idxs.device
        self.desire_window_idxs = self._make_desire_window_idxs(device)

    def _window_desire(self, desire: torch.Tensor) -> torch.Tensor:
        desire = desire.index_select(1, self.desire_window_idxs)
        return desire.reshape(desire.shape[0], self.temporal_size, -1)

    def forward(
        self,
        feats: torch.Tensor,
        desire: torch.Tensor,
        traffic_convention: torch.Tensor,
        action_t: torch.Tensor,
    ) -> torch.Tensor:
        feats = self.mlp1(feats) + feats
        feats = self.mlp2(feats) + feats
        b, t, s, c = feats.shape
        # Temporal attention sees every spatial token. Pool only after attention so
        # each dense output retains one entry per frame for the existing heads/losses.
        feats = feats.reshape(b, t * s, c)
        desire = self.desire_encoder(self._window_desire(desire))
        desire = desire.repeat_interleave(s, dim=1)
        traffic_convention = rearrange(self.traffic_encoder(traffic_convention), "b c -> b () c")
        action_t = rearrange(self.action_t_encoder(action_t), "b c -> b () c")
        temporal_pos = self.temporal_pos_embedding(torch.arange(t, device=feats.device))
        spatial_pos = self.spatial_pos_embedding(torch.arange(s, device=feats.device))
        pos = (temporal_pos[:, None, :] + spatial_pos[None, :, :]).reshape(t * s, c)
        x = feats + rearrange(pos, "ts c -> () ts c") + desire + traffic_convention + action_t
        x = self.transformer(x)
        if self.dense_training_outputs:
            return x.reshape(b, t, s, c).mean(dim=2)
        return x[:, -s:].mean(dim=1)


class Hydra(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        heads: tuple[PathHead, ...]
        head_mlps: dict[str, PathMLP.Config]
        final_layers: dict[str, Linear.Config]
        scale_layers: dict[str, ScaleLayer.Config]

    def __init__(self, config: Config):
        super().__init__()
        self.heads = config.heads
        self.head_mlp = ModuleDict({name: cfg.build() for name, cfg in config.head_mlps.items()})
        self.final_layer = ModuleDict({name: cfg.build() for name, cfg in config.final_layers.items()})
        self.scale_layer = ModuleDict({name: cfg.build() for name, cfg in config.scale_layers.items()})

    def forward(self, in_feats: torch.Tensor) -> dict[str, torch.Tensor]:
        ret = {}
        for name, layer in self.final_layer.items():
            feats = self.head_mlp[name](in_feats) + in_feats if name in self.head_mlp else in_feats
            ret[name] = layer(feats)
        for name, layer in self.scale_layer.items():
            ret[name] = layer(ret[name])
        return ret


class Policy(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        summarizer: PointSummarizer.Config
        hydra: Hydra.Config

    def __init__(self, config: Config):
        super().__init__()
        self.summarizer = config.summarizer.build()
        self.hydra = config.hydra.build()

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.hydra(self.summarizer(features))


class TemporalPolicy(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        temporal_summarizer: TemporalSummarizer.Config
        temporal_hydra: Hydra.Config
        history_idxs: tuple[int, ...]

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.temporal_summarizer = config.temporal_summarizer.build()
        self.temporal_hydra = config.temporal_hydra.build()
        self.register_buffer(
            "history_idxs",
            torch.tensor(config.history_idxs, dtype=torch.long),
            persistent=False,
        )

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        device = buffer_device if buffer_device is not None else self.history_idxs.device
        self.history_idxs = torch.tensor(self.config.history_idxs, dtype=torch.long, device=device)

    def forward(
        self,
        features: torch.Tensor,
        desire_pulse: torch.Tensor,
        traffic_convention: torch.Tensor,
        action_t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        dtype = features.dtype
        summary = self.temporal_summarizer(
            features[:, self.history_idxs],
            desire_pulse.to(dtype),
            traffic_convention[:, -1].to(dtype),
            action_t[:, -1].to(dtype),
        )
        return self.temporal_hydra(summary)


class Vision(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        flavor: str
        input_frame_names: tuple[str, ...]
        in_channels: int
        vision_features: int
        grid_size: tuple[int, int]
        pretrained: bool
        drop_path_rate: float
        mean: float
        std: float

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.encoder = convnext.create_convnext(
            config.flavor,
            pretrained=False,
            in_chans=config.in_channels,
            num_classes=0,
            global_pool="",
            drop_path_rate=config.drop_path_rate,
        )
        # The ConvNeXt head now normalizes without pooling; project its channel-rich
        # 2-D map to the policy width at every spatial location.
        self.proj = nn.Conv2d(self.encoder.num_features, config.vision_features, kernel_size=1)
        self.register_buffer("_mean", torch.empty(1, config.in_channels, 1, 1), persistent=True)
        self.register_buffer("_std", torch.empty(1, config.in_channels, 1, 1), persistent=True)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        device = buffer_device if buffer_device is not None else self._mean.device
        self._mean = torch.full((1, self.config.in_channels, 1, 1), self.config.mean, device=device)
        self._std = torch.full((1, self.config.in_channels, 1, 1), self.config.std, device=device)

    def load_pretrained(self) -> None:
        if not self.config.pretrained:
            return

        state_dict = self._pretrained_state_dict()
        target_state = self.encoder.state_dict()
        load_state = {}
        for name, value in state_dict.items():
            if name.startswith("head."):
                continue
            target = target_state.get(name)
            if target is None:
                continue
            value = self._move_pretrained_value(value, target)
            if tuple(value.shape) != tuple(target.shape):
                continue
            load_state[name] = value

        missing, unexpected = self.encoder.load_state_dict(load_state, strict=False)
        pretrained_name = convnext.pretrained_name(self.config.flavor)
        logger.info(
            f"Loaded {len(load_state)} ConvNeXt tensors from {pretrained_name} "
            f"({len(missing)} missing, {len(unexpected)} unexpected)"
        )

    def _pretrained_state_dict(self) -> dict[str, torch.Tensor]:
        from timm.models._builder import adapt_input_conv, load_state_dict_from_hf

        state_dict = load_state_dict_from_hf(
            f"timm/{convnext.pretrained_name(self.config.flavor)}",
            weights_only=True,
        )
        state_dict = convnext.checkpoint_filter_fn(state_dict, self.encoder)
        if self.config.in_channels != 3:
            state_dict["stem.0.weight"] = adapt_input_conv(self.config.in_channels, state_dict["stem.0.weight"])
        return state_dict

    @staticmethod
    def _move_pretrained_value(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if isinstance(target, DTensor):
            return distribute_tensor(
                value.to(dtype=target.dtype),
                target.device_mesh,
                list(target.placements),
            )
        return value.to(device=target.device, dtype=target.dtype)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat([inputs[name] for name in self.config.input_frame_names], dim=1)
        dtype = next(self.encoder.parameters()).dtype
        x = x.to(dtype)
        x = self.encoder((x - self._mean.to(dtype)) / self._std.to(dtype))
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class PathModel(BaseModel):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        n_frames_input: int
        input_frame_names: tuple[str, ...]
        frame_type: VisionFrameType
        vision: Vision.Config
        point_policy: Policy.Config
        temporal_policy: TemporalPolicy.Config
        unvision: bool
        unvision_decoder: SpatialUnvision.Config

        def update_from_config(self, *, config, **kwargs) -> None:
            parallelism = config.parallelism
            if parallelism.spmd_backend == "full_dtensor":
                raise ValueError("path v1 does not support full DTensor")
            unsupported = {
                "tensor parallel": parallelism.tensor_parallel_degree,
                "context parallel": parallelism.context_parallel_degree,
                "pipeline parallel": parallelism.pipeline_parallel_degree,
                "expert parallel": parallelism.expert_parallel_degree,
            }
            for name, degree in unsupported.items():
                if degree > 1:
                    raise ValueError(f"path v1 does not support {name}")

        def get_nparams_and_flops(self, model: Module, seq_len: int) -> tuple[int, int]:
            nparams = sum(p.numel() for p in model.parameters())
            inputs = PathModel.example_inputs(
                self,
                device=next(model.parameters()).device,
            )
            with torch.no_grad(), FlopCounterMode(display=False) as counter:
                model(inputs)
            # MFU convention estimates backward as twice the counted forward work.
            return nparams, 3 * counter.get_total_flops()

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.vision = config.vision.build()
        self.point_policy = config.point_policy.build()
        self.temporal_policy = config.temporal_policy.build()
        self.unvision = config.unvision_decoder.build() if config.unvision else None

    @staticmethod
    def input_shapes(
        config: PathModel.Config,
        batch_size: int = 1,
    ) -> dict[str, tuple[int, ...]]:
        frame_constants = frame_constants_from_fps(
            n_frames=config.n_frames_input,
            frame_type=config.frame_type,
        )
        history_len = len(frame_constants["history_idxs"])
        temporal_len = frame_constants["temporal_len"]
        shapes = {
            name: (batch_size, history_len, *frame_constants["frame_shapes"][name])
            for name in config.vision.input_frame_names
        }
        shapes.update(
            {
                name: (batch_size, temporal_len, *shape)
                for name, shape in TEMPORAL_INPUTS.items()
                if name != ModelInputs.FEATURES
            }
        )
        return shapes

    @staticmethod
    def input_dtypes(config: PathModel.Config) -> dict[str, torch.dtype]:
        dtypes = dict.fromkeys(config.vision.input_frame_names, torch.uint8)
        for name in TEMPORAL_INPUTS:
            if name != ModelInputs.FEATURES:
                dtypes[name] = torch.float32
        return dtypes

    @classmethod
    def example_inputs(
        cls,
        config: PathModel.Config,
        *,
        batch_size: int = 1,
        device: torch.device | str = "meta",
    ) -> dict[str, torch.Tensor]:
        dtypes = cls.input_dtypes(config)
        return {
            name: torch.zeros(shape, dtype=dtypes[name], device=device)
            for name, shape in cls.input_shapes(config, batch_size).items()
        }

    def verify_module_protocol(self) -> None:
        pass

    def init_states(self, *, buffer_device: torch.device | None = None) -> None:
        super().init_states(buffer_device=buffer_device)
        self._init_plain_modules()
        self.vision.load_pretrained()

    def _init_plain_modules(self) -> None:
        for module in self.modules():
            if isinstance(module, Module):
                continue
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()
        self.vision.encoder.init_path_weights()

    def forward(
        self,
        inputs: dict[str, torch.Tensor] | torch.Tensor,
        big_img: torch.Tensor | None = None,
        desire_pulse: torch.Tensor | None = None,
        traffic_convention: torch.Tensor | None = None,
        action_t: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if isinstance(inputs, torch.Tensor):
            inputs = {
                ModelInputs.IMG: inputs,
                ModelInputs.BIG_IMG: big_img,
                ModelInputs.DESIRE: desire_pulse,
                ModelInputs.TRAFFIC: traffic_convention,
                ModelInputs.ACTION_T: action_t,
            }
        img = inputs[ModelInputs.IMG]
        b, t, *_ = img.shape
        vision_inputs = {
            name: rearrange(inputs[name], "b t c h w -> (b t) c h w", b=b, t=t)
            for name in self.config.vision.input_frame_names
        }
        features = self.vision(vision_inputs)
        features = rearrange(features, "(b t) s c -> b t s c", b=b, t=t)
        outputs = self.point_policy(features) | self.temporal_policy(
            features,
            inputs[ModelInputs.DESIRE],
            inputs[ModelInputs.TRAFFIC],
            inputs[ModelInputs.ACTION_T],
        )
        if self.unvision is not None:
            outputs |= self.unvision(features[:, -1])
        return outputs


def parallelize_path(
    model: PathModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
) -> PathModel:
    if parallelism.spmd_backend == "full_dtensor":
        raise ValueError("path v1 does not support full DTensor")
    if parallel_dims.tp_enabled or parallel_dims.cp_enabled or parallel_dims.pp_enabled or parallel_dims.ep_enabled:
        raise ValueError("path v1 supports data parallelism only")

    model_compile_enabled = compile_config.enable and "model" in compile_config.components
    if ac_config is not None:
        _apply_activation_checkpointing(model, ac_config, dump_folder=dump_folder)

    if model_compile_enabled:
        _apply_compile(model, compile_config)

    names = ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    _apply_fsdp(
        model,
        parallel_dims.get_mesh(names),
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )

    logger.info(
        "Applied HSDP to the path model" if parallel_dims.dp_replicate_enabled else "Applied FSDP to the path model"
    )
    if training.enable_cpu_offload:
        logger.info("Applied CPU Offloading to the path model")
    return model


def _apply_activation_checkpointing(
    model: PathModel,
    ac_config: ActivationCheckpointingConfig,
    *,
    dump_folder: str,
) -> None:
    assert ac_config is not None
    ac_policy: ActivationCheckpointing = ac_config.build(dump_folder=dump_folder)

    if isinstance(ac_policy, MemoryBudgetAC):
        ac_policy.apply(model)
        logger.info("Applied memory-budget activation checkpointing to the path model")
        return

    def wrap(module: nn.Module, fqn: str) -> nn.Module:
        return ac_policy._wrap_block(module, base_fqn=fqn)

    mode = "full" if isinstance(ac_policy, FullAC) else "selective"

    model.vision.encoder.apply_activation_checkpointing(
        wrap,
        mode,
        "vision.encoder",
    )
    model.temporal_policy.temporal_summarizer.transformer.apply_activation_checkpointing(
        wrap,
        "temporal_policy.temporal_summarizer.transformer",
    )
    if model.unvision is not None:
        model.unvision.transformer.apply_activation_checkpointing(wrap, "unvision.transformer")

    logger.info(f"Applied {mode} activation checkpointing to the path model")


def _apply_compile(model: PathModel, compile_config: CompileConfig) -> None:
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
    model.vision.encoder.compile(backend=compile_config.backend)
    model.point_policy.compile(backend=compile_config.backend)
    model.temporal_policy.compile(backend=compile_config.backend)
    if model.unvision is not None:
        model.unvision.compile(backend=compile_config.backend)
    logger.info("Compiling path model components with torch.compile")


def _apply_fsdp(
    model: PathModel,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    enable_symm_mem: bool = False,
) -> None:
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=True,
    )
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy,
        pp_enabled,
    )

    def shard(module: nn.Module, reshard: bool) -> None:
        fully_shard(module, **fsdp_config, reshard_after_forward=reshard)

    model.vision.encoder.apply_fsdp(
        shard,
        reshard_after_forward,
        reshard_after_forward_policy == "always",
    )
    model.temporal_policy.temporal_summarizer.transformer.apply_fsdp(
        shard,
        reshard_after_forward,
    )
    shard(model.vision.encoder, reshard_after_forward)
    shard(model.point_policy, reshard_after_forward)
    shard(model.temporal_policy, reshard_after_forward)
    fully_shard(model, **fsdp_config)

    if enable_symm_mem:
        enable_fsdp_symm_mem(model)
