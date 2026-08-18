# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

from torchtitan.models.common import Embedding, LayerNorm, Linear
from torchtitan.models.common.attention import ScaledDotProductAttention

from .hydra_configs import DRIVING_HEADS, META_HEADS, POSE_HEADS, TEMPORAL_META_HEADS
from .model import (
    Hydra,
    LinearEncoder,
    PathHead,
    PathMLP,
    PathModel,
    PathSelfAttention,
    PathTransformer,
    PathTransformerBlock,
    PointSummarizer,
    Policy,
    ScaleLayer,
    TemporalPolicy,
    TemporalSummarizer,
    Vision,
)
from .model_constants import (
    frame_constants_from_fps,
    FRAME_TYPE,
    INPUT_FRAMES_NAMES,
    ModelInputs,
    N_FRAMES,
    TEMPORAL_INPUTS,
)


POINT_HEADS = tuple(META_HEADS + POSE_HEADS)
TEMPORAL_HEADS = tuple(DRIVING_HEADS + TEMPORAL_META_HEADS)


def model_config(flavor: str = "convnext_xxlarge") -> PathModel.Config:
    vision_features = 512
    frame_constants = frame_constants_from_fps(n_frames=N_FRAMES, frame_type=FRAME_TYPE)
    input_frame_names = tuple(INPUT_FRAMES_NAMES)
    in_channels = sum(frame_constants["frame_shapes"][name][0] for name in input_frame_names)

    return PathModel.Config(
        n_frames_input=N_FRAMES,
        input_frame_names=input_frame_names,
        frame_type=FRAME_TYPE,
        vision=Vision.Config(
            flavor=flavor,
            input_frame_names=input_frame_names,
            in_channels=in_channels,
            vision_features=vision_features,
            pretrained=True,
            drop_path_rate=0.2,
            mean=255 / 2,
            std=255 / 4,
        ),
        point_policy=Policy.Config(
            summarizer=PointSummarizer.Config(
                mlp1=_mlp(vision_features, mlp_mult=2, bias=False, dropout=0.0),
                mlp2=_mlp(vision_features, mlp_mult=2, bias=False, dropout=0.0),
            ),
            hydra=_hydra(POINT_HEADS, in_features=vision_features, mlp_mult=2),
        ),
        temporal_policy=temporal_policy_config(),
    )


def temporal_policy_config(
    *,
    heads: tuple[PathHead, ...] = TEMPORAL_HEADS,
    dropout: float = 0.1,
    dense_training_outputs: bool = True,
) -> TemporalPolicy.Config:
    vision_features = 512
    frame_constants = frame_constants_from_fps()
    history_idxs = tuple(int(index) for index in frame_constants["history_idxs"])
    desire_window_len = frame_constants["desire_window_len"]
    desire_window_starts = tuple(index - history_idxs[0] for index in history_idxs)
    return TemporalPolicy.Config(
        temporal_summarizer=TemporalSummarizer.Config(
            mlp1=_mlp(vision_features, mlp_mult=2, bias=False, dropout=0.0),
            mlp2=_mlp(vision_features, mlp_mult=2, bias=False, dropout=0.0),
            desire_encoder=_encoder(TEMPORAL_INPUTS[ModelInputs.DESIRE][0] * desire_window_len, vision_features),
            desire_window_len=desire_window_len,
            desire_window_starts=desire_window_starts,
            traffic_encoder=_encoder(TEMPORAL_INPUTS[ModelInputs.TRAFFIC][0], vision_features),
            action_t_encoder=_encoder(TEMPORAL_INPUTS[ModelInputs.ACTION_T][0], vision_features),
            transformer=PathTransformer.Config(
                layers=[
                    PathTransformerBlock.Config(
                        attention=_attention(dim=vision_features, n_head=8, dropout=dropout),
                        mlp=_mlp(vision_features, mlp_mult=2, bias=True, dropout=dropout),
                    )
                    for _ in range(4)
                ]
            ),
            pos_embedding=Embedding.Config(
                num_embeddings=len(history_idxs),
                embedding_dim=vision_features,
            ),
            block_size=len(history_idxs),
            dense_training_outputs=dense_training_outputs,
        ),
        temporal_hydra=_hydra(heads, in_features=vision_features, mlp_mult=2),
        history_idxs=history_idxs,
    )


def _mlp(dim: int, *, mlp_mult: float, bias: bool, dropout: float) -> PathMLP.Config:
    hidden = 256 * math.ceil(int(dim * mlp_mult) / 256)
    return PathMLP.Config(
        norm=LayerNorm.Config(normalized_shape=dim),
        c_fc=Linear.Config(in_features=dim, out_features=hidden, bias=bias),
        c_proj=Linear.Config(in_features=hidden, out_features=dim, bias=bias),
        act="gelu_tanh",
        dropout=dropout,
    )


def _encoder(in_features: int, dim: int) -> LinearEncoder.Config:
    return LinearEncoder.Config(
        in_layer=Linear.Config(in_features=in_features, out_features=dim, bias=True),
        out_layer=Linear.Config(in_features=dim, out_features=dim, bias=False),
    )


def _attention(*, dim: int, n_head: int, dropout: float) -> PathSelfAttention.Config:
    head_dim = dim // n_head
    return PathSelfAttention.Config(
        norm=LayerNorm.Config(normalized_shape=dim),
        q_norm=LayerNorm.Config(normalized_shape=head_dim),
        k_norm=LayerNorm.Config(normalized_shape=head_dim),
        c_attn=Linear.Config(in_features=dim, out_features=3 * dim, bias=True),
        c_proj=Linear.Config(in_features=dim, out_features=dim, bias=True),
        inner_attention=ScaledDotProductAttention.Config(),
        n_head=n_head,
        head_dim=head_dim,
        dropout=dropout,
    )


def _hydra(heads: tuple[PathHead, ...], *, in_features: int, mlp_mult: float) -> Hydra.Config:
    return Hydra.Config(
        heads=heads,
        head_mlps={
            head.name: _mlp(in_features, mlp_mult=mlp_mult, bias=False, dropout=0.0) for head in heads if head.mlp
        },
        final_layers={
            head.name: Linear.Config(in_features=in_features, out_features=head.output_size, bias=True)
            for head in heads
        },
        scale_layers={head.name: ScaleLayer.Config(n_features=head.output_size) for head in heads if head.scale},
    )
