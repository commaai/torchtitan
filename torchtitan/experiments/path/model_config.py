# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from dataclasses import replace

from torchtitan.models.common import Embedding, LayerNorm, Linear
from torchtitan.models.common.attention import ScaledDotProductAttention

from .hydra_configs import DRIVING_HEADS, META_HEADS, POSE_HEADS, TEMPORAL_META_HEADS
from .model import (
    Hydra,
    LinearEncoder,
    PathCrossAttention,
    PathHead,
    PathMLP,
    PathModel,
    PathSelfAttention,
    PathTransformer,
    PathTransformerBlock,
    PathTransformerDecoder,
    PathTransformerDecoderBlock,
    PlanVAE,
    PointSummarizer,
    Policy,
    ScaleLayer,
    SpatialUnvision,
    TemporalPolicy,
    TemporalSummarizer,
    Vision,
)
from .plan_vae import PlanLoss, PlanNormalization
from .model_constants import (
    frame_constants_from_fps,
    FRAME_TYPE,
    IDX_N,
    INPUT_FRAMES_NAMES,
    ModelInputs,
    N_FRAMES,
    PLAN_SIZE,
    PLAN_WIDTH,
    SPATIAL_SIZE,
    TEMPORAL_INPUTS,
    VISION_FEATURES,
    VISION_GRID_SIZE,
)


POINT_HEADS = tuple(META_HEADS + POSE_HEADS)
TEMPORAL_HEADS = tuple(DRIVING_HEADS + TEMPORAL_META_HEADS)


def model_config(
    flavor: str = "convnext_xxlarge",
    *,
    training_stage: str = "policy",
    plan_normalization: PlanNormalization = "pooled",
    plan_loss: PlanLoss = "decoded_laplacian",
    sample_plan_vae_posterior: bool = True,
    plan_latent_size: int = 64,
    plan_vae_encoder_layers: int = PlanVAE.N_ENCODER_LAYER,
    plan_vae_decoder_layers: int = PlanVAE.N_DECODER_LAYER,
) -> PathModel.Config:
    vision_features = VISION_FEATURES
    frame_constants = frame_constants_from_fps(n_frames=N_FRAMES, frame_type=FRAME_TYPE)
    input_frame_names = tuple(INPUT_FRAMES_NAMES)
    in_channels = sum(frame_constants["frame_shapes"][name][0] for name in input_frame_names)
    grid_size = VISION_GRID_SIZE

    return PathModel.Config(
        training_stage=training_stage,
        plan_loss=plan_loss,
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
        temporal_policy=temporal_policy_config(
            plan_vae=training_stage in ("plan_vae", "policy"),
            plan_normalization=plan_normalization,
            plan_loss=plan_loss,
            sample_plan_vae_posterior=sample_plan_vae_posterior,
            plan_latent_size=plan_latent_size,
            plan_vae_encoder_layers=plan_vae_encoder_layers,
            plan_vae_decoder_layers=plan_vae_decoder_layers,
        ),
        unvision_decoder=_spatial_unvision_config(in_features=vision_features, grid_size=grid_size),
    )


def temporal_policy_config(
    *,
    heads: tuple[PathHead, ...] = TEMPORAL_HEADS,
    dropout: float = 0.1,
    dense_training_outputs: bool = True,
    plan_vae: bool = False,
    plan_normalization: PlanNormalization = "pooled",
    plan_loss: PlanLoss = "decoded_laplacian",
    sample_plan_vae_posterior: bool = True,
    plan_latent_size: int = 64,
    plan_vae_encoder_layers: int = PlanVAE.N_ENCODER_LAYER,
    plan_vae_decoder_layers: int = PlanVAE.N_DECODER_LAYER,
) -> TemporalPolicy.Config:
    vision_features = VISION_FEATURES
    frame_constants = frame_constants_from_fps()
    history_idxs = tuple(int(index) for index in frame_constants["history_idxs"])
    desire_window_len = frame_constants["desire_window_len"]
    desire_window_starts = tuple(index - history_idxs[0] for index in history_idxs)
    block_size = len(history_idxs)
    spatial_size = SPATIAL_SIZE
    if plan_vae:
        # the plan head predicts a VAE latent (plus a log-sigma per plan value for the
        # decoded-laplacian loss, or a log-var per latent dim for the gaussian NLL
        # loss); the frozen VAE decoder turns the latent into the plan mean
        plan_head_size = {
            "decoded_laplacian": plan_latent_size + PLAN_SIZE,
            "latent_mse": plan_latent_size,
            "latent_nll": 2 * plan_latent_size,
        }[plan_loss]
        heads = tuple(
            head if head.name != "plan" else replace(head, output_size=plan_head_size)
            for head in heads
        )
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
            temporal_pos_embedding=Embedding.Config(
                num_embeddings=block_size,
                embedding_dim=vision_features,
            ),
            spatial_pos_embedding=Embedding.Config(
                num_embeddings=spatial_size,
                embedding_dim=vision_features,
            ),
            temporal_size=block_size,
            spatial_size=spatial_size,
            dense_training_outputs=dense_training_outputs,
        ),
        temporal_hydra=_hydra(heads, in_features=vision_features, mlp_mult=2),
        plan_vae=(
            _plan_vae_config(
                plan_latent_size,
                normalization=plan_normalization,
                sample_posterior_during_training=sample_plan_vae_posterior,
                encoder_layers=plan_vae_encoder_layers,
                decoder_layers=plan_vae_decoder_layers,
            )
            if plan_vae
            else None
        ),
        history_idxs=history_idxs,
    )


def _spatial_unvision_config(
    *,
    in_features: int,
    grid_size: tuple[int, int],
) -> SpatialUnvision.Config:
    dim = SpatialUnvision.N_EMBD
    layers = [
        PathTransformerBlock.Config(
            attention=_attention(dim=dim, n_head=SpatialUnvision.N_HEAD, dropout=0.0, is_causal=False),
            mlp=_mlp(dim=dim, mlp_mult=8 / 3, bias=False, dropout=0.0),
        )
        for _ in range(SpatialUnvision.N_LAYER)
    ]
    return SpatialUnvision.Config(
        in_features=in_features,
        grid_size=grid_size,
        transformer=PathTransformer.Config(layers=layers),
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


def _attention(*, dim: int, n_head: int, dropout: float, is_causal: bool = True) -> PathSelfAttention.Config:
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
        is_causal=is_causal,
    )


def _cross_attention(*, dim: int, n_head: int, dropout: float) -> PathCrossAttention.Config:
    head_dim = dim // n_head
    return PathCrossAttention.Config(
        query_norm=LayerNorm.Config(normalized_shape=dim),
        context_norm=LayerNorm.Config(normalized_shape=dim),
        q_norm=LayerNorm.Config(normalized_shape=head_dim),
        k_norm=LayerNorm.Config(normalized_shape=head_dim),
        q_proj=Linear.Config(in_features=dim, out_features=dim, bias=True),
        kv_proj=Linear.Config(in_features=dim, out_features=2 * dim, bias=True),
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


def _plan_vae_config(
    latent_size: int,
    *,
    dim: int = PlanVAE.N_EMBD,
    normalization: PlanNormalization = "pooled",
    sample_posterior_during_training: bool = True,
    encoder_layers: int = PlanVAE.N_ENCODER_LAYER,
    decoder_layers: int = PlanVAE.N_DECODER_LAYER,
) -> PlanVAE.Config:
    if dim % PlanVAE.N_HEAD != 0:
        raise ValueError(f"Plan VAE dimension {dim} must be divisible by {PlanVAE.N_HEAD} attention heads")
    if encoder_layers <= 0 or decoder_layers <= 0:
        raise ValueError(
            f"Plan VAE layer counts must be positive, got encoder={encoder_layers}, decoder={decoder_layers}"
        )
    encoder = PathTransformer.Config(
        layers=[
            PathTransformerBlock.Config(
                attention=_attention(
                    dim=dim,
                    n_head=PlanVAE.N_HEAD,
                    dropout=0.0,
                    is_causal=False,
                ),
                mlp=_mlp(dim, mlp_mult=8 / 3, bias=True, dropout=0.0),
            )
            for _ in range(encoder_layers)
        ]
    )
    decoder = PathTransformerDecoder.Config(
        layers=[
            PathTransformerDecoderBlock.Config(
                self_attention=_attention(
                    dim=dim,
                    n_head=PlanVAE.N_HEAD,
                    dropout=0.0,
                    is_causal=False,
                ),
                cross_attention=_cross_attention(
                    dim=dim,
                    n_head=PlanVAE.N_HEAD,
                    dropout=0.0,
                ),
                mlp=_mlp(dim, mlp_mult=8 / 3, bias=True, dropout=0.0),
            )
            for _ in range(decoder_layers)
        ]
    )
    pooled_dim = PlanVAE.N_LATENT_TOKENS * dim
    return PlanVAE.Config(
        plan_size=PLAN_SIZE,
        plan_horizon=IDX_N,
        plan_width=PLAN_WIDTH,
        latent_size=latent_size,
        num_latent_tokens=PlanVAE.N_LATENT_TOKENS,
        normalization=normalization,
        sample_posterior_during_training=sample_posterior_during_training,
        input_projection=Linear.Config(in_features=PLAN_WIDTH, out_features=dim, bias=True),
        plan_pos_embedding=Embedding.Config(num_embeddings=IDX_N, embedding_dim=dim),
        encoder=encoder,
        pool_query_embedding=Embedding.Config(
            num_embeddings=PlanVAE.N_LATENT_TOKENS,
            embedding_dim=dim,
        ),
        pool_attention=_cross_attention(dim=dim, n_head=PlanVAE.N_HEAD, dropout=0.0),
        posterior_norm=LayerNorm.Config(normalized_shape=pooled_dim),
        posterior_projection=Linear.Config(in_features=pooled_dim, out_features=2 * latent_size, bias=True),
        latent_projection=Linear.Config(in_features=latent_size, out_features=pooled_dim, bias=True),
        decoder_pos_embedding=Embedding.Config(num_embeddings=IDX_N, embedding_dim=dim),
        decoder=decoder,
        output_norm=LayerNorm.Config(normalized_shape=dim),
        output_projection=Linear.Config(in_features=dim, out_features=PLAN_WIDTH, bias=True),
        output_scale=ScaleLayer.Config(n_features=PLAN_SIZE),
    )
