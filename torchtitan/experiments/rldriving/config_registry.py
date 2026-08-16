# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
import math
import os
from typing import Any, cast
from xx.datasets.constants import BASE_DIR_GT, DEFAULT_TRAIN_LIST

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as _pydantic_dataclass

from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.components.tokenizer import NoOpTokenizer
from torchtitan.config import CompileConfig, DebugConfig, ParallelismConfig, TrainingConfig
from torchtitan.experiments.path.model import Hydra, LinearEncoder, PathHead, PathMLP, TemporalPolicy
from torchtitan.models.common import LayerNorm, Linear
from torchtitan.protocols.model_spec import ModelSpec

from .dataset import RLDrivingDataLoader
from .loss import RLDrivingLoss
from .model import ACTION_HEAD_NAME, Critic, parallelize_rldriving, Q_HEAD_NAME, RLDrivingModel, TwinCritic
from .onnx_checkpoint import RLDrivingOnnxCheckpointManager
from .trainer import RLDrivingLRSchedulersConfig, RLDrivingTrainer
from .validate import RLDrivingValidator


_PathTemporalPolicyConfig = _pydantic_dataclass(
    TemporalPolicy.Config,
    config=ConfigDict(arbitrary_types_allowed=True),
    slots=True,
)


def model_registry(temporal_policy_hparams: dict[str, Any]) -> ModelSpec:
    actor = cast(TemporalPolicy.Config, _PathTemporalPolicyConfig(**temporal_policy_hparams))
    actor.temporal_summarizer.dense_training_outputs = False
    for layer in actor.temporal_summarizer.transformer.layers:
        layer.attention.dropout = 0.0
        layer.mlp.dropout = 0.0
    hydra = actor.temporal_hydra
    hydra.heads = tuple(head for head in hydra.heads if head.name == ACTION_HEAD_NAME)
    for layers in (hydra.head_mlps, hydra.final_layers, hydra.scale_layers):
        for name in tuple(layers):
            if name != ACTION_HEAD_NAME:
                del layers[name]

    dim = actor.temporal_summarizer.pos_embedding.embedding_dim
    hidden = 256 * math.ceil(2 * dim / 256)
    post_action_mlp = PathMLP.Config(
        norm=LayerNorm.Config(normalized_shape=dim),
        c_fc=Linear.Config(in_features=dim, out_features=hidden, bias=False),
        c_proj=Linear.Config(in_features=hidden, out_features=dim, bias=False),
        act="gelu_tanh",
        dropout=0.0,
    )
    critic = Critic.Config(
        temporal_summarizer=copy.deepcopy(actor.temporal_summarizer),
        history_idxs=actor.history_idxs,
        action_encoder=LinearEncoder.Config(
            in_layer=Linear.Config(in_features=2, out_features=dim, bias=True),
            out_layer=Linear.Config(in_features=dim, out_features=dim, bias=False),
        ),
        post_action_mlp1=post_action_mlp,
        post_action_mlp2=copy.deepcopy(post_action_mlp),
        q_hydra=Hydra.Config(
            heads=(PathHead(name=Q_HEAD_NAME, output_size=1, mlp=False, scale=False),),
            head_mlps={},
            final_layers={Q_HEAD_NAME: Linear.Config(in_features=dim, out_features=1, bias=True)},
            scale_layers={},
        ),
    )
    return ModelSpec(
        name="rldriving",
        flavor="default",
        model=RLDrivingModel.Config(actor=actor, critic=TwinCritic.Config(critic=critic)),
        parallelize_fn=parallelize_rldriving,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )


def rldriving() -> RLDrivingTrainer.Config:
    num_epochs = 201
    steps_per_epoch = 64
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    world_size = int(os.environ.get("WORLD_SIZE", str(local_world_size)))
    num_nodes = int(os.environ.get("GROUP_WORLD_SIZE", str(world_size // local_world_size)))
    reporterv2_host = os.getenv("REPORTERV2_HOST")
    reporterv2_training_id = os.getenv("REPORTERV2_TRAINING_ID")
    actor_optim = {"lr": 4e-5, "betas": (0.9, 0.999), "eps": 1e-8}
    critic_optim = {"lr": 2e-4, "betas": (0.9, 0.999), "eps": 1e-8}
    frequent_report_steps = [(epoch + 1) * steps_per_epoch for epoch in range(0, num_epochs, num_epochs // 10)]
    sparse_report_steps = [(epoch + 1) * steps_per_epoch for epoch in range(0, num_epochs, num_epochs // 2)]
    reports = dict.fromkeys(
        (
            "analyse_lat.no_noise",
            "analyse_lat.realistic_noise",
            "analyse_long",
        ),
        frequent_report_steps,
    ) | dict.fromkeys(
        (
            "analyse_unintended_lead_following",
            "analyse_speed_convergence",
            "analyse_platform_oscillation",
            "analyse_nurec",
        ),
        sparse_report_steps,
    )
    return RLDrivingTrainer.Config(
        loss=RLDrivingLoss.Config(
            action_noise=(0.25, 0.25),
            gamma=0.95,
            fps=0.0,
            smooth_lat_cost=0.1,
            smooth_long_cost=0.1,
            curv_cost=100.0,
        ),
        warm_start_checkpoint=os.getenv(
            "RLDRIVING_WARM_START_CHECKPOINT",
            "1acf0a93-3b20-4808-beb4-739aca6bb852/100",
        ),
        tokenizer=NoOpTokenizer.Config(),
        dataloader=RLDrivingDataLoader.Config(
            dataset=DEFAULT_TRAIN_LIST,
            training_id=reporterv2_training_id or "",
            pipeline_dir=BASE_DIR_GT,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
        ),
        optimizer=OptimizersContainer.Config(
            implementation="fused",
            param_groups=[
                ParamGroupConfig(
                    pattern=r"^actor\.temporal_hydra\.(final_layer|scale_layer)\.",
                    optimizer_name="AdamW",
                    optimizer_kwargs={**actor_optim, "weight_decay": 0.0},
                ),
                ParamGroupConfig(
                    pattern=r"^actor\.",
                    optimizer_name="AdamW",
                    optimizer_kwargs={**actor_optim, "weight_decay": 3e-2},
                ),
                ParamGroupConfig(
                    pattern=r"^critic\.(critic1|critic2)\.q_hydra\.(final_layer|scale_layer)\.",
                    optimizer_name="AdamW",
                    optimizer_kwargs={**critic_optim, "weight_decay": 0.0},
                ),
                ParamGroupConfig(
                    pattern=r"^critic\.",
                    optimizer_name="AdamW",
                    optimizer_kwargs={**critic_optim, "weight_decay": 3e-2},
                ),
            ],
        ),
        lr_scheduler=RLDrivingLRSchedulersConfig(
            steps_per_epoch=steps_per_epoch,
            num_epochs=num_epochs,
        ),
        training=TrainingConfig(
            local_batch_size=32,
            global_batch_size=-1,
            seq_len=1,
            max_norm=1.0,
            steps=num_epochs * steps_per_epoch,
            dtype="float32",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=num_nodes,
            data_parallel_shard_degree=local_world_size,
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
            expert_parallel_degree=1,
            enable_sequence_parallel=False,
        ),
        checkpoint=RLDrivingOnnxCheckpointManager.Config(
            keep_latest_k=0,
            enable=True,
            checkpoint_base_folder=f"{reporterv2_host.rstrip('/')}/checkpoint" if reporterv2_host else "",
            export_onnx=True,
            folder=reporterv2_training_id or "checkpoint",
            interval=steps_per_epoch,
        ),
        steps_per_epoch=steps_per_epoch,
        ema_tau=128.0,
        activation_checkpoint=None,
        compile=CompileConfig(enable=True, components=["model"]),
        metrics=MetricsProcessor.Config(
            log_freq=16,
            enable_reporterv2=True,
            save_freq=steps_per_epoch,
        ),
        validator=RLDrivingValidator.Config(
            enable=True,
            freq=steps_per_epoch,
            fps=0,
            reports=reports,
            miniray={"priority": 3},
        ),
        debug=DebugConfig(seed=0),
    )
