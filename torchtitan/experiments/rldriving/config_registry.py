# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from typing import cast

from xx.comma_data.constants import BASE_DIR_GT, DEFAULT_TRAIN_LIST
from xx.ml_tools.constants.model import SUPERCOMBO_FPS

from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.components.tokenizer import NoOpTokenizer
from torchtitan.config import CompileConfig, DebugConfig, ParallelismConfig, TrainingConfig
from torchtitan.protocols.model_spec import ModelSpec

from .dataset import RLDrivingDataLoader
from .loss import RLDrivingLoss
from .model import actor_config, critic_config, parallelize_rldriving, RLDrivingModel
from .onnx_checkpoint import RLDrivingOnnxCheckpointManager
from .trainer import RLDrivingLRSchedulersConfig, RLDrivingTrainer
from .validate import RLDrivingValidator


def model_registry() -> ModelSpec:
    actor = actor_config()
    critic = critic_config(actor)
    return ModelSpec(
        name="rldriving",
        flavor="default",
        model=RLDrivingModel.Config(actor=actor, critic=critic),
        parallelize_fn=parallelize_rldriving,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )


def rldriving() -> RLDrivingTrainer.Config:
    fps = SUPERCOMBO_FPS
    num_epochs = 201
    steps_per_epoch = 64
    model_spec = model_registry()
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    world_size = int(os.environ.get("WORLD_SIZE", str(local_world_size)))
    num_nodes = int(os.environ.get("GROUP_WORLD_SIZE", str(world_size // local_world_size)))
    reporterv2_host = os.getenv("REPORTERV2_HOST")
    reporterv2_training_id = os.getenv("REPORTERV2_TRAINING_ID")
    checkpoint_base_folder = f"{reporterv2_host.rstrip('/')}/checkpoint" if reporterv2_host else ""
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
        model_spec=model_spec,
        loss=RLDrivingLoss.Config(
            action_noise=(0.25, 0.25),
            gamma=0.95,
            fps=fps,
            smooth_lat_cost=0.1,
            smooth_long_cost=0.1,
            curv_cost=100.0,
        ),
        warm_start_checkpoint=os.getenv(
            "RLDRIVING_WARM_START_CHECKPOINT",
            "44b83fa5-2a33-7ee7-40f1-e86e3c24ad36/56320",
        ),
        tokenizer=NoOpTokenizer.Config(),
        dataloader=RLDrivingDataLoader.Config(
            dataset=DEFAULT_TRAIN_LIST,
            training_id=reporterv2_training_id or "",
            pipeline_dir=BASE_DIR_GT,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            fps=fps,
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
        checkpoint=_checkpoint_config(
            cast(RLDrivingModel.Config, model_spec.model),
            base_folder=checkpoint_base_folder,
            folder=reporterv2_training_id or "checkpoint",
            interval=steps_per_epoch,
        ),
        steps_per_epoch=steps_per_epoch,
        train_step_barrier_timeout_seconds=60 * 60,
        ema_tau=128.0,
        fps=fps,
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
            fps=fps,
            reports=reports,
            miniray={"priority": 3},
        ),
        debug=DebugConfig(seed=0),
    )


def _checkpoint_config(
    model: RLDrivingModel.Config,
    *,
    base_folder: str,
    folder: str,
    interval: int,
) -> RLDrivingOnnxCheckpointManager.Config:
    input_shapes = RLDrivingModel.input_shapes(model)
    return RLDrivingOnnxCheckpointManager.Config(
        keep_latest_k=0,
        enable=True,
        checkpoint_base_folder=base_folder,
        export_onnx=True,
        folder=folder,
        interval=interval,
        input_names=list(input_shapes),
        input_shapes=[list(shape) for shape in input_shapes.values()],
        input_dtypes=["float32"] * len(input_shapes),
    )
