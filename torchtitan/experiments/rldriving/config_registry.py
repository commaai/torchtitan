# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from typing import cast, Literal

from xx.comma_data.constants import BASE_DIR_GT, DEFAULT_TRAIN_LIST
from xx.ml_tools.constants.model import SUPERCOMBO_FPS
from xx.release_tests.lib.base_report import BaseReportConfig, ReportFormat
from xx.training.lib.torchtitan.report_runner import Report
from xx.training.rldriving.test import MODEL_REPORTS
from xx.training.path.model_config import model_config as path_model_config

from torchtitan.components.metrics import MetricsProcessor
from torchtitan.config import CompileConfig, DebugConfig, ParallelismConfig, TrainingConfig
from torchtitan.protocols.model_spec import ModelSpec

from .dataset import RLDrivingDataLoader
from .loss import RLDrivingLoss
from .model import actor_config, critic_config, parallelize_rldriving, RLDrivingModel
from .onnx_checkpoint import RLDrivingOnnxCheckpointManager
from .lora import LoRAConfig
from .optimizer import RLDrivingOptimizers
from .tokenizer import RLDrivingTokenizer
from .trainer import RLDrivingLRSchedulersConfig, RLDrivingTrainer


def model_registry(scope: Literal["policy", "all"] = "policy") -> ModelSpec:
    actor = actor_config()
    critic = critic_config(actor)
    path = path_model_config("convnext_xlarge", pretrained=False)
    path.vision.drop_path_rate = 0.0
    path.temporal_policy.temporal_summarizer.dense_training_outputs = False
    return ModelSpec(
        name="rldriving",
        flavor="default",
        model=RLDrivingModel.Config(
            actor=actor, critic=critic, lora=LoRAConfig(scope=scope),
            vision=path.vision, point_policy=path.point_policy, off_policy=path.temporal_policy,
        ),
        parallelize_fn=parallelize_rldriving,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )


def _make_reports(*, fps: int, num_epochs: int, steps_per_epoch: int) -> list[Report]:
    frequent_steps = [(epoch + 1) * steps_per_epoch for epoch in range(0, num_epochs, num_epochs // 10)]
    sparse_steps = [(epoch + 1) * steps_per_epoch for epoch in range(0, num_epochs, num_epochs // 2)]
    steps_by_report = {
        "analyse_lat.no_noise": frequent_steps,
        "analyse_lat.realistic_noise": frequent_steps,
        "analyse_long": frequent_steps,
        "analyse_unintended_lead_following": sparse_steps,
        "analyse_speed_convergence": sparse_steps,
        "analyse_platform_oscillation": sparse_steps,
        "analyse_nurec": sparse_steps,
    }

    def override_config(config: BaseReportConfig, eid: str) -> BaseReportConfig:
        return config.replace(
            rollout={"agent": {"supercombo": eid, "model_trained_fps": fps}},
            save_tmp=False,
            format=ReportFormat.HTML,
        )

    reports = []
    for report_name, (test_cls, config_cls) in MODEL_REPORTS.items():
        reports.append(
            Report(
                test_cls=test_cls,
                test_config=config_cls(),
                config_override_fn=override_config,
                steps=steps_by_report[report_name],
                wait_for_ckpt_keys=["model.onnx"],
            )
        )
    return reports


def rldriving() -> RLDrivingTrainer.Config:
    """Default: LoRA on the temporal action policy using cached vision features."""
    return _rldriving("policy")


def rldriving_lora_policy() -> RLDrivingTrainer.Config:
    return _rldriving("policy")


def rldriving_lora_all() -> RLDrivingTrainer.Config:
    """LoRA on the vision encoder and temporal action policy."""
    return _rldriving("all")


def _rldriving(scope: Literal["policy", "all"]) -> RLDrivingTrainer.Config:
    fps = SUPERCOMBO_FPS
    num_epochs = 201
    steps_per_epoch = 64
    model_spec = model_registry(scope)
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    world_size = int(os.environ.get("WORLD_SIZE", str(local_world_size)))
    num_nodes = int(os.environ.get("GROUP_WORLD_SIZE", str(world_size // local_world_size)))
    reporterv2_host = os.getenv("REPORTERV2_HOST")
    reporterv2_training_id = os.getenv("REPORTERV2_TRAINING_ID")
    checkpoint_base_folder = f"{reporterv2_host.rstrip('/')}/checkpoint" if reporterv2_host else ""
    return RLDrivingTrainer.Config(
        model_spec=model_spec,
        lora=cast(RLDrivingModel.Config, model_spec.model).lora,
        loss=RLDrivingLoss.Config(
            action_noise=(0.25, 0.25),
            gamma=0.95,
            fps=fps,
            smooth_lat_cost=0.15,
            smooth_long_cost=0.05,
            curv_rate_cost=20.0,
        ),
        warm_start_checkpoint=os.getenv(
            "RLDRIVING_WARM_START_CHECKPOINT",
            "b9facbcc-4d47-410e-b3ce-dfcbad12ba92/56320",
        ),
        tokenizer=RLDrivingTokenizer.Config(),
        dataloader=RLDrivingDataLoader.Config(
            dataset=DEFAULT_TRAIN_LIST,
            training_id=reporterv2_training_id or "",
            pipeline_dir=BASE_DIR_GT,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            fps=fps,
        ),
        optimizer=RLDrivingOptimizers.Config(implementation="fused"),
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
        reports=_make_reports(fps=fps, num_epochs=num_epochs, steps_per_epoch=steps_per_epoch),
        miniray={"priority": 3},
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
