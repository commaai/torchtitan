# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from functools import partial
from typing import Literal, TYPE_CHECKING

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.components.tokenizer import NoOpTokenizer
from torchtitan.config import CompileConfig, DebugConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.protocols.model_spec import ModelSpec

from .dataset import COMMA1M_REPO_ID, PathDataLoader
from .model import parallelize_path
from .model_config import model_config as _model_config
from .model_constants import (
    frame_constants_from_fps,
    FRAME_TYPE,
    ModelInputs,
    N_FRAMES,
    SUPERCOMBO_FPS,
    TEMPORAL_INPUTS,
)

if TYPE_CHECKING:
    from .comma1m_trainer import Comma1MPathTrainer
    from .onnx_checkpoint import PathOnnxCheckpointManager
    from .trainer import PathTrainer


def model_registry(flavor: str, *, pretrained: bool = True) -> ModelSpec:
    model = _model_config(flavor)
    model.vision.pretrained = pretrained
    return ModelSpec(
        name="path",
        flavor=flavor,
        model=model,
        parallelize_fn=parallelize_path,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )


def _dp_degrees() -> tuple[int, int]:
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    world_size = int(os.environ.get("WORLD_SIZE", str(local_world_size)))
    num_nodes = int(os.environ.get("GROUP_WORLD_SIZE", str(world_size // local_world_size)))
    return num_nodes, local_world_size


def _path(flavor: str) -> PathTrainer.Config:
    from xx.comma_data.constants import BASE_DIR_GT, DEFAULT_TEST_5K_LIST_TAGGED, DEFAULT_TRAIN_LIST

    from .loss import PathLoss
    from .trainer import PathTrainer
    from .validate import PathValidator

    steps = 1024 * 55
    validation_freq = 1024
    reports = {
        name: [validation_freq, validation_freq * 20, steps]
        for name in (
            "analyse_driving",
            "analyse_lat_no_noise",
            "analyse_cones",
            "analyse_lights",
            "analyse_stop",
            "analyse_hard_brake",
        )
    }
    reports["analyse_dataset"] = [validation_freq]
    mixed_precision_param = "bfloat16"
    num_nodes, local_world_size = _dp_degrees()
    reporterv2_host = os.getenv("REPORTERV2_HOST")
    reporterv2_training_id = os.getenv("REPORTERV2_TRAINING_ID")
    checkpoint_base_folder = f"{reporterv2_host.rstrip('/')}/checkpoint" if reporterv2_host else ""
    fps = SUPERCOMBO_FPS
    plan_only = False
    return PathTrainer.Config(
        loss=PathLoss.Config(),
        model_spec=model_registry(flavor),
        tokenizer=NoOpTokenizer.Config(),
        dataloader=_dataloader_config(
            dataset=DEFAULT_TRAIN_LIST,
            split="train",
            fps=fps,
            plan_only=plan_only,
            limit=2_500_000,
            deterministic_fidxs=False,
            pipeline_dir=BASE_DIR_GT,
            skip=1,
            val_skip=1,
        ),
        optimizer=_optimizer_config(),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=1024,
            total_steps=steps,
            decay_ratio=0.1,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=16,
            global_batch_size=-1,
            seq_len=1,
            steps=steps,
            max_norm=1.0,
            dtype="float32",
            mixed_precision_param=mixed_precision_param,
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
            folder=reporterv2_training_id or "checkpoint",
            base_folder=checkpoint_base_folder,
            interval=validation_freq,
        ),
        activation_checkpoint=FullAC.Config(),
        compile=CompileConfig(enable=True, components=["model"]),
        metrics=MetricsProcessor.Config(log_freq=16, enable_reporterv2=True, save_freq=validation_freq),
        validator=PathValidator.Config(
            enable=True,
            freq=validation_freq,
            steps=32,
            dataloader=_dataloader_config(
                split="val",
                fps=fps,
                plan_only=True,
                dataset=DEFAULT_TEST_5K_LIST_TAGGED,
                limit=6_000,
                deterministic_fidxs=True,
                pipeline_dir=BASE_DIR_GT,
                skip=1,
                val_skip=6,
            ),
            mixed_precision_param=mixed_precision_param,
            reports=reports,
        ),
        debug=DebugConfig(seed=0),
    )


def _comma1m_path(flavor: str, *, pretrained: bool = True) -> Comma1MPathTrainer.Config:
    from .comma1m_trainer import Comma1MPathTrainer
    from .loss import PathMSELoss

    steps = 16
    num_nodes, local_world_size = _dp_degrees()
    dataloader = _dataloader_config(
        dataset=COMMA1M_REPO_ID,
        dataset_path=os.getenv("COMMA1M_DATASET_PATH"),
        split="train",
        fps=SUPERCOMBO_FPS,
        plan_only=True,
        limit=None,
        deterministic_fidxs=False,
        pipeline_dir=None,
        skip=1,
        val_skip=1,
    )
    dataloader.num_writers = 1
    dataloader.shuffle_size = 64
    dataloader.min_mixing = 0
    return Comma1MPathTrainer.Config(
        loss=PathMSELoss.Config(),
        model_spec=model_registry(flavor, pretrained=pretrained),
        tokenizer=NoOpTokenizer.Config(),
        dataloader=dataloader,
        optimizer=_optimizer_config(lr=1e-6),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=0,
            total_steps=steps,
            decay_ratio=0,
            decay_type="linear",
            min_lr_factor=0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=1,
            steps=steps,
            mixed_precision_param="bfloat16",
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=num_nodes,
            data_parallel_shard_degree=local_world_size,
            enable_sequence_parallel=False,
        ),
        checkpoint=CheckpointManager.Config(
            enable=True,
            folder="checkpoint",
            interval=steps,
            enable_first_step_checkpoint=True,
        ),
        activation_checkpoint=FullAC.Config(),
        compile=CompileConfig(enable=True, components=["model", "loss"]),
        metrics=MetricsProcessor.Config(log_freq=1, enable_wandb=True),
        debug=DebugConfig(seed=0),
    )


def _comma1m_ci_path(flavor: str) -> Comma1MPathTrainer.Config:
    config = _comma1m_path(flavor, pretrained=False)
    config.training.steps = 1
    config.training.local_batch_size = 1
    config.training.dtype = "bfloat16"
    config.lr_scheduler.total_steps = 1
    config.dataloader.limit = 1
    config.checkpoint.enable = False
    config.compile.enable = False
    config.metrics.enable_wandb = False
    config.activation_checkpoint = None
    return config


def _dataloader_config(
    *,
    dataset: str,
    dataset_path: str | None = None,
    split: Literal["train", "val"],
    fps: int,
    plan_only: bool,
    limit: int | None,
    deterministic_fidxs: bool,
    pipeline_dir: str | None,
    skip: int,
    val_skip: int,
) -> PathDataLoader.Config:
    return PathDataLoader.Config(
        dataset=dataset,
        dataset_path=dataset_path,
        split=split,
        deterministic_fidxs=deterministic_fidxs,
        fps=fps,
        pipeline_dir=pipeline_dir,
        plan_only=plan_only,
        limit=limit,
        skip=skip,
        val_skip=val_skip,
    )


def _checkpoint_config(
    folder: str,
    base_folder: str,
    interval: int,
) -> PathOnnxCheckpointManager.Config:
    from .onnx_checkpoint import PathOnnxCheckpointManager

    frame_constants = frame_constants_from_fps(n_frames=N_FRAMES, frame_type=FRAME_TYPE)
    temporal_len = frame_constants["temporal_len"]
    vision_input_names = [ModelInputs.IMG, ModelInputs.BIG_IMG]
    temporal_policy_input_names = [
        ModelInputs.FEATURES,
        ModelInputs.DESIRE,
        ModelInputs.TRAFFIC,
        ModelInputs.ACTION_T,
    ]
    input_names = [
        *vision_input_names,
        *temporal_policy_input_names,
    ]
    input_shapes = [
        [1, *frame_constants["frame_shapes"][ModelInputs.IMG]],
        [1, *frame_constants["frame_shapes"][ModelInputs.BIG_IMG]],
        [1, temporal_len, *TEMPORAL_INPUTS[ModelInputs.FEATURES]],
        [1, temporal_len, TEMPORAL_INPUTS[ModelInputs.DESIRE][0]],
        [1, temporal_len, TEMPORAL_INPUTS[ModelInputs.TRAFFIC][0]],
        [1, temporal_len, TEMPORAL_INPUTS[ModelInputs.ACTION_T][0]],
    ]
    return PathOnnxCheckpointManager.Config(
        keep_latest_k=0,  # keep all checkpoints
        enable=True,
        checkpoint_base_folder=base_folder,
        export_onnx=True,
        enable_first_step_checkpoint=True,
        folder=folder,
        interval=interval,
        input_names=input_names,
        input_shapes=input_shapes,
        input_dtypes=["float32"] * len(input_names),
        vision_onnx_compute_dtype="bfloat16",
        temporal_policy_onnx_compute_dtype="float32",
        vision_input_names=vision_input_names,
        temporal_policy_input_names=temporal_policy_input_names,
    )


def _optimizer_config(
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> OptimizersContainer.Config:
    common = {"lr": lr, "betas": betas, "eps": eps}
    no_decay = r"(point_policy\.hydra|temporal_policy\.temporal_hydra)\.(final_layer|scale_layer)"
    return OptimizersContainer.Config(
        implementation="fused_opt_states_bf16",
        param_groups=[
            ParamGroupConfig(
                pattern=no_decay,
                optimizer_name="AdamW",
                optimizer_kwargs={**common, "weight_decay": 0.0},
            ),
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="AdamW",
                optimizer_kwargs={**common, "weight_decay": 1e-3},
            ),
        ],
    )


convnext_atto = partial(_path, "convnext_atto")
convnext_pico_comma1m_ci = partial(_comma1m_ci_path, "convnext_pico")
convnext_xxlarge_comma1m = partial(_comma1m_path, "convnext_xxlarge")
convnext_femto = partial(_path, "convnext_femto")
convnext_pico = partial(_path, "convnext_pico")
convnext_tiny = partial(_path, "convnext_tiny")
convnext_small = partial(_path, "convnext_small")
convnext_quarterxxl = partial(_path, "convnext_quarterxxl")
convnext_thirdxxl = partial(_path, "convnext_thirdxxl")
convnext_base = partial(_path, "convnext_base")
convnext_xxlarge = partial(_path, "convnext_xxlarge")
