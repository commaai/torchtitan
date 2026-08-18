# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from functools import partial
from typing import Literal

from xx.comma_data.constants import BASE_DIR_GT, DEFAULT_TEST_5K_LIST_TAGGED, DEFAULT_TRAIN_LIST

from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.components.tokenizer import NoOpTokenizer
from torchtitan.config import CompileConfig, DebugConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.protocols.model_spec import ModelSpec

from .dataset import PathDataLoader
from .loss import PathLoss
from .model import parallelize_path
from .model_config import model_config as _model_config, VISION_FEATURES, _spatial_size
from .model_constants import (
    frame_constants_from_fps,
    FRAME_TYPE,
    ModelInputs,
    N_FRAMES,
    SUPERCOMBO_FPS,
    TEMPORAL_INPUTS,
)
from .onnx_checkpoint import PathOnnxCheckpointManager
from .trainer import PathTrainer
from .validate import PathValidator


def model_registry(flavor: str, *, unvision: bool = False) -> ModelSpec:
    return ModelSpec(
        name="path",
        flavor=flavor,
        model=_model_config(flavor, unvision=unvision),
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


def _path(flavor: str, *, unvision: bool = False) -> PathTrainer.Config:
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
        model_spec=model_registry(flavor, unvision=unvision),
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
            unvision=unvision,
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
        fps=fps,
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
                unvision=unvision,
            ),
            mixed_precision_param=mixed_precision_param,
            reports=reports,
        ),
        debug=DebugConfig(seed=0),
    )


def _dataloader_config(
    *,
    dataset: str,
    split: Literal["train", "val"],
    fps: int,
    plan_only: bool,
    limit: int | None,
    deterministic_fidxs: bool,
    pipeline_dir: str,
    skip: int,
    val_skip: int,
    unvision: bool | None = None,
) -> PathDataLoader.Config:
    return PathDataLoader.Config(
        dataset=dataset,
        split=split,
        deterministic_fidxs=deterministic_fidxs,
        fps=fps,
        pipeline_dir=pipeline_dir,
        plan_only=plan_only,
        limit=limit,
        skip=skip,
        val_skip=val_skip,
        unvision=unvision,
    )


def _checkpoint_config(folder: str, base_folder: str, interval: int) -> PathOnnxCheckpointManager.Config:
    frame_constants = frame_constants_from_fps(n_frames=N_FRAMES, frame_type=FRAME_TYPE)
    temporal_len = frame_constants["temporal_len"]
    spatial_size = _spatial_size(frame_constants)
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
        [1, temporal_len, spatial_size, VISION_FEATURES],
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


def _optimizer_config() -> OptimizersContainer.Config:
    common = {"lr": 1e-3, "betas": (0.9, 0.95), "eps": 1e-8}
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
convnext_femto = partial(_path, "convnext_femto")
convnext_pico = partial(_path, "convnext_pico")
convnext_tiny = partial(_path, "convnext_tiny")
convnext_small = partial(_path, "convnext_small")
convnext_quarterxxl = partial(_path, "convnext_quarterxxl")
convnext_thirdxxl = partial(_path, "convnext_thirdxxl")
convnext_base = partial(_path, "convnext_base")
convnext_xxlarge = partial(_path, "convnext_xxlarge")
convnext_xxlarge_unvision = partial(_path, "convnext_xxlarge", unvision=True)
