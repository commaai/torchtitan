# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from functools import partial
from xx.training.path.model import parallelize_path
from xx.training.path.model_config import model_config as _model_config
from xx.training.path.model_constants import SUPERCOMBO_FPS

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.components.tokenizer import NoOpTokenizer
from torchtitan.config import CompileConfig, DebugConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.protocols.model_spec import ModelSpec

from .dataloader import Comma1MDataLoader
from .dataset import COMMA1M_REPO_ID
from .loss import PathMSELoss
from .trainer import Comma1MPathTrainer


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="path",
        flavor=flavor,
        model=_model_config(flavor),
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


def _comma1m_path(flavor: str) -> Comma1MPathTrainer.Config:
    steps = 16
    num_nodes, local_world_size = _dp_degrees()
    dataloader = Comma1MDataLoader.Config(
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
        model_spec=model_registry(flavor),
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


convnext_atto = partial(_comma1m_path, "convnext_atto")
convnext_xxlarge = partial(_comma1m_path, "convnext_xxlarge")
