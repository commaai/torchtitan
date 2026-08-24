# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Model and smoke-test configs for the Wan 2.2 VAE experiment."""

from __future__ import annotations

import os

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.components.tokenizer import NoOpTokenizer
from torchtitan.config import CompileConfig, DebugConfig, ParallelismConfig, TrainingConfig
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer

from .dataset import WanVAEDataLoader
from .loss import WanReconstructionLoss
from .model import WanVAE
from .parallelize import parallelize_wan_vae


def _official_config(*, training_stage: str = "frozen") -> WanVAE.Config:
    return WanVAE.Config(training_stage=training_stage)


def _debug_config(*, training_stage: str = "full") -> WanVAE.Config:
    return WanVAE.Config(
        dim=4,
        decoder_dim=4,
        latent_channels=4,
        dim_mult=(1, 2, 2, 2),
        num_res_blocks=1,
        temporal_downsample=(False, True, True),
        latent_mean=(0.0,) * 4,
        latent_std=(1.0,) * 4,
        training_stage=training_stage,
    )


def model_registry(flavor: str = "official") -> ModelSpec:
    configs = {
        "official": lambda: _official_config(training_stage="frozen"),
        "decoder": lambda: _official_config(training_stage="decoder"),
        "full": lambda: _official_config(training_stage="full"),
        "debug": _debug_config,
    }
    try:
        model_config = configs[flavor]()
    except KeyError as exc:
        raise ValueError(f"unsupported Wan VAE flavor {flavor!r}; choose from {sorted(configs)}") from exc
    return ModelSpec(
        name="wan_vae",
        flavor=flavor,
        model=model_config,
        parallelize_fn=parallelize_wan_vae,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )


def _dp_degrees() -> tuple[int, int]:
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    world_size = int(os.environ.get("WORLD_SIZE", str(local_world_size)))
    num_nodes = int(
        os.environ.get(
            "GROUP_WORLD_SIZE",
            str(max(1, world_size // max(1, local_world_size))),
        )
    )
    return num_nodes, local_world_size


def wan_vae_debug() -> Trainer.Config:
    """Tiny end-to-end config for CPU/GPU correctness and compile smoke tests."""

    num_nodes, local_world_size = _dp_degrees()
    return Trainer.Config(
        loss=WanReconstructionLoss.Config(
            mae_weight=1.0,
            mse_weight=1.0,
            lpips_weight=0.0,
        ),
        model_spec=model_registry("debug"),
        tokenizer=NoOpTokenizer.Config(),
        dataloader=WanVAEDataLoader.Config(
            dataset="mock",
            split="train",
            pipeline_dir="",
            image_size=(32, 32),
            clip_frames=9,
            shuffle_size=8,
            num_writers=1,
            num_readers=1,
            mock_data=True,
            mock_segment_batch_size=1,
        ),
        optimizer=default_adamw(lr=1e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=0,
            total_steps=2,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            global_batch_size=-1,
            seq_len=1,
            steps=2,
            max_norm=1.0,
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
        activation_checkpoint=None,
        compile=CompileConfig(enable=True, components=["model"]),
        checkpoint=CheckpointManager.Config(enable=False),
        metrics=MetricsProcessor.Config(log_freq=1),
        debug=DebugConfig(seed=0),
    )


__all__ = ["model_registry", "wan_vae_debug"]
