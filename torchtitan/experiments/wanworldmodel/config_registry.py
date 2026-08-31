# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from dataclasses import fields, replace
from pathlib import Path

from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import ParamGroupConfig, default_adamw
from torchtitan.components.torchpackage_checkpoint import TorchPackageCheckpointManager
from torchtitan.config import CompileConfig, DebugConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.experiments.wan_vae.tokenizer import (
    DEFAULT_WAN_TEXT_CONTEXT_FILENAME,
    DEFAULT_WAN_TEXT_PROMPT,
    WanVAETokenizer,
)
from torchtitan.experiments.worldmodel.config_registry import worldmodel
from torchtitan.experiments.worldmodel.loss import WorldModelLoss
from torchtitan.experiments.worldmodel.trainer import (
    WorldModelFloat8Config,
    WorldModelTrainer,
    WorldModelValidator,
)

from .dataset_config import _dataloader_config
from .model_config import (
    _wan_blocks_only_float8,
    model_registry,
    self_forcing_model_registry,
    WAN_FLOAT8_FILTER_FQNS,
)
from .self_forcing_optimizer import (
    WanSelfForcingLRSchedulers,
    WanSelfForcingOptimizers,
)
from .self_forcing_trainer import WanSelfForcingTrainer
from .trainer import WanWorldModelTrainer, WanWorldModelValidator
from .torchpackage_checkpoint import (
    MODEL_CONFIG_FILE,
    STRUCTURED_LOG_DIR,
    WAN_TRAINING_TORCH_PACKAGE_RECIPE,
    WanTorchPackageCheckpointManager,
)


__all__ = [
    "WAN_FLOAT8_FILTER_FQNS",
    "_dataloader_config",
    "_wan_blocks_only_float8",
    "model_registry",
    "self_forcing_model_registry",
    "worldmodel_wan",
    "worldmodel_wan_debug",
    "worldmodel_wan_self_forcing",
    "worldmodel_wan_self_forcing_debug",
]


DEFAULT_SELF_FORCING_INITIAL_CHECKPOINT = (
    "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/0f65f0b5-ab3c-46b6-b82d-60ff56e0030c/1280"
)


def _wan_checkpoint_dir() -> str:
    default = Path("/raid.unprotected/yassine/Wan2.2-TI2V-5B")
    return os.getenv("WAN_TI2V_5B_CHECKPOINT", str(default))


def _wan_text_context_path(checkpoint_dir: str) -> str:
    default = os.path.join(checkpoint_dir, DEFAULT_WAN_TEXT_CONTEXT_FILENAME)
    return os.getenv("WAN_TEXT_CONTEXT_PATH", default)


def _wan_validator_config(
    config: WorldModelValidator.Config,
    **changes: object,
) -> WanWorldModelValidator.Config:
    values = {
        item.name: getattr(config, item.name)
        for item in fields(WanWorldModelValidator.Config)
        if item.init and hasattr(config, item.name)
    }
    values.update(changes)
    return WanWorldModelValidator.Config(**values)


def _wan_trainer_config(
    config: WorldModelTrainer.Config,
    **changes: object,
) -> WanWorldModelTrainer.Config:
    values = {
        item.name: getattr(config, item.name)
        for item in fields(WanWorldModelTrainer.Config)
        if item.init and hasattr(config, item.name)
    }
    values.update(changes)
    return WanWorldModelTrainer.Config(**values)


def _wan_self_forcing_trainer_config(
    config: WorldModelTrainer.Config,
    **changes: object,
) -> WanSelfForcingTrainer.Config:
    values = {
        item.name: getattr(config, item.name)
        for item in fields(WanSelfForcingTrainer.Config)
        if item.init and hasattr(config, item.name)
    }
    values.update(changes)
    return WanSelfForcingTrainer.Config(**values)


def _wan_checkpoint_config(
    config: TorchPackageCheckpointManager.Config,
    **changes: object,
) -> WanTorchPackageCheckpointManager.Config:
    values = {
        item.name: getattr(config, item.name)
        for item in fields(WanTorchPackageCheckpointManager.Config)
        if item.init and hasattr(config, item.name)
    }
    values.update(
        torch_package_recipe=WAN_TRAINING_TORCH_PACKAGE_RECIPE,
        torch_package_recipe_state_file=MODEL_CONFIG_FILE,
        torch_package_structured_log_dir=STRUCTURED_LOG_DIR,
    )
    values.update(changes)
    return WanTorchPackageCheckpointManager.Config(**values)


def worldmodel_wan() -> WanWorldModelTrainer.Config:
    """Train the pretrained Wan 2.2 TI2V-5B transformer on camera video."""
    base = worldmodel()
    optimizer = replace(
        base.optimizer,
        param_groups=[
            replace(
                group,
                optimizer_kwargs={**group.optimizer_kwargs, "lr": 1e-5},
            )
            for group in base.optimizer.param_groups
        ],
    )
    checkpoint_dir = _wan_checkpoint_dir()
    train_dataloader = _dataloader_config(split="train")
    validation_dataloader = _dataloader_config(split="val")
    return _wan_trainer_config(
        base,
        hf_assets_path=checkpoint_dir,
        loss=WorldModelLoss.Config(plan_loss_weight=0.0),
        tokenizer=WanVAETokenizer.Config(
            compressor_model=checkpoint_dir,
            image_size=train_dataloader.image_size,
            text_context_path=_wan_text_context_path(checkpoint_dir),
            text_prompt=DEFAULT_WAN_TEXT_PROMPT,
        ),
        model_spec=model_registry("wan_ti2v_5b"),
        dataloader=train_dataloader,
        optimizer=optimizer,
        training=replace(
            base.training,
            local_batch_size=4,
            global_batch_size=-1,
            seq_len=1,
        ),
        checkpoint=_wan_checkpoint_config(
            base.checkpoint,
            initial_load_path=checkpoint_dir,
            initial_load_in_hf=True,
            initial_load_model_only=True,
        ),
        validator=_wan_validator_config(
            base.validator,
            dataloader=validation_dataloader,
            local_batch_size_override=4,
            pose_dropout=0.0,
            no_noise_prefill_frames_prob=0.0,
            fake_timesteps_prob=0.0,
        ),
        float8=WorldModelFloat8Config(enable=True),
        activation_checkpoint=FullAC.Config(),
        pose_dropout=0.0,
        no_noise_prefill_frames_prob=0.5,
        fake_timesteps_prob=0.5,
    )


def worldmodel_wan_debug() -> WanWorldModelTrainer.Config:
    """Run the Wan RF training path with a reduced model and mock latents."""
    base = worldmodel_wan()
    dataloader = _dataloader_config(
        split="train",
        dataset="mock",
        pipeline_dir="",
        image_size=(64, 64),
        latent_channels=4,
        latent_size=(4, 4),
        mock_data=True,
        mock_segment_batch_size=1,
        mock_latents=True,
    )
    return replace(
        base,
        hf_assets_path=".",
        dump_folder="./outputs/wanworldmodel_debug",
        tokenizer=WanVAETokenizer.Config(
            compressor_model="",
            image_size=dataloader.image_size,
            text_context_path="",
            text_prompt="",
        ),
        model_spec=model_registry("wan_debug"),
        dataloader=dataloader,
        optimizer=default_adamw(lr=1e-3),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=0,
            total_steps=1,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            global_batch_size=-1,
            seq_len=1,
            steps=1,
            max_norm=1.0,
            dtype="float32",
            mixed_precision_param="float32",
            mixed_precision_reduce="float32",
        ),
        activation_checkpoint=None,
        compile=CompileConfig(enable=False),
        metrics=MetricsProcessor.Config(
            log_freq=1,
            enable_reporterv2=False,
        ),
        checkpoint=_wan_checkpoint_config(
            base.checkpoint,
            enable=False,
            initial_load_path=None,
            initial_load_in_hf=False,
        ),
        validator=replace(
            base.validator,
            enable=False,
            steps=0,
            dataloader=replace(dataloader, split="val"),
        ),
        float8=WorldModelFloat8Config(enable=False),
        debug=DebugConfig(seed=0),
    )


def _self_forcing_optimizer(
    *,
    generator_lr: float,
    fake_score_lr: float,
    fused: bool,
) -> WanSelfForcingOptimizers.Config:
    return WanSelfForcingOptimizers.Config(
        param_groups=[
            ParamGroupConfig(
                pattern=r"^fake_score\.",
                optimizer_name="AdamW",
                optimizer_kwargs={
                    "lr": fake_score_lr,
                    "betas": (0.0, 0.999),
                    "eps": 1e-8,
                    "weight_decay": 1e-2,
                },
            ),
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="AdamW",
                optimizer_kwargs={
                    "lr": generator_lr,
                    "betas": (0.0, 0.999),
                    "eps": 1e-8,
                    "weight_decay": 1e-2,
                },
            ),
        ],
        implementation="fused_opt_states_bf16" if fused else "for-loop",
    )


def _self_forcing_lr_scheduler(
    *,
    total_steps: int | None,
) -> WanSelfForcingLRSchedulers.Config:
    return WanSelfForcingLRSchedulers.Config(
        warmup_steps=0,
        total_steps=total_steps,
        decay_ratio=0.0,
    )


def worldmodel_wan_self_forcing() -> WanSelfForcingTrainer.Config:
    """Run the minimal two-chunk Wan Self-Forcing experiment."""
    base = worldmodel_wan()
    initial_checkpoint = os.getenv(
        "WAN_SELF_FORCING_INITIAL_CHECKPOINT",
        DEFAULT_SELF_FORCING_INITIAL_CHECKPOINT,
    )
    return _wan_self_forcing_trainer_config(
        base,
        model_spec=self_forcing_model_registry("wan_ti2v_5b"),
        optimizer=_self_forcing_optimizer(
            generator_lr=2e-6,
            fake_score_lr=4e-7,
            fused=True,
        ),
        lr_scheduler=_self_forcing_lr_scheduler(
            total_steps=base.lr_scheduler.total_steps,
        ),
        training=replace(
            base.training,
            local_batch_size=1,
            global_batch_size=-1,
        ),
        checkpoint=_wan_checkpoint_config(
            base.checkpoint,
            initial_load_path=initial_checkpoint,
            initial_load_in_hf=False,
            initial_load_model_only=True,
            allow_partial_initial_load=True,
        ),
        validator=replace(
            base.validator,
            enable=False,
            steps=0,
        ),
        teacher_checkpoint_path=_wan_checkpoint_dir(),
        load_teacher_from_hf=True,
        rollout_steps=15,
        rollout_shift=1.0,
        dmd_min_timestep=0.02,
        dmd_max_timestep=0.98,
        dmd_normalizer_min=1e-5,
        fake_score_loss_weight=1.0,
        generator_update_interval=5,
    )


def worldmodel_wan_self_forcing_debug() -> WanSelfForcingTrainer.Config:
    """CPU-sized smoke recipe for the Self-Forcing training step."""
    base = worldmodel_wan_debug()
    return _wan_self_forcing_trainer_config(
        base,
        model_spec=self_forcing_model_registry("wan_debug"),
        optimizer=_self_forcing_optimizer(
            generator_lr=1e-3,
            fake_score_lr=1e-3,
            fused=False,
        ),
        lr_scheduler=_self_forcing_lr_scheduler(
            total_steps=base.lr_scheduler.total_steps,
        ),
        teacher_checkpoint_path="",
        load_teacher_from_hf=False,
        rollout_steps=2,
        rollout_shift=1.0,
        dmd_min_timestep=0.02,
        dmd_max_timestep=0.98,
        dmd_normalizer_min=1e-5,
        fake_score_loss_weight=1.0,
        generator_update_interval=5,
    )
