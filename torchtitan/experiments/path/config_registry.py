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
from .model import PlanVAE
from .model_constants import (
    frame_constants_from_fps,
    FRAME_TYPE,
    ModelInputs,
    N_FRAMES,
    SUPERCOMBO_FPS,
    TEMPORAL_INPUTS,
)
from .plan_vae import PlanLoss, PlanNormalization

if TYPE_CHECKING:
    from .comma1m_trainer import Comma1MPathTrainer
    from .onnx_checkpoint import PathOnnxCheckpointManager
    from .trainer import PathTrainer


def model_registry(
    flavor: str,
    training_stage: Literal["plan_vae", "policy"] = "policy",
    *,
    plan_normalization: PlanNormalization = "pooled",
    plan_loss: PlanLoss = "decoded_laplacian",
    sample_plan_vae_posterior: bool = True,
    plan_latent_size: int = 64,
    plan_vae_encoder_layers: int = PlanVAE.N_ENCODER_LAYER,
    plan_vae_decoder_layers: int = PlanVAE.N_DECODER_LAYER,
) -> ModelSpec:
    return ModelSpec(
        name="path",
        flavor=flavor,
        model=_model_config(
            flavor,
            training_stage=training_stage,
            plan_normalization=plan_normalization,
            plan_loss=plan_loss,
            sample_plan_vae_posterior=sample_plan_vae_posterior,
            plan_latent_size=plan_latent_size,
            plan_vae_encoder_layers=plan_vae_encoder_layers,
            plan_vae_decoder_layers=plan_vae_decoder_layers,
        ),
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


def _path(
    flavor: str,
    training_stage: Literal["plan_vae", "policy"] = "policy",
    *,
    plan_normalization: PlanNormalization = "pooled",
    plan_loss: PlanLoss = "decoded_laplacian",
    deterministic_plan_autoencoder: bool = False,
    plan_latent_size: int = 64,
    plan_vae_encoder_layers: int = PlanVAE.N_ENCODER_LAYER,
    plan_vae_decoder_layers: int = PlanVAE.N_DECODER_LAYER,
    training_steps: int | None = None,
) -> PathTrainer.Config:
    from xx.comma_data.constants import (
        BASE_DIR_GT,
        DEFAULT_TEST_2K_LIST,
        DEFAULT_TEST_5K_LIST_TAGGED,
        DEFAULT_TRAIN_LIST,
    )

    from .loss import PathLoss
    from .trainer import PathTrainer
    from .validate import PathValidator

    train_plan_vae = training_stage == "plan_vae"
    if deterministic_plan_autoencoder and not train_plan_vae:
        raise ValueError("deterministic_plan_autoencoder is only valid for plan_vae training")
    steps = training_steps if training_steps is not None else 1024 * (20 if train_plan_vae else 55)
    validation_freq = 1024
    if train_plan_vae:
        reports = {
            "analyse_driving": [validation_freq, steps // 2, steps],
        }
    else:
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
    mixed_precision_param = "float32" if train_plan_vae else "bfloat16"
    num_nodes, local_world_size = _dp_degrees()
    reporterv2_host = os.getenv("REPORTERV2_HOST")
    reporterv2_training_id = os.getenv("REPORTERV2_TRAINING_ID")
    checkpoint_base_folder = f"{reporterv2_host.rstrip('/')}/checkpoint" if reporterv2_host else ""
    fps = SUPERCOMBO_FPS
    plan_only = train_plan_vae
    validator_dataset = DEFAULT_TEST_2K_LIST if train_plan_vae else DEFAULT_TEST_5K_LIST_TAGGED
    validator_limit = 2_000 if train_plan_vae else 6_000
    validator_val_skip = 1 if train_plan_vae else 6
    model_spec = model_registry(
        flavor,
        training_stage,
        plan_normalization=plan_normalization,
        plan_loss=plan_loss,
        sample_plan_vae_posterior=not deterministic_plan_autoencoder,
        plan_latent_size=plan_latent_size,
        plan_vae_encoder_layers=plan_vae_encoder_layers,
        plan_vae_decoder_layers=plan_vae_decoder_layers,
    )
    return PathTrainer.Config(
        loss=PathLoss.Config(
            vae_reconstruction_weight=1.0,
            vae_sampled_reconstruction_weight=0.0 if deterministic_plan_autoencoder else 0.25,
            vae_kl_weight=0.0 if deterministic_plan_autoencoder else 1e-4,
            vae_kl_free_bits=0.05,
            vae_consistency_weight=0.0 if deterministic_plan_autoencoder else 0.1,
            vae_backward_weight=0.0 if deterministic_plan_autoencoder else 0.05,
            vae_prior_consistency_weight=0.0 if deterministic_plan_autoencoder else 0.01,
            vae_prior_backward_weight=0.0 if deterministic_plan_autoencoder else 0.05,
            vae_position_weight=1.0 if deterministic_plan_autoencoder else 10.0,
            vae_reconstruction_loss=("physical_smooth_l1" if deterministic_plan_autoencoder else "normalized_mse"),
            vae_smooth_l1_beta=1e-3,
            plan_normalization=plan_normalization,
        ),
        model_spec=model_spec,
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
        optimizer=_optimizer_config(training_stage),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=1024,
            total_steps=steps,
            decay_ratio=None if train_plan_vae else 0.1,
            decay_type="cosine" if train_plan_vae else "linear",
            min_lr_factor=0.01 if train_plan_vae else 0.0,
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
            export_onnx=not train_plan_vae,
        ),
        activation_checkpoint=None if train_plan_vae else FullAC.Config(),
        training_stage=training_stage,
        vae_kl_warmup_steps=4096 if train_plan_vae and not deterministic_plan_autoencoder else 0,
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
                dataset=validator_dataset,
                limit=validator_limit,
                deterministic_fidxs=True,
                pipeline_dir=BASE_DIR_GT,
                skip=1,
                val_skip=validator_val_skip,
            ),
            mixed_precision_param=mixed_precision_param,
            reports=reports,
            training_stage=training_stage,
        ),
        debug=DebugConfig(seed=0),
    )


def _comma1m_path(flavor: str) -> Comma1MPathTrainer.Config:
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
    *,
    export_onnx: bool = True,
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
        export_onnx=export_onnx,
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
    training_stage: Literal["plan_vae", "policy"] = "policy",
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> OptimizersContainer.Config:
    common = {"lr": lr, "betas": betas, "eps": eps}
    if training_stage == "plan_vae":
        return OptimizersContainer.Config(
            implementation="fused_opt_states_bf16",
            param_groups=[
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={**common, "lr": 3e-4, "weight_decay": 1e-4},
                ),
            ],
        )
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
convnext_atto_comma1m = partial(_comma1m_path, "convnext_atto")
convnext_xxlarge_comma1m = partial(_comma1m_path, "convnext_xxlarge")
convnext_femto = partial(_path, "convnext_femto")
convnext_pico = partial(_path, "convnext_pico")
convnext_tiny = partial(_path, "convnext_tiny")
convnext_small = partial(_path, "convnext_small")
convnext_quarterxxl = partial(_path, "convnext_quarterxxl")
convnext_thirdxxl = partial(_path, "convnext_thirdxxl")
convnext_base = partial(_path, "convnext_base")
convnext_xlarge = partial(_path, "convnext_xlarge")
convnext_xxlarge = partial(_path, "convnext_xxlarge")

# plan VAE training: the 128-latent, half-depth, per-horizon deterministic autoencoder
convnext_xxlarge_plan_ae_128_half_layers = partial(
    _path,
    "convnext_xxlarge",
    "plan_vae",
    plan_normalization="per_horizon",
    deterministic_plan_autoencoder=True,
    plan_latent_size=128,
    plan_vae_encoder_layers=PlanVAE.N_ENCODER_LAYER // 2,
    plan_vae_decoder_layers=PlanVAE.N_DECODER_LAYER // 2,
    training_steps=10 * 1024,
)

# policy with the master laplacian plan loss, decoded through the frozen plan VAE
convnext_xlarge_plan_decoder_policy = partial(
    _path,
    "convnext_xlarge",
    "policy",
    plan_normalization="per_horizon",
    plan_latent_size=128,
    plan_vae_encoder_layers=PlanVAE.N_ENCODER_LAYER // 2,
    plan_vae_decoder_layers=PlanVAE.N_DECODER_LAYER // 2,
)

# policy with an MSE loss on the plan latent against the frozen encoder of the GT plan
convnext_xlarge_plan_latent_mse_policy = partial(
    _path,
    "convnext_xlarge",
    "policy",
    plan_normalization="per_horizon",
    plan_loss="latent_mse",
    plan_latent_size=128,
    plan_vae_encoder_layers=PlanVAE.N_ENCODER_LAYER // 2,
    plan_vae_decoder_layers=PlanVAE.N_DECODER_LAYER // 2,
)

# policy with a gaussian NLL loss on the plan latent against the frozen encoder of the GT plan
convnext_xlarge_plan_latent_nll_policy = partial(
    _path,
    "convnext_xlarge",
    "policy",
    plan_normalization="per_horizon",
    plan_loss="latent_nll",
    plan_latent_size=128,
    plan_vae_encoder_layers=PlanVAE.N_ENCODER_LAYER // 2,
    plan_vae_decoder_layers=PlanVAE.N_DECODER_LAYER // 2,
)
