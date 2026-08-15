# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from xx.ml_tools.constants.model import ModelInputs

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict, StateDictOptions
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.utils.flop_counter import FlopCounterMode

from torchtitan.config import CompileConfig, ParallelismConfig, TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.fsdp import enable_fsdp_symm_mem, get_fsdp_reshard_after_forward_policy
from torchtitan.experiments.path.model import Hydra, LinearEncoder, PathMLP, TemporalPolicy, TemporalSummarizer
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger


# B: batch, T: temporal steps, D: model width, A: action components.
ACTION_HEAD_NAME = "action"
Q_HEAD_NAME = "q"

TemporalInputs = dict[str, torch.Tensor]
ActorOutputs = dict[str, torch.Tensor]


def _policy_forward(policy: TemporalPolicy, inputs: TemporalInputs) -> ActorOutputs:
    outputs = policy(
        inputs[ModelInputs.FEATURES],
        inputs[ModelInputs.DESIRE],
        inputs[ModelInputs.TRAFFIC],
        inputs[ModelInputs.ACTION_T],
    )
    return {ACTION_HEAD_NAME: outputs[ACTION_HEAD_NAME]}


class Critic(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        temporal_summarizer: TemporalSummarizer.Config
        history_idxs: tuple[int, ...]
        action_encoder: LinearEncoder.Config
        post_action_mlp1: PathMLP.Config
        post_action_mlp2: PathMLP.Config
        q_hydra: Hydra.Config

    def __init__(self, config: Config):
        super().__init__()
        self.temporal_summarizer = config.temporal_summarizer.build()
        self.history_idxs = config.history_idxs
        self.action_encoder = config.action_encoder.build()
        self.post_action_mlp1 = config.post_action_mlp1.build()
        self.post_action_mlp2 = config.post_action_mlp2.build()
        self.q_hydra = config.q_hydra.build()

    def forward(self, inputs: TemporalInputs, action: torch.Tensor) -> torch.Tensor:
        features_BTD = inputs[ModelInputs.FEATURES]
        dtype = features_BTD.dtype
        critic_features_BD = self.temporal_summarizer(
            features_BTD[:, self.history_idxs],
            inputs[ModelInputs.DESIRE].to(dtype),
            inputs[ModelInputs.TRAFFIC][:, -1].to(dtype),
            inputs[ModelInputs.ACTION_T][:, -1].to(dtype),
        )
        critic_features_BD = critic_features_BD + self.post_action_mlp1(critic_features_BD)
        critic_features_BD = critic_features_BD + self.action_encoder(action.to(dtype))
        critic_features_BD = critic_features_BD + self.post_action_mlp2(critic_features_BD)
        q_B1 = self.q_hydra(critic_features_BD)[Q_HEAD_NAME]
        return q_B1.squeeze(-1).clone()


class TwinCritic(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        critic: Critic.Config

    def __init__(self, config: Config):
        super().__init__()
        self.critic1 = config.critic.build()
        self.critic2 = config.critic.build()

    def forward(
        self,
        inputs: TemporalInputs,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.critic1(inputs, action), self.critic2(inputs, action)


class RLDrivingModel(BaseModel):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        actor: TemporalPolicy.Config
        critic: TwinCritic.Config

        def update_from_config(self, *, config, **kwargs) -> None:
            parallelism = config.parallelism
            if parallelism.spmd_backend == "full_dtensor":
                raise ValueError("rldriving does not support full DTensor")
            unsupported = {
                "tensor parallel": parallelism.tensor_parallel_degree,
                "context parallel": parallelism.context_parallel_degree,
                "pipeline parallel": parallelism.pipeline_parallel_degree,
                "expert parallel": parallelism.expert_parallel_degree,
            }
            for name, degree in unsupported.items():
                if degree > 1:
                    raise ValueError(f"rldriving does not support {name}")
            if config.activation_checkpoint is not None:
                raise ValueError("rldriving does not support activation checkpointing")

        def get_nparams_and_flops(self, model: Module, seq_len: int) -> tuple[int, int]:
            rldriving_model = cast(RLDrivingModel, model)
            nparams = sum(parameter.numel() for parameter in rldriving_model.parameters())
            device = next(rldriving_model.parameters()).device
            inputs = {
                name: torch.zeros(shape, dtype=torch.float32, device=device)
                for name, shape in RLDrivingModel.input_shapes(self).items()
            }
            action_dim = self.critic.critic.action_encoder.in_layer.in_features
            action_BA = torch.zeros((1, action_dim), dtype=torch.float32, device=device)
            with torch.no_grad(), FlopCounterMode(display=False) as counter:
                rldriving_model(inputs)
                rldriving_model.critic(inputs, action_BA)
            # MFU convention estimates backward as twice the counted forward work.
            return nparams, 3 * counter.get_total_flops()

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.actor = config.actor.build()
        self.critic = config.critic.build()
        self.target_actor = config.actor.build()
        self.target_critic = config.critic.build()

        self.target_actor.requires_grad_(False).eval()
        self.target_critic.requires_grad_(False).eval()

    @staticmethod
    def input_shapes(
        config: RLDrivingModel.Config,
        batch_size: int = 1,
    ) -> dict[str, tuple[int, ...]]:
        summarizer = config.actor.temporal_summarizer
        temporal_len = max(summarizer.desire_window_starts) + summarizer.desire_window_len
        desire_dim = summarizer.desire_encoder.in_layer.in_features // summarizer.desire_window_len
        return {
            ModelInputs.FEATURES: (
                batch_size,
                temporal_len,
                summarizer.pos_embedding.embedding_dim,
            ),
            ModelInputs.DESIRE: (batch_size, temporal_len, desire_dim),
            ModelInputs.TRAFFIC: (
                batch_size,
                temporal_len,
                summarizer.traffic_encoder.in_layer.in_features,
            ),
            ModelInputs.ACTION_T: (
                batch_size,
                temporal_len,
                summarizer.action_t_encoder.in_layer.in_features,
            ),
        }

    def verify_module_protocol(self) -> None:
        # Path modules contain parameterless torch.nn activations and dropout.
        pass

    def init_states(self, *, buffer_device: torch.device | None = None) -> None:
        super().init_states(buffer_device=buffer_device)
        self.sync_targets()

    @torch.no_grad()
    def sync_targets(self) -> None:
        _copy_model_state(self.actor, self.target_actor)
        _copy_model_state(self.critic, self.target_critic)

    @torch.no_grad()
    def warm_start_critics_from_actor(self) -> None:
        for destination in (
            self.critic.critic1.temporal_summarizer,
            self.critic.critic2.temporal_summarizer,
        ):
            _copy_model_state(self.actor.temporal_summarizer, destination)
        self.sync_targets()

    def train(self, mode: bool = True) -> RLDrivingModel:
        super().train(mode)
        self.target_actor.eval()
        self.target_critic.eval()
        return self

    def forward(self, inputs: TemporalInputs) -> ActorOutputs:
        return _policy_forward(self.actor, inputs)

    def target_forward(self, inputs: TemporalInputs) -> ActorOutputs:
        return _policy_forward(self.target_actor, inputs)


def _copy_model_state(source: nn.Module, destination: nn.Module) -> None:
    options = StateDictOptions(full_state_dict=True)
    source_state = get_model_state_dict(source, options=options)
    set_model_state_dict(destination, source_state, options=options)


def parallelize_rldriving(
    model: RLDrivingModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
) -> RLDrivingModel:
    if compile_config.enable and "model" in compile_config.components:
        torch._dynamo.config.capture_scalar_outputs = True
        model.actor.compile(backend=compile_config.backend)
        model.critic.compile(backend=compile_config.backend)
        model.target_actor.compile(backend=compile_config.backend)
        model.target_critic.compile(backend=compile_config.backend)
        logger.info("Compiling rldriving model components with torch.compile")

    names = ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    mp_policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        cast_forward_inputs=True,
    )
    fsdp_config: dict[str, Any] = {"mesh": parallel_dims.get_mesh(names), "mp_policy": mp_policy}
    if training.enable_cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        parallelism.fsdp_reshard_after_forward,
        parallel_dims.pp_enabled,
    )

    def shard(module: nn.Module, reshard: bool = reshard_after_forward) -> None:
        fully_shard(module, **fsdp_config, reshard_after_forward=reshard)

    for temporal_summarizer in (
        model.actor.temporal_summarizer,
        model.critic.critic1.temporal_summarizer,
        model.critic.critic2.temporal_summarizer,
        model.target_actor.temporal_summarizer,
        model.target_critic.critic1.temporal_summarizer,
        model.target_critic.critic2.temporal_summarizer,
    ):
        temporal_summarizer.transformer.apply_fsdp(shard, reshard_after_forward)

    shard(model.actor)
    shard(model.target_actor)
    for critic in (
        model.critic.critic1,
        model.critic.critic2,
        model.target_critic.critic1,
        model.target_critic.critic2,
    ):
        shard(critic)
    shard(model.critic)
    shard(model.target_critic)
    fully_shard(model, **fsdp_config)

    if parallelism.enable_fsdp_symm_mem:
        enable_fsdp_symm_mem(model)

    logger.info(
        "Applied HSDP to the rldriving model"
        if parallel_dims.dp_replicate_enabled
        else "Applied FSDP to the rldriving model"
    )
    if training.enable_cpu_offload:
        logger.info("Applied CPU Offloading to the rldriving model")
    return model
