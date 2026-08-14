# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.components.loss import BaseLoss
from torchtitan.config import CompileConfig
from torchtitan.tools.logging import logger


# B: batch, A: action components.
ActorOutputs = dict[str, torch.Tensor]
ModelInputs = dict[str, torch.Tensor]
Targets = dict[str, torch.Tensor]
LossResult = tuple[torch.Tensor, dict[str, torch.Tensor]]
PhaseLossFunction = Callable[..., LossResult]

ACTION_OUTPUT = "action"


def _sample_fixed_noise_policy(
    action_pred_BA: torch.Tensor,
    action_noise_A: torch.Tensor,
) -> torch.Tensor:
    action_mean_BA = action_pred_BA[:, :2]
    return action_mean_BA + torch.randn_like(action_mean_BA) * action_noise_A[None, :]


def _critic_loss(
    *,
    next_actor_outputs: ActorOutputs,
    targets: Targets,
    online_critic: nn.Module,
    target_critic: nn.Module,
    current_inputs: ModelInputs,
    next_inputs: ModelInputs,
    action_noise_A: torch.Tensor,
    gamma: float,
) -> LossResult:
    action_reward_B = targets["action_reward"]
    rollout_action_BA = action_reward_B[:, 0:2]
    reward_B = action_reward_B[:, 2]
    done_B = action_reward_B[:, 3]

    q1_rollout_B, q2_rollout_B = online_critic(
        inputs=current_inputs,
        action=rollout_action_BA,
    )

    with torch.no_grad():
        next_action_BA = _sample_fixed_noise_policy(
            next_actor_outputs[ACTION_OUTPUT],
            action_noise_A,
        )
        q1_target_B, q2_target_B = target_critic(
            inputs=next_inputs,
            action=next_action_BA,
        )
        bootstrap_B = torch.minimum(q1_target_B, q2_target_B)
        target_B = reward_B + gamma * (1.0 - done_B) * bootstrap_B
        q_target_abs_gap_B = torch.abs(q1_target_B - q2_target_B)
        q_rollout_abs_gap_B = torch.abs(q1_rollout_B - q2_rollout_B)
        q_target_clip_correction_B = gamma * (1.0 - done_B) * 0.5 * q_target_abs_gap_B

    critic_loss_B = 0.5 * (
        F.mse_loss(q1_rollout_B, target_B, reduction="none") + F.mse_loss(q2_rollout_B, target_B, reduction="none")
    )
    metrics = {
        "critic_loss": critic_loss_B.detach(),
        "q1_rollout": q1_rollout_B.detach(),
        "q2_rollout": q2_rollout_B.detach(),
        "reward": reward_B.detach(),
        "done": done_B.detach(),
        "q_rollout_abs_gap": q_rollout_abs_gap_B.detach(),
        "q_target_abs_gap": q_target_abs_gap_B.detach(),
        "q_target_clip_correction": q_target_clip_correction_B.detach(),
    }
    return critic_loss_B, metrics


def _actor_loss(
    *,
    actor_outputs: ActorOutputs,
    next_actor_outputs: ActorOutputs,
    online_critic: nn.Module,
    current_inputs: ModelInputs,
    targets: Targets,
    action_noise_A: torch.Tensor,
    fps: float,
    smooth_lat_cost: float,
    smooth_long_cost: float,
    curv_cost: float,
    action_bound: float,
    action_bound_loss_weight: float,
) -> LossResult:
    action_pred_BA = actor_outputs[ACTION_OUTPUT]
    sampled_action_BA = _sample_fixed_noise_policy(action_pred_BA, action_noise_A)
    q1_new_B, q2_new_B = online_critic(
        inputs=current_inputs,
        action=sampled_action_BA,
    )
    actor_pi_B = -torch.minimum(q1_new_B, q2_new_B)
    actor_q_abs_gap_B = torch.abs(q1_new_B - q2_new_B)

    not_done_B = 1.0 - targets["action_reward"][:, 3]
    curvature_B = action_pred_BA[:, 0] / targets["speed"].squeeze(-1).square()
    curvature_loss_B = not_done_B * curv_cost * curvature_B.square()

    command_jerk_BA = (next_actor_outputs[ACTION_OUTPUT][:, :2] - action_pred_BA[:, :2]).abs() * fps
    smooth_lat_B = not_done_B * smooth_lat_cost * command_jerk_BA[:, 0].square()
    smooth_long_B = not_done_B * smooth_long_cost * command_jerk_BA[:, 1].square()
    smooth_B = smooth_lat_B + smooth_long_B
    actor_loss_B = actor_pi_B + curvature_loss_B + smooth_B

    action_abs_BA = torch.abs(action_pred_BA[..., :2])
    action_bound_excess_BA = torch.clamp(action_abs_BA - action_bound, min=0.0)
    action_bound_loss_B = action_bound_excess_BA.square().mean(dim=-1)
    loss_B = actor_loss_B + action_bound_loss_weight * action_bound_loss_B

    metrics = {
        "loss": loss_B.detach(),
        "actor_loss": actor_loss_B.detach(),
        "actor_pi": actor_pi_B.detach(),
        "actor_q_abs_gap": actor_q_abs_gap_B.detach(),
        "actor_curv": (not_done_B * curvature_B).detach(),
        "actor_curv_loss": curvature_loss_B.detach(),
        "actor_cmd_lat_jerk": (not_done_B * command_jerk_BA[:, 0]).detach(),
        "actor_cmd_long_jerk": (not_done_B * command_jerk_BA[:, 1]).detach(),
        "actor_smooth_lat_loss": smooth_lat_B.detach(),
        "actor_smooth_long_loss": smooth_long_B.detach(),
        "actor_smooth_loss": smooth_B.detach(),
        "actor_action_bound": action_bound_loss_B.detach(),
        "actor_action_bound_max_abs": action_abs_BA.max(dim=-1).values.detach(),
        "actor_action_bound_max_excess": action_bound_excess_BA.max(dim=-1).values.detach(),
    }
    return loss_B, metrics


class RLDrivingLoss(BaseLoss):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        action_noise: tuple[float, float]
        gamma: float
        fps: float
        smooth_lat_cost: float = 0.0
        smooth_long_cost: float = 0.0
        curv_cost: float = 0.0
        action_bound: float = 10.0
        action_bound_loss_weight: float = 1.0

    def __init__(
        self,
        config: Config,
        *,
        compile_config: CompileConfig | None = None,
    ) -> None:
        self.action_noise_A = torch.tensor(config.action_noise)
        self.gamma = config.gamma
        self.fps = config.fps
        self.smooth_lat_cost = config.smooth_lat_cost
        self.smooth_long_cost = config.smooth_long_cost
        self.curv_cost = config.curv_cost
        self.action_bound = config.action_bound
        self.action_bound_loss_weight = config.action_bound_loss_weight

        self.critic_fn: PhaseLossFunction = _critic_loss
        self.actor_fn: PhaseLossFunction = _actor_loss
        if compile_config is not None and compile_config.enable and "loss" in compile_config.components:
            logger.info("Compiling the rldriving loss functions with torch.compile")
            self.critic_fn = torch.compile(
                self.critic_fn,
                backend=compile_config.backend,
            )
            self.actor_fn = torch.compile(
                self.actor_fn,
                backend=compile_config.backend,
            )

        # RLDrivingTrainer calls the phase-specific methods below. Keep fn set for
        # the BaseLoss protocol without defining an ambiguous combined loss call.
        self.fn = cast(Callable[..., torch.Tensor], self.actor_fn)

    def to(self, device: torch.device) -> RLDrivingLoss:
        self.action_noise_A = self.action_noise_A.to(device)
        return self

    def critic_loss(
        self,
        *,
        next_actor_outputs: ActorOutputs,
        targets: Targets,
        online_critic: nn.Module,
        target_critic: nn.Module,
        current_inputs: ModelInputs,
        next_inputs: ModelInputs,
    ) -> LossResult:
        return self.critic_fn(
            next_actor_outputs=next_actor_outputs,
            targets=targets,
            online_critic=online_critic,
            target_critic=target_critic,
            current_inputs=current_inputs,
            next_inputs=next_inputs,
            action_noise_A=self.action_noise_A,
            gamma=self.gamma,
        )

    def actor_loss(
        self,
        *,
        actor_outputs: ActorOutputs,
        next_actor_outputs: ActorOutputs,
        online_critic: nn.Module,
        current_inputs: ModelInputs,
        targets: Targets,
    ) -> LossResult:
        return self.actor_fn(
            actor_outputs=actor_outputs,
            next_actor_outputs=next_actor_outputs,
            online_critic=online_critic,
            current_inputs=current_inputs,
            targets=targets,
            action_noise_A=self.action_noise_A,
            fps=self.fps,
            smooth_lat_cost=self.smooth_lat_cost,
            smooth_long_cost=self.smooth_long_cost,
            curv_cost=self.curv_cost,
            action_bound=self.action_bound,
            action_bound_loss_weight=self.action_bound_loss_weight,
        )
