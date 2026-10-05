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

ACTION_OUTPUT = "action"


def _sample_fixed_noise_policy(
    action_pred_BA: torch.Tensor,
    action_noise_A: torch.Tensor,
) -> torch.Tensor:
    action_mean_BA = action_pred_BA[:, :2]
    return action_mean_BA + torch.randn_like(action_mean_BA) * action_noise_A


@torch.no_grad()
def _source_rewards(
    target_critic: nn.Module,
    current_inputs: ModelInputs,
    bootstrap_inputs: ModelInputs,
    actions_BNA: torch.Tensor,
) -> torch.Tensor:
    n_step = actions_BNA.shape[1]
    history_length = next(iter(current_inputs.values())).shape[1]
    sequence_inputs = {
        name: torch.cat((value, bootstrap_inputs[name][:, -n_step:]), dim=1) for name, value in current_inputs.items()
    }
    probabilities = []
    for offset in range(n_step):
        inputs = {name: value[:, offset : offset + history_length] for name, value in sequence_inputs.items()}
        critic1, critic2 = target_critic(inputs=inputs, action=actions_BNA[:, offset])
        probabilities.append(0.5 * (critic1["off_policy"].float().sigmoid() + critic2["off_policy"].float().sigmoid()))
    return torch.stack(probabilities, dim=1)


def _critic_loss(
    *,
    config: RLDrivingLoss.Config,
    bootstrap_actor_outputs: ActorOutputs,
    targets: Targets,
    online_critic: nn.Module,
    target_critic: nn.Module,
    current_inputs: ModelInputs,
    bootstrap_inputs: ModelInputs,
    action_noise_A: torch.Tensor,
) -> LossResult:
    action_reward_B = targets["action_reward"]
    rollout_action_BA = action_reward_B[:, 0:2]
    rewards_BN = _source_rewards(target_critic, current_inputs, bootstrap_inputs, targets["n_step_action"])

    critic1, critic2 = online_critic(
        inputs=current_inputs,
        action=rollout_action_BA,
    )
    q1_rollout_B, q2_rollout_B = critic1["q"], critic2["q"]
    off_policy_B = targets["is_off_policy"].squeeze(-1)
    # Equal reference/actor batches retain the mean RL loss over actor samples.
    on_policy_weight_B = 2.0 * (1.0 - off_policy_B)

    with torch.no_grad():
        bootstrap_action_BA = _sample_fixed_noise_policy(
            bootstrap_actor_outputs[ACTION_OUTPUT],
            action_noise_A,
        )
        target1, target2 = target_critic(
            inputs=bootstrap_inputs,
            action=bootstrap_action_BA,
        )
        q1_target_B, q2_target_B = target1["q"], target2["q"]
        bootstrap_B = torch.minimum(q1_target_B, q2_target_B)
        discounts_N = config.gamma ** torch.arange(
            rewards_BN.shape[1], device=rewards_BN.device, dtype=rewards_BN.dtype
        )
        discounted_reward_B = (rewards_BN * discounts_N).sum(dim=1)
        bootstrap_discount = config.gamma ** rewards_BN.shape[1]
        target_B = discounted_reward_B + bootstrap_discount * bootstrap_B
        q_target_abs_gap_B = torch.abs(q1_target_B - q2_target_B)
        q_rollout_abs_gap_B = torch.abs(q1_rollout_B - q2_rollout_B)
        q_target_clip_correction_B = bootstrap_discount * 0.5 * q_target_abs_gap_B

    critic_loss_B = 0.5 * (
        F.mse_loss(q1_rollout_B, target_B, reduction="none") + F.mse_loss(q2_rollout_B, target_B, reduction="none")
    )
    metrics = {
        "environment_reward": targets["n_step_reward"][:, 0].detach(),
        "source_reward": rewards_BN[:, 0],
        "critic_loss": critic_loss_B.detach(),
        "q1_rollout": q1_rollout_B.detach(),
        "q2_rollout": q2_rollout_B.detach(),
        "reward": rewards_BN[:, 0].detach(),
        "discounted_n_step_reward": discounted_reward_B.detach(),
        "q_rollout_abs_gap": q_rollout_abs_gap_B.detach(),
        "q_target_abs_gap": q_target_abs_gap_B.detach(),
        "q_target_clip_correction": q_target_clip_correction_B.detach(),
    }
    metrics = {name: value * on_policy_weight_B for name, value in metrics.items()}
    source_loss_B = 0.5 * (
        F.binary_cross_entropy_with_logits(critic1["off_policy"], off_policy_B, reduction="none")
        + F.binary_cross_entropy_with_logits(critic2["off_policy"], off_policy_B, reduction="none")
    )
    with torch.no_grad():
        probability1_B = critic1["off_policy"].sigmoid()
        probability2_B = critic2["off_policy"].sigmoid()
        probability_B = 0.5 * (probability1_B + probability2_B)
        metrics.update(
            {
                "source_loss": source_loss_B.detach(),
                "source_accuracy": 0.5
                * (
                    ((probability1_B >= 0.5) == off_policy_B.bool()).float()
                    + ((probability2_B >= 0.5) == off_policy_B.bool()).float()
                ),
                "off_policy_probability_on_policy": probability_B * on_policy_weight_B,
                "off_policy_probability_reference": probability_B * 2.0 * off_policy_B,
                "off_policy_fraction": off_policy_B,
            }
        )
    return critic_loss_B * on_policy_weight_B + source_loss_B, metrics


def _actor_loss(
    *,
    config: RLDrivingLoss.Config,
    actor_outputs: ActorOutputs,
    next_actor_outputs: ActorOutputs,
    online_critic: nn.Module,
    current_inputs: ModelInputs,
    targets: Targets,
) -> LossResult:
    action_pred_BA = actor_outputs[ACTION_OUTPUT]
    next_action_pred_BA = next_actor_outputs[ACTION_OUTPUT]
    critic1, critic2 = online_critic(
        inputs=current_inputs,
        action=action_pred_BA[:, :2],
    )
    q1_new_B, q2_new_B = critic1["q"], critic2["q"]
    actor_pi_B = -q1_new_B
    actor_q_abs_gap_B = torch.abs(q1_new_B - q2_new_B)

    curvature_B = action_pred_BA[:, 0] / targets["speed"].squeeze(-1).square()
    next_curvature_B = next_action_pred_BA[:, 0] / targets["next_speed"].squeeze(-1).square()
    curvature_rate_B = (next_curvature_B - curvature_B) * config.fps
    curvature_rate_loss_B = config.curv_rate_cost * curvature_rate_B.square()

    command_jerk_BA = (next_action_pred_BA[:, :2] - action_pred_BA[:, :2]).abs() * config.fps
    smooth_lat_B = config.smooth_lat_cost * command_jerk_BA[:, 0].square()
    smooth_long_B = config.smooth_long_cost * command_jerk_BA[:, 1].square()
    smooth_B = smooth_lat_B + smooth_long_B
    actor_loss_B = actor_pi_B + curvature_rate_loss_B + smooth_B

    action_abs_BA = torch.abs(action_pred_BA[..., :2])
    action_bound_excess_BA = torch.clamp(action_abs_BA - config.action_bound, min=0.0)
    action_bound_loss_B = action_bound_excess_BA.square().mean(dim=-1)
    loss_B = actor_loss_B + config.action_bound_loss_weight * action_bound_loss_B

    metrics = {
        "loss": loss_B.detach(),
        "actor_loss": actor_loss_B.detach(),
        "actor_pi": actor_pi_B.detach(),
        "actor_q_abs_gap": actor_q_abs_gap_B.detach(),
        "actor_curv": curvature_B.detach(),
        "actor_curv_rate": curvature_rate_B.detach(),
        "actor_curv_rate_abs": curvature_rate_B.abs().detach(),
        "actor_curv_rate_loss": curvature_rate_loss_B.detach(),
        "actor_cmd_lat_jerk": command_jerk_BA[:, 0].detach(),
        "actor_cmd_long_jerk": command_jerk_BA[:, 1].detach(),
        "actor_smooth_lat_loss": smooth_lat_B.detach(),
        "actor_smooth_long_loss": smooth_long_B.detach(),
        "actor_smooth_loss": smooth_B.detach(),
        "actor_action_bound": action_bound_loss_B.detach(),
        "actor_action_bound_max_abs": action_abs_BA.max(dim=-1).values.detach(),
        "actor_action_bound_max_excess": action_bound_excess_BA.max(dim=-1).values.detach(),
    }
    on_policy_weight_B = 2.0 * (1.0 - targets["is_off_policy"].squeeze(-1))
    return loss_B * on_policy_weight_B, {name: value * on_policy_weight_B for name, value in metrics.items()}


class RLDrivingLoss(BaseLoss):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        action_noise: tuple[float, float]
        gamma: float
        fps: float
        smooth_lat_cost: float = 0.0
        smooth_long_cost: float = 0.0
        curv_rate_cost: float = 0.0
        action_bound: float = 10.0
        action_bound_loss_weight: float = 1.0

    def __init__(
        self,
        config: Config,
        *,
        compile_config: CompileConfig | None = None,
    ) -> None:
        self.config = config
        self.action_noise_A = torch.tensor(config.action_noise)

        self.critic_fn = _critic_loss
        self.actor_fn = _actor_loss
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

        self.fn = cast(Callable[..., torch.Tensor], self.actor_fn)

    def to(self, device: torch.device) -> RLDrivingLoss:
        self.action_noise_A = self.action_noise_A.to(device)
        return self

    def critic_loss(
        self,
        *,
        bootstrap_actor_outputs: ActorOutputs,
        targets: Targets,
        online_critic: nn.Module,
        target_critic: nn.Module,
        current_inputs: ModelInputs,
        bootstrap_inputs: ModelInputs,
    ) -> LossResult:
        return self.critic_fn(
            config=self.config,
            bootstrap_actor_outputs=bootstrap_actor_outputs,
            targets=targets,
            online_critic=online_critic,
            target_critic=target_critic,
            current_inputs=current_inputs,
            bootstrap_inputs=bootstrap_inputs,
            action_noise_A=self.action_noise_A,
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
            config=self.config,
            actor_outputs=actor_outputs,
            next_actor_outputs=next_actor_outputs,
            online_critic=online_critic,
            current_inputs=current_inputs,
            targets=targets,
        )
