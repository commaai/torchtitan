# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from torchtitan.experiments.rldriving.loss import _actor_loss, _critic_loss


class _FixedCritic(nn.Module):
    def __init__(self, q1, q2):
        super().__init__()
        self.q1 = q1
        self.q2 = q2

    def forward(self, *, inputs, action):
        del inputs, action
        return self.q1, self.q2


class TestRLDrivingLoss(unittest.TestCase):
    def test_critic_loss_only_bootstraps_from_valid_next_observation(self):
        targets = {
            "action_reward": torch.tensor(
                [
                    [0.1, 0.2, 3.0],
                    [0.3, 0.4, 5.0],
                ]
            ),
            "next_observation_valid": torch.tensor([[1.0], [0.0]]),
        }

        loss, metrics = _critic_loss(
            next_actor_outputs={"action": torch.zeros(2, 4)},
            targets=targets,
            online_critic=_FixedCritic(
                torch.tensor([1.0, 2.0], requires_grad=True),
                torch.tensor([4.0, 8.0], requires_grad=True),
            ),
            target_critic=_FixedCritic(
                torch.tensor([10.0, 20.0]),
                torch.tensor([6.0, 30.0]),
            ),
            current_inputs={},
            next_inputs={},
            action_noise_A=torch.zeros(2),
            gamma=0.5,
        )

        torch.testing.assert_close(loss, torch.tensor([14.5, 9.0]))
        torch.testing.assert_close(metrics["q_rollout_abs_gap"], torch.tensor([3.0, 6.0]))
        torch.testing.assert_close(metrics["q_target_abs_gap"], torch.tensor([4.0, 10.0]))
        torch.testing.assert_close(metrics["q_target_clip_correction"], torch.tensor([1.0, 0.0]))
        self.assertEqual(
            set(metrics),
            {
                "critic_loss",
                "q1_rollout",
                "q2_rollout",
                "reward",
                "next_observation_valid",
                "q_rollout_abs_gap",
                "q_target_abs_gap",
                "q_target_clip_correction",
            },
        )

    def test_actor_smoothness_only_uses_valid_next_observation(self):
        targets = {
            "speed": torch.full((2, 1), 2.0),
            "next_observation_valid": torch.tensor([[1.0], [0.0]]),
        }
        actor_outputs = {"action": torch.tensor([[4.0, 1.0], [4.0, 1.0]])}
        next_actor_outputs = {"action": torch.tensor([[5.0, 3.0], [8.0, 5.0]])}

        loss, metrics = _actor_loss(
            actor_outputs=actor_outputs,
            next_actor_outputs=next_actor_outputs,
            online_critic=_FixedCritic(torch.zeros(2), torch.zeros(2)),
            current_inputs={},
            targets=targets,
            fps=1.0,
            smooth_lat_cost=2.0,
            smooth_long_cost=3.0,
            curv_cost=0.0,
            action_bound=100.0,
            action_bound_loss_weight=0.0,
        )

        torch.testing.assert_close(loss, torch.tensor([14.0, 0.0]))
        torch.testing.assert_close(metrics["actor_cmd_lat_jerk"], torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(metrics["actor_cmd_long_jerk"], torch.tensor([2.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
