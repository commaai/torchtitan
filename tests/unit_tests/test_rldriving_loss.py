# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from torchtitan.experiments.rldriving.loss import _critic_loss


class _FixedCritic(nn.Module):
    def __init__(self, q1, q2):
        super().__init__()
        self.q1 = q1
        self.q2 = q2

    def forward(self, *, inputs, action):
        del inputs, action
        return self.q1, self.q2


class TestRLDrivingLoss(unittest.TestCase):
    def test_critic_loss_always_bootstraps_without_done(self):
        targets = {
            "action_reward": torch.tensor(
                [
                    [0.1, 0.2, 3.0],
                    [0.3, 0.4, 5.0],
                ]
            )
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

        torch.testing.assert_close(loss, torch.tensor([14.5, 109.0]))
        torch.testing.assert_close(metrics["q_rollout_abs_gap"], torch.tensor([3.0, 6.0]))
        torch.testing.assert_close(metrics["q_target_abs_gap"], torch.tensor([4.0, 10.0]))
        torch.testing.assert_close(metrics["q_target_clip_correction"], torch.tensor([1.0, 2.5]))
        self.assertEqual(
            set(metrics),
            {
                "critic_loss",
                "q1_rollout",
                "q2_rollout",
                "reward",
                "q_rollout_abs_gap",
                "q_target_abs_gap",
                "q_target_clip_correction",
            },
        )


if __name__ == "__main__":
    unittest.main()
