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
    def forward(self, *, inputs, action):
        del inputs, action
        return (
            torch.tensor([1.0, 2.0], requires_grad=True),
            torch.tensor([4.0, 8.0], requires_grad=True),
        )


class TestRLDrivingLoss(unittest.TestCase):
    def test_critic_loss_does_not_bootstrap(self):
        targets = {
            "action_reward": torch.tensor(
                [
                    [0.1, 0.2, 3.0],
                    [0.3, 0.4, 5.0],
                ]
            )
        }

        loss, metrics = _critic_loss(
            targets=targets,
            online_critic=_FixedCritic(),
            current_inputs={},
        )

        torch.testing.assert_close(loss, torch.tensor([2.5, 9.0]))
        torch.testing.assert_close(metrics["q_rollout_abs_gap"], torch.tensor([3.0, 6.0]))
        self.assertEqual(
            set(metrics),
            {"critic_loss", "q1_rollout", "q2_rollout", "reward", "q_rollout_abs_gap"},
        )


if __name__ == "__main__":
    unittest.main()
