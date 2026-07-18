# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import unittest

import torch

from torchtitan.experiments.path.attention_diagnostics import (
    _max_attention_logit,
    PlanViTAttentionDiagnostics,
    PlanViTAttentionDiagnosticsConfig,
)
from torchtitan.experiments.path.config_registry import _vit_model_config


class TestAttentionDiagnostics(unittest.TestCase):
    def test_max_attention_logit_matches_pre_softmax_scores(self):
        q = torch.tensor(
            [[[[1.0, 2.0]], [[-1.0, 3.0]], [[2.0, -2.0]]]],
            dtype=torch.float32,
        )
        k = torch.tensor(
            [[[[2.0, 0.0]], [[1.0, 4.0]], [[-3.0, 1.0]]]],
            dtype=torch.float32,
        )
        expected = (
            torch.matmul(q.transpose(1, 2), k.transpose(1, 2).transpose(-2, -1))
            * math.sqrt(0.5)
        ).amax()
        actual = _max_attention_logit(q, k, scale=None, is_causal=False)
        torch.testing.assert_close(actual, expected)

    def test_causal_max_ignores_future_scores(self):
        q = torch.tensor([[[[1.0]], [[-1.0]]]])
        k = torch.tensor([[[[1.0]], [[100.0]]]])
        actual = _max_attention_logit(q, k, scale=1.0, is_causal=True)
        self.assertEqual(actual.item(), 1.0)

    def test_invalid_head_shape_fails(self):
        q = torch.zeros(1, 2, 1, 4)
        k = torch.zeros(1, 2, 2, 4)
        with self.assertRaisesRegex(ValueError, "matching q and k heads"):
            _max_attention_logit(q, k, scale=None, is_causal=False)

    def test_tracker_records_periodic_and_last_logged_steps(self):
        model = _vit_model_config("w64", mup=True).build()
        model.init_weights(buffer_device=torch.device("cpu"))
        tracker = PlanViTAttentionDiagnostics(
            PlanViTAttentionDiagnosticsConfig(
                enable=True, start_step=20, interval=30
            ),
            [model],
            total_steps=49,
            log_freq=10,
        )
        attention = model.blocks[0].attention
        q = torch.ones(1, 2, 1, 64)
        k = torch.ones_like(q)
        v = torch.zeros_like(q)

        tracker.begin_step(10, will_log=True)
        attention.inner_attention(q, k, v, is_causal=False)
        model.plan_head(torch.ones(1, 64))
        self.assertEqual(tracker.metrics(None), {})

        tracker.begin_step(20, will_log=True)
        attention.inner_attention(q, k, v, is_causal=False)
        model.plan_head(torch.ones(1, 64))
        metrics = tracker.metrics(None)
        self.assertEqual(metrics["model/attention_logit_max"], 8.0)
        self.assertEqual(metrics["model/attention_logit_max_layer0"], 8.0)
        self.assertTrue(math.isfinite(metrics["model/plan_output_abs_mean"]))

        tracker.begin_step(40, will_log=True)
        attention.inner_attention(q, k, v, is_causal=False)
        model.plan_head(torch.ones(1, 64))
        self.assertEqual(tracker.metrics(None)["model/attention_logit_max"], 8.0)
        tracker.close()

    def test_tracker_schedule_must_align_with_metric_logging(self):
        model = _vit_model_config("w64", mup=True).build()
        with self.assertRaisesRegex(ValueError, "multiples of metrics.log_freq"):
            PlanViTAttentionDiagnostics(
                PlanViTAttentionDiagnosticsConfig(
                    enable=True, start_step=21, interval=30
                ),
                [model],
                total_steps=100,
                log_freq=10,
            )


if __name__ == "__main__":
    unittest.main()
