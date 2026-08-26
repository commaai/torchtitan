# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any
from xx.release_tests.lib.base_report import ReportFormat
from xx.training.lib.checkpoint import wait_for_checkpoint
from xx.training.rldriving.test import MODEL_REPORTS, SCALAR_REPORTS

import torch.distributed as dist
import torch.nn as nn

from torchtitan.components.metrics import MetricsProcessor
from xx.training.lib.torchtitan.report_runner import ReportRunner, ReportSpec
from torchtitan.components.validate import BaseValidator


class RLDrivingValidator(BaseValidator):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseValidator.Config):
        enable: bool
        fps: int
        reports: dict[str, list[int]] = field(default_factory=dict)
        miniray: dict[str, Any] = field(default_factory=dict)

    config: Config  # pyrefly: ignore [bad-override]

    def __init__(
        self,
        config: Config,
        *,
        metrics_processor: MetricsProcessor,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__(config=config)
        self.metrics_processor = metrics_processor
        self.training_id = os.getenv("REPORTERV2_TRAINING_ID") or "local"
        self.miniray = {
            **config.miniray,
            "job_group": f"rldriving_validation_{self.training_id}",
        }
        self.report_runner = ReportRunner(metrics_processor=metrics_processor, enabled=dist.get_rank() == 0)

    def validate(self, model_parts: list[nn.Module], step: int) -> None:
        del model_parts
        self._submit_reports(step)

    def close(self) -> None:
        self.report_runner.close()

    def _submit_reports(self, step: int) -> None:
        current_checkpoint = f"{self.training_id}/{step}"

        def _run_report(TestCls: type, test_config: Any, include_scalars: bool) -> tuple[Any, ...]:
            wait_for_checkpoint(current_checkpoint)
            test = TestCls(test_config)
            html = test.run_report()
            return (html, test.scalars) if include_scalars else (html,)

        report_specs = {}
        for report_name, (TestCls, ReportConfigCls) in MODEL_REPORTS.items():
            include_scalars = report_name in SCALAR_REPORTS
            report_config = ReportConfigCls(
                rollout={
                    "agent": {
                        "supercombo": current_checkpoint,
                        "model_trained_fps": self.config.fps,
                    }
                },
                report_name=f"rldriving_{report_name}",
                save_tmp=False,
                format=ReportFormat.HTML,
                miniray=self.miniray,
            )
            report_specs[report_name] = ReportSpec(
                output_names=(report_name, f"{report_name}_scalar") if include_scalars else (report_name,),
                output_types=("html", "scalar") if include_scalars else ("html",),
                steps=self.config.reports.get(report_name, []),
                func=_run_report,
                arguments=[TestCls, report_config, include_scalars],
            )

        self.report_runner.submit_due(step=step, report_specs=report_specs)
