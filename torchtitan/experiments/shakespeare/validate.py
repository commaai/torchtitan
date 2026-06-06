# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import torch
import torch.nn as nn

from torchtitan.components.validate import Validator
from torchtitan.tools.logging import logger


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


class ShakespeareValidator(Validator):
    @dataclass(kw_only=True, slots=True)
    class Config(Validator.Config):
        pass

    def __init__(self, config: Config, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self._report_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="shakespeare-report",
        )
        self._report_futures: list[Future[None]] = []

    def validate(self, model_parts: list[nn.Module], step: int) -> None:
        logger.info("Running Shakespeare validation at step %s", step)
        super().validate(model_parts, step)
        self._schedule_report(step)

    def _schedule_report(self, step: int) -> None:
        if _rank() != 0:
            logger.debug("Skipping Shakespeare validation report on nonzero rank")
            return

        self._report_futures = [
            future for future in self._report_futures if not future.done()
        ]
        future = self._report_executor.submit(self._write_report, step)
        future.add_done_callback(lambda future: self._log_report_result(future, step))
        self._report_futures.append(future)
        logger.info("Scheduled async Shakespeare validation report for step %s", step)

    def _write_report(self, step: int) -> None:
        html = self._build_report_html(step)
        self.metrics_processor.write_report(
            html,
            step=step,
            name="hello_world",
            output_type="html",
        )

    def _build_report_html(self, step: int) -> str:
        return "<html><body>hello world</body></html>"

    def _log_report_result(self, future: Future[None], step: int) -> None:
        try:
            future.result()
        except Exception:
            logger.exception(
                "Failed to write Shakespeare validation report for step %s",
                step,
            )
        else:
            logger.info("Finished Shakespeare validation report for step %s", step)

    def close(self) -> None:
        self._report_executor.shutdown(wait=True)
        self._report_futures.clear()
        super().close()
