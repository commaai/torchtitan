# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import Any

import torch.distributed as dist
import tyro

from torchtitan.components.checkpoint import (
    DATALOADER,
    LR_SCHEDULER,
    OPTIMIZER,
    TRAIN_STATE,
    CheckpointManager,
)
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.metrics import BaseLogger
from torchtitan.config.manager import custom_registry
from torchtitan.observability import structured_logger as sl
from torchtitan.tools.logging import init_logger, logger
from xx.datasets.constants import BASE_DIR_GT_10M

from .. import config_registry
from ..trainer import PathTrainer


class NoOpDataLoader(BaseDataLoader):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        pass

    def __init__(self, config: Config, **kwargs: Any) -> None:
        pass

    def __iter__(self) -> Iterator[Any]:
        return iter(())

    def close(self) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass


class EvalMetricLogger(BaseLogger):
    def __init__(self, run_id: str, eval_id: str) -> None:
        self.run_id = run_id
        self.eval_id = eval_id

    def log(self, metrics: dict[str, Any], step: int) -> None:
        from reporterv2 import write_metrics

        row = {
            "step": step,
            "epoch": step,
            "ts": time.time(),
            **{f"{self.eval_id}/{name}": value for name, value in metrics.items()},
        }
        write_metrics(self.run_id, {}, [row])


class EvalTrainer(PathTrainer):
    @dataclass(kw_only=True, slots=True)
    class Config(PathTrainer.Config):
        checkpoint: CheckpointManager.Config = field(default_factory=CheckpointManager.Config)
        run_id: str = ""
        checkpoint_steps: list[int] | None = None
        eval_id: str = ""

    config: Config

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if dist.get_rank() == 0:
            self.metrics_processor.logger = EvalMetricLogger(
                run_id=config.run_id,
                eval_id=config.eval_id,
            )

    def evaluate(self) -> None:
        checkpoint_steps = self.config.checkpoint_steps
        if checkpoint_steps is None:
            logger.info(f"Autodiscovering checkpoints in {self.checkpointer.folder}")
            checkpoint_steps = sorted(
                self.checkpointer._list_checkpoint_steps(
                    self.checkpointer.folder,
                    require_metadata=True,
                )
            )
            logger.info(
                f"Found {len(checkpoint_steps)} checkpoints: {checkpoint_steps}"
            )

        for checkpoint_step in checkpoint_steps:
            logger.info(f"Evaluating checkpoint {checkpoint_step}")
            sl.set_step(checkpoint_step, relative_step=0)
            self.validator.unique_segment_counter.reset()
            if not self.checkpointer.load(step=checkpoint_step):
                raise RuntimeError(f"failed to load checkpoint step {checkpoint_step}")
            self.validator.validate(self.model_parts, checkpoint_step)
        logger.info("Evaluation completed")


def _promote_config(config: PathTrainer.Config) -> EvalTrainer.Config:
    values = {config_field.name: getattr(config, config_field.name) for config_field in fields(config)}
    return EvalTrainer.Config(**values)


def parse_args(args: Sequence[str] | None = None) -> EvalTrainer.Config:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--config", required=True)
    parsed, remaining = parser.parse_known_args(args)
    base_config = getattr(config_registry, parsed.config)()
    return tyro.cli(
        EvalTrainer.Config,
        args=remaining,
        default=_promote_config(base_config),
        registry=custom_registry,
    )


def _prepare_config(
    config: EvalTrainer.Config,
    *,
    reporterv2_host: str,
) -> EvalTrainer.Config:
    if not config.run_id:
        raise ValueError("--run-id is required")
    if not config.eval_id:
        raise ValueError("--eval-id is required")

    checkpoint = CheckpointManager.Config(
        enable=True,
        folder=f"{reporterv2_host.rstrip('/')}/checkpoint/{config.run_id}",
        checkpoint_id_format=config.checkpoint.checkpoint_id_format,
        load_only=True,
        exclude_from_loading=[OPTIMIZER, LR_SCHEDULER, DATALOADER, TRAIN_STATE],
        initial_load_model_only=False,
    )
    validator = replace(
        config.validator,
        dataloader=replace(
            config.validator.dataloader,
            dataset="datasets/lists/prune10m_val.txt",
            pipeline_dir=BASE_DIR_GT_10M,
            plan_only=True,
            val_skip=6, # 4 samples per row at 5 fps (prune10m_val rows are 10s slices)
        ),
        reports={},
        save_predictions=True,
        steps=32,
        prediction_file_prefix=f"{config.eval_id}.val_preds",
    )
    return replace(
        config,
        checkpoint=checkpoint,
        dataloader=NoOpDataLoader.Config(),
        metrics=replace(
            config.metrics,
            enable_tensorboard=False,
            enable_wandb=False,
            enable_reporterv2=False,
        ),
        validator=validator,
    )


def main() -> None:
    init_logger()
    reporterv2_host = os.getenv("REPORTERV2_HOST")
    if not reporterv2_host:
        raise ValueError("REPORTERV2_HOST must be set")

    config = _prepare_config(parse_args(), reporterv2_host=reporterv2_host)
    os.environ["REPORTERV2_TRAINING_ID"] = config.run_id
    sl.init_structured_logger(
        source="evaluation",
        output_dir=config.dump_folder,
        enable=config.debug.enable_structured_logging,
    )

    trainer = config.build()
    try:
        trainer.evaluate()
    finally:
        trainer.close()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
