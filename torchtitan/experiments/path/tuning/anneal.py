from __future__ import annotations

import argparse
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, fields, replace
from datetime import timedelta
from typing import Any, Literal

import tyro
import torch.distributed as dist
from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader

from torchtitan.components.checkpoint import (
    DATALOADER,
    LR_SCHEDULER,
    MODEL,
    OPTIMIZER,
    TRAIN_STATE,
    CheckpointManager,
)
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.config.manager import custom_registry
from torchtitan.distributed import utils as dist_utils
from torchtitan.observability import structured_logger as sl
from torchtitan.tools.logging import init_logger, logger

from .. import config_registry
from ..trainer import PathTrainer


def _checkpoint_paths(checkpointer: CheckpointManager, step: int) -> set[tuple[str, ...]]:
    checkpoint_id = checkpointer._create_checkpoint_id(step)
    metadata = FsspecReader(checkpoint_id).read_metadata()
    paths = {tuple(path) for path in metadata.planner_data.values()}
    model_paths = {(key,) for key in checkpointer.states[MODEL].state_dict()}
    if missing := model_paths - paths:
        raise RuntimeError(f"checkpoint {step} is missing {len(missing)} model states")
    return paths


def _has_state(paths: set[tuple[str, ...]], *prefix: str) -> bool:
    return any(path[: len(prefix)] == prefix for path in paths)


class AnnealingTrainer(PathTrainer):
    @dataclass(kw_only=True, slots=True)
    class Config(PathTrainer.Config):
        checkpoint: CheckpointManager.Config = field(default_factory=CheckpointManager.Config)
        checkpoint_run_id: str = ""
        checkpoint_steps: list[int] = field(default_factory=list)
        annealing_steps: int = 128
        annealing_decay_type: Literal["linear", "sqrt", "cosine"] = "linear"

    config: Config

    def _rebuild_optimizer(self) -> None:
        self.optimizers.zero_grad()
        for optimizer in self.optimizers:
            optimizer.state.clear()
        self.optimizers = self.config.optimizer.build(model_parts=self.model_parts)
        post_build = self.config.model_spec.post_optimizer_build_fn
        if post_build is not None:
            post_build(self.optimizers, self.model_parts, self.parallel_dims)
        self.metrics_processor.optimizers = self.optimizers
        self.checkpointer.states[OPTIMIZER] = self.optimizers

    def _rebuild_dataloader(self) -> Iterator[Any]:
        self.dataloader.close()
        dist.barrier()
        batch_mesh = self.parallel_dims.get_optional_mesh("batch")
        self.dataloader = self.config.dataloader.build(
            dp_world_size=batch_mesh.size() if batch_mesh is not None else 1,
            dp_rank=batch_mesh.get_local_rank() if batch_mesh is not None else 0,
            tokenizer=self.tokenizer,
            seq_len=self.config.training.seq_len,
            local_batch_size=self.config.training.local_batch_size,
        )
        self.checkpointer.states[DATALOADER] = self.dataloader
        return self.batch_generator(self.dataloader)

    def _rebuild_lr_scheduler(self) -> None:
        self.lr_schedulers = self.config.lr_scheduler.build(
            optimizers=self.optimizers,
            training_steps=self.config.annealing_steps,
        )
        self.checkpointer.states[LR_SCHEDULER] = self.lr_schedulers

    def _anneal_checkpoint(
        self,
        checkpoint_step: int,
        *,
        data_iterator: Iterator[Any],
        profiler: Any,
        set_train_timeout: bool,
    ) -> Iterator[Any]:
        paths = _checkpoint_paths(self.checkpointer, checkpoint_step)
        has_optimizer = _has_state(paths, OPTIMIZER)
        cold_start = not has_optimizer and not _has_state(paths, TRAIN_STATE)

        self.step = checkpoint_step
        self.ntokens_seen = 0
        self.unique_segment_counter.reset()
        self._rebuild_optimizer()
        if cold_start:
            data_iterator = self._rebuild_dataloader()
        if not self.checkpointer.load(step=checkpoint_step):
            raise RuntimeError(f"failed to load checkpoint step {checkpoint_step}")
        if self.step != checkpoint_step:
            raise RuntimeError(f"checkpoint {checkpoint_step} restored train step {self.step}")
        if not has_optimizer:
            # DCP materializes target optimizer state even when the source has none.
            self._rebuild_optimizer()
        self._rebuild_lr_scheduler()

        sl.set_step(self.step, relative_step=0)
        self.metrics_processor.logger.log(
            {"dataset/unique_segments_seen": self.unique_segment_counter.last_global_count},
            step=self.step,
        )
        self.validator.validate(self.model_parts, self.step)

        for relative_step in range(1, self.config.annealing_steps + 1):
            self.step += 1
            sl.set_step(self.step, relative_step=relative_step)
            with sl.log_trace_span("step"):
                self.gc_handler.run(self.step)
                self.train_step(data_iterator)
                profiler.step()

                if set_train_timeout and relative_step == 1:
                    dist_utils.set_pg_timeouts(
                        timeout=timedelta(seconds=self.config.comm.train_timeout_seconds),
                        parallel_dims=self.parallel_dims,
                    )

        self.validator.validate(self.model_parts, self.step)
        return data_iterator

    def train(self) -> None:
        sl.log_trace_instant("training_start")
        data_iterator = self.batch_generator(self.dataloader)
        for index, checkpoint_step in enumerate(self.config.checkpoint_steps):
            with self.config.profiler.build(
                global_step=checkpoint_step,
                base_folder=self.config.dump_folder,
            ) as profiler:
                logger.info(f"Annealing checkpoint {checkpoint_step}")
                data_iterator = self._anneal_checkpoint(
                    checkpoint_step,
                    data_iterator=data_iterator,
                    profiler=profiler,
                    set_train_timeout=index == 0,
                )
        logger.info("Annealing completed")


def _promote_config(config: PathTrainer.Config) -> AnnealingTrainer.Config:
    values = {config_field.name: getattr(config, config_field.name) for config_field in fields(config)}
    return AnnealingTrainer.Config(**values)


def parse_args(args: Sequence[str] | None = None) -> AnnealingTrainer.Config:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--config", required=True)
    parsed, remaining = parser.parse_known_args(args)
    config_name = parsed.config
    config_fn = getattr(config_registry, config_name, None)
    if config_fn is None or not callable(config_fn):
        raise ValueError(f"unknown PATH config {config_name!r}")
    base_config = config_fn()
    if not isinstance(base_config, PathTrainer.Config):
        raise ValueError(f"PATH config {config_name!r} did not return PathTrainer.Config")
    default_config = _promote_config(base_config)
    return tyro.cli(
        AnnealingTrainer.Config,
        args=remaining,
        default=default_config,
        registry=custom_registry,
    )


def _prepare_config(
    config: AnnealingTrainer.Config,
    *,
    reporterv2_host: str,
    reporterv2_training_id: str,
) -> AnnealingTrainer.Config:
    if not config.checkpoint_run_id:
        raise ValueError("--checkpoint-run-id is required")
    if not config.checkpoint_steps:
        raise ValueError("--checkpoint-steps must not be empty")
    if any(step < 1 for step in config.checkpoint_steps):
        raise ValueError("--checkpoint-steps must contain full checkpoint steps")
    if config.checkpoint_run_id == reporterv2_training_id:
        raise ValueError("source and annealing ReporterV2 run IDs must differ")
    if config.annealing_steps < 1:
        raise ValueError("--annealing-steps must be positive")
    if not config.validator.enable:
        raise ValueError("annealing requires validation to be enabled")

    for previous_step, next_step in zip(config.checkpoint_steps, config.checkpoint_steps[1:]):
        if next_step <= previous_step + config.annealing_steps:
            raise ValueError("checkpoint steps must be increasing and their annealing ranges must not overlap")

    source_folder = f"{reporterv2_host.rstrip('/')}/checkpoint/{config.checkpoint_run_id}"
    checkpoint = CheckpointManager.Config(
        enable=True,
        folder=source_folder,
        checkpoint_id_format=config.checkpoint.checkpoint_id_format,
        load_only=True,
        exclude_from_loading=[LR_SCHEDULER, DATALOADER],
        initial_load_model_only=False,
        allow_partial_initial_load=True,
    )
    return replace(
        config,
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=0,
            total_steps=config.annealing_steps,
            decay_ratio=None,
            decay_type=config.annealing_decay_type,
            min_lr_factor=0.0,
        ),
        training=replace(
            config.training,
            steps=config.checkpoint_steps[-1] + config.annealing_steps,
        ),
        checkpoint=checkpoint,
        metrics=replace(
            config.metrics,
            log_freq=1,
            save_freq=1,
            enable_tensorboard=False,
            enable_wandb=False,
            enable_reporterv2=True,
        ),
        validator=replace(
            config.validator,
            reports={},
            save_predictions=False,
        ),
    )


def main() -> None:
    init_logger()
    reporterv2_host = os.getenv("REPORTERV2_HOST")
    reporterv2_training_id = os.getenv("REPORTERV2_TRAINING_ID")
    if not reporterv2_host or not reporterv2_training_id:
        raise ValueError("REPORTERV2_HOST and REPORTERV2_TRAINING_ID must both be set")

    config = _prepare_config(
        parse_args(),
        reporterv2_host=reporterv2_host,
        reporterv2_training_id=reporterv2_training_id,
    )
    sl.init_structured_logger(
        source="training",
        output_dir=config.dump_folder,
        enable=config.debug.enable_structured_logging,
    )

    trainer: AnnealingTrainer | None = None
    try:
        trainer = config.build()
        trainer.train()
    except BaseException:
        if trainer is not None:
            trainer.close()
        raise
    else:
        trainer.close()
        if dist.is_initialized():
            dist.destroy_process_group()
        logger.info("Process group destroyed")


if __name__ == "__main__":
    main()
