# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import timedelta
from functools import cache
from typing import Any, cast, Literal

from xx.common.helpers import parse_info
from xx.ml_tools.constants.model import TEMPORAL_INPUTS
from xx.training.lib.checkpoint import Checkpoint
from xx.training.rldriving.dataloader import RolloutContext

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader
from torch.distributed.elastic.multiprocessing.errors import record
from torch.optim.lr_scheduler import LambdaLR

from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.unique_counter import StringUniqueCounter
from torchtitan.distributed import utils as dist_utils
from torchtitan.observability import structured_logger as sl
from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer

from .dataset import RLDrivingDataLoader
from .loss import RLDrivingLoss
from .model import RLDrivingModel
from .onnx_checkpoint import RLDrivingOnnxCheckpointManager
from .validate import RLDrivingValidator


Batch = tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]
PreparedBatch = tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]


_get_path_checkpoint = cache(Checkpoint)


@dataclass(kw_only=True, slots=True)
class RLDrivingLRSchedulersConfig(LRSchedulersContainer.Config):
    steps_per_epoch: int
    num_epochs: int
    actor_delay_epochs: float = 15.0
    actor_warmup_fraction: float = 0.1
    cooldown_fraction: float = 0.4
    min_lr_factor: float = 0.05
    critic_switch_epoch: float = 15.0
    critic_second_lr: float = 4e-5

    # pyrefly: ignore [bad-override]
    def build(self, *, optimizers, training_steps):
        return RLDrivingLRSchedulers(self, optimizers=optimizers)


class RLDrivingLRSchedulers(LRSchedulersContainer):
    Config = RLDrivingLRSchedulersConfig

    def __init__(
        self,
        config: Config,
        *,
        optimizers: OptimizersContainer,
    ) -> None:
        self.config = config
        self.optimizer_container = optimizers
        self.optimizer = next(iter(optimizers))
        self.schedulers = [
            LambdaLR(
                self.optimizer,
                [self._lr_lambda(group) for group in self.optimizer.param_groups],
            )
        ]

    def _lr_lambda(self, group):
        phase = group["param_names"][0].split(".", 1)[0]
        base_lr = float(group["lr"])

        def lr_lambda(current_step: int) -> float:
            config = self.config
            epoch = current_step / config.steps_per_epoch
            max_epoch = config.num_epochs - 1.0
            cooldown_start = max_epoch * (1.0 - config.cooldown_fraction)
            if phase == "actor":
                if epoch < config.actor_delay_epochs:
                    return 0.0
                warmup_end = config.actor_delay_epochs + max_epoch * config.actor_warmup_fraction
                if epoch < warmup_end:
                    return (epoch - config.actor_delay_epochs) / (max_epoch * config.actor_warmup_fraction)
                if epoch < cooldown_start:
                    return 1.0
                progress = min(1.0, (epoch - cooldown_start) / (max_epoch - cooldown_start))
                return 1.0 + progress * (config.min_lr_factor - 1.0)

            if epoch < config.critic_switch_epoch:
                return 1.0
            lr = config.critic_second_lr
            if epoch >= cooldown_start:
                progress = min(1.0, (epoch - cooldown_start) / (max_epoch - cooldown_start))
                lr *= 1.0 + progress * (config.min_lr_factor - 1.0)
            return lr / base_lr

        return lr_lambda

    def step_phase(self, phase: Literal["actor", "critic"]) -> None:
        all_param_groups = self.optimizer.param_groups
        self.optimizer.param_groups = [
            group for group in all_param_groups if group["param_names"][0].startswith(f"{phase}.")
        ]
        self.optimizer_container.step()
        self.optimizer.param_groups = all_param_groups


class RLDrivingTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        loss: RLDrivingLoss.Config  # pyrefly: ignore [bad-override]
        dataloader: RLDrivingDataLoader.Config  # pyrefly: ignore [bad-override]
        checkpoint: RLDrivingOnnxCheckpointManager.Config  # pyrefly: ignore [bad-override]
        lr_scheduler: RLDrivingLRSchedulers.Config  # pyrefly: ignore [bad-override]
        validator: RLDrivingValidator.Config  # pyrefly: ignore [bad-override]
        warm_start_checkpoint: str
        steps_per_epoch: int
        train_step_barrier_timeout_seconds: int
        ema_tau: float
        fps: int

        def __post_init__(self) -> None:
            Trainer.Config.__post_init__(self)
            if self.codedir:
                self.dataloader.codedir = self.codedir
                self.validator.miniray = {**self.validator.miniray, "codedir": self.codedir}
            if self.steps_per_epoch != self.dataloader.steps_per_epoch:
                raise ValueError("trainer and dataloader steps_per_epoch must match")
            if self.ema_tau < 1.0:
                raise ValueError("ema_tau must be at least 1")

    config: Config  # pyrefly: ignore [bad-override]
    loss_fn: RLDrivingLoss  # pyrefly: ignore [bad-override]
    dataloader: RLDrivingDataLoader  # pyrefly: ignore [bad-override]
    lr_schedulers: RLDrivingLRSchedulers  # pyrefly: ignore [bad-override]
    validator: RLDrivingValidator  # pyrefly: ignore [bad-override]

    def __init__(self, config: Config):
        super().__init__(config)
        if self.gradient_accumulation_steps != 1:
            raise ValueError("rldriving does not support gradient accumulation")
        self.train_step_barrier_group: dist.ProcessGroup | None = None
        if dist.get_world_size() > 1:
            self.train_step_barrier_group = dist.new_group(
                backend="gloo", timeout=timedelta(seconds=config.train_step_barrier_timeout_seconds)
            )
        training_id = os.getenv("REPORTERV2_TRAINING_ID") or "local"
        self.unique_segment_counter = StringUniqueCounter(f"unique_ids:{training_id}:rldriving:train")
        self.loss_fn.to(self.device)
        self.model = cast(RLDrivingModel, self.model_parts[0])
        dcp.load(
            {"temporal_policy": self.model.actor},
            storage_reader=FsspecReader(_get_path_checkpoint(config.warm_start_checkpoint).url_or_file()),
        )
        self.model.warm_start_critics_from_actor()

    # pyrefly: ignore [bad-override]
    def batch_generator(self, data_iterable: Iterable[Batch]) -> Iterator[Batch]:
        data_iterator = iter(data_iterable)
        while True:
            data_load_start = time.perf_counter()
            try:
                batch = next(data_iterator)
            except StopIteration as ex:
                raise DataloaderExhaustedError() from ex
            batch_size = next(iter(batch[0].values())).shape[0]
            self.metrics_processor.ntokens_since_last_log += batch_size
            self.metrics_processor.data_loading_times.append(time.perf_counter() - data_load_start)
            yield batch

    def prepare_batch(self, batch: Batch) -> PreparedBatch:
        inputs, targets, metadata = batch
        # info is a serialized host-side column, not a model tensor.
        inputs = {name: value.to(self.device) for name, value in inputs.items() if name != "info"}
        targets = {name: value.to(self.device) for name, value in targets.items()}
        metadata = {name: value.to(self.device) for name, value in metadata.items()}
        current_inputs = {name: inputs[name].float() for name in TEMPORAL_INPUTS}
        next_inputs = {
            name: torch.cat((inputs[name][:, 1:], inputs[f"next_{name}"]), dim=1).float() for name in current_inputs
        }
        return current_inputs, next_inputs, targets, metadata

    # pyrefly: ignore [bad-override]
    def train_step(self, data_iterator: Iterator[Batch]) -> None:
        steps_per_epoch = self.config.steps_per_epoch
        rollout_epoch = ((self.step - 1) // steps_per_epoch) * steps_per_epoch + 1
        self.dataloader.attach_training_context(RolloutContext(epoch=rollout_epoch))
        batch = next(data_iterator)
        info = batch[0].get("info")
        if info is not None:
            self.unique_segment_counter.update(parse_info(value)["name"] for value in info.cpu().numpy())
        current_inputs, next_inputs, targets, metadata = self.prepare_batch(batch)
        batch_size = next(iter(current_inputs.values())).shape[0]
        self.ntokens_seen += batch_size
        local_samples = torch.tensor(batch_size, dtype=torch.float32, device=self.device)

        lr_metrics = self.lr_schedulers.get_metrics()
        metric_sums: dict[str, torch.Tensor] = {}
        self.optimizers.zero_grad()
        if self.train_step_barrier_group is not None:
            dist.barrier(group=self.train_step_barrier_group)
        with self.train_context():
            actor_outputs = self.model(current_inputs)
            next_actor_outputs = self.model(next_inputs)
            actor_loss_B, actor_metrics = self.loss_fn.actor_loss(
                actor_outputs=actor_outputs,
                next_actor_outputs=next_actor_outputs,
                online_critic=self.model.critic,
                current_inputs=current_inputs,
                targets=targets,
            )
            actor_loss = actor_loss_B.sum() / local_samples
            actor_loss.backward()
        actor_loss = actor_loss.detach()
        self._accumulate_metrics(metric_sums, actor_metrics)
        del actor_outputs, next_actor_outputs, actor_loss_B, actor_metrics
        actor_grad_norm = self._clip_phase_grad_norm(self.model.actor)
        self.checkpointer.maybe_wait_for_staging()
        self.lr_schedulers.step_phase("actor")

        self.optimizers.zero_grad()
        with self.train_context():
            with torch.no_grad():
                next_actor_outputs = self.model.target_forward(next_inputs)
            critic_loss_B, critic_metrics = self.loss_fn.critic_loss(
                next_actor_outputs=next_actor_outputs,
                targets=targets,
                online_critic=self.model.critic,
                target_critic=self.model.target_critic,
                current_inputs=current_inputs,
                next_inputs=next_inputs,
            )
            critic_loss = critic_loss_B.sum() / local_samples
            critic_loss.backward()
        critic_loss = critic_loss.detach()
        self._accumulate_metrics(metric_sums, critic_metrics)
        del next_actor_outputs, critic_loss_B, critic_metrics
        critic_grad_norm = self._clip_phase_grad_norm(self.model.critic)
        self.lr_schedulers.step_phase("critic")
        self.optimizers.zero_grad()

        with torch.no_grad():
            decay = 1.0 - 1.0 / self.config.ema_tau
            for online, target in (
                (self.model.actor, self.model.target_actor),
                (self.model.critic, self.model.target_critic),
            ):
                for online_param, target_param in zip(online.parameters(), target.parameters()):
                    target_param.mul_(decay).add_(online_param, alpha=1.0 - decay)
                for online_buffer, target_buffer in zip(online.buffers(), target.buffers()):
                    target_buffer.copy_(online_buffer)
        self.lr_schedulers.step()

        if self.step == 0 or not self.metrics_processor.should_log(self.step):
            return

        loss = actor_loss + critic_loss
        loss_mesh = self.parallel_dims.get_optional_mesh("loss")
        if loss_mesh is not None:
            global_samples = float(dist_utils.dist_sum(local_samples, loss_mesh))
            global_avg_loss = dist_utils.dist_sum(loss * local_samples, loss_mesh) / global_samples
            global_max_loss = dist_utils.dist_max(loss, loss_mesh)
            global_samples_seen = dist_utils.dist_sum(
                torch.tensor(self.ntokens_seen, dtype=torch.int64, device=self.device),
                loss_mesh,
            )
            metric_averages = {
                name: dist_utils.dist_sum(value, loss_mesh) / global_samples for name, value in metric_sums.items()
            }
        else:
            global_avg_loss = global_max_loss = float(loss.item())
            global_samples_seen = self.ntokens_seen
            metric_averages = {name: float(value.item()) / batch_size for name, value in metric_sums.items()}

        batch_mesh = self.parallel_dims.get_optional_mesh("batch")
        unique_segments_seen = (
            self.unique_segment_counter.global_count(batch_mesh.get_group())
            if batch_mesh is not None
            else self.unique_segment_counter.local_count()
        )

        metadata_averages = {}
        for name, value in metadata.items():
            value = value.float()
            finite = torch.isfinite(value)
            value_sum = torch.where(finite, value, 0.0).sum()
            value_count = finite.sum()
            if loss_mesh is not None:
                total = dist_utils.dist_sum(value_sum, loss_mesh)
                count = dist_utils.dist_sum(value_count, loss_mesh)
            else:
                total = float(value_sum.item())
                count = float(value_count.item())
            metadata_averages[name] = total / count if count else float("nan")

        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            float(torch.maximum(actor_grad_norm, critic_grad_norm).item()),
            extra_metrics={
                "n_samples_seen": global_samples_seen,
                "actor_grad_norm": float(actor_grad_norm.item()),
                "critic_grad_norm": float(critic_grad_norm.item()),
                "dataset/unique_segments_seen": unique_segments_seen,
                **lr_metrics,
                **{f"rldriving/{name}": value for name, value in metric_averages.items()},
                **{f"sim/{name}": value for name, value in metadata_averages.items()},
            },
        )

    def _clip_phase_grad_norm(self, module: nn.Module) -> torch.Tensor:
        return dist_utils.clip_grad_norm_(
            module.parameters(),
            self.config.training.max_norm,
            foreach=True,
        )

    @staticmethod
    def _accumulate_metrics(
        sums: dict[str, torch.Tensor],
        metrics: dict[str, torch.Tensor],
    ) -> None:
        for name, value in metrics.items():
            sums[name] = sums.get(name, torch.zeros((), device=value.device)) + value.float().sum()

    @record
    def train(self) -> None:
        config = self.config
        sl.log_trace_instant("training_start")
        loaded = self.checkpointer.load(step=config.checkpoint.load_step)
        if not loaded:
            self.checkpointer.save(0)
        self.set_runtime_seed()
        loaded_step = self.step
        logger.info(f"Training starts at step {self.step + 1}")

        with config.profiler.build(
            global_step=self.step,
            base_folder=config.dump_folder,
        ) as profiler:
            data_iterator = self.batch_generator(self.dataloader)
            while self.should_continue_training():
                self.step += 1
                sl.set_step(self.step, relative_step=self.step - loaded_step)
                with sl.log_trace_span("step"):
                    self.gc_handler.run(self.step)
                    try:
                        self.train_step(data_iterator)
                    except DataloaderExhaustedError:
                        logger.warning("Ran out of data; last step was canceled.")
                        break
                    self.checkpointer.save(
                        self.step,
                        last_step=(self.step == config.training.steps),
                    )
                    if config.validator.enable and self.validator.should_validate(self.step):
                        self.validator.validate(self.model_parts, self.step)
                    profiler.step()
                    if self.step - loaded_step == 1:
                        dist_utils.set_pg_timeouts(
                            timeout=timedelta(seconds=config.comm.train_timeout_seconds),
                            parallel_dims=self.parallel_dims,
                        )

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)
        logger.info("Training completed")

    def close(self) -> None:
        self.dataloader.close()
        if self.config.validator.enable:
            self.validator.close()
        super().close()
        if self.train_step_barrier_group is not None:
            dist.destroy_process_group(self.train_step_barrier_group)
            self.train_step_barrier_group = None

    def state_dict(self) -> dict[str, Any]:
        return {
            **super().state_dict(),
            "unique_segment_counter": self.unique_segment_counter.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        if "unique_segment_counter" in state_dict:
            self.unique_segment_counter.load_state_dict(state_dict["unique_segment_counter"])
