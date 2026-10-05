# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Any, Literal
from xx.comma_data.constants import BASE_DIR_GT

from xx.common.basedir import XX_BASEDIR
from xx.training.lib.dataloader import DataLoader
from xx.training.rldriving.config import DatasetConfig
from xx.training.rldriving.dataloader import get_dataset, RolloutContext

import torch

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.tokenizer import BaseTokenizer


class RLDrivingDataLoader(BaseDataLoader):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        dataset: str
        fps: int
        training_id: str = ""
        shuffle_size: int = 50_000
        min_mixing: float = 0.9
        num_writers: int = 1
        num_readers: int = 1
        limit: int | None = 500_000

        codedir: str | None = XX_BASEDIR
        pipeline_dir: str | None = BASE_DIR_GT
        queue_priority: int = 5
        max_queue_size: int = 256
        max_fq_size: int = 8192

        train_skip: int = 1
        epochs: int = 0
        steps_per_epoch: int = 1
        save_cache: bool = False
        load_caches: list[str] = field(default_factory=list)

        zero_desire: bool = False
        photo_noise_model: Literal["NONE", "VISION"] = "VISION"
        pre_worldmodel_warmup_seconds: int = 7
        min_simulation_seconds: int = 6
        max_simulation_seconds: int = 7
        worldmodel_future_size_seconds: int = 1
        worldmodel_context_size_seconds: int = 2

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        seq_len: int,
        local_batch_size: int,
        snapshot_every_n_steps: int | None = 1,
        validation_steps: int = 1,
        **kwargs: Any,
    ) -> None:
        del tokenizer, seq_len, snapshot_every_n_steps, validation_steps, kwargs
        from gigashuffle import DataloaderConfig

        if local_batch_size % 2:
            raise ValueError("Reference/actor training requires an even local batch size")
        local_batch_size //= 2
        local_rank = int(os.environ.get("LOCAL_RANK", dp_rank))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", dp_world_size))
        node_rank = int(os.environ.get("GROUP_RANK", dp_rank // local_world_size))

        xx_config = DatasetConfig(
            dataset=config.dataset_path or config.dataset,
            training_id=config.training_id,
            bs=local_batch_size,
            nproc_per_node=local_world_size,
            nnodes=dp_world_size // local_world_size,
            node_rank=node_rank,
            shuffle_size=str(config.shuffle_size // 2),
            min_mixing=config.min_mixing,
            num_writers=config.num_writers,
            num_readers=config.num_readers,
            limit=config.limit,
            codedir=config.codedir,
            pipeline_dir=config.pipeline_dir,
            queue_priority=config.queue_priority,
            max_queue_size=config.max_queue_size // 2,
            max_fq_size=config.max_fq_size // 2,
            train_skip=config.train_skip,
            epochs=config.epochs,
            steps_per_epoch=config.steps_per_epoch,
            save_cache=config.save_cache,
            load_caches=list(config.load_caches),
            fps=config.fps,
            zero_desire=config.zero_desire,
            photo_noise_model=config.photo_noise_model,
            pre_worldmodel_warmup_seconds=config.pre_worldmodel_warmup_seconds,
            min_simulation_seconds=config.min_simulation_seconds,
            max_simulation_seconds=config.max_simulation_seconds,
            worldmodel_future_size_seconds=config.worldmodel_future_size_seconds,
            worldmodel_context_size_seconds=config.worldmodel_context_size_seconds,
        )
        self.datasets = [
            get_dataset(xx_config.replace(worldmodel_reference=reference), local_rank=local_rank)
            for reference in (False, True)
        ]
        loader_config = DataloaderConfig(
            bs=local_batch_size,
            shuffle_size=config.shuffle_size // 2,
            min_mixing=config.min_mixing,
            num_writers=config.num_writers,
            num_readers=config.num_readers,
            fill_once=False,
            local_rank=local_rank,
            global_rank=dp_rank,
            local_world_size=local_world_size,
            global_world_size=dp_world_size,
            queue_name=f"{config.training_id or 'rldriving'}-train-node{node_rank}",
        )
        self._loader_configs = [
            replace(loader_config, queue_name=f"{loader_config.queue_name}-{source}") for source in ("sim", "wm")
        ]
        self.loaders: list[Any] = []
        self._iterators: list[Any] = []

    # pyrefly: ignore [bad-override]
    def __iter__(
        self,
    ) -> Iterator[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]]:
        if not self.loaders:
            self.loaders = [DataLoader(dataset, config) for dataset, config in zip(self.datasets, self._loader_configs)]
        self._iterators = [iter(loader) for loader in self.loaders]
        try:
            for sim, reference in zip(*self._iterators):
                inputs, targets = (
                    {name: torch.cat((left[name], right[name])) for name in left.keys() & right.keys()}
                    for left, right in zip(sim[:2], reference[:2])
                )
                yield inputs, targets, sim[2]
        finally:
            for iterator in self._iterators:
                iterator.close()
            self._iterators = []

    def attach_training_context(self, context: RolloutContext) -> None:
        for dataset in self.datasets:
            dataset.context = context
        for loader in self.loaders:
            loader.attach_training_context(context)

    def close(self) -> None:
        for iterator in self._iterators:
            iterator.close()
        self._iterators = []
        for loader in self.loaders:
            loader._shutdown_workers()

    def state_dict(self) -> dict[str, int]:
        return {}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        return
