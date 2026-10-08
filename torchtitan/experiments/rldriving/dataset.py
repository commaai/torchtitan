# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
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
            shuffle_size=str(config.shuffle_size),
            min_mixing=config.min_mixing,
            num_writers=config.num_writers,
            num_readers=config.num_readers,
            limit=config.limit,
            codedir=config.codedir,
            pipeline_dir=config.pipeline_dir,
            queue_priority=config.queue_priority,
            max_queue_size=config.max_queue_size,
            max_fq_size=config.max_fq_size,
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
        self.dataset = get_dataset(xx_config, local_rank=local_rank)
        self._loader_config = DataloaderConfig(
            bs=local_batch_size,
            shuffle_size=config.shuffle_size,
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
        self.loader: Any | None = None
        self._iterator: Any | None = None

    # pyrefly: ignore [bad-override]
    def __iter__(
        self,
    ) -> Iterator[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]]:
        if self.loader is None:
            self.loader = DataLoader(self.dataset, self._loader_config)
        iterator: Any = iter(self.loader)
        self._iterator = iterator
        try:
            for inputs, targets, metadata in iterator:
                yield inputs, targets, metadata
        finally:
            iterator.close()
            if self._iterator is iterator:
                self._iterator = None

    def attach_training_context(self, context: RolloutContext) -> None:
        self.dataset.context = context
        if self.loader is not None:
            self.loader.attach_training_context(context)

    def close(self) -> None:
        if self._iterator is not None:
            self._iterator.close()
            self._iterator = None
        if self.loader is not None:
            self.loader._shutdown_workers()

    def state_dict(self) -> dict[str, int]:
        return {}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        return
