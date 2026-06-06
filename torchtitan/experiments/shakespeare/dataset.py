# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import torch
import torch.distributed as dist
from gigashuffle import DataloaderConfig, MultiprocessShuffledDataloader
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import get_worker_info, IterableDataset

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.tools.logging import logger


SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
DEFAULT_DATASET_PATH = "./assets/datasets/shakespeare/input.txt"


class ByteTokenizer(BaseTokenizer):
    """Tokenizer shim for the byte-level Shakespeare dataset."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenizer.Config):
        pass

    def __init__(
        self,
        config: Config | None = None,
        *,
        tokenizer_path: str | None = None,
    ) -> None:
        super().__init__()
        self.bos_id = None
        self.eos_id = None

    def encode(self, *args, **kwargs) -> list[int]:
        text = args[0] if args else kwargs.pop("text")
        add_bos = kwargs.pop("add_bos", False)
        add_eos = kwargs.pop("add_eos", False)
        if add_bos or add_eos:
            raise ValueError("ByteTokenizer does not define BOS/EOS tokens")
        return list(text.encode("utf-8"))

    def decode(self, *args, **kwargs) -> str:
        tokens = args[0] if args else kwargs.pop("tokens")
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().tolist()
        return bytes(int(token) % 256 for token in tokens).decode(
            "utf-8", errors="replace"
        )

    def get_vocab_size(self) -> int:
        return 256


class ShakespeareDataset(IterableDataset, Stateful):
    """Yields byte-level language-modeling chunks for gigashuffle writers."""

    def __init__(
        self,
        *,
        dataset_path: str | None,
        split: str,
        seq_len: int,
        chunk_size: int,
        stride: int,
        val_fraction: float,
        infinite: bool,
        download: bool,
    ) -> None:
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if stride < 0:
            raise ValueError(f"stride must be non-negative, got {stride}")
        if not 0.0 < val_fraction < 1.0:
            raise ValueError(
                f"val_fraction must be between 0 and 1, got {val_fraction}"
            )

        self.dataset_path = dataset_path or DEFAULT_DATASET_PATH
        self.split = self._normalize_split(split)
        self.seq_len = seq_len
        self.chunk_size = chunk_size
        self.stride = stride or seq_len
        self.val_fraction = val_fraction
        self.infinite = infinite
        self.download = download
        self._epoch = 0
        self._cursor = 0

        self._data = self._load_split_bytes()
        if len(self._data) <= self.seq_len:
            raise ValueError(
                f"Shakespeare {self.split} split has {len(self._data)} bytes, "
                f"but seq_len={self.seq_len} requires at least {self.seq_len + 1}"
            )

    @staticmethod
    def _normalize_split(split: str) -> str:
        split = split.lower()
        if split == "validation":
            split = "val"
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        return split

    def _materialize_dataset_file(self) -> Path:
        path = Path(self.dataset_path)
        if path.exists():
            return path
        if not self.download:
            raise FileNotFoundError(
                f"Shakespeare dataset file {path} does not exist. "
                "Set dataloader.download=True or provide dataloader.dataset_path."
            )

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading Shakespeare dataset to %s", path)
            urlretrieve(SHAKESPEARE_URL, path)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        return path

    def _load_split_bytes(self) -> bytes:
        path = self._materialize_dataset_file()
        data = path.read_bytes()
        split_idx = int(len(data) * (1.0 - self.val_fraction))
        if self.split == "train":
            return data[:split_idx]
        return data[split_idx:]

    def _num_sequences(self) -> int:
        return 1 + (len(self._data) - self.seq_len - 1) // self.stride

    def _sequence_at(self, sequence_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = sequence_idx * self.stride
        window = self._data[start : start + self.seq_len + 1]
        x = torch.tensor(list(window[:-1]), dtype=torch.long)
        y = torch.tensor(list(window[1:]), dtype=torch.long)
        return x, y

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        num_sequences = self._num_sequences()
        positions = torch.arange(self.seq_len, dtype=torch.int32)

        while True:
            inputs = []
            labels = []
            batch_positions = []
            while len(inputs) < self.chunk_size:
                sequence_idx = self._cursor * num_workers + worker_id
                if sequence_idx >= num_sequences:
                    if not self.infinite:
                        break
                    self._epoch += 1
                    self._cursor = 0
                    sequence_idx = worker_id
                self._cursor += 1

                x, y = self._sequence_at(sequence_idx)
                inputs.append(x)
                labels.append(y)
                batch_positions.append(positions)

            if not inputs:
                logger.warning("Shakespeare %s split has run out of data", self.split)
                break

            yield [
                {
                    "input": torch.stack(inputs),
                    "labels": torch.stack(labels),
                    "positions": torch.stack(batch_positions),
                }
            ]

            if len(inputs) < self.chunk_size and not self.infinite:
                logger.warning("Shakespeare %s split has run out of data", self.split)
                break

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self._epoch, "cursor": self._cursor}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return
        self._epoch = state_dict["epoch"]
        self._cursor = state_dict["cursor"]


class ShakespeareDataLoader(BaseDataLoader):
    """TorchTitan dataloader adapter backed by gigashuffle."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        dataset: str = "train"
        """Dataset split to use: train or val."""

        dataset_path: str | None = DEFAULT_DATASET_PATH
        """Path to the Shakespeare input.txt file."""

        infinite: bool = True
        """Whether to loop the split indefinitely."""

        download: bool = True
        """Download the Shakespeare dataset when dataset_path is missing."""

        val_fraction: float = 0.1
        """Fraction of the corpus reserved for validation."""

        stride: int = 0
        """Byte stride between sequences. 0 defaults to seq_len."""

        chunk_size: int = 64
        """Number of sequences produced by each writer dataset sample."""

        shuffle_size: int = 8192
        """Number of individual sequences in the gigashuffle shared buffer."""

        min_mixing: float = 0.5
        """Fraction of shuffle_size to fill before readers emit batches."""

        num_writers: int = 2
        """Number of gigashuffle writer processes per rank."""

        num_readers: int = 2
        """Number of gigashuffle reader processes per rank."""

        writer_max_retries: int = 100
        """Maximum empty-sample retries in gigashuffle writers."""

        fill_once: bool = False
        """Use gigashuffle's single ordered pass mode."""

        redis_host: str = "localhost"
        redis_port: int = 6379
        redis_db: int = 6
        queue_name: str = ""
        """Redis queue name. Empty derives a shared name from the run id."""

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
        **kwargs,
    ) -> None:
        self.config = config
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank

        world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        if world_size != dp_world_size:
            raise ValueError(
                "ShakespeareDataLoader currently supports data-parallel-only "
                f"meshes. Got distributed world_size={world_size} and "
                f"data-parallel world_size={dp_world_size}."
            )

        dataset = ShakespeareDataset(
            dataset_path=config.dataset_path,
            split=config.dataset,
            seq_len=seq_len,
            chunk_size=config.chunk_size,
            stride=config.stride,
            val_fraction=config.val_fraction,
            infinite=config.infinite,
            download=config.download,
        )

        global_rank = (
            dist.get_rank()
            if dist.is_available() and dist.is_initialized()
            else dp_rank
        )
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        queue_name = config.queue_name or self._default_queue_name(config.dataset)

        gigashuffle_config = DataloaderConfig(
            bs=local_batch_size,
            shuffle_size=config.shuffle_size,
            min_mixing=config.min_mixing,
            num_writers=config.num_writers,
            num_readers=config.num_readers,
            writer_max_retries=config.writer_max_retries,
            fill_once=config.fill_once,
            local_rank=local_rank,
            global_rank=global_rank,
            local_world_size=local_world_size,
            global_world_size=world_size,
            redis_host=config.redis_host,
            redis_port=config.redis_port,
            redis_db=config.redis_db,
            queue_name=queue_name,
        )
        self._loader = MultiprocessShuffledDataloader(
            dataset,
            config=gigashuffle_config,
        )

    @staticmethod
    def _default_queue_name(split: str) -> str:
        run_name = (
            os.environ.get("REPORTERV2_TRAINING_ID")
            or os.environ.get("TORCHELASTIC_RUN_ID")
            or os.environ.get("SLURM_JOB_ID")
            or "local"
        )
        return f"torchtitan-shakespeare-{run_name}-{split}"

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        for buffer in self._loader:
            sample = buffer[0]
            input_dict = {
                "input": sample["input"],
                "positions": sample["positions"],
            }
            labels = sample["labels"]
            yield input_dict, labels

    def state_dict(self) -> dict[str, Any]:
        return self._loader.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._loader.load_state_dict(state_dict)
