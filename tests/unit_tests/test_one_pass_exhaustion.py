from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.trainer import Trainer
from xx.training.lib.dataloader import OnePassDatasetExhaustedError


class _ExhaustingTrainer(Trainer):
    def __init__(self, *, one_pass: bool) -> None:
        self.config = SimpleNamespace(
            checkpoint=SimpleNamespace(load_step=-1),
            dataloader=SimpleNamespace(one_pass=one_pass),
            dump_folder="/tmp",
            training=SimpleNamespace(steps=2),
            profiler=SimpleNamespace(build=lambda **_kwargs: nullcontext()),
        )
        self.step = 0
        self.ntokens_seen = 0
        self.checkpointer = Mock()
        self.gc_handler = Mock()

    def batch_generator(self, _dataloader):
        return iter(())

    def train_step(self, _data_iterator) -> None:
        raise DataloaderExhaustedError


def test_one_pass_exhaustion_is_a_hard_training_failure(monkeypatch):
    trainer = _ExhaustingTrainer(one_pass=True)
    trainer.dataloader = ()
    monkeypatch.setattr("torchtitan.trainer.sl.set_step", Mock())

    with pytest.raises(RuntimeError, match="attempted step 1"):
        trainer.train()

    trainer.checkpointer.save.assert_not_called()


def test_default_exhaustion_keeps_existing_graceful_stop(monkeypatch):
    trainer = _ExhaustingTrainer(one_pass=False)
    trainer.dataloader = ()
    monkeypatch.setattr("torchtitan.trainer.sl.set_step", Mock())
    monkeypatch.setattr("torchtitan.trainer.torch.distributed.get_rank", lambda: 1)

    trainer.train()

    assert trainer.step == 1
    trainer.checkpointer.save.assert_not_called()


def test_multiprocess_one_pass_failure_cannot_save_a_short_checkpoint(monkeypatch):
    trainer = _ExhaustingTrainer(one_pass=True)
    trainer.dataloader = ()
    trainer.train_step = Mock(
        side_effect=OnePassDatasetExhaustedError(
            "one-pass aggregate exhausted"
        )
    )
    monkeypatch.setattr("torchtitan.trainer.sl.set_step", Mock())

    with pytest.raises(OnePassDatasetExhaustedError, match="aggregate exhausted"):
        trainer.train()

    trainer.checkpointer.save.assert_not_called()
