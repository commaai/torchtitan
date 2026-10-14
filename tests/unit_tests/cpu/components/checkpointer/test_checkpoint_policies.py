# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import queue
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.metadata import Metadata

from torchtitan.components.checkpointer import CheckpointManager, MODEL
from torchtitan.components.checkpointer.dcp import (
    AsyncMode,
    CHECKPOINT_SIZE_LIMIT_BYTES,
    CHECKPOINT_UPLOAD_TIMEOUT_SECONDS,
)


@pytest.fixture
def manager(tmp_path):
    model = torch.nn.Linear(16, 16)
    state = SimpleNamespace(state_dict=lambda: {}, load_state_dict=lambda _: None)
    manager = CheckpointManager(
        CheckpointManager.Config(
            folder=str(tmp_path / "output"), keep_latest_k=0, last_save_model_only=False
        ),
        model_parts=[model],
        optimizers=state,
        lr_schedulers=state,
        ema=None,
        extra_states={},
        sd_adapter=None,
    )
    yield manager
    manager.close()


@pytest.mark.parametrize("prefix", ["step-", "", "epoch_", "epoch.+"])
def test_prefix_save_discovery_and_retention(manager, prefix):
    manager.checkpoint_id_format = prefix
    for step in (1, 2, 10):
        assert manager.save(step, last_step=True)
    assert manager._create_checkpoint_id(10) == f"{manager.folder}/{prefix}10"
    assert manager._find_load_step() == 10
    assert manager._find_load_step(max_step=5) == 2
    assert manager._parse_step(prefix + "01") is None
    assert manager._parse_step("foreign-100") is None
    assert manager.load(step=10)

    manager.keep_latest_k = 2
    manager.purge_queue = queue.Queue()
    manager.purge_thread = mock.Mock(is_alive=lambda: False)
    with mock.patch("torch.distributed.get_rank", return_value=0):
        manager._purge_stale_checkpoints(saving_step=10)
    assert manager.purge_queue.get_nowait() == manager._create_checkpoint_id(1)
    assert manager.purge_queue.empty()


@pytest.mark.parametrize("prefix", ["", "epoch.+"])
def test_prefix_staging_cleanup_preserves_inflight_save(manager, prefix):
    manager.checkpoint_id_format = prefix
    folder = Path(manager.folder)
    for name in (f".staging-{prefix}1", f".staging-{prefix}2", f"{prefix}2"):
        (folder / name).mkdir(parents=True)
    manager.keep_latest_k = 2
    manager.purge_queue = queue.Queue()
    manager.purge_thread = mock.Mock(is_alive=lambda: False)
    with mock.patch("torch.distributed.get_rank", return_value=0):
        manager._purge_stale_checkpoints(saving_step=2, staging_dir_prefix=".staging-")
    assert manager.purge_queue.get_nowait() == str(folder / f".staging-{prefix}1")
    assert manager.purge_queue.empty()


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("resume", [False, True])
def test_partial_loading_keeps_initialized_missing_keys(
    manager, tmp_path, partial, resume
):
    model = manager.states[MODEL].model[0]
    initial_bias = model.bias.detach().clone()
    checkpoint_id = (
        manager._create_checkpoint_id(1) if resume else str(tmp_path / "initial")
    )
    dcp.save({"weight": torch.ones_like(model.weight)}, checkpoint_id=checkpoint_id)
    if not resume:
        manager.initial_load_path = checkpoint_id
    manager.allow_partial_initial_load = partial
    if partial:
        assert manager.load()
        torch.testing.assert_close(model.weight, torch.ones_like(model.weight))
        torch.testing.assert_close(model.bias, initial_bias)
    else:
        with pytest.raises(CheckpointException, match="Missing key"):
            manager.load()


def metadata_with_size(size):
    return Metadata(
        state_dict_metadata={},
        storage_data={0: SimpleNamespace(length=size, relative_path="shard.distcp")},
    )


def test_size_limit_boundary_and_logging(manager):
    with mock.patch("torchtitan.components.checkpointer.dcp.logger.info") as log:
        manager._log_dcp_size(metadata_with_size(CHECKPOINT_SIZE_LIMIT_BYTES))
    log.assert_called_once_with(
        "total DCP checkpoint size: %.3f GB (%d shards).", 100.0, 1
    )
    with pytest.raises(RuntimeError, match="exceed the size limit"):
        manager._log_dcp_size(metadata_with_size(CHECKPOINT_SIZE_LIMIT_BYTES + 1))


@pytest.mark.parametrize("mode", list(AsyncMode))
def test_size_checked_on_sync_and_async_completion(manager, mode):
    metadata = metadata_with_size(CHECKPOINT_SIZE_LIMIT_BYTES + 1)
    manager.async_mode = mode
    if mode is AsyncMode.DISABLED:
        with mock.patch.object(dcp, "save", return_value=metadata):
            with pytest.raises(RuntimeError, match="size limit"):
                manager.dcp_save({}, "unused", mode)
    else:
        manager.save_future = Future()
        manager.save_future.set_result(metadata)
        with pytest.raises(RuntimeError, match="size limit"):
            manager.maybe_wait_for_saving()
        assert manager.save_future is None


def test_remote_timeout_and_local_writer_selection(manager):
    with (
        mock.patch(
            "torch.distributed.checkpoint._fsspec_filesystem.FsspecWriter"
        ) as writer,
        mock.patch.object(dcp, "save", return_value=metadata_with_size(0)) as save,
    ):
        manager.dcp_save({}, "memory://checkpoints/1", AsyncMode.DISABLED)
        writer.assert_called_once_with(
            "memory://checkpoints/1", timeout=CHECKPOINT_UPLOAD_TIMEOUT_SECONDS
        )
        assert save.call_args.kwargs["storage_writer"] is writer.return_value
        writer.reset_mock()
        manager.dcp_save({}, manager._create_checkpoint_id(2), AsyncMode.DISABLED)
        writer.assert_not_called()
        assert save.call_args.kwargs["storage_writer"] is None
