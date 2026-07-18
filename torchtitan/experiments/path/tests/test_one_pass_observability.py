from __future__ import annotations

import os
import tempfile
from multiprocessing import get_context

import pytest
import torch
import torch.distributed as dist

from torchtitan.experiments.path.one_pass_observability import (
    OwnerShardedOnePassAudit,
    estimated_owner_bytes,
    stable_numeric_owner,
)


def audit(world_size: int = 1) -> OwnerShardedOnePassAudit:
    return OwnerShardedOnePassAudit(
        num_writers=6,
        total_samples=64,
        world_size=world_size,
        expected_samples_per_source=7.9,
        host_budget_bytes=512 * 1024 * 1024,
    )


def test_memory_preflight_is_bounded_and_rejects_small_budget():
    estimate = estimated_owner_bytes(
        total_samples=24_140_544, world_size=32
    )
    assert 350 * 1024 * 1024 < estimate < 400 * 1024 * 1024
    with pytest.raises(RuntimeError, match="configured host budget"):
        OwnerShardedOnePassAudit(
            num_writers=6,
            total_samples=24_140_544,
            world_size=32,
            expected_samples_per_source=7.9,
            host_budget_bytes=estimate - 1,
        )


def test_one_sample_per_source_is_the_fail_closed_memory_bound():
    total = 32_000
    worst = estimated_owner_bytes(total_samples=total, world_size=32)
    expected = estimated_owner_bytes(
        total_samples=total, world_size=32, min_samples_per_source=7.9
    )
    assert worst > expected


def test_numeric_sample_ownership_is_not_confined_to_sample_index_bits():
    owners = {stable_numeric_owner(sample_id, 32) for sample_id in range(256)}
    assert len(owners) > 24


def test_single_rank_exact_counts_and_stable_attestation():
    checker = audit()
    checker.observe(["a", "a", "b"], [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 1, 0])
    result = checker.sync()
    assert (result.segments, result.occurrences, result.samples) == (2, 2, 3)
    assert checker.shard_attestation() == checker.shard_attestation()


def test_duplicate_occurrence_sample_fails():
    checker = audit()
    checker.observe(["a"], [0], [0], [0], [0])
    checker.sync()
    checker.observe(["a"], [0], [0], [0], [0])
    with pytest.raises(RuntimeError, match="consumed twice"):
        checker.sync()


def test_same_segment_from_new_occurrence_fails():
    checker = audit()
    checker.observe(["a"], [0], [0], [0], [0])
    checker.sync()
    checker.observe(["a"], [0], [0], [1], [0])
    with pytest.raises(RuntimeError, match="mapped to occurrences"):
        checker.sync()


def _distributed_violation_worker(rank: int, world_size: int, init_file: str, queue) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        checker = audit(world_size)
        checker.observe(["same"], [0], [0], [0], [0])
        try:
            checker.sync(group=dist.group.WORLD, device=torch.device("cpu"))
        except RuntimeError as error:
            queue.put((rank, str(error)))
        else:
            queue.put((rank, "NO ERROR"))
    finally:
        dist.destroy_process_group()


@pytest.mark.timeout(20)
def test_cross_rank_duplicate_fails_every_rank_without_hang():
    ctx = get_context("spawn")
    queue = ctx.Queue()
    with tempfile.NamedTemporaryFile(delete=False) as file:
        init_file = file.name
    try:
        processes = [
            ctx.Process(target=_distributed_violation_worker, args=(rank, 2, init_file, queue))
            for rank in range(2)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=15)
            assert process.exitcode == 0
        results = [queue.get(timeout=2) for _ in processes]
        assert {rank for rank, _ in results} == {0, 1}
        assert all("one-pass consumption audit failed" in message for _, message in results)
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)
