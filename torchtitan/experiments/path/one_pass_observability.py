from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


SAMPLE_ID_BYTES_ESTIMATE = 72
BIJECTION_ENTRY_BYTES_ESTIMATE = 400
FIXED_OVERHEAD_BYTES_ESTIMATE = 16 * 1024 * 1024
DEFAULT_HOST_BUDGET_BYTES = 512 * 1024 * 1024


def estimated_owner_bytes(
    *, total_samples: int, world_size: int, min_samples_per_source: float = 1
) -> int:
    if min(total_samples, world_size, min_samples_per_source) < 1:
        raise ValueError("audit memory inputs must be positive")
    samples = (total_samples + world_size - 1) // world_size
    occurrences_total = int((total_samples + min_samples_per_source - 1) // min_samples_per_source)
    occurrences = (occurrences_total + world_size - 1) // world_size
    return (
        FIXED_OVERHEAD_BYTES_ESTIMATE
        + samples * SAMPLE_ID_BYTES_ESTIMATE
        + occurrences * BIJECTION_ENTRY_BYTES_ESTIMATE
    )


def stable_segment_owner(name: str, world_size: int) -> int:
    digest = hashlib.sha256(name.encode()).digest()
    return int.from_bytes(digest[:8], "big") % world_size


def stable_numeric_owner(value: int, world_size: int) -> int:
    digest = hashlib.sha256(struct.pack(">Q", value)).digest()
    return int.from_bytes(digest[:8], "big") % world_size


def pack_occurrence(global_rank: int, writer: int, sequence: int, num_writers: int) -> int:
    if not (0 <= global_rank < 2**24 and 0 <= writer < num_writers < 2**8):
        raise ValueError("source rank/writer is outside the collision-free encoding")
    if not 0 <= sequence < 2**32:
        raise ValueError("source sequence is outside the collision-free encoding")
    return ((global_rank * num_writers + writer) << 32) | sequence


def pack_sample(occurrence: int, sample_index: int) -> int:
    if not 0 <= sample_index < 2**8:
        raise ValueError("source sample index is outside the collision-free encoding")
    value = (occurrence << 8) | sample_index
    if value >= 2**63:
        raise ValueError("source sample identity does not fit signed int64")
    return value


@dataclass(frozen=True, slots=True)
class AuditResult:
    segments: int
    occurrences: int
    samples: int
    consumed_samples: int
    n_checks: int


class OwnerShardedOnePassAudit:
    def __init__(
        self,
        *,
        num_writers: int,
        total_samples: int,
        world_size: int,
        expected_samples_per_source: float,
        host_budget_bytes: int,
    ) -> None:
        estimate = estimated_owner_bytes(
            total_samples=total_samples,
            world_size=world_size,
        )
        expected_estimate = estimated_owner_bytes(
            total_samples=total_samples,
            world_size=world_size,
            min_samples_per_source=expected_samples_per_source,
        )
        if estimate > host_budget_bytes:
            raise RuntimeError(
                "one-pass audit estimated owner memory exceeds its configured host "
                f"budget: estimate={estimate} budget={host_budget_bytes}"
            )
        self.num_writers = num_writers
        self.estimated_owner_bytes = estimate
        self.expected_owner_bytes = expected_estimate
        self.host_budget_bytes = host_budget_bytes
        self.segment_to_occurrence: dict[str, int] = {}
        self.occurrence_to_segment: dict[int, str] = {}
        self.sample_ids: set[int] = set()
        self.pending_pairs: list[tuple[str, int]] = []
        self.pending_samples: list[int] = []
        self.local_consumed_samples = 0
        self.n_checks = 0

    def observe(
        self,
        segment_names: list[str],
        global_ranks: list[int],
        writers: list[int],
        sequences: list[int],
        sample_indices: list[int],
    ) -> None:
        lengths = {
            len(segment_names),
            len(global_ranks),
            len(writers),
            len(sequences),
            len(sample_indices),
        }
        if len(lengths) != 1:
            raise ValueError("one-pass metadata fields have different batch lengths")
        for name, rank, writer, sequence, sample_index in zip(
            segment_names,
            global_ranks,
            writers,
            sequences,
            sample_indices,
            strict=True,
        ):
            occurrence = pack_occurrence(rank, writer, sequence, self.num_writers)
            self.pending_pairs.append((str(name), occurrence))
            self.pending_samples.append(pack_sample(occurrence, sample_index))
        self.local_consumed_samples += len(segment_names)

    @staticmethod
    def _group_info(group) -> tuple[int, int]:
        if not dist.is_available() or not dist.is_initialized():
            return 1, 0
        return dist.get_world_size(group), dist.get_rank(group)

    @staticmethod
    def _gather(payload: Any, group, world_size: int) -> list[Any]:
        if world_size == 1:
            return [payload]
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, payload, group=group)
        return gathered

    def sync(self, *, group=None, device: torch.device | None = None) -> AuditResult:
        world_size, owner_rank = self._group_info(group)
        payload = {
            "pairs": sorted(set(self.pending_pairs)),
            "samples": list(self.pending_samples),
            "consumed": self.local_consumed_samples,
        }
        gathered = self._gather(payload, group, world_size)
        violations: list[str] = []
        for rank_payload in gathered:
            for name, occurrence in rank_payload["pairs"]:
                if stable_segment_owner(name, world_size) == owner_rank:
                    previous = self.segment_to_occurrence.get(name)
                    if previous is not None and previous != occurrence:
                        violations.append(
                            f"segment {name!r} mapped to occurrences {previous} and {occurrence}"
                        )
                    self.segment_to_occurrence[name] = occurrence
                if stable_numeric_owner(occurrence, world_size) == owner_rank:
                    previous_name = self.occurrence_to_segment.get(occurrence)
                    if previous_name is not None and previous_name != name:
                        violations.append(
                            f"occurrence {occurrence} mapped to {previous_name!r} and {name!r}"
                        )
                    self.occurrence_to_segment[occurrence] = name
            for sample_id in rank_payload["samples"]:
                if stable_numeric_owner(sample_id, world_size) != owner_rank:
                    continue
                if sample_id in self.sample_ids:
                    violations.append(f"sample identity {sample_id} was consumed twice")
                self.sample_ids.add(sample_id)

        live_bytes = (
            FIXED_OVERHEAD_BYTES_ESTIMATE
            + len(self.sample_ids) * SAMPLE_ID_BYTES_ESTIMATE
            + (len(self.segment_to_occurrence) + len(self.occurrence_to_segment))
            * (BIJECTION_ENTRY_BYTES_ESTIMATE // 2)
        )
        if live_bytes > self.host_budget_bytes:
            violations.append(
                "one-pass audit live owner memory estimate exceeds its configured "
                f"budget: estimate={live_bytes} budget={self.host_budget_bytes}"
            )

        flag_device = device or torch.device("cpu")
        violation_flag = torch.tensor(bool(violations), dtype=torch.int64, device=flag_device)
        if world_size > 1:
            dist.all_reduce(violation_flag, op=dist.ReduceOp.MAX, group=group)
        if int(violation_flag.item()):
            all_violations = self._gather(violations, group, world_size)
            messages = [message for rank_messages in all_violations for message in rank_messages]
            raise RuntimeError("one-pass consumption audit failed: " + "; ".join(messages[:16]))

        counts = torch.tensor(
            [
                len(self.segment_to_occurrence),
                len(self.occurrence_to_segment),
                len(self.sample_ids),
                self.local_consumed_samples,
            ],
            dtype=torch.int64,
            device=flag_device,
        )
        if world_size > 1:
            dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=group)
        segments, occurrences, samples, consumed = map(int, counts.tolist())
        relation_violation = int(segments != occurrences or samples != consumed)
        relation_flag = torch.tensor(relation_violation, dtype=torch.int64, device=flag_device)
        if world_size > 1:
            dist.all_reduce(relation_flag, op=dist.ReduceOp.MAX, group=group)
        if int(relation_flag.item()):
            raise RuntimeError(
                "one-pass consumption audit count mismatch: "
                f"segments={segments} occurrences={occurrences} "
                f"samples={samples} consumed={consumed}"
            )
        self.pending_pairs.clear()
        self.pending_samples.clear()
        self.n_checks += 1
        return AuditResult(segments, occurrences, samples, consumed, self.n_checks)

    def shard_attestation(self) -> dict[str, Any]:
        digest = hashlib.sha256(b"one-pass-owner-shard-v1\0")
        for name, occurrence in sorted(self.segment_to_occurrence.items()):
            encoded = name.encode()
            digest.update(struct.pack(">I", len(encoded)))
            digest.update(encoded)
            digest.update(struct.pack(">Q", occurrence))
        digest.update(b"\0occurrences\0")
        for occurrence, name in sorted(self.occurrence_to_segment.items()):
            encoded = name.encode()
            digest.update(struct.pack(">Q", occurrence))
            digest.update(struct.pack(">I", len(encoded)))
            digest.update(encoded)
        digest.update(b"\0samples\0")
        for sample_id in sorted(self.sample_ids):
            digest.update(struct.pack(">Q", sample_id))
        return {
            "segments": len(self.segment_to_occurrence),
            "occurrences": len(self.occurrence_to_segment),
            "samples": len(self.sample_ids),
            "sha256": digest.hexdigest(),
        }
