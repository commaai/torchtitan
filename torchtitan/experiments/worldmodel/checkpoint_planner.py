# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses
import re

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner_helpers import (
    _create_chunk_list,
    create_read_items_for_chunk_list,
)

_SCALE_SHIFT_FQN = re.compile(r"^(?:blocks\.\d+|final_layer)\.scale_shift_table$")


class WorldModelSuffixCropLoadPlanner(dcp.DefaultLoadPlanner):
    """Load a shorter worldmodel by retaining the checkpoint's context suffix.

    Worldmodel training lays out temporal inputs as ``[future, context]``. When
    future frames are removed, the retained context therefore lives at the end
    of temporal state tensors. DCP normally rejects differently sized tensors;
    this planner remaps the approved tensors to that suffix while preserving the
    default behavior, including hard failures, for every other tensor.
    """

    def create_local_plan(self):
        if self.metadata is None:
            raise AssertionError("load metadata was not initialized")

        original_metadata = self.metadata
        adjusted_entries = dict(original_metadata.state_dict_metadata)
        crops = {}

        for fqn, obj in self.state_dict.items():
            checkpoint_metadata = adjusted_entries.get(fqn)
            size = getattr(obj, "size", None)
            if not isinstance(checkpoint_metadata, TensorStorageMetadata) or not callable(size):
                continue

            saved_size = tuple(checkpoint_metadata.size)
            current_size = tuple(size())
            if saved_size == current_size:
                continue

            approved = fqn == "pos_embed" or _SCALE_SHIFT_FQN.fullmatch(fqn) is not None
            can_suffix_crop = (
                approved
                and len(saved_size) == len(current_size)
                and len(saved_size) >= 2
                and saved_size[:1] == current_size[:1]
                and saved_size[1] > current_size[1]
                and saved_size[2:] == current_size[2:]
            )
            if not can_suffix_crop:
                # Let DefaultLoadPlanner produce its normal size-mismatch error.
                continue

            shift = [0] * len(saved_size)
            shift[1] = saved_size[1] - current_size[1]
            crops[fqn] = (checkpoint_metadata, torch.Size(shift), obj)

            # Bypass only DefaultLoadPlanner's global-size equality check. The
            # read items for this key are rebuilt against the original metadata.
            adjusted_entries[fqn] = dataclasses.replace(checkpoint_metadata, size=size())

        self.metadata = dataclasses.replace(original_metadata, state_dict_metadata=adjusted_entries)
        try:
            default_plan = super().create_local_plan()
        finally:
            self.metadata = original_metadata

        items = [item for item in default_plan.items if item.dest_index.fqn not in crops]
        for fqn, (checkpoint_metadata, shift, obj) in crops.items():
            local_chunks = _create_chunk_list(obj)
            shifted_chunks = [
                ChunkStorageMetadata(
                    offsets=torch.Size(offset + delta for offset, delta in zip(chunk.offsets, shift)),
                    sizes=chunk.sizes,
                )
                for chunk in local_chunks
            ]

            for item in create_read_items_for_chunk_list(fqn, checkpoint_metadata, shifted_chunks):
                local_index = item.dest_index.index
                if local_index is None:
                    raise AssertionError("tensor read item has no local chunk index")
                items.append(
                    dataclasses.replace(
                        item,
                        # Tensor lookup uses the real, unshifted destination shard.
                        dest_index=MetadataIndex(
                            fqn,
                            local_chunks[local_index].offsets,
                            local_index,
                        ),
                    )
                )

        return dataclasses.replace(default_plan, items=items)


__all__ = ["WorldModelSuffixCropLoadPlanner"]
