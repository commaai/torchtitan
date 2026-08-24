# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""FSDP/HSDP and torch.compile support for the Wan VAE."""

from __future__ import annotations

from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

from torchtitan.config import CompileConfig, ParallelismConfig, TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.fsdp import (
    disable_fsdp_gradient_division,
    enable_fsdp_symm_mem,
    get_fsdp_reshard_after_forward_policy,
)
from torchtitan.tools.logging import logger

from .model import AttentionBlock, WanVAE


def parallelize_wan_vae(
    model: WanVAE,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: Any,
    dump_folder: str,
) -> WanVAE:
    del dump_folder
    if parallelism.spmd_backend == "full_dtensor":
        raise ValueError("Wan VAE does not support full DTensor")
    if parallel_dims.tp_enabled or parallel_dims.cp_enabled or parallel_dims.pp_enabled or parallel_dims.ep_enabled:
        raise ValueError("Wan VAE supports FSDP/HSDP only")
    if ac_config is not None:
        # Streaming feature caches are mutated during forward. Replaying a block
        # with activation checkpointing would mutate those caches a second time.
        raise ValueError("Wan VAE streaming does not yet support activation checkpointing")

    if compile_config.enable and "model" in compile_config.components:
        _apply_compile(model, compile_config)

    dp_mesh = parallel_dims.get_activated_mesh(["dp_replicate", "fsdp"])
    if dp_mesh is None:
        dp_mesh = parallel_dims.get_mesh("fsdp")
    _apply_fsdp(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )
    hsdp_enabled = parallel_dims.dp_replicate_enabled and parallel_dims.fsdp_enabled
    logger.info("Applied HSDP to Wan VAE" if hsdp_enabled else "Applied FSDP to Wan VAE")
    return model


def _apply_compile(model: WanVAE, compile_config: CompileConfig) -> None:
    # The streaming wrappers mutate a Python feature-cache list between causal
    # chunks. Dynamo does not reliably publish those mutations across an FSDP
    # boundary, so keep the small cache controller eager and compile the pure,
    # compute-heavy kernels beneath it. Compilation remains in place and does
    # not add a state-dict prefix.
    compiled = 0
    for module in (model.conv1, model.conv2):
        module.compile(
            backend=compile_config.backend,
            fullgraph=True,
            dynamic=True,
        )
        compiled += 1
    for module in model.modules():
        if isinstance(module, AttentionBlock):
            module.compile(
                backend=compile_config.backend,
                fullgraph=True,
                dynamic=True,
            )
            compiled += 1
    logger.info("Compiling %d Wan VAE latent-projection and attention kernels", compiled)


def _apply_fsdp(
    model: WanVAE,
    dp_mesh: DeviceMesh,
    *,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool,
    reshard_after_forward_policy: str,
    enable_symm_mem: bool,
) -> None:
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=True,
    )
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy,
        pp_enabled,
    )

    # Keep the causal cache inside one FSDP forward boundary. Nested FSDP hooks
    # rebuild argument pytrees and would give each stage a copy of the mutable
    # cache list, losing writes before the following temporal chunk.
    fully_shard(
        model,
        **fsdp_config,
        reshard_after_forward=reshard_after_forward,
    )

    if enable_symm_mem:
        enable_fsdp_symm_mem(model)
    disable_fsdp_gradient_division(model)
