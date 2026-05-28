from typing import Any

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper as ptd_checkpoint_wrapper
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

from torchtitan.config import ActivationCheckpointConfig, CompileConfig, ParallelismConfig, TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.fsdp import enable_fsdp_symm_mem
from torchtitan.tools.logging import logger


def parallelize_worldmodel(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    del dump_folder
    if parallel_dims.tp_enabled or parallel_dims.pp_enabled or parallel_dims.cp_enabled or parallel_dims.ep_enabled:
        raise NotImplementedError("worldmodel v1 supports FSDP/HSDP only; TP, PP, CP, and EP are not supported")

    if ac_config.mode != "none":
        apply_ac(model, ac_config)

    if compile_config.enable and "model" in compile_config.components:
        apply_compile(model, compile_config)

    dp_mesh = parallel_dims.get_activated_mesh(["dp_replicate", "fsdp"])
    assert dp_mesh is not None
    apply_fsdp(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        cpu_offload=training.enable_cpu_offload,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )
    return model


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cpu_offload: bool = False,
    enable_symm_mem: bool = False,
):
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    embds = [
        model.x_embedder,
        model.position_scale,
        model.euler_scale,
        model.augments_pos_ref_augment_embedder,
        model.ref_augment_from_augments_euler_embedder,
        model.pose_mask_embedder,
        model.t_embedder,
        model.fidx_embedder,
    ]
    for module in embds:
        fully_shard(module, **fsdp_config)

    for idx, block in enumerate(model.blocks):
        fully_shard(block, **fsdp_config, reshard_after_forward=(idx < len(model.blocks) - 1))

    if model.plan_head is not None:
        for idx, block in enumerate(model.plan_head.mlps):
            fully_shard(block, **fsdp_config, reshard_after_forward=(idx < len(model.plan_head.mlps) - 1))
        fully_shard(model.plan_head, **fsdp_config)

    if model.final_layer is not None:
        fully_shard(model.final_layer, **fsdp_config, reshard_after_forward=False)

    fully_shard(model, **fsdp_config)

    if enable_symm_mem:
        enable_fsdp_symm_mem(model)

    logger.info("Applied fully_shard to the worldmodel")

def apply_compile(model: nn.Module, compile_config: CompileConfig):
    embds = [
        model.x_embedder,
        model.position_scale,
        model.euler_scale,
        model.augments_pos_ref_augment_embedder,
        model.ref_augment_from_augments_euler_embedder,
        model.pose_mask_embedder,
        model.t_embedder,
        model.fidx_embedder,
    ]
    if model.final_layer is not None:
        embds.append(model.final_layer)
    for module in embds:
        module.compile(backend=compile_config.backend, fullgraph=True)
    for block in model.blocks:
        block.compile(backend=compile_config.backend, fullgraph=True)
    if model.plan_head is not None:
        for block in model.plan_head.mlps:
            block.compile(backend=compile_config.backend, fullgraph=True)

    logger.info("Compiled worldmodel blocks with torch.compile")


def apply_ac(model: nn.Module, ac_config: ActivationCheckpointConfig):
    for layer_id, block in model.blocks.named_children():
        wrapped = ptd_checkpoint_wrapper(
            block,
            preserve_rng_state=ac_config.preserve_rng_state,
        )
        model.blocks.register_module(layer_id, wrapped)

    if model.plan_head is not None:
        for layer_id, block in model.plan_head.mlps.named_children():
            wrapped = ptd_checkpoint_wrapper(block, preserve_rng_state=ac_config.preserve_rng_state)
            model.plan_head.mlps.register_module(layer_id, wrapped)

    logger.info("Applied activation checkpointing to the worldmodel")
