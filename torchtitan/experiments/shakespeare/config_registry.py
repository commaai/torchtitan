# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import torch
import torch.nn as nn
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Placement, Replicate, Shard

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.onnx_checkpoint import OnnxCheckpointManager
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.quantization import Float8LinearConverter
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.context_parallel import apply_cp_to_forward
from torchtitan.distributed.fsdp import (
    disable_fsdp_gradient_division,
    enable_fsdp_symm_mem,
    get_fsdp_reshard_after_forward_policy,
)
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.models.common import (
    compute_ffn_hidden_dim,
    Decoder,
    Embedding,
    Linear,
    RMSNorm,
    RoPE,
    TransformerBlock,
)
from torchtitan.models.common.attention import AttentionMasksType, VarlenAttention
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_ffn_config,
    make_gqa_config,
)
from torchtitan.models.common.decoder_sharding import (
    norm_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
    set_gqa_attention_sharding,
    set_gqa_inner_attention_local_map,
)
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.utils import (
    get_dense_model_nparams_and_flops,
    validate_converter_order,
)
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer

from .dataset import ByteTokenizer, ShakespeareDataLoader
from .validate import ShakespeareValidator


_VOCAB_SIZE = 256
_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=0.02)}


class ShakespeareTransformerBlock(TransformerBlock):
    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        assert config.feed_forward is not None
        self.feed_forward = config.feed_forward.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        h = x + self.attention(
            self.attention_norm(x), freqs_cis, attention_masks, positions
        )
        return h + self.feed_forward(self.ffn_norm(h))


class ShakespeareModel(Decoder):
    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 384
        vocab_size: int = _VOCAB_SIZE

        def update_from_config(self, *, config, **kwargs) -> None:
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

            if parallelism.context_parallel_degree > 1 and isinstance(
                self.layers[0].attention.inner_attention, VarlenAttention.Config
            ):
                raise NotImplementedError(
                    "Context Parallel only supports SDPA and FlexAttention. "
                    "Varlen attention is not supported with CP."
                )

            _set_shakespeare_sharding_config(
                self,
                loss_parallel=not parallelism.disable_loss_parallel,
                enable_sp=parallelism.enable_sequence_parallel,
            )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            return get_dense_model_nparams_and_flops(
                model,
                n_layers=len(self.layers),
                n_heads=self.layers[0].attention.n_heads,
                head_dims=2 * (self.dim // self.layers[0].attention.n_heads),
                seq_len=seq_len,
            )


def _set_shakespeare_sharding_config(
    config: ShakespeareModel.Config,
    *,
    loss_parallel: bool,
    enable_sp: bool,
) -> None:
    set_decoder_sharding_config(
        config,
        loss_parallel=loss_parallel,
        enable_sp=enable_sp,
    )
    for layer_cfg in config.layers:
        norm = norm_config(enable_sp=enable_sp)
        layer_cfg.attention_norm.sharding_config = norm
        layer_cfg.ffn_norm.sharding_config = norm
        attn_x_placement: Placement = Shard(1) if enable_sp else Replicate()

        set_gqa_attention_sharding(layer_cfg.attention, enable_sp=enable_sp)
        set_gqa_inner_attention_local_map(layer_cfg.attention.inner_attention)

        assert layer_cfg.feed_forward is not None
        set_dense_ffn_sharding(
            layer_cfg.feed_forward,
            attn_x_placement=attn_x_placement,
            enable_sp=enable_sp,
        )


def parallelize_shakespeare(
    model: ShakespeareModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    if parallelism.full_dtensor:
        raise ValueError("Shakespeare experiment does not support full_dtensor yet")

    _validate_no_gradient_accumulation(training, parallel_dims)

    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    if parallel_dims.cp_enabled:
        layers = cast(Any, model.layers)
        apply_cp_to_forward(
            [block.attention.inner_attention for block in layers.values()],
            parallel_dims.get_mesh("cp"),
        )
    if parallel_dims.tp_enabled:
        model.parallelize(parallel_dims)
        maybe_enable_async_tp(
            parallelism,
            compile_config,
            parallel_dims.get_mesh("tp"),
        )

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )
    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            base_folder=dump_folder,
        )
    if model_compile_enabled:
        apply_compile(model, compile_config)

    names = ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    dp_mesh = parallel_dims.get_mesh(names)
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

    if parallel_dims.dp_replicate_enabled:
        logger.info("Applied HSDP to the Shakespeare model")
    else:
        logger.info("Applied FSDP to the Shakespeare model")
    return model


def _validate_no_gradient_accumulation(
    training: TrainingConfig,
    parallel_dims: ParallelDims,
) -> None:
    if training.global_batch_size < 0:
        return

    batch_degree = (
        parallel_dims.dp_replicate * parallel_dims.dp_shard
        if parallel_dims.dp_enabled
        else 1
    )
    expected_global_batch_size = training.local_batch_size * batch_degree
    if training.global_batch_size != expected_global_batch_size:
        raise ValueError(
            "Shakespeare experiment does not support gradient accumulation. "
            "Leave training.global_batch_size=-1 or set it to "
            f"training.local_batch_size * data-parallel degree "
            f"({expected_global_batch_size}); got {training.global_batch_size}."
        )


def _apply_fsdp(
    model: ShakespeareModel,
    dp_mesh,
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
        cast_forward_inputs=False,
    )
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy,
        pp_enabled,
    )

    if model.tok_embeddings is not None:
        fully_shard(
            cast(nn.Module, model.tok_embeddings),
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    if model.norm is not None and model.lm_head is not None:
        fully_shard(
            [cast(nn.Module, model.norm), cast(nn.Module, model.lm_head)],
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )
    layers = cast(Any, model.layers)
    for transformer_block in layers.values():
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    fully_shard(model, **fsdp_config)

    if enable_symm_mem:
        enable_fsdp_symm_mem(model)
    disable_fsdp_gradient_division(model)


def _output_linear_init(dim: int) -> dict[str, Callable]:
    std = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=std, a=-3 * std, b=3 * std),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _build_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    hidden_dim: int,
    attn_backend: str,
) -> list[TransformerBlock.Config]:
    inner_attention, mask_type = get_attention_config(attn_backend)
    return [
        ShakespeareTransformerBlock.Config(
            attention_norm=RMSNorm.Config(
                normalized_shape=dim,
                param_init=_NORM_INIT,
            ),
            ffn_norm=RMSNorm.Config(
                normalized_shape=dim,
                param_init=_NORM_INIT,
            ),
            attention=make_gqa_config(
                dim=dim,
                n_heads=n_heads,
                wqkv_param_init=_LINEAR_INIT,
                wo_param_init=_depth_init(layer_id),
                inner_attention=inner_attention,
                mask_type=mask_type,
                rope_backend="complex",
            ),
            feed_forward=make_ffn_config(
                dim=dim,
                hidden_dim=hidden_dim,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_depth_init(layer_id),
            ),
        )
        for layer_id in range(n_layers)
    ]


def _model_config(
    *,
    dim: int = 384,
    n_layers: int = 6,
    n_heads: int = 6,
    attn_backend: str = "sdpa",
) -> ShakespeareModel.Config:
    return ShakespeareModel.Config(
        dim=dim,
        vocab_size=_VOCAB_SIZE,
        tok_embeddings=Embedding.Config(
            num_embeddings=_VOCAB_SIZE,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=_VOCAB_SIZE,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=4096,
            theta=10000,
            backend="complex",
            scaling="none",
        ),
        layers=_build_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            hidden_dim=compute_ffn_hidden_dim(dim, multiple_of=256),
            attn_backend=attn_backend,
        ),
    )


def model_registry(
    flavor: str = "default",
    *,
    attn_backend: str = "sdpa",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    if flavor != "default":
        raise ValueError(f"Unsupported Shakespeare flavor {flavor!r}")
    config = _model_config(attn_backend=attn_backend)
    if converters is not None:
        validate_converter_order(converters)
        for converter in converters:
            converter.build().convert(config)
    return ModelSpec(
        name="shakespeare",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_shakespeare,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )


def _node_count() -> int:
    return int(os.environ.get("NNODES") or os.environ.get("SLURM_NNODES") or "1")


def shakespeare() -> Trainer.Config:
    seq_len = 256
    compile_config = CompileConfig(enable=True)

    return Trainer.Config(
        loss=CrossEntropyLoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry(
            converters=[
                Float8LinearConverter.Config(
                    model_compile_enabled=(
                        compile_config.enable and "model" in compile_config.components
                    ),
                )
            ],
        ),
        tokenizer=ByteTokenizer.Config(),
        dataloader=ShakespeareDataLoader.Config(
            dataset="train",
        ),
        optimizer=OptimizersContainer.Config(
            lr=3e-4,
            beta1=0.9,
            beta2=0.95,
            weight_decay=0.1,
            implementation="fused_opt_states_bf16",
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=50,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=32,
            seq_len=seq_len,
            steps=1000,
            max_norm=1.0,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=_node_count(),
            data_parallel_shard_degree=8,
        ),
        checkpoint=OnnxCheckpointManager.Config(
            enable=False,
            interval=500,
            last_save_model_only=False,
            input_names=["tokens"],
            output_names=["logits"],
            input_shapes=[[1, seq_len]],
            input_dtypes=["int64"],
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
        compile=compile_config,
        metrics=MetricsProcessor.Config(
            log_freq=10,
            enable_reporterv2=True,
        ),
        validator=ShakespeareValidator.Config(
            enable=True,
            freq=100,
            steps=10,
            dataloader=ShakespeareDataLoader.Config(
                dataset="val",
                shuffle_size=2048,
                min_mixing=0.25,
            ),
        ),
        debug=DebugConfig(seed=42),
    )
