# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torchtitan.models.llama3 as llama3
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
    TrainingConfig,
)
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer

from .dataset import ByteTokenizer, ShakespeareDataLoader
from .validate import ShakespeareValidator


_VOCAB_SIZE = 256
_LLAMA3_FLAVOR = "debugmodel"


def _adapt_llama3_config_for_shakespeare(
    config: llama3.Llama3Model.Config,
) -> llama3.Llama3Model.Config:
    config.vocab_size = _VOCAB_SIZE
    config.enable_weight_tying = True
    config.tok_embeddings.num_embeddings = _VOCAB_SIZE
    config.tok_embeddings.param_init = llama3._EMBEDDING_SKIP_INIT
    config.lm_head.out_features = _VOCAB_SIZE
    return config


def _model_config(
    *,
    attn_backend: str = "sdpa",
) -> llama3.Llama3Model.Config:
    llama3_spec = llama3.model_registry(_LLAMA3_FLAVOR, attn_backend=attn_backend)
    config = llama3_spec.model
    assert isinstance(config, llama3.Llama3Model.Config)
    return _adapt_llama3_config_for_shakespeare(config)


def model_registry(
    flavor: str = "default",
    *,
    attn_backend: str = "sdpa",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    if flavor != "default":
        raise ValueError(f"Unsupported Shakespeare flavor {flavor!r}")
    llama3_spec = llama3.model_registry(_LLAMA3_FLAVOR, attn_backend=attn_backend)
    config = llama3_spec.model
    assert isinstance(config, llama3.Llama3Model.Config)
    config = _adapt_llama3_config_for_shakespeare(config)
    if converters is not None:
        validate_converter_order(converters)
        for converter in converters:
            converter.build().convert(config)
    return ModelSpec(
        name="shakespeare",
        flavor=flavor,
        model=config,
        parallelize_fn=llama3_spec.parallelize_fn,
        pipelining_fn=llama3_spec.pipelining_fn,
        post_optimizer_build_fn=llama3_spec.post_optimizer_build_fn,
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
