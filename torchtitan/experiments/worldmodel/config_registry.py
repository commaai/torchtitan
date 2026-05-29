from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.quantization import Float8LinearConverter
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments.worldmodel.dataloader import WorldModelDataLoader
from torchtitan.experiments.worldmodel.loss import WorldModelLoss
from torchtitan.experiments.worldmodel.trainer import WorldModelTrainer
from torchtitan.experiments.worldmodel.validate import WorldModelValidator

from . import model_registry

WORLD_MODEL_FLOAT8_FILTER_FQNS = [
    "x_embedder",
    "augments_pos_ref_augment_embedder",
    "ref_augment_from_augments_euler_embedder",
    "pose_mask_embedder",
    "t_embedder",
    "fidx_embedder",
    "final_layer",
    "plan_head",
]


def _blocks_only_float8(*, emulate: bool = False, model_compile_enabled: bool = False) -> Float8LinearConverter.Config:
    return Float8LinearConverter.Config(
        emulate=emulate,
        filter_fqns=WORLD_MODEL_FLOAT8_FILTER_FQNS,
        enable_fsdp_float8_all_gather=True,
        precompute_float8_dynamic_scale_for_fsdp=True,
        model_compile_enabled=model_compile_enabled,
    )


def _base_dataloader(
    *,
    infinite: bool = True,
    num_workers: int = 2,
    persistent_workers: bool = True,
    prefetch_factor: int | None = 16,
    skip: int = 100,
) -> WorldModelDataLoader.Config:
    return WorldModelDataLoader.Config(
        infinite=infinite,
        mock_data=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        multiprocessing_context="spawn",
        in_order=False,
        in_channels=32,
        latent_size=(16, 32),
        context_size_frames=10,
        future_size_frames=5,
        max_future_frames=50,
        inference_conditioning_frames=14,
        fps=5,
        skip=skip,
    )


def worldmodel() -> WorldModelTrainer.Config:
    dataloader = _base_dataloader()
    return WorldModelTrainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        loss=WorldModelLoss.Config(plan_loss_weight=0.1),
        model_spec=model_registry("base", converters=[_blocks_only_float8(model_compile_enabled=True)]),
        optimizer=OptimizersContainer.Config(
            name="AdamW",
            lr=5e-4,
            beta1=0.9,
            beta2=0.95,
            weight_decay=1e-2,
            implementation="fused_opt_states_bf16",
        ),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=512, decay_ratio=0.1, decay_type="cosine"),
        training=TrainingConfig(
            local_batch_size=16,
            steps=512 * 100,
            dtype="float32",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="bfloat16",
            max_norm=1.0,
        ),
        dataloader=dataloader,
        metrics=MetricsProcessor.Config(enable_reporterv2=True, log_freq=1, save_freq=512),
        # TODO: 16/8 hard coded?
        parallelism=ParallelismConfig(data_parallel_replicate_degree=32, data_parallel_shard_degree=8),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        compile=CompileConfig(enable=True, components=["model", "loss"]),
        checkpoint=CheckpointManager.Config(
            enable=True,
            interval=512 * 10,
            storage_backend="reporterv2",
            async_mode="async",
            exclude_from_saving=[
                "optimizer",
                "lr_scheduler",
                "dataloader",
                "train_state",
            ],
            exclude_from_loading=[
                "optimizer",
                "lr_scheduler",
                "dataloader",
                "train_state",
            ],
            enable_first_step_checkpoint=True,
        ),
        validator=WorldModelValidator.Config(
            enable=True,
            freq=512,
            steps=8,
            cache_data=True,
            dataloader=_base_dataloader(
                infinite=False,
                persistent_workers=False,
                prefetch_factor=2,
            ),
        ),
    )
