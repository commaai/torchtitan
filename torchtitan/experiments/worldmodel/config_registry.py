from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import MSELoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.quantization import Float8LinearConverter
from torchtitan.config import ActivationCheckpointConfig, CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.experiments.worldmodel.dataloader import WorldModelDataLoader
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
        model_compile_enabled=model_compile_enabled,
    )


def _debug_dataloader(*, infinite: bool = True) -> WorldModelDataLoader.Config:
    return WorldModelDataLoader.Config(
        infinite=infinite,
        mock_data=True,
        mock_segment_batch_size=8,
        num_workers=2,
        prefetch_factor=16,
        persistent_workers=True,
        multiprocessing_context="spawn",
        in_channels=32,
        latent_size=(16, 32),
        context_size_frames=10,
        future_size_frames=5,
        max_future_frames=15,
        inference_conditioning_frames=14,
        train_skip=1,
        val_skip=1,
        compressor_model="dummy-compressor",
        image_size=(32, 64),
    )


def _base_dataloader(*, infinite: bool = True) -> WorldModelDataLoader.Config:
    return WorldModelDataLoader.Config(
        infinite=infinite,
        mock_data=False,
        num_workers=4,
        prefetch_factor=16,
        persistent_workers=True,
        multiprocessing_context="spawn",
        in_order=False,
        in_channels=32,
        latent_size=(16, 32),
        context_size_frames=10,
        future_size_frames=5,
        max_future_frames=50,
        inference_conditioning_frames=14,
        fps=5,
        train_skip=100,
        val_skip=800,
        compressor_model="4672da0d-19f5-44f8-a5fb-2215981c9c0e",
    )


def worldmodel_debugmodel() -> WorldModelTrainer.Config:
    dataloader = _debug_dataloader()
    return WorldModelTrainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        loss=MSELoss.Config(),
        model_spec=model_registry("debug"),
        optimizer=OptimizersContainer.Config(
            name="AdamW",
            lr=8e-4,
            beta1=0.9,
            beta2=0.95,
            weight_decay=1e-2,
            implementation="fused_opt_states_bf16",
        ),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=1, decay_ratio=0.0),
        training=TrainingConfig(
            local_batch_size=2,
            steps=10,
            dtype="float32",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
            max_norm=1.0,
        ),
        dataloader=dataloader,
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        checkpoint=CheckpointManager.Config(enable=False, interval=10, last_save_model_only=False),
        validator=WorldModelValidator.Config(
            enable=True,
            freq=5,
            steps=1,
            dataloader=_debug_dataloader(infinite=False),
            no_noise_conditioning_frames_prob=1.0,
            fake_timesteps_prob=0.0,
        ),
    )


def worldmodel() -> WorldModelTrainer.Config:
    dataloader = _base_dataloader()
    return WorldModelTrainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        loss=MSELoss.Config(),
        model_spec=model_registry("base"),
        optimizer=OptimizersContainer.Config(
            name="AdamW",
            lr=3e-4,
            beta1=0.9,
            beta2=0.95,
            weight_decay=1e-1,
            implementation="fused_opt_states_bf16",
        ),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=128, decay_ratio=0.1, decay_type="cosine"),
        training=TrainingConfig(
            local_batch_size=16,
            steps=512 * 10,
            dtype="float32",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
            max_norm=1.0,
        ),
        dataloader=dataloader,
        metrics=MetricsProcessor.Config(log_freq=100),
        parallelism=ParallelismConfig(data_parallel_replicate_degree=1, data_parallel_shard_degree=-1),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        compile=CompileConfig(enable=True),
        checkpoint=CheckpointManager.Config(enable=False, interval=1000, last_save_model_only=False),
        validator=WorldModelValidator.Config(
            enable=False,
            freq=1000,
            steps=10,
            dataloader=_base_dataloader(infinite=False),
        ),
    )


def worldmodel_hsdp() -> WorldModelTrainer.Config:
    config = worldmodel()
    config.parallelism = ParallelismConfig(data_parallel_replicate_degree=2, data_parallel_shard_degree=-1)
    return config


def worldmodel_debugmodel_float8() -> WorldModelTrainer.Config:
    config = worldmodel_debugmodel()
    config.model_spec = model_registry(
        "debug",
        converters=[_blocks_only_float8(emulate=True)],
    )
    return config


def worldmodel_float8() -> WorldModelTrainer.Config:
    config = worldmodel()
    config.compile = CompileConfig(enable=True)
    model_compile_enabled = config.compile.enable and "model" in config.compile.components
    config.model_spec = model_registry(
        "base",
        converters=[_blocks_only_float8(model_compile_enabled=model_compile_enabled)],
    )
    return config
