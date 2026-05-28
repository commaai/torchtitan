from typing import Any

from torchtitan.experiments.worldmodel.dataloader import WorldModelDataLoader
from torchtitan.experiments.worldmodel.model import WorldModel


def validate_and_finalize_worldmodel_config(config: Any) -> None:
    if not isinstance(config.dataloader, WorldModelDataLoader.Config):
        raise TypeError("worldmodel requires WorldModelDataLoader.Config")
    if config.model_spec is None:
        raise ValueError("worldmodel requires a model_spec")

    model_config = config.model_spec.model
    if not isinstance(model_config, WorldModel.Config):
        raise TypeError("worldmodel model_spec must contain WorldModel.Config")

    total_frames = config.dataloader.context_size_frames + config.dataloader.future_size_frames
    if model_config.input_size[0] != total_frames:
        raise ValueError(
            "model.input_size[0] must equal dataloader context + future frames "
            f"({model_config.input_size[0]} != {total_frames})"
        )
    if config.dataloader.inference_conditioning_frames > total_frames:
        raise ValueError("inference_conditioning_frames must fit in input frames")
    if model_config.input_size[1:] != config.dataloader.latent_size:
        raise ValueError("model.input_size spatial dimensions must match dataloader.latent_size")
    if model_config.in_channels != config.dataloader.in_channels:
        raise ValueError("model.in_channels must match dataloader.in_channels")
    if (
        config.parallelism.tensor_parallel_degree > 1
        or config.parallelism.pipeline_parallel_degree > 1
        or config.parallelism.context_parallel_degree > 1
        or config.parallelism.expert_parallel_degree > 1
    ):
        raise NotImplementedError("worldmodel v1 supports FSDP/HSDP only; TP, PP, CP, and EP are not supported")

    model_config._sync_derived_fields()
    config.training.seq_len = model_config.num_patches
