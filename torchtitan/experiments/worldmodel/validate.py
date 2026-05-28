from dataclasses import dataclass, field, replace
from typing import cast

import torch
import torch.nn as nn
from torch.distributed.pipelining.schedules import _PipelineSchedule

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.loss import LossFunction
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.components.validate import ValidationContext, Validator
from torchtitan.config import ParallelismConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.experiments.worldmodel.compressor import load_compressor_encoder
from torchtitan.experiments.worldmodel.dataloader import WorldModelDataLoader
from torchtitan.experiments.worldmodel.model import WorldModel
from torchtitan.experiments.worldmodel.schedulers import RFScheduler
from torchtitan.experiments.worldmodel.step import compute_worldmodel_losses, prepare_worldmodel_batch


class WorldModelValidator(Validator):
    @dataclass(kw_only=True, slots=True)
    class Config(Validator.Config):
        dataloader: BaseDataLoader.Config = field(default_factory=lambda: WorldModelDataLoader.Config(mock_data=True, infinite=False))
        pose_dropout: float = 0.0
        plan_loss_weight: float = 0.1
        noise_scheduler_steps: int = 10

    def __init__(
        self,
        config: Config,
        *,
        parallelism: ParallelismConfig,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        parallel_dims: ParallelDims,
        loss_fn: LossFunction,
        validation_context: ValidationContext,
        local_batch_size: int,
        metrics_processor: MetricsProcessor,
        pp_schedule: _PipelineSchedule | None = None,
        pp_has_first_stage: bool | None = None,
        pp_has_last_stage: bool | None = None,
        **kwargs,
    ):
        del parallelism, loss_fn, pp_schedule, pp_has_first_stage, pp_has_last_stage
        del kwargs
        self.config = config
        self.tokenizer = tokenizer
        self.parallel_dims = parallel_dims
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.local_batch_size = local_batch_size
        self.validation_context = validation_context
        self.metrics_processor = metrics_processor
        self.dl_config = replace(config.dataloader, infinite=config.steps != -1)
        self.compressor_encoder: torch.nn.Module | None = None

    def _load_compressor_encoder(self, model: WorldModel, device: torch.device) -> torch.nn.Module:
        if self.compressor_encoder is not None:
            return self.compressor_encoder
        dtype = next(param.dtype for param in model.parameters() if param.is_floating_point())
        self.compressor_encoder = load_compressor_encoder(
            compressor_model=self.dl_config.compressor_model,
            encoder_path=self.dl_config.compressor_encoder_path,
            device=device,
            dtype=dtype,
        )
        return self.compressor_encoder

    @torch.no_grad()
    def validate(self, model_parts: list[nn.Module], step: int) -> None:
        model = cast(WorldModel, model_parts[0])
        model.eval()

        scheduler = RFScheduler(steps=self.config.noise_scheduler_steps).to(device=next(model.parameters()).device)
        discrete_timesteps = scheduler.timesteps[:-1]

        validation_dataloader = self.dl_config.build(
            dp_world_size=self.dp_world_size,
            dp_rank=self.dp_rank,
            tokenizer=self.tokenizer,
            local_batch_size=self.local_batch_size,
            val=True,
        )

        losses = []
        num_steps = 0
        device = next(model.parameters()).device
        compressor_encoder = self._load_compressor_encoder(model, device)
        for input_dict, targets in validation_dataloader:
            if self.config.steps != -1 and num_steps >= self.config.steps:
                break

            model_inputs, targets = prepare_worldmodel_batch(
                model,
                input_dict,
                targets,
                device=device,
                scheduler=scheduler,
                discrete_timesteps=discrete_timesteps,
                compressor_encoder=compressor_encoder,
                pose_dropout=self.config.pose_dropout,
                train=False,
            )
            with self.validation_context():
                outputs = model(**model_inputs)
                per_sample_loss, _terms = compute_worldmodel_losses(outputs, targets, batch_size=model_inputs["x"].shape[0], plan_loss_weight=self.config.plan_loss_weight)
            losses.append(per_sample_loss.mean().detach())
            num_steps += 1

        if losses:
            loss = torch.stack(losses).mean()
            if self.parallel_dims.dp_cp_enabled:
                loss = torch.tensor(dist_utils.dist_mean(loss, self.parallel_dims.get_optional_mesh("loss")), device=device)
            self.metrics_processor.log_validation(loss=float(loss.item()), step=step)

        model.train()
