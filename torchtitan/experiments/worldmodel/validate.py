from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Any, cast

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
from torchtitan.experiments.worldmodel.loss import WorldModelLoss
from torchtitan.experiments.worldmodel.model import WorldModel
from torchtitan.experiments.worldmodel.schedulers import RFScheduler
from torchtitan.experiments.worldmodel.step import prepare_worldmodel_batch
from torchtitan.tools.logging import logger

WorldModelBatch = tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]


def _clone_batch_to_cpu(batch: WorldModelBatch) -> WorldModelBatch:
    input_dict, targets = batch
    return (
        {key: value.detach().cpu().clone() for key, value in input_dict.items()},
        {key: value.detach().cpu().clone() for key, value in targets.items()},
    )


class WorldModelValidator(Validator):
    @dataclass(kw_only=True, slots=True)
    class Config(Validator.Config):
        dataloader: BaseDataLoader.Config = field(
            default_factory=lambda: WorldModelDataLoader.Config(
                mock_data=True,
                infinite=False,
            )
        )
        pose_dropout: float = 0.0
        noise_scheduler_steps: int = 10
        cache_data: bool = True
        extra_methods: list[str] = field(default_factory=lambda: ["html_report"])

        def __post_init__(self) -> None:
            if self.steps < 0:
                raise ValueError("worldmodel validation steps must be >= 0")

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
        seq_len: int | None = None,
        pp_schedule: _PipelineSchedule | None = None,
        pp_has_first_stage: bool | None = None,
        pp_has_last_stage: bool | None = None,
        **kwargs,
    ):
        del parallelism, pp_schedule, pp_has_first_stage, pp_has_last_stage
        del kwargs
        self.config = config
        self.loss_fn = cast(WorldModelLoss, loss_fn)
        self.tokenizer = tokenizer
        self.parallel_dims = parallel_dims
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.local_batch_size = local_batch_size
        self.seq_len = seq_len
        self.validation_context = validation_context
        self.metrics_processor = metrics_processor
        self.dl_config = replace(config.dataloader, infinite=False)
        if isinstance(self.dl_config, WorldModelDataLoader.Config):
            self.dl_config = replace(self.dl_config, persistent_workers=False)
        self.compressor_encoder: torch.nn.Module | None = None
        self._dtype: torch.dtype | None = None
        self._cached_batches: list[WorldModelBatch] = []

    def _load_compressor_encoder(
        self,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.nn.Module:
        if self.compressor_encoder is not None:
            return self.compressor_encoder
        self.compressor_encoder = load_compressor_encoder(
            compressor_model=self.dl_config.compressor_model,
            device=device,
            dtype=dtype,
        )
        return self.compressor_encoder

    def _iter_validation_batches(self) -> Iterator[WorldModelBatch]:
        if self.config.steps == 0:
            logger.info("worldmodel validation skipped because steps=0")
            return
        if self.config.cache_data and self._cached_batches:
            logger.info(
                f"worldmodel validation reusing {len(self._cached_batches)} cached CPU batches"
            )
            yield from self._cached_batches
            return
        if self.config.cache_data:
            logger.info(
                f"worldmodel validation filling CPU batch cache with up to {self.config.steps} batches"
            )
        else:
            logger.info(
                f"worldmodel validation building dataloader for {self.config.steps} batches without cache"
            )
        validation_dataloader = self.dl_config.build(
            dp_world_size=self.dp_world_size,
            dp_rank=self.dp_rank,
            tokenizer=self.tokenizer,
            local_batch_size=self.local_batch_size,
            val=True,
        )
        validation_iterator = iter(validation_dataloader)
        num_steps = 0
        try:
            for step_idx, batch in zip(range(self.config.steps), validation_iterator):
                if self.config.cache_data:
                    batch = _clone_batch_to_cpu(batch)
                    self._cached_batches.append(batch)
                    logger.info(
                        f"worldmodel validation cached CPU batch {len(self._cached_batches)}/{self.config.steps}"
                    )
                num_steps = step_idx + 1
                yield batch
        finally:
            del validation_iterator
            del validation_dataloader
            logger.info(
                f"worldmodel validation dataloader released after {num_steps} batches"
            )
        if self.config.cache_data:
            logger.info(
                f"worldmodel validation cache fill complete: {len(self._cached_batches)} batches"
            )

    def _run_metrics_validation(
        self,
        model: WorldModel,
        step: int,
    ) -> tuple[float, dict[str, Any]] | None:
        if self.config.steps == 0:
            return None

        scheduler = RFScheduler(steps=self.config.noise_scheduler_steps).to(
            device=next(model.parameters()).device
        )
        discrete_timesteps = scheduler.timesteps[:-1]

        term_sums: dict[str, torch.Tensor] = {}
        num_steps = 0
        val_tokens = 0
        device = next(model.parameters()).device
        compressor_encoder = self._load_compressor_encoder(self._dtype, device)
        for input_dict, targets in self._iter_validation_batches():
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
                dtype=self._dtype,
            )
            with self.validation_context():
                outputs = model(**model_inputs)
                _per_sample_loss, terms = self.loss_fn(outputs, targets)
            for name, term in terms.items():
                term_sums[name] = term_sums.get(
                    name,
                    torch.zeros((), device=device),
                ) + term.mean().detach()
            bsz = next(iter(input_dict.values())).shape[0]
            val_tokens += bsz * (self.seq_len or 1)
            num_steps += 1

        if num_steps == 0:
            return None

        metrics = {name: value / num_steps for name, value in term_sums.items()}
        loss = metrics["loss"]
        if self.parallel_dims.dp_cp_enabled:
            loss_value = dist_utils.dist_mean(
                loss,
                self.parallel_dims.get_optional_mesh("loss"),
            )
            global_val_tokens = dist_utils.dist_sum(
                torch.tensor(val_tokens, dtype=torch.int64, device=device),
                self.parallel_dims.get_optional_mesh("loss"),
            )
        else:
            loss_value = float(loss.item())
            global_val_tokens = val_tokens

        extra_metrics: dict[str, Any] = {
            "data/n_tokens_seen/val": global_val_tokens,
            "dataloader/val_cache_size/": len(self._cached_batches),
        }
        self.metrics_processor.ntokens_since_last_log += val_tokens
        for name, term in metrics.items():
            if name == "loss":
                continue
            if self.parallel_dims.dp_cp_enabled:
                term_value = dist_utils.dist_mean(
                    term,
                    self.parallel_dims.get_optional_mesh("loss"),
                )
            else:
                term_value = float(term.item())
            extra_metrics[f"worldmodel/{name}/val"] = term_value

        self.metrics_processor.log_validation(
            loss=loss_value,
            step=step,
            extra_metrics=extra_metrics,
        )
        return loss_value, extra_metrics

    def _run_html_report(
        self,
        step: int,
        result: tuple[float, dict[str, Any]] | None,
    ) -> None:
        if result is None:
            return
        loss, extra_metrics = result
        rows = "".join(
            f"<li>{name}={value:.4f}</li>"
            for name, value in sorted(extra_metrics.items())
            if isinstance(value, float)
        )
        html = f"<h1>Epoch {step}</h1><p>val_loss={loss:.4f}</p><ul>{rows}</ul>"
        self.metrics_processor.write_report( html, step, "report", "html")

    @torch.no_grad()
    def validate(self, model_parts: list[nn.Module], step: int) -> None:
        model = cast(WorldModel, model_parts[0])
        model.eval()
        logger.info(
            "worldmodel validation start: "
            f"step={step}, steps={self.config.steps}, cache_data={self.config.cache_data}, "
            f"cached_batches={len(self._cached_batches)}, extra_methods={self.config.extra_methods}"
        )
        try:
            result = self._run_metrics_validation(model, step)

            if "html_report" in self.config.extra_methods:
                self._run_html_report(step, result)
        finally:
            model.train()
            logger.info(f"worldmodel validation finished: step={step}")
