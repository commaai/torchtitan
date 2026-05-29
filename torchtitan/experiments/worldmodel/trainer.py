import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import cast
import torch
from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.worldmodel.compressor import load_compressor_encoder
from torchtitan.experiments.worldmodel.config import validate_and_finalize_worldmodel_config
from torchtitan.experiments.worldmodel.dataloader import WorldModelDataLoader
from torchtitan.experiments.worldmodel.loss import WorldModelLoss
from torchtitan.experiments.worldmodel.model import WorldModel
from torchtitan.experiments.worldmodel.schedulers import RFScheduler
from torchtitan.experiments.worldmodel.step import prepare_worldmodel_batch
from torchtitan.experiments.worldmodel.validate import WorldModelValidator
from torchtitan.observability import structured_logger as sl
from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer

class WorldModelTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        dataloader: WorldModelDataLoader.Config = field(default_factory=WorldModelDataLoader.Config)  # pyrefly: ignore [bad-override]
        pose_dropout: float = 0.1
        noise_scheduler_steps: int = 10

        def __post_init__(self) -> None:
            Trainer.Config.__post_init__(self)
            validate_and_finalize_worldmodel_config(self)

    def __init__(self, config: Config):
        self._last_loss_terms: dict[str, torch.Tensor] = {}
        super().__init__(config)
        dist_utils.set_determinism(self.parallel_dims, self.device,config.debug, distinct_seed_mesh_dims=["fsdp", "dp_replicate"])
        self._dtype = TORCH_DTYPE_MAP[config.training.mixed_precision_param] if self.parallel_dims.dp_shard_enabled else TORCH_DTYPE_MAP[config.training.dtype]
        self.train_noise_scheduler = RFScheduler(steps=config.noise_scheduler_steps).to(device=self.device)
        self.discrete_timesteps = self.train_noise_scheduler.timesteps[:-1]
        self.compressor_encoder = load_compressor_encoder(
            compressor_model=config.dataloader.compressor_model,
            device=self.device,
            dtype=self._dtype,
        )
        if config.validator.enable:
            validator = cast(WorldModelValidator, self.validator)
            validator.compressor_encoder = self.compressor_encoder
            validator._dtype = self._dtype
            logger.info(f"Reusing trainer compressor encoder for worldmodel validation with dtype {self._dtype}")

    def batch_generator(self, data_iterable: Iterable[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]) -> Iterator[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]:
        data_iterator = iter(data_iterable)
        while True:
            data_load_start = time.perf_counter()
            try:
                batch = next(data_iterator)
            except StopIteration as ex:
                raise DataloaderExhaustedError() from ex
            input_dict, targets = batch
            bsz = input_dict["imgs"].shape[0]
            ntokens_batch = bsz * self.config.training.seq_len
            self.metrics_processor.ntokens_since_last_log += ntokens_batch
            self.metrics_processor.data_loading_times.append(time.perf_counter() - data_load_start)
            yield input_dict, targets

    def forward_backward_step(self, *, input_dict: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> torch.Tensor:
        assert len(self.model_parts) == 1
        model = cast(WorldModel, self.model_parts[0])

        with sl.log_trace_span("worldmodel_prepare_batch"):
            model_inputs, targets = prepare_worldmodel_batch(
                model,
                input_dict,
                targets,
                device=self.device,
                scheduler=self.train_noise_scheduler,
                discrete_timesteps=self.discrete_timesteps,
                compressor_encoder=self.compressor_encoder,
                pose_dropout=self.config.pose_dropout,
                train=True,
                dtype=self._dtype,
            )

        bsz = model_inputs["x"].shape[0]
        self.ntokens_seen += bsz * self.config.training.seq_len

        with self.train_context():
            with sl.log_trace_span("worldmodel_forward"):
                outputs = model(**model_inputs)
            with sl.log_trace_span("worldmodel_loss"):
                per_sample_loss, terms = cast(WorldModelLoss, self.loss_fn)(outputs, targets)

            loss = per_sample_loss.mean()
            self._last_loss_terms = terms
            del outputs, per_sample_loss
            with sl.log_trace_span("worldmodel_backward"):
                loss.backward()

        return loss

    def train_step(self, data_iterator: Iterator[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]):
        with sl.log_trace_span("worldmodel_zero_grad"):
            self.optimizers.zero_grad()
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]

        if self.gradient_accumulation_steps > 1:
            raise ValueError("worldmodel v1 does not support gradient accumulation.")

        with sl.log_trace_span("worldmodel_fetch_batch"):
            input_dict, targets = next(data_iterator)
        with sl.log_trace_span("worldmodel_forward_backward"):
            self.forward_backward_step(input_dict=input_dict, targets=targets)

        with sl.log_trace_span("worldmodel_optim"):
            grad_norm = dist_utils.clip_grad_norm_(
                [p for m in self.model_parts for p in m.parameters()],
                self.config.training.max_norm,
                foreach=True,
                pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
                ep_enabled=self.parallel_dims.ep_enabled,
            )
            self.checkpointer.maybe_wait_for_staging()
            self.optimizers.step()
            self.lr_schedulers.step()

        if not self.metrics_processor.should_log(self.step):
            return

        local_loss = self._last_loss_terms["loss"].mean()
        if self.parallel_dims.dp_cp_enabled:
            loss_mesh = self.parallel_dims.get_optional_mesh("loss")
            global_avg_loss = dist_utils.dist_mean(local_loss, loss_mesh)
            global_max_loss = dist_utils.dist_max(local_loss, loss_mesh)
            global_ntokens_seen = dist_utils.dist_sum(torch.tensor(self.ntokens_seen, dtype=torch.int64, device=self.device), loss_mesh)
        else:
            global_avg_loss = global_max_loss = float(local_loss.detach().item())
            global_ntokens_seen = self.ntokens_seen

        extra_metrics = {
            "data/n_tokens_seen/": global_ntokens_seen,
            "metrics/lr/": lr,
        }
        for name, term in self._last_loss_terms.items():
            if name == "loss":
                continue
            term_mean = term.mean()
            if self.parallel_dims.dp_cp_enabled:
                term_value = dist_utils.dist_mean(term_mean, self.parallel_dims.get_optional_mesh("loss"))
            else:
                term_value = float(term_mean.item())
            extra_metrics[f"worldmodel/{name}/"] = term_value

        iterator = self.dataloader._iterator
        prefetch_capacity = iterator._prefetch_factor * iterator._num_workers
        prefetch_in_flight = iterator._tasks_outstanding
        prefetch_buffered_or_in_flight = len(iterator._task_info)
        extra_metrics.update(
            {
                "dataloader/prefetch_capacity/": prefetch_capacity,
                "dataloader/prefetch_in_flight/": prefetch_in_flight,
                "dataloader/prefetch_buffered_or_in_flight/": prefetch_buffered_or_in_flight,
                "dataloader/prefetch_fill_ratio/": prefetch_buffered_or_in_flight / prefetch_capacity,
                "dataloader/prefetch_min_worker_tasks/": min(iterator._workers_num_tasks),
                "dataloader/prefetch_max_worker_tasks/": max(iterator._workers_num_tasks),
            }
        )

        self.metrics_processor.log(self.step, global_avg_loss, global_max_loss, float(grad_norm.item()), extra_metrics=extra_metrics)
