from __future__ import annotations

import json
import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from itertools import islice
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.report_runner import ReportRunner, ReportSpec
from torchtitan.components.unique_counter import StringUniqueCounter
from torchtitan.components.validate import BaseValidator
from torchtitan.config import ParallelismConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.tools.logging import logger
from xx.common.helpers import parse_info
from xx.training.lib.checkpoint import wait_for_checkpoint
from xx.release_tests.lib.base_report import ReportFormat
from xx.training.path.test import (
    DATASET_REPORTS,
    MODEL_REPORTS,
)

from .loss import PathLoss
from .tokenizer import PathTokenizer

ValidationContext = Callable[[], AbstractContextManager[None]]


def segment_names_and_fidxs_from_info(
    info: torch.Tensor,
) -> list[tuple[str, int]]:
    infos = (parse_info(x) for x in info.cpu().numpy())
    return [(info["name"], int(info["fidx"])) for info in infos]


def global_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


class PathValidator(BaseValidator):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseValidator.Config):
        enable: bool
        steps: int
        dataloader: BaseDataLoader.Config
        mixed_precision_param: str
        reports: dict[str, list[int]] = field(default_factory=dict)
        miniray: dict[str, Any] = field(default_factory=dict)
        save_predictions: bool = False
        prediction_file_prefix: str = "val_preds"

    def __init__(
        self,
        config: Config,
        *,
        parallelism: ParallelismConfig,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: PathTokenizer,
        parallel_dims: ParallelDims,
        loss_fn: PathLoss,
        validation_context: ValidationContext,
        metrics_processor: MetricsProcessor,
        seq_len: int,
        local_batch_size: int,
        **kwargs: Any,
    ) -> None:
        del parallelism, kwargs
        super().__init__(config=config)
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.tokenizer = tokenizer
        self.parallel_dims = parallel_dims
        self.loss_fn = loss_fn
        self.validation_context = validation_context
        self.metrics_processor = metrics_processor
        self.seq_len = seq_len
        self.local_batch_size = local_batch_size
        self.tokenizer_dtype = TORCH_DTYPE_MAP[config.mixed_precision_param]
        self.miniray = dict(self.config.miniray)
        self.dataloader = self.config.dataloader.build(
            dp_world_size=self.dp_world_size,
            dp_rank=self.dp_rank,
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
            local_batch_size=self.local_batch_size,
            validation_steps=self.config.steps,
        )
        # TODO centralize training_id
        self.training_id = os.getenv("REPORTERV2_TRAINING_ID") or "local"
        self.unique_segment_counter = StringUniqueCounter(f"unique_ids:{self.training_id}:path:validation")
        self.report_runner = ReportRunner(metrics_processor=self.metrics_processor, enabled=global_rank() == 0)

    @torch.no_grad()
    def validate(self, model_parts: list[nn.Module], step: int) -> None:
        for model in model_parts:
            model.eval()

        device = next(model_parts[0].parameters()).device
        try:
            total_loss = torch.zeros((), device=device)
            total_samples = torch.zeros((), device=device)
            metric_sums: dict[str, torch.Tensor] = {}
            validation_segment_names: set[str] = set()
            prediction_rows: list[tuple[str, int]] = []
            prediction_batches: dict[str, list[torch.Tensor]] = {}
            target_batches: dict[str, list[torch.Tensor]] = {}
            batch_mesh = self.parallel_dims.get_optional_mesh("batch")
            loss_mesh = self.parallel_dims.get_optional_mesh("loss")
            batches = iter(self.dataloader)
            if self.config.steps != -1:
                batches = islice(batches, self.config.steps)
            for input_dict, targets in batches:
                self.metrics_processor.ntokens_since_last_log += next(iter(input_dict.values())).shape[0]
                info = input_dict.get("info")
                info_rows = (
                    segment_names_and_fidxs_from_info(info)
                    if info is not None
                    else []
                )
                validation_segment_names.update(name for name, _ in info_rows)
                input_dict = {k: v.to(device) for k, v in input_dict.items()}
                input_dict = self.tokenizer.reconstruct(
                    input_dict,
                    device=device,
                    dtype=self.tokenizer_dtype,
                )
                targets = {k: v.to(device) for k, v in targets.items()}
                local_samples = torch.tensor(
                    next(iter(input_dict.values())).shape[0], dtype=torch.float32, device=device
                )
                global_samples = (
                    dist_utils.dist_sum(local_samples, batch_mesh) if batch_mesh is not None else local_samples
                )

                with self.validation_context():
                    pred = model_parts[0](input_dict)
                    loss_vec, metrics = self.loss_fn(pred, targets)

                if self.config.save_predictions:
                    prediction_rows.extend(info_rows)
                    for name, value in pred.items():
                        prediction_batches.setdefault(name, []).append(value.cpu())
                    for name, value in targets.items():
                        target_batches.setdefault(name, []).append(value.cpu())

                loss_sum = loss_vec.float().sum()
                batch_metric_sums = {k: v.float().sum() for k, v in metrics.items() if k != "loss"}
                if self.parallel_dims.dp_cp_enabled:
                    loss_sum = dist_utils.dist_sum(loss_sum, loss_mesh)
                    batch_metric_sums = {
                        name: dist_utils.dist_sum(batch_metric_sums[name], loss_mesh)
                        for name in sorted(batch_metric_sums)
                    }
                total_loss += loss_sum
                total_samples += global_samples
                for name, value in batch_metric_sums.items():
                    metric_sums[name] = metric_sums.get(name, torch.zeros((), device=device)) + value

            self.unique_segment_counter.update(validation_segment_names)

            samples = torch.as_tensor(total_samples, dtype=torch.float32, device=device)
            loss = float((torch.as_tensor(total_loss, dtype=torch.float32, device=device) / samples).item())
            extra_metrics = {
                f"validation_metrics/path/{k}": float(
                    (torch.as_tensor(v, dtype=torch.float32, device=device) / samples).item()
                )
                for k, v in metric_sums.items()
            }
            extra_metrics["validation_metrics/dataset/unique_segments_seen"] = (
                self.unique_segment_counter.global_count(batch_mesh.get_group())
                if batch_mesh is not None
                else self.unique_segment_counter.local_count()
            )
            self.metrics_processor.log_validation(loss=loss, step=step, extra_metrics=extra_metrics)
            if self.config.save_predictions:
                self._save_predictions(
                    prediction_rows,
                    prediction_batches,
                    target_batches,
                    step,
                    batch_mesh,
                )
            self._submit_reports(step)
        finally:
            for model in model_parts:
                model.train()

    def close(self) -> None:
        try:
            self.report_runner.close()
        finally:
            self.dataloader.close()

    def _save_predictions(
        self,
        rows: list[tuple[str, int]],
        prediction_batches: dict[str, list[torch.Tensor]],
        target_batches: dict[str, list[torch.Tensor]],
        step: int,
        batch_mesh: Any,
    ) -> None:
        from reporterv2.storage import store_put
        from safetensors.torch import save as save_safetensors

        group = batch_mesh.get_group() if batch_mesh is not None else None
        rank = global_rank()
        tensors = {
            "fidx": torch.tensor([fidx for _, fidx in rows]),
            **{
                f"predictions/{name}": torch.cat(values)
                for name, values in prediction_batches.items()
            },
            **{
                f"targets/{name}": torch.cat(values)
                for name, values in target_batches.items()
            },
        }
        folder = (
            f"runs/{self.training_id}/reports/"
            f"{self.config.prediction_file_prefix}.{step}"
        )
        shard_file = f"__{rank}_0.safetensors"
        shard = save_safetensors(tensors)
        logger.info(
            f"Saving prediction shard {shard_file} ({len(shard) / 2**20:.1f} MiB)"
        )
        store_put(f"{folder}/{shard_file}", shard)

        shard_metadata = {
            "file": shard_file,
            "names": [name for name, _ in rows],
        }
        if group is not None:
            shards: list[Any] | None = (
                [None] * dist.get_world_size(group) if rank == 0 else None
            )
            dist.gather_object(shard_metadata, shards, dst=0, group=group)
        else:
            shards = [shard_metadata]
        if rank != 0:
            return

        store_put(f"{folder}/.metadata", json.dumps({"shards": shards}))
        logger.info(f"Saved prediction metadata for step {step}")

    def _submit_reports(self, step: int) -> None:
        current_checkpoint = f"{self.training_id}/{step}"

        def _run_report(TestCls: type, test_config: Any, wait_for_checkpoint_keys: list[str]) -> tuple[Any, ...]:
            for k in wait_for_checkpoint_keys:
                wait_for_checkpoint(current_checkpoint, k)
            return (TestCls(test_config).run_report(),)

        dataloader_config = self.config.dataloader
        report_specs: dict[str, ReportSpec] = {}
        for report_name, (TestCls, ReportConfigCls) in MODEL_REPORTS.items():
            report_config = ReportConfigCls(
                rollout={"agent": {"supercombo": current_checkpoint, "model_trained_fps": dataloader_config.fps}},
                report_name=f"path_{report_name}",
                save_tmp=False,
                format=ReportFormat.HTML,
                miniray=self.miniray,
            )
            report_specs[report_name] = ReportSpec(
                output_names=(report_name,),
                output_types=("html",),
                steps=self.config.reports.get(report_name, []),
                func=_run_report,
                arguments=[TestCls, report_config, ["vision.onnx", "vision.onnx.data", "temporal_policy.onnx"]],
            )

        for report_name, (TestCls, ReportConfigCls) in DATASET_REPORTS.items():
            report_config = ReportConfigCls(
                route_list=dataloader_config.dataset,
                pipeline_dir=dataloader_config.pipeline_dir,
                report_name=f"path_{report_name}",
                save_tmp=False,
                format=ReportFormat.HTML,
                miniray=self.miniray,
            )
            report_specs[report_name] = ReportSpec(
                output_names=(report_name,),
                output_types=("html",),
                steps=self.config.reports.get(report_name, []),
                func=_run_report,
                arguments=[TestCls, report_config, []],
            )

        self.report_runner.submit_due(step=step, report_specs=report_specs)
