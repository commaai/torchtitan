from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.nn as nn

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.report_runner import ReportRunner, ReportSpec
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.components.unique_counter import StringUniqueCounter
from torchtitan.components.validate import BaseValidator
from torchtitan.config import ParallelismConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.tools.logging import logger
from xx.common.helpers import parse_info
from xx.training.lib.checkpoint import wait_for_checkpoint
from xx.release_tests.lib.base_report import ReportFormat
from xx.training.path.test import (
    DATASET_REPORTS,
    MODEL_REPORTS,
    MODEL_REPORT_ROUTE_LISTS,
)

from .model_constants import IDX_N, PLAN_WIDTH
from .plan_vae import (
    PLAN_VAE_RECONSTRUCTION,
    prepare_plan_latent_batch,
    PlanNormalization,
    prepare_plan_vae_batch,
    unnormalize_plan,
)

from .loss import PathLoss

ValidationContext = Callable[[], AbstractContextManager[None]]


def segment_names_and_fidxs_from_info(
    info: torch.Tensor,
) -> list[tuple[str, int]]:
    infos = (parse_info(x) for x in info.cpu().numpy())
    return [(info["name"], int(info["fidx"])) for info in infos]


def global_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


PlanVAEReportPayload = tuple[list[tuple[str, int]], torch.Tensor, torch.Tensor]


def plan_vae_analyse_driving_data(
    payloads: list[PlanVAEReportPayload],
    *,
    normalization: PlanNormalization = "pooled",
) -> dict[str, dict[str, dict[str, Any]]]:
    segment_rows: dict[str, list[tuple[int, torch.Tensor, torch.Tensor]]] = defaultdict(list)
    for rows, normalized_predictions, targets in payloads:
        if len(rows) != normalized_predictions.shape[0] or len(rows) != targets.shape[0]:
            raise ValueError(
                "Plan VAE report metadata and tensors have different batch sizes: "
                f"rows={len(rows)}, predictions={normalized_predictions.shape[0]}, targets={targets.shape[0]}"
            )
        predictions = unnormalize_plan(
            normalized_predictions.float(),
            normalization=normalization,
        ).unflatten(-1, (IDX_N, PLAN_WIDTH))
        targets = targets.float().unflatten(-1, (IDX_N, PLAN_WIDTH))
        predictions = predictions.reshape(len(rows), -1, IDX_N, PLAN_WIDTH)[:, -1]
        targets = targets.reshape(len(rows), -1, IDX_N, PLAN_WIDTH)[:, -1]
        for (segment, fidx), prediction, target in zip(rows, predictions, targets, strict=True):
            segment_rows[segment].append((fidx, prediction, target))

    data = {}
    for segment, rows in segment_rows.items():
        ordered_rows = sorted(rows, key=lambda row: row[0])
        data[segment] = {
            "pred": {"plan": torch.stack([row[1] for row in ordered_rows]).numpy()},
            "true": {"plan": torch.stack([row[2] for row in ordered_rows]).numpy()},
        }
    return data


def run_plan_vae_analyse_driving_report(
    test_cls: type,
    test_config: Any,
    data: dict[str, dict[str, dict[str, Any]]],
) -> tuple[str]:
    writer = test_cls(test_config).make_report(data)
    return (writer.export_html(tofile=False, use_virtual_webgl=test_config.use_virtual_webgl),)


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
        training_stage: Literal["plan_vae", "policy"]

    def __init__(
        self,
        config: Config,
        *,
        parallelism: ParallelismConfig,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
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
        self.training_stage = config.training_stage
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
        vae_report_due = self.training_stage == "plan_vae" and step in self.config.reports.get("analyse_driving", [])
        plan_normalization: PlanNormalization = "pooled"
        if self.training_stage == "plan_vae":
            diffusion_plan = model_parts[0].temporal_policy.plan_vae
            assert diffusion_plan is not None
            plan_normalization = diffusion_plan.normalization
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
                batch_size = targets["plan"].shape[0]
                self.metrics_processor.ntokens_since_last_log += batch_size
                info = input_dict.get("info")
                info_rows = segment_names_and_fidxs_from_info(info) if info is not None else []
                validation_segment_names.update(name for name, _ in info_rows)
                if self.training_stage == "plan_vae":
                    input_dict = {}
                    targets = {"plan": targets["plan"]}
                input_dict = {k: v.to(device) for k, v in input_dict.items()}
                targets = {k: v.to(device) for k, v in targets.items()}
                local_samples = torch.tensor(batch_size, dtype=torch.float32, device=device)
                global_samples = (
                    dist_utils.dist_sum(local_samples, batch_mesh) if batch_mesh is not None else local_samples
                )

                with self.validation_context():
                    if self.training_stage == "plan_vae":
                        input_dict, targets = prepare_plan_vae_batch(
                            input_dict,
                            targets,
                            normalization=plan_normalization,
                        )
                    elif model_parts[0].config.plan_loss in ("latent_mse", "latent_nll"):
                        plan_vae = model_parts[0].temporal_policy.plan_vae
                        assert plan_vae is not None
                        targets = prepare_plan_latent_batch(
                            targets,
                            plan_vae.encode_mean,
                            normalization=plan_vae.normalization,
                        )
                    pred = model_parts[0](input_dict)
                    loss_vec, metrics = self.loss_fn(pred, targets)

                if self.config.save_predictions:
                    prediction_rows.extend(info_rows)
                    for name, value in pred.items():
                        prediction_batches.setdefault(name, []).append(value.cpu())
                    for name, value in targets.items():
                        target_batches.setdefault(name, []).append(value.cpu())
                elif vae_report_due:
                    prediction_rows.extend(info_rows)
                    prediction_batches.setdefault(PLAN_VAE_RECONSTRUCTION, []).append(
                        pred[PLAN_VAE_RECONSTRUCTION].cpu()
                    )
                    target_batches.setdefault("plan", []).append(targets["plan"].cpu())

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
                f"validation_metrics/{k}": float(
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
            vae_report_data = None
            if vae_report_due:
                vae_report_data = self._gather_plan_vae_report_data(
                    prediction_rows,
                    prediction_batches,
                    target_batches,
                    batch_mesh,
                    plan_normalization,
                )
            if self.config.save_predictions:
                self._save_predictions(
                    prediction_rows,
                    prediction_batches,
                    target_batches,
                    step,
                    batch_mesh,
                )
            self._submit_reports(step, vae_report_data)
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

    def _gather_plan_vae_report_data(
        self,
        rows: list[tuple[str, int]],
        prediction_batches: dict[str, list[torch.Tensor]],
        target_batches: dict[str, list[torch.Tensor]],
        batch_mesh: Any,
        normalization: PlanNormalization,
    ) -> dict[str, dict[str, dict[str, Any]]] | None:
        payload: PlanVAEReportPayload = (
            rows,
            torch.cat(prediction_batches[PLAN_VAE_RECONSTRUCTION]),
            torch.cat(target_batches["plan"]),
        )
        rank = global_rank()
        if batch_mesh is not None:
            group = batch_mesh.get_group()
            payloads: list[PlanVAEReportPayload | None] | None = (
                [None] * dist.get_world_size(group) if rank == 0 else None
            )
            dist.gather_object(payload, payloads, dst=0, group=group)
        else:
            payloads = [payload]
        if rank != 0:
            return None
        assert payloads is not None and all(item is not None for item in payloads)
        return plan_vae_analyse_driving_data(
            [item for item in payloads if item is not None],
            normalization=normalization,
        )

    def _submit_reports(
        self,
        step: int,
        vae_report_data: dict[str, dict[str, dict[str, Any]]] | None = None,
    ) -> None:
        current_checkpoint = f"{self.training_id}/{step}"

        def _run_report(TestCls: type, test_config: Any, wait_for_checkpoint_keys: list[str]) -> tuple[Any, ...]:
            for k in wait_for_checkpoint_keys:
                wait_for_checkpoint(current_checkpoint, k)
            return (TestCls(test_config).run_report(),)

        dataloader_config = self.config.dataloader
        report_specs: dict[str, ReportSpec] = {}
        if self.training_stage == "plan_vae":
            TestCls, ReportConfigCls = MODEL_REPORTS["analyse_driving"]
            report_config = ReportConfigCls(
                plan_only=True,
                route_list=MODEL_REPORT_ROUTE_LISTS["analyse_driving"],
                rollout={"agent": {"supercombo": current_checkpoint}},
                report_name="path_analyse_driving",
                save_tmp=False,
                format=ReportFormat.HTML,
                miniray=self.miniray,
            )
            report_specs["analyse_driving"] = ReportSpec(
                output_names=("analyse_driving",),
                output_types=("html",),
                steps=self.config.reports.get("analyse_driving", []),
                func=run_plan_vae_analyse_driving_report,
                arguments=[TestCls, report_config, vae_report_data],
            )
            self.report_runner.submit_due(step=step, report_specs=report_specs)
            return

        for report_name, (TestCls, ReportConfigCls) in MODEL_REPORTS.items():
            report_route = (
                {"route_list": MODEL_REPORT_ROUTE_LISTS[report_name]}
                if report_name in MODEL_REPORT_ROUTE_LISTS
                else {}
            )
            report_config = ReportConfigCls(
                rollout={"agent": {"supercombo": current_checkpoint, "model_trained_fps": dataloader_config.fps}},
                report_name=f"path_{report_name}",
                save_tmp=False,
                format=ReportFormat.HTML,
                miniray=self.miniray,
                **report_route,
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
