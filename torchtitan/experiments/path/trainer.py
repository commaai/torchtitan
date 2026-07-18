# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import base64
import getpass
import hashlib
import json
import os
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import torch

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.components.unique_counter import StringUniqueCounter
from torchtitan.distributed import utils as dist_utils
from torchtitan.observability import structured_logger as sl
from torchtitan.trainer import Trainer

from .attention_diagnostics import (
    PlanViTAttentionDiagnostics,
    PlanViTAttentionDiagnosticsConfig,
)
from .loss import PathLoss
from .one_pass_observability import (
    DEFAULT_HOST_BUDGET_BYTES,
    AuditResult,
    OwnerShardedOnePassAudit,
)
from .onnx_checkpoint import PathOnnxCheckpointManager
from .validate import PathValidator, segment_names_from_info


def _training_id_path_component(training_id: str) -> str:
    """Encode one Reporter run identity as a single, path-safe component."""
    encoded = base64.urlsafe_b64encode(training_id.encode()).decode().rstrip("=")
    return f"run-{encoded}"


def final_checkpoint_config(
    *, flavor: str, stem: str, seed: int | None, steps: int
) -> CheckpointManager.Config:
    reporterv2_host = os.getenv("REPORTERV2_HOST")
    report_user = os.getenv("REPORT_USER") or getpass.getuser()
    if reporterv2_host:
        training_id = os.getenv("REPORTERV2_TRAINING_ID")
        folder_parts = [report_user]
        if training_id:
            folder_parts.append(_training_id_path_component(training_id))
        folder_parts.extend((flavor, f"{stem}_s{seed}"))
        return PathOnnxCheckpointManager.Config(
            enable=True,
            checkpoint_base_folder=f"{reporterv2_host.rstrip('/')}/checkpoint",
            checkpoint_id_format="step-",
            folder="/".join(folder_parts),
            interval=steps,
            keep_latest_k=0,
        )
    return CheckpointManager.Config(
        enable=True,
        folder=(
            f"/raid.unprotected/reports/{report_user}_reports"
            f"/prune_10m/vit/checkpoints/{flavor}/{stem}_s{seed}"
        ),
        interval=steps,
        keep_latest_k=0,
    )


class PathTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        loss: PathLoss.Config
        validator: PathValidator.Config
        checkpoint: PathOnnxCheckpointManager.Config
        miniray: dict[str, Any] = field(default_factory=dict)
        fps: int
        attention_diagnostics: PlanViTAttentionDiagnosticsConfig = field(
            default_factory=PlanViTAttentionDiagnosticsConfig
        )
        one_pass_audit_host_budget_bytes: int = DEFAULT_HOST_BUDGET_BYTES

        def __post_init__(self) -> None:
            Trainer.Config.__post_init__(self)
            if self.codedir:
                self.miniray = {**self.miniray, "codedir": self.codedir}
                self.validator.miniray = {
                    **self.validator.miniray,
                    "codedir": self.codedir,
                }
            if self.dataloader.limit and not self.checkpoint.enable:
                stem = os.path.splitext(os.path.basename(self.dataloader.dataset))[0]
                self.checkpoint = final_checkpoint_config(
                    flavor=self.model_spec.flavor,
                    stem=stem,
                    seed=self.debug.seed,
                    steps=self.training.steps,
                )

    def __init__(self, config: Config):
        super().__init__(config)
        training_id = os.getenv("REPORTERV2_TRAINING_ID") or "local"
        self.unique_segment_counter = StringUniqueCounter(
            f"unique_ids:{training_id}:path:train"
        )
        self.one_pass_audit: OwnerShardedOnePassAudit | None = None
        if config.dataloader.one_pass:
            batch_mesh = self.parallel_dims.get_optional_mesh("batch")
            batch_world_size = batch_mesh.size() if batch_mesh is not None else 1
            self.one_pass_audit = OwnerShardedOnePassAudit(
                num_writers=config.dataloader.num_writers,
                total_samples=(
                    config.training.steps * config.training.global_batch_size
                ),
                world_size=batch_world_size,
                expected_samples_per_source=7.9,
                host_budget_bytes=config.one_pass_audit_host_budget_bytes,
            )
        self.attention_diagnostics = PlanViTAttentionDiagnostics(
            config.attention_diagnostics,
            self.model_parts,
            total_steps=config.training.steps,
            log_freq=config.metrics.log_freq,
        )
        self.loss_fn.to(self.device)

    def batch_generator(
        self,
        data_iterable: Iterable[
            tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
        ],
    ) -> Iterator[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]:
        data_iterator = iter(data_iterable)
        while True:
            data_load_start = time.perf_counter()
            try:
                input_dict, targets = next(data_iterator)
            except StopIteration as ex:
                raise DataloaderExhaustedError() from ex
            self.metrics_processor.ntokens_since_last_log += next(
                iter(input_dict.values())
            ).shape[0]
            self.metrics_processor.data_loading_times.append(
                time.perf_counter() - data_load_start
            )
            yield input_dict, targets

    @sl.log_trace_span("post_dataloading_process")
    def post_dataloading_process(
        self,
        input_dict: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self.ntokens_seen += next(iter(input_dict.values())).shape[0]
        return input_dict, labels

    @sl.log_trace_span("fwd_bwd")
    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
        local_samples: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        inputs, labels = self.post_dataloading_process(input_dict, labels)
        assert len(self.model_parts) == 1
        with self.train_context():
            pred = self.model_parts[0](inputs)
            loss_vec, metrics = self.loss_fn(pred, labels)
            loss = loss_vec.sum() / local_samples
            del pred
            loss.backward()
        return loss, metrics

    def train_step(
        self,
        data_iterator: Iterator[
            tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
        ],
    ) -> None:
        self.optimizers.zero_grad()
        will_log = self.metrics_processor.should_log(self.step) or (
            self.one_pass_audit is not None
            and self.step == self.config.training.steps
        )
        self.attention_diagnostics.begin_step(self.step, will_log=will_log)
        lr_metrics = self.lr_schedulers.get_metrics()
        parallel_dims = self.parallel_dims
        batch_mesh = parallel_dims.get_optional_mesh("batch")

        microbatches = []
        step_segment_names: set[str] = set()
        step_audit_records: list[
            tuple[list[str], list[int], list[int], list[int], list[int]]
        ] = []
        local_samples = torch.tensor(0, dtype=torch.int64)
        for _ in range(self.gradient_accumulation_steps):
            with sl.log_trace_span("fetching_batch"):
                input_dict, targets = next(data_iterator)
                local_samples += next(iter(input_dict.values())).shape[0]
                if "info" in input_dict:
                    segment_names = segment_names_from_info(input_dict["info"])
                    step_segment_names.update(segment_names)
                    if self.one_pass_audit is not None:
                        from xx.training.lib.dataloader import (
                            ONE_PASS_SOURCE_GLOBAL_RANK,
                            ONE_PASS_SOURCE_KEYS,
                            ONE_PASS_SOURCE_SAMPLE_INDEX,
                            ONE_PASS_SOURCE_SEQUENCE,
                            ONE_PASS_SOURCE_WRITER,
                        )

                        missing = [key for key in ONE_PASS_SOURCE_KEYS if key not in input_dict]
                        if missing:
                            raise RuntimeError(
                                f"one-pass consumed batch is missing source metadata: {missing}"
                            )
                        metadata = {key: input_dict.pop(key) for key in ONE_PASS_SOURCE_KEYS}
                        step_audit_records.append(
                            (
                                segment_names,
                                metadata[ONE_PASS_SOURCE_GLOBAL_RANK].tolist(),
                                metadata[ONE_PASS_SOURCE_WRITER].tolist(),
                                metadata[ONE_PASS_SOURCE_SEQUENCE].tolist(),
                                metadata[ONE_PASS_SOURCE_SAMPLE_INDEX].tolist(),
                            )
                        )
                microbatches.append((input_dict, targets))
        sl.log_trace_scalar({"local_samples": int(local_samples)})

        local_samples = local_samples.to(self.device)
        if batch_mesh is not None:
            global_samples = dist_utils.dist_sum(local_samples, batch_mesh)
        else:
            global_samples = local_samples.float()
        global_samples = torch.as_tensor(
            global_samples, dtype=torch.float32, device=self.device
        )
        global_samples_value = float(global_samples.item())

        accumulated_losses = []
        metric_sums: dict[str, torch.Tensor] = {}
        for input_dict, targets in microbatches:
            input_dict = {k: v.to(self.device) for k, v in input_dict.items()}
            targets = {k: v.to(self.device) for k, v in targets.items()}
            loss, metrics = self.forward_backward_step(
                input_dict=input_dict,
                labels=targets,
                local_samples=local_samples,
            )
            accumulated_losses.append(loss.detach())
            for name, value in metrics.items():
                if name == "loss":
                    continue
                metric_sums[name] = (
                    metric_sums.get(name, torch.zeros((), device=self.device))
                    + value.float().sum()
                )

        with sl.log_trace_span("optim"):
            grad_norm = dist_utils.clip_grad_norm_(
                [p for m in self.model_parts for p in m.parameters()],
                self.config.training.max_norm,
                foreach=True,
                pp_mesh=parallel_dims.get_optional_mesh("pp"),
                ep_enabled=parallel_dims.ep_enabled,
            )
            self.checkpointer.maybe_wait_for_staging()
            self.optimizers.step()
            self.lr_schedulers.step()

        if self.one_pass_audit is None:
            self.unique_segment_counter.update(step_segment_names)
        else:
            for record in step_audit_records:
                self.one_pass_audit.observe(*record)

        loss = torch.sum(torch.stack(accumulated_losses))
        if not will_log:
            return

        if parallel_dims.dp_cp_enabled:
            loss_mesh = parallel_dims.get_optional_mesh("loss")
            local_loss_sum = loss * local_samples
            global_avg_loss, global_max_loss, global_samples_seen = (
                dist_utils.dist_sum(local_loss_sum.detach(), loss_mesh)
                / global_samples_value,
                dist_utils.dist_max(loss.detach(), loss_mesh),
                dist_utils.dist_sum(
                    torch.tensor(
                        self.ntokens_seen, dtype=torch.int64, device=self.device
                    ),
                    loss_mesh,
                ),
            )
            metric_sums = {
                k: dist_utils.dist_sum(v, loss_mesh) for k, v in metric_sums.items()
            }
        else:
            global_avg_loss = global_max_loss = float(loss.detach().item())
            global_samples_seen = self.ntokens_seen

        path_metrics = {
            f"path/{k}": float(
                (
                    torch.as_tensor(v, dtype=torch.float32, device=self.device)
                    / global_samples
                ).item()
            )
            for k, v in metric_sums.items()
        }
        audit_result: AuditResult | None = None
        if self.one_pass_audit is not None:
            audit_result = self.one_pass_audit.sync(
                group=batch_mesh.get_group() if batch_mesh is not None else None,
                device=self.device,
            )
            unique_segments_seen = audit_result.segments
            if audit_result.consumed_samples != int(global_samples_seen):
                raise RuntimeError(
                    "one-pass audit disagrees with trainer sample count: "
                    f"audit={audit_result.consumed_samples} trainer={int(global_samples_seen)}"
                )
            if self.step == self.config.training.steps:
                self._publish_one_pass_audit(audit_result, batch_mesh)
        else:
            unique_segments_seen = (
                self.unique_segment_counter.global_count(batch_mesh.get_group())
                if batch_mesh is not None
                else self.unique_segment_counter.local_count()
            )
        dataset_metrics = {
            "dataset/unique_segments_seen": unique_segments_seen,
        }
        if audit_result is not None:
            dataset_metrics |= {
                "dataset/unique_source_occurrences": audit_result.occurrences,
                "dataset/unique_source_samples": audit_result.samples,
                "dataset/one_pass_audit_checks": audit_result.n_checks,
            }
        extra_metrics = {
            "n_samples_seen": global_samples_seen,
            **lr_metrics,
            **path_metrics,
            **dataset_metrics,
            **self.attention_diagnostics.metrics(batch_mesh),
        }
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            float(grad_norm.item()),
            extra_metrics=extra_metrics,
        )

    def _publish_one_pass_audit(self, result: AuditResult, batch_mesh) -> None:
        assert self.one_pass_audit is not None
        group = batch_mesh.get_group() if batch_mesh is not None else None
        world_size = batch_mesh.size() if batch_mesh is not None else 1
        global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        local = {"global_rank": global_rank, **self.one_pass_audit.shard_attestation()}
        attestations: list[dict[str, Any] | None] = [None] * world_size
        if world_size > 1:
            torch.distributed.all_gather_object(attestations, local, group=group)
        else:
            attestations[0] = local

        publication_error = ""
        if global_rank == 0:
            try:
                from reporterv2.storage import store_put
                from xx.training.lib import dataloader as loader_module

                def source_sha(path: str) -> str:
                    with open(path, "rb") as file:
                        return hashlib.sha256(file.read()).hexdigest()

                document = {
                    "schema_version": 1,
                    "claim": "exact consumed-side one-pass proof completed before final checkpoint",
                    "step": self.step,
                    "global_totals": {
                        "segments": result.segments,
                        "occurrences": result.occurrences,
                        "samples": result.samples,
                        "consumed_samples": result.consumed_samples,
                    },
                    "n_checks": result.n_checks,
                    "owner_shards": attestations,
                    "memory": {
                        "estimated_owner_bytes": self.one_pass_audit.estimated_owner_bytes,
                        "expected_owner_bytes_at_7p9_samples_per_source": self.one_pass_audit.expected_owner_bytes,
                        "configured_host_budget_bytes": self.one_pass_audit.host_budget_bytes,
                    },
                    "source_sha256": {
                        "training/lib/dataloader.py": source_sha(loader_module.__file__),
                        "torchtitan/torchtitan/experiments/path/one_pass_observability.py": source_sha(
                            os.path.join(os.path.dirname(__file__), "one_pass_observability.py")
                        ),
                        "torchtitan/torchtitan/experiments/path/trainer.py": source_sha(__file__),
                    },
                    "residue": "counts and deterministic owner-shard hashes only; full shards are not uploaded",
                    "later_reaudit": "not possible from hashes; any failed or disputed proof requires a refire",
                }
                training_id = os.getenv("REPORTERV2_TRAINING_ID") or "local"
                store_put(
                    f"runs/{training_id}/reports/one_pass_audit.{self.step}.json",
                    json.dumps(document, sort_keys=True, indent=2),
                )
            except Exception as error:
                publication_error = repr(error)
        errors = [publication_error]
        if world_size > 1:
            torch.distributed.broadcast_object_list(errors, src=0, group=group)
        if errors[0]:
            raise RuntimeError(f"failed to publish one-pass audit evidence: {errors[0]}")

    def close(self) -> None:
        attention_diagnostics = getattr(self, "attention_diagnostics", None)
        if attention_diagnostics is not None:
            attention_diagnostics.close()
        self.dataloader.close()
        if self.config.validator.enable:
            self.validator.close()
        super().close()

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["unique_segment_counter"] = self.unique_segment_counter.state_dict()
        validator_unique_segment_counter = getattr(
            getattr(self, "validator", None), "unique_segment_counter", None
        )
        if validator_unique_segment_counter is not None:
            state[
                "validation_unique_segment_counter"
            ] = validator_unique_segment_counter.state_dict()
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        if "unique_segment_counter" in state_dict:
            self.unique_segment_counter.load_state_dict(
                state_dict["unique_segment_counter"]
            )
        validator_unique_segment_counter = getattr(
            getattr(self, "validator", None), "unique_segment_counter", None
        )
        if (
            validator_unique_segment_counter is not None
            and "validation_unique_segment_counter" in state_dict
        ):
            validator_unique_segment_counter.load_state_dict(
                state_dict["validation_unique_segment_counter"]
            )
