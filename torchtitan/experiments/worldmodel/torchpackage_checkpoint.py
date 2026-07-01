# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import io
import os
import posixpath
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader
from torch.package import PackageExporter

from torchtitan.components import fs
from torchtitan.components.checkpoint import CheckpointManager, MODEL
from torchtitan.experiments.worldmodel.model import WorldModel
from torchtitan.experiments.worldmodel.model_config import model_registry
from torchtitan.experiments.worldmodel.model_for_inference import WorldModelForInference
from torchtitan.models.common.nn_modules import Linear
from torchtitan.observability import structured_logger as sl
from torchtitan.tools.logging import init_logger, logger


os.environ.setdefault("NCCL_P2P_DISABLE", "1")

REPORTERV2_HOST = os.getenv(
    "REPORTERV2_HOST", "mkv://data-gen.comma.life:3080/reporterv2"
)
MODEL_CONFIG_FILE = "_torchpackage_model_config.pt"
STRUCTURED_LOG_DIR = os.getenv(
    "TORCHTITAN_STRUCTURED_LOG_DIR", "./outputs/worldmodel_torchpackage_checkpoint"
)
PACKAGE_NAME = "model.torchpackage"

TORCH_EXPORT_INTERN_MODULES = [
    "torchtitan.config.**",
    "torchtitan.distributed",
    "torchtitan.distributed.compile",
    "torchtitan.distributed.parallel_dims",
    "torchtitan.distributed.spmd_types",
    "torchtitan.distributed.utils",
    "torchtitan.experiments.worldmodel.model_for_inference",
    "torchtitan.experiments.worldmodel.model",
    "torchtitan.experiments.worldmodel.schedulers",
    "torchtitan.models.common.attention",
    "torchtitan.models.common.embedding",
    "torchtitan.models.common.nn_modules",
    "torchtitan.models.common.rope",
    "torchtitan.observability.**",
    "torchtitan.protocols.**",
    "torchtitan.tools.logging",
    "torchtitan.tools.utils",
]
TORCH_EXPORT_STRIP_FUTURE_ANNOTATIONS_MODULES = [
    "torchtitan.distributed.parallel_dims",
    "torchtitan.models.common.embedding",
    "torchtitan.protocols.module",
    "torchtitan.protocols.model_spec",
]
TORCH_EXPORT_EXTERN_MODULES = [
    "torch.**",
    "torchao.**",
    "numpy.**",
    "einops.**",
    "spmd_types.**",
    "typing_extensions.**",
    "tyro.**",
    "docstring_parser.**",
    "typeguard.**",
    "dataclasses.**",
    "collections.**",
    "argparse.**",
    "sys.**",
]
TORCH_EXPORT_DENY_MODULES = ["openpilot.**", "cereal", "cereal.**", "capnp", "capnp.**"]
TORCH_EXPORT_MOCK_MODULES = ["**"]
TORCH_EXPORT_CONFIG_INIT_SOURCE = """
import torch

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

from .configs import (
    CommConfig,
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from .configurable import Configurable

__all__ = [
    "Configurable",
    "TORCH_DTYPE_MAP",
    "CompileConfig",
    "ParallelismConfig",
    "CommConfig",
    "TrainingConfig",
    "DebugConfig",
]
""".lstrip()


def build_meta_model(
    model_config: WorldModel.Config, *, dtype: torch.dtype = torch.bfloat16
) -> WorldModelForInference:
    with torch.device("meta"):
        return WorldModelForInference(model_config).to(dtype=dtype).eval()


def copy_model_config(model_config: WorldModel.Config) -> WorldModel.Config:
    model_config = copy.deepcopy(model_config)
    for _fqn, linear_config, parent, attr in list(model_config.traverse(Linear.Config)):
        if type(linear_config) is Linear.Config:
            continue
        new_config = Linear.Config(
            in_features=linear_config.in_features,
            out_features=linear_config.out_features,
            bias=linear_config.bias,
            param_init=linear_config.param_init,
            sharding_config=linear_config.sharding_config,
        )
        if isinstance(parent, list):
            parent[attr] = new_config
        else:
            setattr(parent, attr, new_config)
    return model_config


def load_model_config(path: str) -> WorldModel.Config:
    with fs.open_file(path, "rb") as handle:
        model_config = torch.load(
            io.BytesIO(handle.read()),
            map_location="cpu",
            weights_only=False,
        )
    if not isinstance(model_config, WorldModel.Config):
        raise TypeError(
            f"{path} contained {type(model_config).__name__}, "
            "expected WorldModel.Config."
        )
    return model_config


def convert_state_dict_to_fp8(
    model_config: WorldModel.Config,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        Float8MMConfig,
        quantize_,
    )
    from torchao.quantization.granularity import PerTensor

    with torch.device("cpu"):
        model = WorldModelForInference(model_config).to(dtype=torch.bfloat16).eval()
    model.load_state_dict(state_dict, strict=True, assign=True)
    state_dict.clear()
    del state_dict
    quantize_(
        model.blocks,
        Float8DynamicActivationFloat8WeightConfig(
            granularity=PerTensor(),
            mm_config=Float8MMConfig(),
        ),
    )
    converted = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    del model
    gc.collect()
    return converted


def build_package(
    *,
    model_config: WorldModel.Config,
    state_dict: dict[str, torch.Tensor],
    step: int,
) -> bytes:

    with sl.log_trace_span("worldmodel_package_convert_fp8"):
        state_dict = convert_state_dict_to_fp8(model_config, state_dict)

    with sl.log_trace_span("worldmodel_package_model_io"):
        io_model_config = copy.deepcopy(model_config)
        if io_model_config.transformer.attention_impl == "FLEX":
            io_model_config.transformer.attention_impl = "SDPA"
        if io_model_config.plan_head.attention_impl == "FLEX":
            io_model_config.plan_head.attention_impl = "SDPA"
        io_model = build_meta_model(io_model_config)
        assert io_model.config.transformer.attention_mask != "NONE"
        num_conditioning_frames = io_model.config.input_size[0] - 1
        model_io = io_model.get_model_io(
            dtype=torch.bfloat16,
            steps=1,
            num_conditioning_frames=num_conditioning_frames,
        )
        del io_model, io_model_config
        model = build_meta_model(model_config)

    with sl.log_trace_span("worldmodel_package_build"):
        package_buffer = io.BytesIO()
        with PackageExporter(package_buffer, debug=True) as exporter:
            exporter.intern(TORCH_EXPORT_INTERN_MODULES)
            exporter.extern(TORCH_EXPORT_EXTERN_MODULES)
            exporter.deny(TORCH_EXPORT_DENY_MODULES)
            exporter.save_source_string(
                "torchtitan.config",
                TORCH_EXPORT_CONFIG_INIT_SOURCE,
                is_package=True,
            )
            for module_name in TORCH_EXPORT_STRIP_FUTURE_ANNOTATIONS_MODULES:
                spec = importlib.util.find_spec(module_name)
                if spec is None or spec.origin is None:
                    raise ModuleNotFoundError(module_name)
                source = Path(spec.origin).read_text()
                source = source.replace("from __future__ import annotations\n\n", "")
                if module_name == "torchtitan.distributed.parallel_dims":
                    source = source.replace(
                        ") -> ParallelDims:\n", ') -> "ParallelDims":\n'
                    )
                exporter.save_source_string(module_name, source)
            exporter.mock(
                TORCH_EXPORT_MOCK_MODULES,
                exclude=TORCH_EXPORT_INTERN_MODULES
                + TORCH_EXPORT_EXTERN_MODULES
                + TORCH_EXPORT_DENY_MODULES,
            )
            exporter.save_pickle("model", "model.pkl", model)
            del model
            exporter.save_pickle(
                "meta", "meta.pkl", {"model_io": model_io, "step": step}
            )
            del model_io

            state_dict_buffer = io.BytesIO()
            torch.save(state_dict, state_dict_buffer)
            state_dict_bytes = state_dict_buffer.getvalue()
            del state_dict_buffer
            state_dict.clear()
            del state_dict
            gc.collect()
            exporter.save_binary("assets", "state_dict.pt", state_dict_bytes)
            del state_dict_bytes
        package = package_buffer.getvalue()
        del package_buffer
    sl.log_trace_scalar({"worldmodel_package.package_bytes": len(package)})
    return package


def export_torch_package(
    *,
    checkpoint_path: str,
    output_path: str,
    model_config: WorldModel.Config,
    step: int,
    model_config_path: str | None = None,
) -> None:
    sl.set_step(step)
    logger.info("Packaging worldmodel checkpoint step=%s", step)
    logger.info("DCP checkpoint path: %s", checkpoint_path)
    if model_config_path is not None:
        logger.info("Worldmodel config path: %s", model_config_path)
    logger.info("Torch package output path: %s", output_path)

    with sl.log_trace_span("worldmodel_package_load_dcp"):
        model = build_meta_model(model_config)
        state_dict = {
            name: torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
            for name, tensor in model.state_dict().items()
        }
        dcp.load(
            state_dict,
            storage_reader=FsspecReader(checkpoint_path),
            checkpoint_id=checkpoint_path,
        )
        del model

    try:
        package = build_package(
            model_config=model_config,
            state_dict=state_dict,
            step=step,
        )
    finally:
        del state_dict
        gc.collect()
    package_bytes = len(package)

    with sl.log_trace_span("worldmodel_package_write"):
        with fs.open_file(output_path, "wb") as handle:
            handle.write(package)
    del package
    gc.collect()
    logger.info(
        "Saved %.2f GiB torch package to %s", package_bytes / (1024**3), output_path
    )


class WorldModelTorchPackageCheckpointManager(CheckpointManager):
    """Checkpoint manager that writes inference torch packages alongside DCP."""

    @dataclass(kw_only=True, slots=True)
    class Config(CheckpointManager.Config):
        checkpoint_base_folder: str = ""
        """Override the trainer dump folder for checkpoint writes."""

        export_torch_package: bool = False
        """Export a rank-0 torch package artifact alongside written checkpoints."""

        torch_package_file: str = PACKAGE_NAME
        """File name for the torch package artifact."""

        torch_package_model_config_file: str = MODEL_CONFIG_FILE
        """File name for the serialized WorldModel config used by the packager."""

        torch_package_async: bool = True
        """Run packaging in a rank-0 subprocess after DCP has completed."""

        torch_package_wait_on_close: bool = True
        """Wait for active torch package subprocesses when closing the manager."""

        torch_package_max_concurrent: int = 1
        """Maximum rank-0 torch package subprocesses to run at once."""

        torch_package_structured_log_dir: str = STRUCTURED_LOG_DIR
        """Structured log directory used by the package worker."""

        def __post_init__(self) -> None:
            CheckpointManager.Config.__post_init__(self)
            if self.export_torch_package and not self.torch_package_file:
                raise ValueError("torch_package_file cannot be empty.")
            if self.export_torch_package and not self.torch_package_model_config_file:
                raise ValueError("torch_package_model_config_file cannot be empty.")
            if self.torch_package_max_concurrent < 1:
                raise ValueError("torch_package_max_concurrent must be at least 1.")

    def __init__(self, config: Config, **kwargs) -> None:
        if config.checkpoint_base_folder:
            kwargs["base_folder"] = config.checkpoint_base_folder
        super().__init__(config, **kwargs)
        self.export_torch_package_enabled = config.export_torch_package
        self.torch_package_file = config.torch_package_file
        self.torch_package_model_config_file = config.torch_package_model_config_file
        self.torch_package_async = config.torch_package_async
        self.torch_package_wait_on_close = config.torch_package_wait_on_close
        self.torch_package_max_concurrent = config.torch_package_max_concurrent
        self.torch_package_structured_log_dir = config.torch_package_structured_log_dir
        self._torch_package_processes: list[subprocess.Popen] = []

    @torch.no_grad()
    def save(self, curr_step: int, last_step: bool = False) -> bool:
        saved = super().save(curr_step, last_step)
        if saved and self.export_torch_package_enabled:
            self._schedule_torch_package(curr_step)
        return saved

    def close(self) -> None:
        if getattr(self, "torch_package_wait_on_close", False):
            for process in getattr(self, "_torch_package_processes", []):
                return_code = process.wait()
                if return_code == 0:
                    logger.info("Torch package export pid=%s completed.", process.pid)
                else:
                    logger.error(
                        "Torch package export pid=%s failed with return code %s.",
                        process.pid,
                        return_code,
                    )
            self._torch_package_processes = []
        else:
            self._reap_torch_package_processes()
        super().close()

    def _schedule_torch_package(self, curr_step: int) -> None:
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return

        model_parts = self.states[MODEL].model
        if len(model_parts) != 1:
            raise ValueError("Worldmodel torch packages do not support PP.")

        checkpoint_path = self._create_checkpoint_id(curr_step)
        output_path = fs.join_path(checkpoint_path, self.torch_package_file)
        model_config_path = fs.join_path(
            checkpoint_path, self.torch_package_model_config_file
        )
        model_config = getattr(model_parts[0], "config", None)
        if not isinstance(model_config, WorldModel.Config):
            raise TypeError(
                "Worldmodel torch package export requires the model to expose "
                "WorldModel.Config as model.config."
            )
        model_config = copy_model_config(model_config)
        with sl.log_trace_span("worldmodel_package_save_config"):
            buffer = io.BytesIO()
            torch.save(model_config, buffer)
            model_config_bytes = buffer.getvalue()
            del buffer
            with fs.open_file(model_config_path, "wb") as handle:
                handle.write(model_config_bytes)
            del model_config_bytes, model_config

        save_future = self.save_future
        if save_future is None:
            self._start_torch_package(
                checkpoint_path=checkpoint_path,
                output_path=output_path,
                model_config_path=model_config_path,
                step=curr_step,
            )
            return

        def start_after_dcp(future) -> None:
            try:
                future.result()
            except Exception:
                logger.exception(
                    "Skipping torch package export for %s because DCP save failed.",
                    checkpoint_path,
                )
                return
            self._start_torch_package(
                checkpoint_path=checkpoint_path,
                output_path=output_path,
                model_config_path=model_config_path,
                step=curr_step,
            )

        save_future.add_done_callback(start_after_dcp)
        logger.info(
            "Queued torch package export for %s after async DCP completion.",
            output_path,
        )

    def _start_torch_package(
        self,
        *,
        checkpoint_path: str,
        output_path: str,
        model_config_path: str,
        step: int,
    ) -> None:
        if not self.torch_package_async:
            model_config = load_model_config(model_config_path)
            try:
                export_torch_package(
                    checkpoint_path=checkpoint_path,
                    output_path=output_path,
                    model_config=model_config,
                    step=step,
                    model_config_path=model_config_path,
                )
            finally:
                del model_config
                gc.collect()
            return

        self._reap_torch_package_processes()
        if len(self._torch_package_processes) >= self.torch_package_max_concurrent:
            logger.warning(
                "Skipping torch package export for %s because %s package worker(s) "
                "are already running.",
                output_path,
                len(self._torch_package_processes),
            )
            return

        cmd = [
            sys.executable,
            "-m",
            "torchtitan.experiments.worldmodel.torchpackage_checkpoint",
            "--checkpoint-path",
            checkpoint_path,
            "--output-path",
            output_path,
            "--model-config-path",
            model_config_path,
            "--step",
            str(step),
            "--structured-log-dir",
            self.torch_package_structured_log_dir,
        ]
        env = os.environ.copy()
        env.setdefault("NCCL_P2P_DISABLE", "1")
        process = subprocess.Popen(cmd, env=env, start_new_session=True)
        self._torch_package_processes.append(process)
        logger.info(
            "Started torch package export pid=%s for %s", process.pid, output_path
        )

    def _reap_torch_package_processes(self) -> None:
        active = []
        for process in self._torch_package_processes:
            return_code = process.poll()
            if return_code is None:
                active.append(process)
            elif return_code == 0:
                logger.info("Torch package export pid=%s completed.", process.pid)
            else:
                logger.error(
                    "Torch package export pid=%s failed with return code %s.",
                    process.pid,
                    return_code,
                )
        self._torch_package_processes = active


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package a worldmodel DCP checkpoint for inference."
    )
    parser.add_argument("checkpoint_path", nargs="?")
    parser.add_argument("output_path", nargs="?", default=None)
    parser.add_argument("--flavor", default="base")
    parser.add_argument("--checkpoint-path", dest="flag_checkpoint_path")
    parser.add_argument("--output-path", dest="flag_output_path")
    parser.add_argument("--model-config-path")
    parser.add_argument("--step", type=int)
    parser.add_argument("--structured-log-dir", default=STRUCTURED_LOG_DIR)
    args = parser.parse_args()
    init_logger()
    sl.init_structured_logger(
        source="worldmodel_torchpackage_checkpoint",
        output_dir=args.structured_log_dir,
    )
    with sl.log_trace_span("worldmodel_package_total"):
        if args.model_config_path is not None:
            if args.flag_checkpoint_path is None:
                parser.error("--checkpoint-path is required with --model-config-path")
            if args.flag_output_path is None:
                parser.error("--output-path is required with --model-config-path")
            if args.step is None:
                parser.error("--step is required with --model-config-path")
            model_config = load_model_config(args.model_config_path)
            try:
                export_torch_package(
                    checkpoint_path=args.flag_checkpoint_path,
                    output_path=args.flag_output_path,
                    model_config=model_config,
                    step=args.step,
                    model_config_path=args.model_config_path,
                )
            finally:
                del model_config
                gc.collect()
            return

        checkpoint_path = args.flag_checkpoint_path or args.checkpoint_path
        output_path = args.flag_output_path or args.output_path
        if checkpoint_path is None:
            parser.error("checkpoint_path is required")

        step = int(posixpath.basename(checkpoint_path.rstrip("/")))
        if "://" not in checkpoint_path and not os.path.exists(checkpoint_path):
            parts = checkpoint_path.strip("/").split("/")
            if len(parts) == 2:
                run_id, checkpoint_step = parts
                checkpoint_path = posixpath.join(
                    REPORTERV2_HOST.rstrip("/"),
                    "checkpoint",
                    run_id,
                    checkpoint_step,
                )
        if output_path is None:
            output_path = posixpath.join(checkpoint_path.rstrip("/"), PACKAGE_NAME)

        with sl.log_trace_span("worldmodel_package_load_config"):
            model_config = copy_model_config(model_registry(args.flavor).model)
        export_torch_package(
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            model_config=model_config,
            step=step,
        )
        del model_config
        gc.collect()


if __name__ == "__main__":
    main()
