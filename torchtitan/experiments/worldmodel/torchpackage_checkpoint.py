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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.package import PackageExporter

from torchtitan.components import fs
from torchtitan.components.torchpackage_checkpoint import (
    export_torch_package as export_recipe_torch_package,
    load_recipe_state,
    TorchPackageCheckpointManager,
)
from torchtitan.experiments.worldmodel.model import WorldModel
from torchtitan.experiments.worldmodel.model_for_inference import (
    BF16_KV_CACHE_DTYPE,
    FP8_KV_CACHE_DTYPE,
    KVCacheDType,
    WEIGHT_FORMATS,
    WeightFormat,
    WorldModelForInference,
)
from torchtitan.observability import structured_logger as sl
from torchtitan.tools.logging import init_logger


os.environ.setdefault("NCCL_P2P_DISABLE", "1")

PACKAGE_NAME = "model.torchpackage"
FORMAT_PACKAGE_NAME = "model.{weight_format}.torchpackage"
DEFAULT_WEIGHT_FORMAT: WeightFormat = "fp8_nvfp4"
MODEL_CONFIG_FILE = "_torchpackage_model_config.pt"
STRUCTURED_LOG_DIR = os.getenv(
    "TORCHTITAN_STRUCTURED_LOG_DIR",
    "/tmp/torchtitan_train/worldmodel_torchpackage_checkpoint",
)
WORLD_MODEL_TORCH_PACKAGE_RECIPE = (
    "torchtitan.experiments.worldmodel.torchpackage_checkpoint:" "WorldModelTorchPackageRecipe"
)
WORLD_MODEL_TRAINING_TORCH_PACKAGE_RECIPE = (
    "torchtitan.experiments.worldmodel.torchpackage_checkpoint:" "WorldModelTrainingTorchPackageRecipe"
)

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


@dataclass(frozen=True, slots=True)
class WeightFormatSpec:
    kv_cache_dtype: KVCacheDType
    minimum_compute_capability: tuple[int, int] | None


WEIGHT_FORMAT_SPECS: dict[WeightFormat, WeightFormatSpec] = {
    "bf16": WeightFormatSpec(BF16_KV_CACHE_DTYPE, None),
    "fp8": WeightFormatSpec(FP8_KV_CACHE_DTYPE, (8, 9)),
    "fp8_nvfp4": WeightFormatSpec(FP8_KV_CACHE_DTYPE, (10, 0)),
}


def build_meta_model(
    model_config: WorldModel.Config,
    *,
    dtype: torch.dtype = torch.bfloat16,
    default_kv_cache_dtype: KVCacheDType = FP8_KV_CACHE_DTYPE,
) -> WorldModelForInference:
    with torch.device("meta"):
        return (
            WorldModelForInference(
                model_config,
                default_kv_cache_dtype=default_kv_cache_dtype,
            )
            .to(dtype=dtype)
            .eval()
        )


def validate_model_config(state: Any) -> WorldModel.Config:
    if not isinstance(state, WorldModel.Config):
        raise TypeError(f"Worldmodel torch package state must be WorldModel.Config, " f"got {type(state).__name__}.")
    return state


def load_model_config(path: str) -> WorldModel.Config:
    return validate_model_config(load_recipe_state(path))


def convert_state_dict_for_inference(
    model_config: WorldModel.Config,
    state_dict: dict[str, torch.Tensor],
    *,
    weight_format: WeightFormat = DEFAULT_WEIGHT_FORMAT,
) -> dict[str, torch.Tensor]:
    with torch.device("cpu"):
        model = WorldModelForInference(model_config).to(dtype=torch.bfloat16).eval()
    model.load_state_dict(state_dict, strict=True, assign=True)
    state_dict.clear()
    del state_dict
    model.quantize_for_inference(weight_format)
    converted = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    del model
    gc.collect()
    return converted


def build_package(
    *,
    model_config: WorldModel.Config,
    state_dict: dict[str, torch.Tensor],
    step: int,
    weight_format: WeightFormat = DEFAULT_WEIGHT_FORMAT,
) -> bytes:
    format_spec = WEIGHT_FORMAT_SPECS[weight_format]

    with sl.log_trace_span(f"worldmodel_package_convert_{weight_format}"):
        state_dict = convert_state_dict_for_inference(
            model_config,
            state_dict,
            weight_format=weight_format,
        )

    with sl.log_trace_span("worldmodel_package_model_io"):
        io_model_config = copy.deepcopy(model_config)
        if io_model_config.transformer.attention_impl == "FLEX":
            io_model_config.transformer.attention_impl = "SDPA"
        if io_model_config.plan_head.attention_impl == "FLEX":
            io_model_config.plan_head.attention_impl = "SDPA"
        io_model = build_meta_model(io_model_config)
        assert io_model.config.transformer.attention_mask != "NONE"
        num_prefill_frames = io_model.config.input_size[0] - 1
        model_io = io_model.get_model_io(
            dtype=torch.bfloat16,
            steps=1,
            num_prefill_frames=num_prefill_frames,
            cfg=2.0,
        )
        del io_model, io_model_config
        model = build_meta_model(
            model_config,
            default_kv_cache_dtype=format_spec.kv_cache_dtype,
        )

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
                module_source = Path(spec.origin).read_text()
                module_source = module_source.replace("from __future__ import annotations\n\n", "")
                if module_name == "torchtitan.distributed.parallel_dims":
                    module_source = module_source.replace(") -> ParallelDims:\n", ') -> "ParallelDims":\n')
                exporter.save_source_string(module_name, module_source)
            exporter.mock(
                TORCH_EXPORT_MOCK_MODULES,
                exclude=TORCH_EXPORT_INTERN_MODULES + TORCH_EXPORT_EXTERN_MODULES + TORCH_EXPORT_DENY_MODULES,
            )
            exporter.save_pickle("model", "model.pkl", model)
            del model
            metadata = {
                "model_io": model_io,
                "step": step,
                "weight_format": weight_format,
                "kv_cache_dtype": format_spec.kv_cache_dtype,
                "minimum_compute_capability": format_spec.minimum_compute_capability,
            }
            exporter.save_pickle("meta", "meta.pkl", metadata)
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


@dataclass(slots=True)
class WorldModelTorchPackageRecipe:
    weight_format: WeightFormat = DEFAULT_WEIGHT_FORMAT

    def build_empty_state_dict(self, state: Any) -> dict[str, torch.Tensor]:
        model_config = validate_model_config(state)
        model = build_meta_model(model_config)
        try:
            return {
                name: torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
                for name, tensor in model.state_dict().items()
            }
        finally:
            del model
            gc.collect()

    def build_package(
        self,
        *,
        state: Any,
        state_dict: dict[str, torch.Tensor],
        step: int,
    ) -> dict[str, bytes]:
        model_config = validate_model_config(state)
        return {
            PACKAGE_NAME: build_package(
                model_config=model_config,
                state_dict=state_dict,
                step=step,
                weight_format=self.weight_format,
            )
        }


class WorldModelTrainingTorchPackageRecipe(WorldModelTorchPackageRecipe):
    weight_formats = WEIGHT_FORMATS

    def build_package(
        self,
        *,
        state: Any,
        state_dict: dict[str, torch.Tensor],
        step: int,
    ) -> dict[str, bytes]:
        model_config = validate_model_config(state)
        return {
            FORMAT_PACKAGE_NAME.format(weight_format=weight_format): build_package(
                model_config=model_config,
                state_dict=state_dict.copy(),
                step=step,
                weight_format=weight_format,
            )
            for weight_format in self.weight_formats
        }


def export_torch_package(
    checkpoint_path: str,
    *,
    weight_format: WeightFormat = DEFAULT_WEIGHT_FORMAT,
    model_flavor: str | None = None,
) -> None:
    recipe_state_path = fs.join_path(checkpoint_path, MODEL_CONFIG_FILE)
    step = fs.basename(checkpoint_path)
    assert step.isdigit(), f"checkpoint path {checkpoint_path} does not end with a step number."
    if model_flavor is None:
        model_config = load_model_config(recipe_state_path)
    else:
        from torchtitan.experiments.worldmodel.model_config import model_registry

        model_config = model_registry(model_flavor).model
    try:
        export_recipe_torch_package(
            recipe=WorldModelTorchPackageRecipe(weight_format=weight_format),
            checkpoint_path=checkpoint_path,
            recipe_state=model_config,
            step=int(step),
            recipe_state_path=(recipe_state_path if model_flavor is None else None),
        )
    finally:
        del model_config
        gc.collect()


class WorldModelTorchPackageCheckpointManager(TorchPackageCheckpointManager):
    """Worldmodel checkpoint manager configured with the worldmodel recipe."""

    @dataclass(kw_only=True, slots=True)
    class Config(TorchPackageCheckpointManager.Config):
        torch_package_recipe: str = WORLD_MODEL_TRAINING_TORCH_PACKAGE_RECIPE
        torch_package_recipe_state_file: str = MODEL_CONFIG_FILE
        torch_package_structured_log_dir: str = STRUCTURED_LOG_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Package a worldmodel DCP checkpoint for inference.")
    parser.add_argument("checkpoint_path")
    parser.add_argument(
        "--weight-format",
        choices=WEIGHT_FORMATS,
        default=DEFAULT_WEIGHT_FORMAT,
    )
    parser.add_argument("--model-flavor")
    args = parser.parse_args()

    init_logger()
    sl.init_structured_logger(
        source="worldmodel_torchpackage_checkpoint",
        output_dir=STRUCTURED_LOG_DIR,
    )
    with sl.log_trace_span("worldmodel_package_total"):
        export_torch_package(
            args.checkpoint_path,
            weight_format=args.weight_format,
            model_flavor=args.model_flavor,
        )


if __name__ == "__main__":
    main()
