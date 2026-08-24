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
from torchtitan.experiments.worldmodel.model_wan import WanModel
from torchtitan.experiments.worldmodel.model_wan_for_inference import (
    PACKAGED_TEXT_CONTEXT_BUFFER,
    WanModelForInference,
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
_WORLD_MODEL_TORCH_PACKAGE_MODULE = "torchtitan.experiments.worldmodel.torchpackage_checkpoint"
WORLD_MODEL_TORCH_PACKAGE_RECIPE = f"{_WORLD_MODEL_TORCH_PACKAGE_MODULE}:WorldModelTorchPackageRecipe"
WORLD_MODEL_TRAINING_TORCH_PACKAGE_RECIPE = f"{_WORLD_MODEL_TORCH_PACKAGE_MODULE}:WorldModelTrainingTorchPackageRecipe"

TORCH_EXPORT_INTERN_MODULES = [
    "torchtitan.config.**",
    "torchtitan.distributed",
    "torchtitan.distributed.compile",
    "torchtitan.distributed.parallel_dims",
    "torchtitan.distributed.spmd_types",
    "torchtitan.distributed.utils",
    "torchtitan.experiments.worldmodel.model_for_inference",
    "torchtitan.experiments.worldmodel.model",
    "torchtitan.experiments.worldmodel.model_wan_for_inference",
    "torchtitan.experiments.worldmodel.model_wan",
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
    "safetensors.**",
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
    # Training-only Wan helpers import these lazily; packaged inference never
    # invokes them and should not pull their dependency trees into the archive.
    "torchtitan.distributed.activation_checkpoint",
    "torchtitan.distributed.fsdp",
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


@dataclass(slots=True)
class WanTorchPackageConfig:
    """Wan architecture and its fixed raw prompt context for package export."""

    model_config: WanModel.Config
    text_context_LC: torch.Tensor | None = None
    text_prompt: str = ""


TorchPackageConfig = WorldModel.Config | WanModel.Config | WanTorchPackageConfig


WEIGHT_FORMAT_SPECS: dict[WeightFormat, WeightFormatSpec] = {
    "bf16": WeightFormatSpec(BF16_KV_CACHE_DTYPE, None),
    "fp8": WeightFormatSpec(FP8_KV_CACHE_DTYPE, (8, 9)),
    "fp8_nvfp4": WeightFormatSpec(FP8_KV_CACHE_DTYPE, (10, 0)),
}


def build_meta_model(
    model_config: WorldModel.Config | WanModel.Config,
    *,
    dtype: torch.dtype = torch.bfloat16,
    default_kv_cache_dtype: KVCacheDType = FP8_KV_CACHE_DTYPE,
) -> WorldModelForInference | WanModelForInference:
    with torch.device("meta"):
        if isinstance(model_config, WanModel.Config):
            model = WanModelForInference(
                model_config,
                default_kv_cache_dtype=default_kv_cache_dtype,
            )
        else:
            model = WorldModelForInference(
                model_config,
                default_kv_cache_dtype=default_kv_cache_dtype,
            )
        return model.to(dtype=dtype).eval()


def validate_model_config(state: Any) -> WorldModel.Config | WanModel.Config:
    if isinstance(state, WanTorchPackageConfig):
        state = state.model_config
    if not isinstance(state, (WorldModel.Config, WanModel.Config)):
        raise TypeError(
            "Worldmodel torch package state must be WorldModel.Config, "
            f"WanModel.Config, or WanTorchPackageConfig, got {type(state).__name__}."
        )
    return state


def validate_package_config(state: Any) -> TorchPackageConfig:
    validate_model_config(state)
    if isinstance(state, WanTorchPackageConfig) and state.text_context_LC is not None:
        _normalize_wan_text_context(
            state.model_config,
            state.text_context_LC,
        )
    return state


def load_package_config(path: str) -> TorchPackageConfig:
    return validate_package_config(load_recipe_state(path))


def load_model_config(path: str) -> TorchPackageConfig:
    return load_package_config(path)


def _normalize_wan_text_context(
    model_config: WanModel.Config,
    text_context_LC: torch.Tensor | None,
) -> torch.Tensor | None:
    if text_context_LC is None:
        return None
    if text_context_LC.ndim != 2 or text_context_LC.size(1) != model_config.text_dim:
        raise ValueError(
            f"Wan package text context must have shape [L, {model_config.text_dim}], "
            f"got {tuple(text_context_LC.shape)}"
        )
    if not 0 < text_context_LC.size(0) <= model_config.text_len:
        raise ValueError(
            f"Wan package text context length must be in [1, {model_config.text_len}], "
            f"got {text_context_LC.size(0)}"
        )
    if not torch.isfinite(text_context_LC).all():
        raise ValueError("Wan package text context contains non-finite values")
    return text_context_LC.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()


def convert_state_dict_for_inference(
    model_config: WorldModel.Config | WanModel.Config,
    state_dict: dict[str, torch.Tensor],
    *,
    weight_format: WeightFormat = DEFAULT_WEIGHT_FORMAT,
) -> dict[str, torch.Tensor]:
    with torch.device("cpu"):
        if isinstance(model_config, WanModel.Config):
            model = WanModelForInference(model_config).to(dtype=torch.bfloat16).eval()
        else:
            model = WorldModelForInference(model_config).to(dtype=torch.bfloat16).eval()
    model.load_state_dict(state_dict, strict=True, assign=True)
    state_dict.clear()
    del state_dict
    model.to(dtype=torch.bfloat16)
    model.quantize_for_inference(weight_format)
    converted = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    del model
    gc.collect()
    return converted


def build_package(
    *,
    model_config: WorldModel.Config | WanModel.Config,
    state_dict: dict[str, torch.Tensor],
    step: int,
    weight_format: WeightFormat = DEFAULT_WEIGHT_FORMAT,
    text_context_LC: torch.Tensor | None = None,
    text_prompt: str = "",
) -> bytes:
    format_spec = WEIGHT_FORMAT_SPECS[weight_format]
    if text_context_LC is not None and not isinstance(model_config, WanModel.Config):
        raise TypeError("text_context_LC is supported only for Wan torch packages")
    packaged_text_context_LC = (
        _normalize_wan_text_context(model_config, text_context_LC)
        if isinstance(model_config, WanModel.Config)
        else None
    )

    with sl.log_trace_span(f"worldmodel_package_convert_{weight_format}"):
        state_dict = convert_state_dict_for_inference(
            model_config,
            state_dict,
            weight_format=weight_format,
        )

    with sl.log_trace_span("worldmodel_package_model_io"):
        if isinstance(model_config, WanModel.Config):
            io_model = build_meta_model(model_config)
            model_io = io_model.get_model_io(
                dtype=torch.bfloat16,
                steps=1,
                num_prefill_frames=1,
                cfg=2.0,
            )
            io_model_config = None
        else:
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
        if packaged_text_context_LC is not None:
            assert isinstance(model, WanModelForInference)
            model.set_packaged_text_context(
                torch.empty_like(
                    packaged_text_context_LC,
                    device="meta",
                )
            )
            state_dict[PACKAGED_TEXT_CONTEXT_BUFFER] = packaged_text_context_LC

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
            strip_future_annotations_modules = TORCH_EXPORT_STRIP_FUTURE_ANNOTATIONS_MODULES
            if isinstance(model_config, WanModel.Config):
                strip_future_annotations_modules = [
                    *strip_future_annotations_modules,
                    "torchtitan.experiments.worldmodel.model_wan",
                ]
            for module_name in strip_future_annotations_modules:
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
                "text_prompt": text_prompt if packaged_text_context_LC is not None else "",
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
        package_config = validate_package_config(state)
        model_config = validate_model_config(package_config)
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
        package_config = validate_package_config(state)
        model_config = validate_model_config(package_config)
        text_context_LC = (
            package_config.text_context_LC
            if isinstance(package_config, WanTorchPackageConfig)
            else None
        )
        text_prompt = (
            package_config.text_prompt
            if isinstance(package_config, WanTorchPackageConfig)
            else ""
        )
        return {
            PACKAGE_NAME: build_package(
                model_config=model_config,
                state_dict=state_dict,
                step=step,
                weight_format=self.weight_format,
                text_context_LC=text_context_LC,
                text_prompt=text_prompt,
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
        package_config = validate_package_config(state)
        model_config = validate_model_config(package_config)
        text_context_LC = (
            package_config.text_context_LC
            if isinstance(package_config, WanTorchPackageConfig)
            else None
        )
        text_prompt = (
            package_config.text_prompt
            if isinstance(package_config, WanTorchPackageConfig)
            else ""
        )
        return {
            FORMAT_PACKAGE_NAME.format(weight_format=weight_format): build_package(
                model_config=model_config,
                state_dict=state_dict.copy(),
                step=step,
                weight_format=weight_format,
                text_context_LC=text_context_LC,
                text_prompt=text_prompt,
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
    step_name = fs.basename(checkpoint_path)
    step = step_name.removeprefix("step-")
    assert step.isdigit(), f"checkpoint path {checkpoint_path} does not end with a step number."
    if model_flavor is None:
        package_config = load_model_config(recipe_state_path)
    else:
        from torchtitan.experiments.worldmodel.model_config import model_registry

        package_config = model_registry(model_flavor).model
    try:
        export_recipe_torch_package(
            recipe=WorldModelTorchPackageRecipe(weight_format=weight_format),
            checkpoint_path=checkpoint_path,
            recipe_state=package_config,
            step=int(step),
            recipe_state_path=(recipe_state_path if model_flavor is None else None),
        )
    finally:
        del package_config
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
