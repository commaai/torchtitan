# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import gc
import subprocess
import time
from collections.abc import Generator
from dataclasses import dataclass
from types import ModuleType

import torch
from torch._utils import _get_available_device_type, _get_device_module

from torchtitan.observability import structured_logger as sl
from torchtitan.tools.logging import logger


_DTypeLike = str | torch.dtype


def has_cuda_capability(major: int, minor: int) -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (
        major,
        minor,
    )


def has_rocm_capability(major: int, minor: int) -> bool:
    is_rocm = torch.cuda.is_available() and torch.version.hip is not None
    return is_rocm and torch.cuda.get_device_capability() >= (
        major,
        minor,
    )


def get_device_info() -> tuple[str, ModuleType]:
    device_type = _get_available_device_type() or "cuda"
    device_module = _get_device_module(device_type)  # default device_module:torch.cuda
    return device_type, device_module


device_type, device_module = get_device_info()


# used to avoid stragglers in garbage collection
class GarbageCollection:
    def __init__(self, gc_freq: int = 1000, debug: bool = False):
        assert gc_freq > 0, "gc_freq must be a positive integer"
        self.gc_freq = gc_freq
        self.debug = debug
        gc.disable()
        self.collect("Initial GC collection")
        if debug:
            from torch.utils.viz._cycles import warn_tensor_cycles

            if torch.distributed.get_rank() == 0:
                warn_tensor_cycles()

    @sl.log_trace_span("gc_collect")
    def run(self, step_count: int) -> bool:
        """Run a GC cycle if this step should collect. Returns True when a
        collection actually ran, False otherwise."""
        if self.debug:
            self.collect(
                "Force GC to perform collection to obtain debug information",
                generation=2,
            )
            gc.collect()
            sl.add_step_tag("gc")
            return True
        if step_count > 1 and step_count % self.gc_freq == 0:
            self.collect("Performing periodic GC collection")
            sl.add_step_tag("gc")
            return True
        return False

    @staticmethod
    def collect(reason: str, generation: int = 1):
        begin = time.monotonic()
        gc.collect(generation)
        logger.info("[GC] %s took %.2f seconds", reason, time.monotonic() - begin)


_PEAK_FLOPS = {
    "A100": {
        ("float32", "float32"): 19.5e12,
        ("bfloat16", "float32"): 312e12,
    },
    "A6000": {
        ("float32", "float32"): 38.7e12,
    },
    "H100 NVL": {
        ("float32", "float32"): 67e12,
        ("bfloat16", "float32"): 835e12,
    },
    "H100 PCIe": {
        ("float32", "float32"): 51e12,
        ("bfloat16", "float32"): 756e12,
    },
    "H100": {
        ("float32", "float32"): 67e12,
        ("bfloat16", "float32"): 989e12,
    },
    "H200": {
        ("float32", "float32"): 67e12,
        ("bfloat16", "float32"): 989e12,
    },
    "H20": {
        ("bfloat16", "float32"): 148e12,
    },
    "GB200": {
        ("bfloat16", "float32"): 2.5e15,
    },
    "GB300": {
        ("bfloat16", "float32"): 2.5e15,
    },
    "B300": {
        ("bfloat16", "float32"): 2.25e15,
    },
    "B200": {
        ("bfloat16", "float32"): 2.25e15,
    },
    "RTX 5090": {
        ("float32", "float32"): 104.8e12,
        ("bfloat16", "float32"): 209.5e12,
        ("bfloat16", "bfloat16"): 419e12,
        ("float8", "bfloat16"): 838e12,
    },
    "MI355X": {
        ("bfloat16", "float32"): 2500e12,
    },
    "MI300X": {
        ("bfloat16", "float32"): 1300e12,
    },
    "MI325X": {
        ("bfloat16", "float32"): 1300e12,
    },
    "MI250X": {
        ("bfloat16", "float32"): 191.5e12,
    },
    "l40s": {
        ("float32", "float32"): 91.6e12,
        ("bfloat16", "float32"): 362e12,
    },
}


def _normalize_peak_flops_dtype(dtype: _DTypeLike) -> str:
    return str(dtype).lower().removeprefix("torch.")


def get_peak_flops(
    device_name: str,
    mul_dtype: _DTypeLike = torch.bfloat16,
    acc_dtype: _DTypeLike = torch.float32,
) -> float:
    mul_dtype = _normalize_peak_flops_dtype(mul_dtype)
    acc_dtype = _normalize_peak_flops_dtype(acc_dtype)

    try:
        # Run the lspci command and capture the output
        result = subprocess.run(["lspci"], stdout=subprocess.PIPE, text=True)
        # Filter the output for lines containing both "NVIDIA" and "H100"
        filtered_lines = [
            line
            for line in result.stdout.splitlines()
            if "NVIDIA" in line and "H100" in line
        ]
        # Join all filtered lines into a single string
        device_name = " ".join(filtered_lines) or device_name
    except FileNotFoundError as e:
        logger.warning(f"Error running lspci: {e}, fallback to use device_name")
    if "Data Center GPU Max 1550" in device_name:
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        device_flops = {("bfloat16", "float32"): 512 * max_comp_units * 1300 * 10**6}
    elif "neuron" in device_name:
        neuron_device_name = device_module.get_device_properties().name
        if neuron_device_name in ("trn1", "trn1n", "inf2"):
            device_flops = {("bfloat16", "float32"): 90e12}
        elif neuron_device_name in ("trn2", "trn2n", "trn2u", "trn3", "trn3u"):
            device_flops = {("bfloat16", "float32"): 79e12 * 2}
        else:
            logger.warning(
                f"Unknown neuron device: {neuron_device_name}, fallback to trn2/trn3"
            )
            device_flops = {("bfloat16", "float32"): 79e12 * 2}
    else:
        device_flops = next(
            (
                flops
                for name, flops in _PEAK_FLOPS.items()
                if name.lower() in device_name.lower()
            ),
            None,
        )

    if device_flops is None:
        logger.warning(f"Peak flops undefined for: {device_name}, fallback to A100")
        device_flops = _PEAK_FLOPS["A100"]

    key = (mul_dtype, acc_dtype)
    for fallback_key in (
        key,
        (mul_dtype, "float32"),
        (mul_dtype, "bfloat16"),
        ("bfloat16", acc_dtype),
        ("bfloat16", "float32"),
        ("float32", "float32"),
    ):
        if fallback_key in device_flops:
            if fallback_key != key:
                logger.warning(
                    f"Peak flops undefined for {device_name} with multiply={mul_dtype}, "
                    f"accumulate={acc_dtype}; using multiply={fallback_key[0]}, "
                    f"accumulate={fallback_key[1]} peak."
                )
            return device_flops[fallback_key]

    logger.warning(
        f"Peak flops undefined for: {device_name}, fallback to A100 BF16/FP32"
    )
    return _PEAK_FLOPS["A100"][("bfloat16", "float32")]


@dataclass(frozen=True)
class Color:
    black = "\033[30m"
    red = "\033[31m"
    green = "\033[32m"
    yellow = "\033[33m"
    blue = "\033[34m"
    magenta = "\033[35m"
    cyan = "\033[36m"
    white = "\033[37m"
    reset = "\033[39m"
    orange = "\033[38;2;180;60;0m"
    turquoise = "\033[38;2;54;234;195m"


@dataclass(frozen=True)
class NoColor:
    black = ""
    red = ""
    green = ""
    yellow = ""
    blue = ""
    magenta = ""
    cyan = ""
    white = ""
    reset = ""
    orange = ""
    turquoise = ""


assert set(NoColor.__dataclass_fields__.keys()) == set(
    Color.__dataclass_fields__.keys()
), "NoColor must have the same fields as Color."


def check_if_feature_in_pytorch(
    feature_name: str,
    pull_request: str,
    min_nightly_version: str | None = None,
) -> None:
    if "git" in torch.__version__:  # pytorch is built from source
        # notify users to check if the pull request is included in their pytorch
        logger.warning(
            "Detected that the pytorch is built from source. Please make sure the PR "
            f"({pull_request}) is included in pytorch for correct {feature_name}."
        )
    elif min_nightly_version is not None and torch.__version__ < min_nightly_version:
        logger.warning(
            f"Detected that the pytorch version {torch.__version__} is older than "
            f"{min_nightly_version}. Please upgrade a newer version to include the "
            f"change in ({pull_request}) for correct {feature_name}."
        )


@contextlib.contextmanager
def set_default_dtype(dtype: torch.dtype) -> Generator[None, None, None]:
    """
    Context manager to set torch's default dtype.

    Args:
        dtype (torch.dtype): The desired default dtype inside the context manager.

    Returns:
        ContextManager: context manager for setting default dtype.

    Example:
        >>> with set_default_dtype(torch.bfloat16):
        >>>     x = torch.tensor([1, 2, 3])
        >>>     x.dtype
        torch.bfloat16


    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)
