# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Verify Wan VAE checkpoint compatibility, output geometry, and causality."""

from __future__ import annotations

import argparse
import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType

import torch

from .model import WAN_VAE_MEAN, WAN_VAE_STD, WanVAE


def _autocast(device: torch.device, dtype: torch.dtype):
    if dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _load_official_module(source_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("wan22_official_vae", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import official VAE source from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.inference_mode()
def verify_checkpoint(
    checkpoint: Path,
    *,
    device: torch.device,
    compute_dtype: torch.dtype,
    official_source: Path | None = None,
) -> dict[str, object]:
    model = WanVAE.from_pretrained(checkpoint, device=device)

    # Geometry contract at the requested training resolution. The memory-safe
    # official four-frame stream is used here.
    generator = torch.Generator(device="cpu").manual_seed(0)
    video = torch.randn(
        1,
        3,
        57,
        256,
        512,
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    with _autocast(device, compute_dtype):
        latents = model.encode(video, chunk_size=4)
    expected_shape = (1, 48, 15, 16, 32)
    if tuple(latents.shape) != expected_shape:
        raise AssertionError(f"57x256x512 encoded to {tuple(latents.shape)}, expected {expected_shape}")
    del video, latents
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # A full-resolution coalesced encode is unnecessarily memory hungry. Test
    # chunk equivalence over all 57 timesteps at a smaller spatial grid; the
    # temporal operators and feature-cache transitions are unchanged.
    small_video = torch.randn(
        1,
        3,
        57,
        32,
        32,
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    with _autocast(device, compute_dtype):
        chunked = model.encode(small_video, chunk_size=4)
        coalesced = model.encode(small_video, chunk_size=None)
    chunk_max_abs = float((chunked - coalesced).abs().max().cpu())
    chunk_mean_abs = float((chunked - coalesced).abs().mean().cpu())
    # Different temporal extents select different convolution kernels. Their
    # accumulation order is not bit-identical even though the causal operation
    # is equivalent (observed relative maxima: <0.04% FP32, <0.8% BF16).
    tolerance = 4e-2 if compute_dtype == torch.bfloat16 else 2e-3
    if chunk_max_abs > tolerance:
        raise AssertionError(f"chunked/coalesced encodes differ by {chunk_max_abs}, tolerance {tolerance}")

    result: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_keys": len(model.state_dict()),
        "geometry": {
            "input": (1, 3, 57, 256, 512),
            "latents": expected_shape,
        },
        "chunk_equivalence": {
            "max_abs": chunk_max_abs,
            "mean_abs": chunk_mean_abs,
            "compute_dtype": str(compute_dtype),
        },
    }

    if official_source is not None:
        official_module = _load_official_module(official_source)
        with torch.device("meta"):
            official = official_module.WanVAE_(
                dim=160,
                dec_dim=256,
                z_dim=48,
                dim_mult=[1, 2, 4, 4],
                num_res_blocks=2,
                attn_scales=[],
                temperal_downsample=[False, True, True],
                dropout=0.0,
            )
        state_dict = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        official.load_state_dict(state_dict, strict=True, assign=True)
        official.to(device).eval().requires_grad_(False)
        scale = [
            torch.tensor(WAN_VAE_MEAN, device=device, dtype=compute_dtype),
            torch.tensor(WAN_VAE_STD, device=device, dtype=compute_dtype).reciprocal(),
        ]
        with _autocast(device, compute_dtype):
            reference = official.encode(small_video, scale).float()
            native_reconstruction = model.decode(chunked, clamp=False, output_dtype=torch.float32)
            reference_reconstruction = official.decode(reference, scale).float()
        reference_max_abs = float((chunked - reference).abs().max().cpu())
        reference_mean_abs = float((chunked - reference).abs().mean().cpu())
        decode_max_abs = float((native_reconstruction - reference_reconstruction).abs().max().cpu())
        decode_mean_abs = float((native_reconstruction - reference_reconstruction).abs().mean().cpu())
        if reference_max_abs > tolerance:
            raise AssertionError(f"native/upstream encodes differ by {reference_max_abs}, tolerance {tolerance}")
        if decode_max_abs > tolerance:
            raise AssertionError(f"native/upstream decodes differ by {decode_max_abs}, tolerance {tolerance}")
        result["official_parity"] = {
            "source": str(official_source),
            "encode_max_abs": reference_max_abs,
            "encode_mean_abs": reference_mean_abs,
            "decode_max_abs": decode_max_abs,
            "decode_mean_abs": decode_mean_abs,
        }

    return result


def main() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root / "weights/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=("float32", "bfloat16"),
        default="bfloat16" if torch.cuda.is_available() else "float32",
    )
    parser.add_argument(
        "--official-source",
        type=Path,
        help="optional path to Wan2.2/wan/modules/vae2_2.py for output parity",
    )
    args = parser.parse_args()
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[args.compute_dtype]
    result = verify_checkpoint(
        args.checkpoint,
        device=torch.device(args.device),
        compute_dtype=dtype,
        official_source=args.official_source,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
