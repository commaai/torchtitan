# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Cache one Wan UMT5 prompt embedding for text-encoder-free training."""

from __future__ import annotations

import argparse
import importlib
import pathlib
import sys
import types

import torch
from safetensors.torch import save_file

from .tokenizer import (
    DEFAULT_WAN_TEXT_CONTEXT_FILENAME,
    DEFAULT_WAN_TEXT_PROMPT,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, required=True)
    parser.add_argument("--wan-repo", type=pathlib.Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_WAN_TEXT_PROMPT)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir.resolve()
    wan_repo = args.wan_repo.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else checkpoint_dir / DEFAULT_WAN_TEXT_CONTEXT_FILENAME
    )
    if output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite existing context: {output}")
    if not (wan_repo / "wan/modules/t5.py").is_file():
        raise FileNotFoundError(f"not a Wan2.2 checkout: {wan_repo}")

    # Import only the T5 files. Wan's package initializers eagerly import its
    # full generation stack and optional CLI dependencies such as easydict.
    wan_package = types.ModuleType("wan")
    wan_package.__file__ = str(wan_repo / "wan/__init__.py")
    wan_package.__path__ = [str(wan_repo / "wan")]
    wan_package.__package__ = "wan"
    sys.modules["wan"] = wan_package
    modules_package = types.ModuleType("wan.modules")
    modules_package.__file__ = str(wan_repo / "wan/modules/__init__.py")
    modules_package.__path__ = [str(wan_repo / "wan/modules")]
    modules_package.__package__ = "wan.modules"
    sys.modules["wan.modules"] = modules_package
    tokenizers_module = types.ModuleType("wan.modules.tokenizers")

    class LocalHuggingfaceTokenizer:
        def __init__(self, name: pathlib.Path, seq_len: int, clean: str) -> None:
            from transformers import AutoTokenizer

            if clean != "whitespace":
                raise ValueError(f"unsupported text cleaning mode: {clean}")
            self.tokenizer = AutoTokenizer.from_pretrained(name)
            self.seq_len = seq_len

        def __call__(self, texts: list[str], **kwargs):
            return_mask = kwargs.pop("return_mask", False)
            cleaned = [" ".join(text.split()) for text in texts]
            tokens = self.tokenizer(
                cleaned,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.seq_len,
                **kwargs,
            )
            if return_mask:
                return tokens.input_ids, tokens.attention_mask
            return tokens.input_ids

    tokenizers_module.HuggingfaceTokenizer = LocalHuggingfaceTokenizer
    sys.modules["wan.modules.tokenizers"] = tokenizers_module
    T5EncoderModel = importlib.import_module("wan.modules.t5").T5EncoderModel

    device = torch.device(args.device)
    encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=checkpoint_dir / "models_t5_umt5-xxl-enc-bf16.pth",
        tokenizer_path=checkpoint_dir / "google/umt5-xxl",
    )
    with torch.inference_mode():
        context_LC = encoder([args.prompt], device)[0].cpu().contiguous()
    if context_LC.ndim != 2 or context_LC.shape[1] != 4096:
        raise RuntimeError(f"unexpected UMT5 context shape: {tuple(context_LC.shape)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"context": context_LC},
        str(output),
        metadata={
            "encoder": "umt5-xxl",
            "prompt": args.prompt,
            "text_len": "512",
        },
    )
    print(
        {
            "output": str(output),
            "prompt": args.prompt,
            "shape": tuple(context_LC.shape),
            "dtype": str(context_LC.dtype),
        }
    )


if __name__ == "__main__":
    main()
