# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.utils.hooks import RemovableHandle

from torchtitan.distributed import utils as dist_utils

from .model import PathSelfAttention
from .vit import PlanViT


@dataclass(kw_only=True, slots=True)
class PlanViTAttentionDiagnosticsConfig:
    enable: bool = False
    start_step: int = 2_000
    interval: int = 10_000

    def __post_init__(self) -> None:
        if self.start_step < 1:
            raise ValueError("attention diagnostics start_step must be positive")
        if self.interval < 1:
            raise ValueError("attention diagnostics interval must be positive")


def _max_attention_logit(
    q_BLNH: torch.Tensor,
    k_BLNH: torch.Tensor,
    *,
    scale: float | None,
    is_causal: bool,
) -> torch.Tensor:
    """Return the largest pre-softmax attention score in one local batch."""
    if q_BLNH.ndim != 4 or k_BLNH.ndim != 4:
        raise ValueError("attention diagnostics require q and k shaped (B, L, N, H)")
    if q_BLNH.shape[0] != k_BLNH.shape[0]:
        raise ValueError("attention diagnostics require matching q and k batches")
    if q_BLNH.shape[2:] != k_BLNH.shape[2:]:
        raise ValueError("attention diagnostics require matching q and k heads")

    score_scale = q_BLNH.shape[-1] ** -0.5 if scale is None else scale
    q_BNLH = q_BLNH.detach().float().transpose(1, 2)
    k_BNLH = k_BLNH.detach().float().transpose(1, 2)
    scores_BNLS = torch.matmul(q_BNLH, k_BNLH.transpose(-2, -1)) * score_scale
    if is_causal:
        causal_LS = torch.ones(
            scores_BNLS.shape[-2:], dtype=torch.bool, device=scores_BNLS.device
        ).tril()
        scores_BNLS.masked_fill_(~causal_LS, -torch.inf)
    return scores_BNLS.amax()


class PlanViTAttentionDiagnostics:
    """Sparsely measure attention and readout scale without changing model outputs."""

    def __init__(
        self,
        config: PlanViTAttentionDiagnosticsConfig,
        model_parts: list[nn.Module],
        *,
        total_steps: int,
        log_freq: int,
    ) -> None:
        self.config = config
        self.total_steps = total_steps
        self.log_freq = log_freq
        self._active = False
        self._local_max: torch.Tensor | None = None
        self._local_layer0_max: torch.Tensor | None = None
        self._output_abs_mean_sum: torch.Tensor | None = None
        self._output_calls = 0
        self._handles: list[RemovableHandle] = []

        if not config.enable:
            return
        if config.start_step % log_freq != 0 or config.interval % log_freq != 0:
            raise ValueError(
                "attention diagnostics start_step and interval must be multiples "
                "of metrics.log_freq"
            )
        if len(model_parts) != 1 or not isinstance(model_parts[0], PlanViT):
            raise ValueError("attention diagnostics currently support PlanViT only")

        attention_modules = [
            module
            for module in model_parts[0].modules()
            if isinstance(module, PathSelfAttention)
        ]
        if not attention_modules:
            raise ValueError("PlanViT has no PathSelfAttention modules to diagnose")
        for layer_id, module in enumerate(attention_modules):
            self._handles.append(
                module.inner_attention.register_forward_pre_hook(
                    self._make_hook(
                        layer_id=layer_id, is_causal=module.is_causal
                    ),
                    with_kwargs=True,
                )
            )
        self._handles.append(
            model_parts[0].plan_head.register_forward_hook(self._output_hook)
        )

    @property
    def enabled(self) -> bool:
        return self.config.enable

    def _make_hook(self, *, layer_id: int, is_causal: bool):
        def hook(_module, args, kwargs) -> None:
            if not self._active:
                return
            q_BLNH, k_BLNH = args[:2]
            local_max = _max_attention_logit(
                q_BLNH,
                k_BLNH,
                scale=kwargs.get("scale"),
                is_causal=is_causal,
            )
            self._local_max = (
                local_max
                if self._local_max is None
                else torch.maximum(self._local_max, local_max)
            )
            if layer_id == 0:
                self._local_layer0_max = (
                    local_max
                    if self._local_layer0_max is None
                    else torch.maximum(self._local_layer0_max, local_max)
                )

        return hook

    def _output_hook(self, _module, _args, output: torch.Tensor) -> None:
        if not self._active:
            return
        output_abs_mean = output.detach().float().abs().mean()
        self._output_abs_mean_sum = (
            output_abs_mean
            if self._output_abs_mean_sum is None
            else self._output_abs_mean_sum + output_abs_mean
        )
        self._output_calls += 1

    def begin_step(self, step: int, *, will_log: bool) -> None:
        last_logged_step = self.total_steps - (self.total_steps % self.log_freq)
        if last_logged_step == 0:
            last_logged_step = self.total_steps
        periodic = step >= self.config.start_step and (
            step - self.config.start_step
        ) % self.config.interval == 0
        self._active = self.enabled and will_log and (
            periodic or step == last_logged_step
        )
        self._local_max = None
        self._local_layer0_max = None
        self._output_abs_mean_sum = None
        self._output_calls = 0

    def metrics(self, batch_mesh: DeviceMesh | None) -> dict[str, float]:
        if not self._active:
            return {}
        if self._local_max is None:
            raise RuntimeError(
                "attention diagnostics were active but observed no attention calls"
            )
        if self._local_layer0_max is None:
            raise RuntimeError(
                "attention diagnostics were active but did not observe layer 0"
            )
        if self._output_abs_mean_sum is None or self._output_calls == 0:
            raise RuntimeError(
                "attention diagnostics were active but observed no PlanViT output"
            )
        local_output_abs_mean = self._output_abs_mean_sum / self._output_calls
        global_max = (
            dist_utils.dist_max(self._local_max, batch_mesh)
            if batch_mesh is not None
            else float(self._local_max.item())
        )
        global_layer0_max = (
            dist_utils.dist_max(self._local_layer0_max, batch_mesh)
            if batch_mesh is not None
            else float(self._local_layer0_max.item())
        )
        global_output_abs_mean = (
            dist_utils.dist_mean(local_output_abs_mean, batch_mesh)
            if batch_mesh is not None
            else float(local_output_abs_mean.item())
        )
        return {
            "model/attention_logit_max": global_max,
            "model/attention_logit_max_layer0": global_layer0_max,
            "model/plan_output_abs_mean": global_output_abs_mean,
        }

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
