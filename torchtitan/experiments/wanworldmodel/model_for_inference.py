# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Portions Copyright 2024-2025 The Alibaba Wan Team Authors and licensed
# under the Apache License, Version 2.0.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.experiments.worldmodel.model_for_inference import (
    _resolve_kv_cache_dtype,
    BF16_KV_CACHE_DTYPE,
    FP8_KV_CACHE_DTYPE,
    KVCacheDType,
    WeightFormat,
)
from torchtitan.experiments.wanworldmodel.model import (
    _autocast_disabled,
    _compact_padded_context,
    _gate_wan,
    _modulate_wan,
    _normalize_frame_indices,
    frame_block_causal_mask,
    sinusoidal_embedding_1d,
    WanAttentionBlock,
    WanCrossAttention,
    WanModel,
    WanSelfAttention,
)
from torchtitan.experiments.worldmodel.schedulers import RFScheduler


PACKAGED_TEXT_CONTEXT_BUFFER = "packaged_text_context"
WAN_INFERENCE_SAMPLERS = ("ode", "sde")
DEFAULT_WAN_INFERENCE_STEPS = 15
DEFAULT_WAN_INFERENCE_SCHEDULE = "linear"
DEFAULT_WAN_INFERENCE_SHIFT = 1.0


def _frame_indices_to_semantic_positions(
    frame_indices_BF: torch.Tensor,
    spatial_grid: tuple[int, int],
) -> torch.Tensor:
    """Expand latent-frame coordinates to Wan's flattened 3D token positions."""
    height, width = spatial_grid
    spatial_tokens = height * width
    spatial_positions_S = torch.arange(
        spatial_tokens,
        device=frame_indices_BF.device,
        dtype=torch.int64,
    )
    return (
        frame_indices_BF[:, :, None] * spatial_tokens
        + spatial_positions_S.view(1, 1, -1)
    ).flatten(1)


def _rope_apply_positions(
    x_BSND: torch.Tensor,
    semantic_pos_S: torch.Tensor,
    spatial_grid: tuple[int, int],
    freqs_MD: torch.Tensor,
) -> torch.Tensor:
    """Apply Wan 3D RoPE at arbitrary flattened semantic positions."""
    height, width = spatial_grid
    spatial_tokens = height * width
    if semantic_pos_S.ndim == 1:
        if semantic_pos_S.numel() != x_BSND.size(1):
            raise ValueError(
                f"semantic_pos must have shape [{x_BSND.size(1)}], got "
                f"{tuple(semantic_pos_S.shape)}"
            )
        positions_BS = semantic_pos_S.view(1, -1)
    elif semantic_pos_S.shape == x_BSND.shape[:2]:
        positions_BS = semantic_pos_S
    else:
        raise ValueError(
            "semantic_pos must have shape "
            f"[{x_BSND.size(1)}] or [{x_BSND.size(0)}, {x_BSND.size(1)}], "
            f"got {tuple(semantic_pos_S.shape)}"
        )

    positions_BS = positions_BS.to(device=x_BSND.device, dtype=torch.long)
    frames_BS = torch.div(positions_BS, spatial_tokens, rounding_mode="floor")
    spatial_pos_BS = positions_BS.remainder(spatial_tokens)
    heights_BS = torch.div(spatial_pos_BS, width, rounding_mode="floor")
    widths_BS = spatial_pos_BS.remainder(width)
    complex_dim = x_BSND.size(3) // 2
    temporal_freqs, height_freqs, width_freqs = freqs_MD.split(
        (complex_dim - 2 * (complex_dim // 3), complex_dim // 3, complex_dim // 3),
        dim=1,
    )
    position_freqs_BSD = torch.cat(
        (
            temporal_freqs.index_select(0, frames_BS.flatten()).unflatten(
                0,
                frames_BS.shape,
            ),
            height_freqs.index_select(0, heights_BS.flatten()).unflatten(
                0,
                heights_BS.shape,
            ),
            width_freqs.index_select(0, widths_BS.flatten()).unflatten(
                0,
                widths_BS.shape,
            ),
        ),
        dim=2,
    )
    x_BSND_complex = torch.view_as_complex(x_BSND.to(torch.float64).reshape(*x_BSND.shape[:-1], -1, 2))
    rotated_BSND = torch.view_as_real(
        x_BSND_complex * position_freqs_BSD.unsqueeze(2)
    ).flatten(3)
    return rotated_BSND.float()


def _inference_sdpa(
    q_BSND: torch.Tensor,
    k_BLND: torch.Tensor,
    v_BLND: torch.Tensor,
    *,
    input_mask: torch.Tensor | None,
    compute_dtype: torch.dtype,
    key_bias_BL: torch.Tensor | None = None,
) -> torch.Tensor:
    q_BNSD = q_BSND.to(compute_dtype).transpose(1, 2)
    k_BNLD = k_BLND.to(compute_dtype).transpose(1, 2)
    v_BNLD = v_BLND.to(compute_dtype).transpose(1, 2)
    if input_mask is not None:
        if input_mask.ndim == 2:
            input_mask = input_mask.view(1, 1, *input_mask.shape)
        elif input_mask.ndim == 3:
            input_mask = input_mask.unsqueeze(1)
        if input_mask.ndim != 4:
            raise ValueError(f"attention mask must have 2, 3, or 4 dimensions, got {input_mask.ndim}")
    if key_bias_BL is not None:
        if key_bias_BL.shape != (q_BNSD.size(0), k_BNLD.size(2)):
            raise ValueError(
                "key bias must have shape "
                f"[{q_BNSD.size(0)}, {k_BNLD.size(2)}], got {tuple(key_bias_BL.shape)}"
            )
        key_bias_B11L = key_bias_BL.to(device=q_BNSD.device, dtype=compute_dtype).view(
            q_BNSD.size(0), 1, 1, k_BNLD.size(2)
        )
        if input_mask is None:
            input_mask = key_bias_B11L
        elif input_mask.dtype == torch.bool:
            input_mask = key_bias_B11L.masked_fill(~input_mask, float("-inf"))
        else:
            input_mask = input_mask.to(compute_dtype) + key_bias_B11L
    output_BNSD = F.scaled_dot_product_attention(
        q_BNSD,
        k_BNLD,
        v_BNLD,
        attn_mask=input_mask,
        dropout_p=0.0,
        is_causal=False,
    )
    return output_BNSD.transpose(1, 2).contiguous()


class WanKVCache(nn.Module):
    """Non-persistent per-layer KV storage for Wan latent tokens."""

    def __init__(
        self,
        max_batch_size: int,
        max_seq_length: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        self.dtype = dtype
        cache_shape = (max_batch_size, max_seq_length, num_heads, head_dim)
        self.register_buffer(
            "k_cache",
            torch.zeros(cache_shape, dtype=dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(cache_shape, dtype=dtype, device=device),
            persistent=False,
        )

    def cache(
        self,
        cache_pos_S: torch.Tensor,
        k_BSND: torch.Tensor,
        v_BSND: torch.Tensor,
        cache_seq_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = k_BSND.size(0)
        self.k_cache[:batch_size, cache_pos_S] = k_BSND.to(self.dtype)
        self.v_cache[:batch_size, cache_pos_S] = v_BSND.to(self.dtype)
        return (
            self.k_cache[:batch_size, :cache_seq_length],
            self.v_cache[:batch_size, :cache_seq_length],
        )


class WanInferenceSelfAttention(WanSelfAttention):
    kv_cache: WanKVCache | None

    def forward(
        self,
        x_BSC: torch.Tensor,
        grid_size: tuple[int, int, int],
        freqs_MD: torch.Tensor,
        *,
        frame_indices_BF: torch.Tensor | None = None,
        semantic_pos_S: torch.Tensor | None = None,
        spatial_grid: tuple[int, int] | None = None,
        cache_pos_S: torch.Tensor | None = None,
        cache_seq_length: int | None = None,
        input_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if semantic_pos_S is None:
            if cache_pos_S is not None:
                raise ValueError("cached Wan attention requires semantic positions")
            return super().forward(
                x_BSC,
                grid_size,
                freqs_MD,
                frame_indices_BF=frame_indices_BF,
                input_mask=input_mask,
            )
        if frame_indices_BF is not None:
            raise ValueError(
                "use either frame_indices_BF or semantic_pos_S, not both"
            )
        if spatial_grid is None:
            raise ValueError("spatial_grid is required with semantic positions")

        batch_size, seq_len = x_BSC.shape[:2]
        q_BSND = self.norm_q(self.q(x_BSC)).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k_BSND = self.norm_k(self.k(x_BSC)).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v_BSND = self.v(x_BSC).view(batch_size, seq_len, self.num_heads, self.head_dim)
        compute_dtype = v_BSND.dtype
        q_BSND = _rope_apply_positions(q_BSND, semantic_pos_S, spatial_grid, freqs_MD).to(compute_dtype)
        k_BSND = _rope_apply_positions(k_BSND, semantic_pos_S, spatial_grid, freqs_MD).to(compute_dtype)

        if cache_pos_S is not None:
            if self.training:
                raise RuntimeError("Wan KV cache is only supported for inference")
            if cache_seq_length is None:
                raise ValueError("cache_seq_length is required with cache_pos")
            if self.kv_cache is None:
                raise RuntimeError("Wan KV cache must be initialized before cached attention")
            k_BSND, v_BSND = self.kv_cache.cache(
                cache_pos_S,
                k_BSND,
                v_BSND,
                cache_seq_length,
            )

        output_BSND = _inference_sdpa(
            q_BSND,
            k_BSND,
            v_BSND,
            input_mask=input_mask,
            compute_dtype=compute_dtype,
        )
        return self.o(output_BSND.flatten(2))


class WanInferenceCrossAttention(WanCrossAttention):
    def forward(
        self,
        x_BSC: torch.Tensor,
        context_BLC: torch.Tensor,
        context_lens_B: torch.Tensor | None,
        context_key_bias_BL: torch.Tensor | None = None,
        *,
        input_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_mask is None:
            return super().forward(
                x_BSC,
                context_BLC,
                context_lens_B,
                context_key_bias_BL,
            )
        batch_size, query_len = x_BSC.shape[:2]
        context_len = context_BLC.size(1)
        q_BSND = self.norm_q(self.q(x_BSC)).view(batch_size, query_len, self.num_heads, self.head_dim)
        k_BLND = self.norm_k(self.k(context_BLC)).view(batch_size, context_len, self.num_heads, self.head_dim)
        v_BLND = self.v(context_BLC).view(batch_size, context_len, self.num_heads, self.head_dim)
        output_BSND = _inference_sdpa(
            q_BSND,
            k_BLND,
            v_BLND,
            input_mask=input_mask,
            compute_dtype=v_BLND.dtype,
            key_bias_BL=context_key_bias_BL,
        )
        return self.o(output_BSND.flatten(2))


class WanInferenceAttentionBlock(WanAttentionBlock):
    def forward(
        self,
        x_BSC: torch.Tensor,
        e_BK6C: torch.Tensor,
        grid_size: tuple[int, int, int],
        freqs_MD: torch.Tensor,
        context_BLC: torch.Tensor | None,
        context_lens_B: torch.Tensor | None,
        context_key_bias_BL: torch.Tensor | None = None,
        *,
        frame_indices_BF: torch.Tensor | None = None,
        semantic_pos_S: torch.Tensor | None = None,
        spatial_grid: tuple[int, int] | None = None,
        cache_pos_S: torch.Tensor | None = None,
        cache_seq_length: int | None = None,
        input_mask: torch.Tensor | None = None,
        cross_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            semantic_pos_S is None
            and context_BLC is not None
            and cache_pos_S is None
            and cross_attention_mask is None
        ):
            return super().forward(
                x_BSC,
                e_BK6C,
                grid_size,
                freqs_MD,
                context_BLC,
                context_lens_B,
                context_key_bias_BL,
                frame_indices_BF=frame_indices_BF,
                input_mask=input_mask,
            )
        if frame_indices_BF is not None:
            raise ValueError(
                "use either frame_indices_BF or semantic_pos_S, not both"
            )

        if e_BK6C.dtype != torch.float32:
            raise ValueError(f"Wan modulation must be float32, got {e_BK6C.dtype}")
        with _autocast_disabled(x_BSC):
            modulation_BK6C = self.modulation.float().unsqueeze(0) + e_BK6C
            shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = modulation_BK6C.chunk(6, dim=2)

        attention_input_BSC = _modulate_wan(
            self.norm1(x_BSC),
            shift_msa,
            scale_msa,
            grid_size,
        )
        attention_output_BSC = self.self_attn(
            attention_input_BSC.to(self.self_attn.q.weight.dtype),
            grid_size,
            freqs_MD,
            semantic_pos_S=semantic_pos_S,
            spatial_grid=spatial_grid,
            cache_pos_S=cache_pos_S,
            cache_seq_length=cache_seq_length,
            input_mask=input_mask,
        )
        with _autocast_disabled(x_BSC):
            x_BSC = x_BSC.float() + _gate_wan(attention_output_BSC, gate_msa, grid_size)

        if context_BLC is not None:
            cross_attention_dtype = self.cross_attn.q.weight.dtype
            cross_attention_output_BSC = self.cross_attn(
                self.norm3(x_BSC).to(cross_attention_dtype),
                context_BLC.to(cross_attention_dtype),
                context_lens_B,
                context_key_bias_BL,
                input_mask=cross_attention_mask,
            )
            with _autocast_disabled(x_BSC):
                x_BSC = x_BSC.float() + cross_attention_output_BSC.float()
        ffn_input_BSC = _modulate_wan(
            self.norm2(x_BSC),
            shift_ffn,
            scale_ffn,
            grid_size,
        )
        ffn_output_BSC = self.ffn(ffn_input_BSC.to(self.ffn[0].weight.dtype))
        with _autocast_disabled(x_BSC):
            return x_BSC.float() + _gate_wan(ffn_output_BSC, gate_ffn, grid_size)


class WanModelForInference(WanModel):
    """Block-causal Wan with prefix KV caching and diffusion decode."""

    def __init__(
        self,
        config: WanModel.Config,
        *,
        default_kv_cache_dtype: KVCacheDType = FP8_KV_CACHE_DTYPE,
        default_inference_sampler: str = "ode",
        default_inference_steps: int = DEFAULT_WAN_INFERENCE_STEPS,
        default_inference_schedule: str = DEFAULT_WAN_INFERENCE_SCHEDULE,
        default_inference_shift: float = DEFAULT_WAN_INFERENCE_SHIFT,
    ):
        super().__init__(config)
        self.attention_mask = "BLOCKWISE_LOWER_TRIANGLE"
        if self.patch_size[0] != 1:
            raise ValueError("Wan prefix caching currently requires temporal patch_size=1")
        for block in self.blocks:
            block.__class__ = WanInferenceAttentionBlock
            block.self_attn.__class__ = WanInferenceSelfAttention
            block.cross_attn.__class__ = WanInferenceCrossAttention
            block.self_attn.kv_cache = None
        self.default_kv_cache_dtype = default_kv_cache_dtype
        if default_inference_sampler not in WAN_INFERENCE_SAMPLERS:
            raise ValueError(
                f"unknown Wan inference sampler {default_inference_sampler!r}; "
                f"expected one of {WAN_INFERENCE_SAMPLERS}"
            )
        if default_inference_steps <= 0:
            raise ValueError(
                "default Wan inference steps must be positive, got "
                f"{default_inference_steps}"
            )
        if not default_inference_schedule:
            raise ValueError("default Wan inference schedule cannot be empty")
        if default_inference_shift <= 0:
            raise ValueError(
                "default Wan inference shift must be positive, got "
                f"{default_inference_shift}"
            )
        self.default_inference_sampler = default_inference_sampler
        self.default_inference_steps = default_inference_steps
        self.default_inference_schedule = default_inference_schedule
        self.default_inference_shift = default_inference_shift
        self.max_batch_size = -1
        self.max_seq_length = -1
        self.cache_dtype: torch.dtype | None = None
        self.inference_masks: dict[
            tuple[str, int, int, int, bool],
            tuple[torch.Tensor | None, torch.Tensor | None],
        ] = {}

    @staticmethod
    def input_shapes(
        config: WanModel.Config,
        batch_size: int = 1,
        *,
        latent_frames: int = 2,
        latent_height: int | None = None,
        latent_width: int | None = None,
    ) -> dict[str, tuple[int, ...]]:
        latent_height = latent_height or config.patch_size[1] * 2
        latent_width = latent_width or config.patch_size[2] * 2
        return {
            "latents": (
                batch_size,
                latent_frames,
                config.in_dim,
                latent_height,
                latent_width,
            )
        }

    @staticmethod
    def input_dtypes(dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.dtype]:
        return {"latents": dtype}

    @classmethod
    def example_inputs(
        cls,
        config: WanModel.Config,
        *,
        batch_size: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "meta",
        latent_frames: int = 2,
        latent_height: int | None = None,
        latent_width: int | None = None,
    ) -> dict[str, torch.Tensor]:
        shape = cls.input_shapes(
            config,
            batch_size,
            latent_frames=latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
        )["latents"]
        return {"latents": torch.randn(shape, dtype=dtype, device=device)}

    def get_model_io(
        self,
        *,
        batch_size: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        latent_frames: int = 2,
        latent_height: int | None = None,
        latent_width: int | None = None,
        num_prefill_frames: int = 1,
        **_: Any,
    ) -> dict[str, dict[str, torch.Size] | dict[str, torch.dtype]]:
        input_shape = torch.Size(
            self.input_shapes(
                self.config,
                batch_size,
                latent_frames=latent_frames,
                latent_height=latent_height,
                latent_width=latent_width,
            )["latents"]
        )
        output_shape = torch.Size(
            (
                input_shape[0],
                input_shape[1] - num_prefill_frames,
                input_shape[2],
                input_shape[3],
                input_shape[4],
            )
        )
        return {
            "in_shape": {"latents": input_shape},
            "in_dtype": {"latents": dtype},
            "out_shape": {"latents": output_shape},
            "out_dtype": {"latents": dtype},
        }

    def get_null_text_embedding(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the raw fixed null context used by context-free fine-tuning."""
        context_BLC = torch.zeros((batch_size, 1, self.text_dim), device=device, dtype=dtype)
        context_BLC[..., 0] = 1
        return context_BLC

    def set_packaged_text_context(self, context_LC: torch.Tensor) -> None:
        """Install the raw prompt context carried by an inference package."""
        if context_LC.ndim != 2 or context_LC.size(1) != self.text_dim:
            raise ValueError(f"packaged context must have shape [L, {self.text_dim}], got {tuple(context_LC.shape)}")
        if not 0 < context_LC.size(0) <= self.text_len:
            raise ValueError(f"packaged context length must be in [1, {self.text_len}], got {context_LC.size(0)}")
        if context_LC.device.type != "meta" and not torch.isfinite(context_LC).all():
            raise ValueError("packaged context contains non-finite values")
        context_LC = context_LC.contiguous()
        if PACKAGED_TEXT_CONTEXT_BUFFER in self._buffers:
            self._buffers[PACKAGED_TEXT_CONTEXT_BUFFER] = context_LC
        else:
            self.register_buffer(
                PACKAGED_TEXT_CONTEXT_BUFFER,
                context_LC,
                persistent=True,
            )

    def has_packaged_text_context(self) -> bool:
        return PACKAGED_TEXT_CONTEXT_BUFFER in self._buffers

    def _project_context(
        self,
        context: torch.Tensor | Sequence[torch.Tensor] | None,
        *,
        batch_size: int,
        device: torch.device,
        use_packaged_context: bool = True,
    ) -> torch.Tensor:
        projected_context_BLC, _ = self._project_context_with_bias(
            context,
            batch_size=batch_size,
            device=device,
            use_packaged_context=use_packaged_context,
        )
        return projected_context_BLC

    def _project_context_with_bias(
        self,
        context: torch.Tensor | Sequence[torch.Tensor] | None,
        *,
        batch_size: int,
        device: torch.device,
        use_packaged_context: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        weight_dtype = self.text_embedding[0].weight.dtype
        if context is None and use_packaged_context and self.has_packaged_text_context():
            context_LC = self._buffers[PACKAGED_TEXT_CONTEXT_BUFFER]
            context_list = [context_LC] * batch_size
        elif context is None:
            context_list = list(
                self.get_null_text_embedding(
                    batch_size,
                    device=device,
                    dtype=weight_dtype,
                )
            )
        elif isinstance(context, torch.Tensor):
            if context.ndim == 2 and batch_size == 1:
                context = context.unsqueeze(0)
            if context.ndim != 3 or context.size(0) != batch_size:
                raise ValueError(
                    f"context tensor must have shape [{batch_size}, L, {self.text_dim}], got {tuple(context.shape)}"
                )
            context_list = list(context)
        else:
            if len(context) != batch_size:
                raise ValueError(f"context batch {len(context)} does not match latent batch {batch_size}")
            context_list = list(context)

        context_on_device = []
        for context_LC in context_list:
            if context_LC.ndim != 2 or context_LC.size(1) != self.text_dim:
                raise ValueError(f"each context must have shape [L, {self.text_dim}], got {tuple(context_LC.shape)}")
            if context_LC.size(0) > self.text_len:
                raise ValueError(f"context length {context_LC.size(0)} exceeds text_len {self.text_len}")
            context_on_device.append(context_LC.to(device=device, dtype=weight_dtype))
        compact_context_BLC, context_key_bias_BL = _compact_padded_context(
            context_on_device,
            text_len=self.text_len,
            text_dim=self.text_dim,
        )
        return self.text_embedding(compact_context_BLC), context_key_bias_BL

    def _patchify(self, latents_BFCHW: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        if latents_BFCHW.ndim != 5 or latents_BFCHW.size(2) != self.in_dim:
            raise ValueError(f"latents must have shape [B, F, {self.in_dim}, H, W], got {tuple(latents_BFCHW.shape)}")
        latents_BCFHW = latents_BFCHW.permute(0, 2, 1, 3, 4)
        patches_BCFHW = self.patch_embedding(latents_BCFHW)
        grid_size = tuple(patches_BCFHW.shape[2:])
        return patches_BCFHW.flatten(2).transpose(1, 2), grid_size

    def _compact_timesteps(
        self,
        timesteps: torch.Tensor,
        *,
        batch_size: int,
        frames: int,
        spatial_tokens: int,
    ) -> torch.Tensor:
        seq_len = frames * spatial_tokens
        if timesteps.ndim == 1:
            if timesteps.numel() != batch_size:
                raise ValueError(f"timestep batch {timesteps.numel()} does not match {batch_size}")
            return timesteps[:, None]
        if timesteps.shape == (batch_size, frames):
            return timesteps
        if timesteps.shape == (batch_size, seq_len):
            return timesteps
        raise ValueError(
            f"timesteps must have shape [{batch_size}], [{batch_size}, {frames}], "
            f"or [{batch_size}, {seq_len}], got {tuple(timesteps.shape)}"
        )

    def _unpatchify_uniform(
        self,
        x_BSC: torch.Tensor,
        grid_size: tuple[int, int, int],
    ) -> torch.Tensor:
        frames, height, width = grid_size
        patch_frames, patch_height, patch_width = self.patch_size
        x_BFHWPTPHWPC = x_BSC.view(
            x_BSC.size(0),
            frames,
            height,
            width,
            patch_frames,
            patch_height,
            patch_width,
            self.out_dim,
        )
        x_BCFPTHPHWPW = torch.einsum(
            "bfhwpqrc->bcfphqwr",
            x_BFHWPTPHWPC,
        )
        x_BCFHW = x_BCFPTHPHWPW.reshape(
            x_BSC.size(0),
            self.out_dim,
            frames * patch_frames,
            height * patch_height,
            width * patch_width,
        )
        return x_BCFHW.permute(0, 2, 1, 3, 4).float()

    def _forward_chunk(
        self,
        latents_BFCHW: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        semantic_pos_S: torch.Tensor,
        input_mask: torch.Tensor | None,
        context_BLC: torch.Tensor,
        context_key_bias_BL: torch.Tensor | None = None,
        cross_attention_mask: torch.Tensor | None = None,
        cache_pos_S: torch.Tensor | None = None,
        cache_seq_length: int | None = None,
    ) -> torch.Tensor:
        x_BSC, grid_size = self._patchify(latents_BFCHW)
        batch_size, seq_len = x_BSC.shape[:2]
        frames, height, width = grid_size
        if semantic_pos_S.ndim == 1:
            valid_semantic_shape = semantic_pos_S.shape == (seq_len,)
        else:
            valid_semantic_shape = semantic_pos_S.shape == (batch_size, seq_len)
        if not valid_semantic_shape:
            raise ValueError(
                "semantic_pos must have shape "
                f"[{seq_len}] or [{batch_size}, {seq_len}], got "
                f"{tuple(semantic_pos_S.shape)}"
            )
        t_BK = self._compact_timesteps(
            timesteps,
            batch_size=batch_size,
            frames=frames,
            spatial_tokens=height * width,
        )
        conditioning_len = t_BK.size(1)
        with _autocast_disabled(t_BK):
            timestep_embedding_BKC = sinusoidal_embedding_1d(
                self.freq_dim,
                t_BK.flatten(),
            ).unflatten(0, (batch_size, conditioning_len))
            time_dtype = self.time_embedding[0].weight.dtype
            e_BKC = self.time_embedding(timestep_embedding_BKC.to(time_dtype)).float()
            projection_dtype = self.time_projection[1].weight.dtype
            e_BK6C = (
                self.time_projection(e_BKC.to(projection_dtype))
                .float()
                .unflatten(
                    2,
                    (6, self.dim),
                )
            )

        freqs_MD = self._get_rope_freqs(x_BSC.device)
        for block in self.blocks:
            x_BSC = block(
                x_BSC,
                e_BK6C,
                grid_size,
                freqs_MD,
                context_BLC,
                None,
                context_key_bias_BL,
                semantic_pos_S=semantic_pos_S,
                spatial_grid=(height, width),
                cache_pos_S=cache_pos_S,
                cache_seq_length=cache_seq_length,
                input_mask=input_mask,
                cross_attention_mask=cross_attention_mask,
            )
        return self._unpatchify_uniform(self.head(x_BSC, e_BKC, grid_size), grid_size)

    def forward_causal(
        self,
        latents_BFCHW: torch.Tensor,
        timesteps_BF: torch.Tensor,
        context: torch.Tensor | Sequence[torch.Tensor] | None = None,
        frame_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the fine-tuning path with frame-block-causal attention."""
        batch_size, frames = latents_BFCHW.shape[:2]
        height = latents_BFCHW.size(3) // self.patch_size[1]
        width = latents_BFCHW.size(4) // self.patch_size[2]
        spatial_tokens = height * width
        compact_pos_S = torch.arange(
            frames * spatial_tokens,
            device=latents_BFCHW.device,
        )
        frame_indices_BF = _normalize_frame_indices(
            frame_indices,
            batch_size=batch_size,
            frames=frames,
            device=latents_BFCHW.device,
        )
        semantic_pos_BS = _frame_indices_to_semantic_positions(
            frame_indices_BF,
            (height, width),
        )
        input_mask = frame_block_causal_mask(
            compact_pos_S,
            compact_pos_S,
            spatial_tokens,
        ).view(1, 1, compact_pos_S.numel(), compact_pos_S.numel())
        context_BLC, context_key_bias_BL = self._project_context_with_bias(
            context,
            batch_size=batch_size,
            device=latents_BFCHW.device,
        )
        return self._forward_chunk(
            latents_BFCHW,
            timesteps_BF,
            semantic_pos_S=semantic_pos_BS,
            input_mask=input_mask,
            context_BLC=context_BLC,
            context_key_bias_BL=context_key_bias_BL,
        )

    def _has_compatible_caches(
        self,
        max_batch_size: int,
        max_seq_length: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> bool:
        if self.max_batch_size < max_batch_size or self.max_seq_length < max_seq_length or self.cache_dtype != dtype:
            return False
        return all(
            isinstance(block.self_attn.kv_cache, WanKVCache)
            and block.self_attn.kv_cache.dtype == dtype
            and block.self_attn.kv_cache.k_cache.device == device
            for block in self.blocks
        )

    def setup_caches(
        self,
        max_batch_size: int,
        max_seq_length: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if self._has_compatible_caches(
            max_batch_size,
            max_seq_length,
            dtype=dtype,
            device=device,
        ):
            return
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.cache_dtype = dtype
        for block in self.blocks:
            block.self_attn.kv_cache = WanKVCache(
                max_batch_size,
                max_seq_length,
                self.num_heads,
                self.dim // self.num_heads,
                dtype,
                device,
            )

    def cleanup_caches(self) -> None:
        for block in self.blocks:
            block.self_attn.kv_cache = None
        self.max_batch_size = -1
        self.max_seq_length = -1
        self.cache_dtype = None

    def clear_inference_masks(self) -> None:
        self.inference_masks = {}

    @torch.no_grad()
    def get_inference_masks(
        self,
        device: torch.device,
        prefix_tokens: int,
        decode_tokens: int,
        spatial_tokens: int,
        cfg_enabled: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        key = (
            str(device),
            prefix_tokens,
            decode_tokens,
            spatial_tokens,
            cfg_enabled,
        )
        if key in self.inference_masks:
            return self.inference_masks[key]

        branches = 2 if cfg_enabled else 1
        prefix_pos_P = torch.arange(prefix_tokens, device=device)
        decode_pos_D = torch.arange(
            prefix_tokens,
            prefix_tokens + decode_tokens,
            device=device,
        )
        query_pos_Q = decode_pos_D.repeat(branches)
        key_pos_K = torch.cat((prefix_pos_P, decode_pos_D.repeat(branches)))
        cache_seq_length = key_pos_K.numel()

        prefill_mask = None
        if prefix_tokens:
            prefill_mask = frame_block_causal_mask(
                prefix_pos_P,
                key_pos_K,
                spatial_tokens,
            )
            prefill_mask &= torch.arange(cache_seq_length, device=device).view(1, -1) < prefix_tokens
            prefill_mask = prefill_mask.view(1, 1, prefix_tokens, cache_seq_length)

        decode_mask = frame_block_causal_mask(
            query_pos_Q,
            key_pos_K,
            spatial_tokens,
        )
        if branches == 2:
            query_branch_Q = torch.arange(branches, device=device).repeat_interleave(decode_tokens)
            key_branch_K = torch.cat(
                (
                    torch.full((prefix_tokens,), -1, device=device),
                    torch.arange(branches, device=device).repeat_interleave(decode_tokens),
                )
            )
            decode_mask &= (key_branch_K.view(1, -1) < 0) | (key_branch_K.view(1, -1) == query_branch_Q.view(-1, 1))
        decode_mask = decode_mask.view(1, 1, branches * decode_tokens, cache_seq_length)
        if device.type != "meta" and bool(decode_mask.all()):
            decode_mask = None

        masks = (prefill_mask, decode_mask)
        self.inference_masks[key] = masks
        return masks

    def _packed_context(
        self,
        context: torch.Tensor | Sequence[torch.Tensor] | None,
        *,
        batch_size: int,
        decode_tokens: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        null_context_BLC, null_key_bias_BL = self._project_context_with_bias(
            None,
            batch_size=batch_size,
            device=device,
            use_packaged_context=False,
        )
        conditional_context_BLC, conditional_key_bias_BL = self._project_context_with_bias(
            context,
            batch_size=batch_size,
            device=device,
        )
        packed_context_BL2C = torch.cat(
            (null_context_BLC, conditional_context_BLC),
            dim=1,
        )
        query_branch_Q = torch.arange(2, device=device).repeat_interleave(decode_tokens)
        key_branch_K = torch.cat(
            (
                torch.zeros(null_context_BLC.size(1), device=device, dtype=torch.long),
                torch.ones(conditional_context_BLC.size(1), device=device, dtype=torch.long),
            )
        )
        branch_mask_11QK = (query_branch_Q.view(-1, 1) == key_branch_K.view(1, -1)).view(
            1, 1, decode_tokens * 2, key_branch_K.numel()
        )
        if null_key_bias_BL is None:
            null_key_bias_BL = torch.zeros(
                (batch_size, null_context_BLC.size(1)),
                device=device,
                dtype=torch.float32,
            )
        if conditional_key_bias_BL is None:
            conditional_key_bias_BL = torch.zeros(
                (batch_size, conditional_context_BLC.size(1)),
                device=device,
                dtype=torch.float32,
            )
        packed_key_bias_B11K = torch.cat(
            (null_key_bias_BL, conditional_key_bias_BL),
            dim=1,
        ).view(batch_size, 1, 1, -1)
        cross_attention_mask = torch.where(
            branch_mask_11QK,
            packed_key_bias_B11K,
            packed_key_bias_B11K.new_tensor(float("-inf")),
        )
        return packed_context_BL2C, cross_attention_mask

    @torch.inference_mode()
    def prefill(
        self,
        latents_BFCHW: torch.Tensor,
        *,
        cache_seq_length: int,
        input_mask: torch.Tensor | None,
        context: torch.Tensor | Sequence[torch.Tensor] | None = None,
        frame_indices: torch.Tensor | None = None,
    ) -> None:
        if latents_BFCHW.size(1) == 0:
            return
        batch_size, frames = latents_BFCHW.shape[:2]
        height = latents_BFCHW.size(3) // self.patch_size[1]
        width = latents_BFCHW.size(4) // self.patch_size[2]
        prefix_tokens = frames * height * width
        prefix_pos_P = torch.arange(prefix_tokens, device=latents_BFCHW.device)
        frame_indices_BF = _normalize_frame_indices(
            frame_indices,
            batch_size=batch_size,
            frames=frames,
            device=latents_BFCHW.device,
        )
        semantic_pos_BP = _frame_indices_to_semantic_positions(
            frame_indices_BF,
            (height, width),
        )
        context_BLC, context_key_bias_BL = self._project_context_with_bias(
            context,
            batch_size=batch_size,
            device=latents_BFCHW.device,
        )
        self._forward_chunk(
            latents_BFCHW,
            torch.zeros(
                (batch_size, frames),
                device=latents_BFCHW.device,
                dtype=torch.float32,
            ),
            semantic_pos_S=semantic_pos_BP,
            input_mask=input_mask,
            context_BLC=context_BLC,
            context_key_bias_BL=context_key_bias_BL,
            cache_pos_S=prefix_pos_P,
            cache_seq_length=cache_seq_length,
        )

    @torch.inference_mode()
    def decode(
        self,
        latents_BFCHW: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        num_prefill_frames: int,
        input_mask: torch.Tensor | None,
        context: torch.Tensor | Sequence[torch.Tensor] | None = None,
        frame_indices: torch.Tensor | None = None,
        cfg: float = 0.0,
    ) -> torch.Tensor:
        batch_size, frames = latents_BFCHW.shape[:2]
        height = latents_BFCHW.size(3) // self.patch_size[1]
        width = latents_BFCHW.size(4) // self.patch_size[2]
        spatial_tokens = height * width
        prefix_tokens = num_prefill_frames * spatial_tokens
        decode_tokens = frames * spatial_tokens
        branches = 2 if cfg > 0.0 else 1
        cache_seq_length = prefix_tokens + branches * decode_tokens
        frame_indices_BF = _normalize_frame_indices(
            frame_indices,
            batch_size=batch_size,
            frames=frames,
            device=latents_BFCHW.device,
        )
        if frame_indices is None:
            frame_indices_BF = frame_indices_BF + num_prefill_frames
        semantic_pos_BS = _frame_indices_to_semantic_positions(
            frame_indices_BF,
            (height, width),
        ).repeat(1, branches)
        cache_pos_S = torch.arange(
            prefix_tokens,
            cache_seq_length,
            device=latents_BFCHW.device,
        )

        if branches == 2:
            packed_latents_BFCHW = torch.cat(
                (latents_BFCHW, latents_BFCHW),
                dim=1,
            )
            if timesteps.ndim == 1:
                packed_timesteps = timesteps
            else:
                packed_timesteps = torch.cat((timesteps, timesteps), dim=1)
            context_BLC, cross_attention_mask = self._packed_context(
                context,
                batch_size=batch_size,
                decode_tokens=decode_tokens,
                device=latents_BFCHW.device,
            )
            context_key_bias_BL = None
        else:
            packed_latents_BFCHW = latents_BFCHW
            packed_timesteps = timesteps
            context_BLC, context_key_bias_BL = self._project_context_with_bias(
                context,
                batch_size=batch_size,
                device=latents_BFCHW.device,
            )
            cross_attention_mask = None

        output_BFCHW = self._forward_chunk(
            packed_latents_BFCHW,
            packed_timesteps,
            semantic_pos_S=semantic_pos_BS,
            input_mask=input_mask,
            context_BLC=context_BLC,
            context_key_bias_BL=context_key_bias_BL,
            cross_attention_mask=cross_attention_mask,
            cache_pos_S=cache_pos_S,
            cache_seq_length=cache_seq_length,
        )
        if branches == 1:
            return output_BFCHW
        unconditional_BFCHW, conditional_BFCHW = output_BFCHW.chunk(2, dim=1)
        return unconditional_BFCHW + cfg * (conditional_BFCHW - unconditional_BFCHW)

    @torch.inference_mode()
    def forward_n_steps(
        self,
        x_BFCHW: torch.Tensor,
        *,
        num_prefill_frames: int,
        input_mask: torch.Tensor | None,
        scheduler: RFScheduler,
        steps: int,
        timestep_scale: float = 1000.0,
        context: torch.Tensor | Sequence[torch.Tensor] | None = None,
        frame_indices: torch.Tensor | None = None,
        cfg: float = 0.0,
        inference_sampler: str = "ode",
        generator: torch.Generator | None = None,
        return_trajectory: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor | None]:
        """Denoise a suffix while overwriting only its physical cache slots."""
        if inference_sampler not in WAN_INFERENCE_SAMPLERS:
            raise ValueError(
                f"unknown Wan inference sampler {inference_sampler!r}; "
                f"expected one of {WAN_INFERENCE_SAMPLERS}"
            )
        batch_size = x_BFCHW.size(0)
        device = x_BFCHW.device
        dtype = x_BFCHW.dtype
        autocast_enabled = dtype in (torch.float16, torch.bfloat16) and device.type in ("cpu", "cuda")
        autocast_dtype = dtype if autocast_enabled else torch.bfloat16
        trajectory = [x_BFCHW.clone()] if return_trajectory else None
        model_output: dict[str, torch.Tensor] = {}
        for step_idx in range(steps):
            timestep_B = torch.full(
                (batch_size,),
                scheduler.timesteps[step_idx] * timestep_scale,
                device=device,
                dtype=torch.float32,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                velocity_BFCHW = self.decode(
                    x_BFCHW,
                    timestep_B,
                    num_prefill_frames=num_prefill_frames,
                    input_mask=input_mask,
                    context=context,
                    frame_indices=frame_indices,
                    cfg=cfg,
                )
            model_output = {"sample": velocity_BFCHW}
            if inference_sampler == "ode":
                # Wan predicts the forward flow epsilon-x_0. RFScheduler.step
                # integrates x_0-epsilon, so convert conventions exactly once
                # at this boundary.
                x_BFCHW = scheduler.step(
                    -velocity_BFCHW,
                    step_idx,
                    x_BFCHW,
                ).to(dtype)
            else:
                # Self-Forcing trains on this random-exit SDE trajectory, not
                # the deterministic RF/ODE trajectory above. Each evaluation
                # predicts x_0, then the next point is sampled from the forward
                # process at the next discrete timestep.
                timestep = scheduler.timesteps[step_idx]
                predicted_x0_BFCHW = (
                    x_BFCHW - timestep * velocity_BFCHW
                ).to(dtype)
                if step_idx == steps - 1:
                    x_BFCHW = predicted_x0_BFCHW
                else:
                    next_timestep = scheduler.timesteps[step_idx + 1]
                    noise_BFCHW = torch.randn(
                        predicted_x0_BFCHW.shape,
                        device=device,
                        dtype=dtype,
                        generator=generator,
                    )
                    x_BFCHW = (
                        (1 - next_timestep) * predicted_x0_BFCHW
                        + next_timestep * noise_BFCHW
                    ).to(dtype)
            if trajectory is not None:
                trajectory.append(x_BFCHW.clone())
        return (
            x_BFCHW,
            model_output,
            torch.stack(trajectory, dim=1) if trajectory is not None else None,
        )

    @torch.inference_mode()
    def generate(
        self,
        latents: torch.Tensor,
        *,
        steps: int | None = None,
        num_prefill_frames: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        inference_schedule: str | None = None,
        shift: float | None = None,
        timestep_scale: float = 1000.0,
        context: torch.Tensor | Sequence[torch.Tensor] | None = None,
        frame_indices: torch.Tensor | None = None,
        cfg: float = 0.0,
        inference_sampler: str | None = None,
        generator: torch.Generator | None = None,
        return_trajectory: bool = False,
        kv_cache_dtype: KVCacheDType | None = None,
        **scheduler_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        if latents.ndim != 5 or latents.size(2) != self.in_dim:
            raise ValueError(f"latents must have shape [B, F, {self.in_dim}, H, W], got {tuple(latents.shape)}")
        batch_size, frames = latents.shape[:2]
        if not 0 <= num_prefill_frames < frames:
            raise ValueError(f"num_prefill_frames={num_prefill_frames} must be in [0, {frames - 1}]")
        steps = self.default_inference_steps if steps is None else steps
        inference_schedule = (
            self.default_inference_schedule
            if inference_schedule is None
            else inference_schedule
        )
        shift = self.default_inference_shift if shift is None else shift
        inference_sampler = (
            self.default_inference_sampler
            if inference_sampler is None
            else inference_sampler
        )
        if inference_sampler not in WAN_INFERENCE_SAMPLERS:
            raise ValueError(
                f"unknown Wan inference sampler {inference_sampler!r}; "
                f"expected one of {WAN_INFERENCE_SAMPLERS}"
            )
        frame_indices_BF = _normalize_frame_indices(
            frame_indices,
            batch_size=batch_size,
            frames=frames,
            device=latents.device,
        )
        if steps <= 0:
            return {"latents": latents[:, num_prefill_frames:].to(dtype=dtype)}

        latents = latents.to(dtype=dtype)
        device = latents.device
        height = latents.size(3) // self.patch_size[1]
        width = latents.size(4) // self.patch_size[2]
        spatial_tokens = height * width
        prefix_tokens = num_prefill_frames * spatial_tokens
        decode_tokens = (frames - num_prefill_frames) * spatial_tokens
        branches = 2 if cfg > 0.0 else 1
        cache_seq_length = prefix_tokens + branches * decode_tokens
        requested_cache_dtype = self.default_kv_cache_dtype if kv_cache_dtype is None else kv_cache_dtype
        self.setup_caches(
            batch_size,
            cache_seq_length,
            dtype=_resolve_kv_cache_dtype(requested_cache_dtype),
            device=device,
        )
        prefill_mask, decode_mask = self.get_inference_masks(
            device,
            prefix_tokens,
            decode_tokens,
            spatial_tokens,
            cfg > 0.0,
        )
        autocast_enabled = dtype in (torch.float16, torch.bfloat16) and device.type in ("cpu", "cuda")
        autocast_dtype = dtype if autocast_enabled else torch.bfloat16
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ):
            self.prefill(
                latents[:, :num_prefill_frames],
                cache_seq_length=cache_seq_length,
                input_mask=prefill_mask,
                context=context,
                frame_indices=frame_indices_BF[:, :num_prefill_frames],
            )

        scheduler = RFScheduler(
            steps=steps,
            inference_schedule=inference_schedule,
            **scheduler_kwargs,
        ).to(device=device)
        if shift <= 0:
            raise ValueError(f"shift must be positive, got {shift}")
        if shift != 1.0:
            shifted = shift * scheduler.timesteps / (1 + (shift - 1) * scheduler.timesteps)
            scheduler.timesteps.copy_(shifted)
            scheduler.dt.copy_(-torch.diff(shifted))

        x_BFCHW, _, trajectory = self.forward_n_steps(
            latents[:, num_prefill_frames:],
            num_prefill_frames=num_prefill_frames,
            input_mask=decode_mask,
            scheduler=scheduler,
            steps=steps,
            timestep_scale=timestep_scale,
            context=context,
            frame_indices=frame_indices_BF[:, num_prefill_frames:],
            cfg=cfg,
            inference_sampler=inference_sampler,
            generator=generator,
            return_trajectory=return_trajectory,
        )

        outputs = {"latents": x_BFCHW}
        if trajectory is not None:
            outputs["trajectory"] = trajectory
        return outputs

    def compile_for_inference(self) -> None:
        for block in self.blocks:
            block.compile(mode="max-autotune-no-cudagraphs")

    def quantize_for_inference(
        self,
        weight_format: WeightFormat = "fp8_nvfp4",
    ) -> None:
        if weight_format == "bf16":
            return
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, Float8MMConfig, quantize_

        if weight_format == "fp8":
            from torchao.quantization.granularity import PerTensor
            from torchao.quantization.quantize_.common.kernel_preference import KernelPreference

            fp8_config = Float8DynamicActivationFloat8WeightConfig(
                granularity=PerTensor(),
                mm_config=Float8MMConfig(),
                kernel_preference=KernelPreference.TORCH,
            )
            quantize_(self.blocks, fp8_config)
            return
        if weight_format != "fp8_nvfp4":
            raise ValueError(f"unknown Wan weight format {weight_format}")

        from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
        from torchao.quantization.granularity import PerBlock

        fp8_config = Float8DynamicActivationFloat8WeightConfig(
            granularity=[PerBlock([1, 128]), PerBlock([128, 128])],
        )

        def is_attention_linear(module: nn.Module, fqn: str) -> bool:
            return isinstance(module, nn.Linear) and (".self_attn." in f".{fqn}." or ".cross_attn." in f".{fqn}.")

        def is_ffn_linear(module: nn.Module, fqn: str) -> bool:
            return isinstance(module, nn.Linear) and ".ffn." in f".{fqn}."

        quantize_(self.blocks, fp8_config, filter_fn=is_attention_linear)
        nvfp4_config = NVFP4DynamicActivationNVFP4WeightConfig(
            use_dynamic_per_tensor_scale=True,
            use_triton_kernel=False,
        )
        for fqn, module in self.blocks.named_modules():
            if is_ffn_linear(module, fqn):
                module.to(device="cuda")
                quantize_(module, nvfp4_config)
                module.to(device="cpu")


KVCache = WanKVCache


__all__ = [
    "BF16_KV_CACHE_DTYPE",
    "DEFAULT_WAN_INFERENCE_SCHEDULE",
    "DEFAULT_WAN_INFERENCE_SHIFT",
    "DEFAULT_WAN_INFERENCE_STEPS",
    "FP8_KV_CACHE_DTYPE",
    "KVCacheDType",
    "KVCache",
    "WanKVCache",
    "WanModelForInference",
    "WAN_INFERENCE_SAMPLERS",
    "frame_block_causal_mask",
]
