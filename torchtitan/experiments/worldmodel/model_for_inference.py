import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask

from torchtitan.experiments.worldmodel.model import (
    SelfAttention,
    TensorOrMask,
    WorldModel,
    _cast_if_autocast_enabled,
    _dense_mask,
    _mask_fn,
)
from torchtitan.experiments.worldmodel.schedulers import RFScheduler


def _is_sm_at_least_89(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_available() and torch.cuda.get_device_capability(device) >= (8, 9)


def _cache_dtype_for_device(device: torch.device) -> torch.dtype:
    return torch.float8_e4m3fn if _is_sm_at_least_89(device) else torch.bfloat16


def _create_block_mask_fn(device: torch.device):
    return create_block_mask if device.type == "meta" else torch.compile(create_block_mask)


@dataclass(frozen=True, slots=True)
class InputPosMaskPair:
    input_pos: torch.Tensor
    cache_pos: torch.Tensor
    cache_seq_length: int
    input_mask: TensorOrMask | None


@dataclass(frozen=True, slots=True)
class BranchLayout:
    branch_count: int
    prefix_tokens: int
    tokens_per_branch: int
    input_pos: torch.Tensor
    cache_pos: torch.Tensor
    cache_seq_length: int

    @property
    def packed_tokens(self) -> int:
        return self.branch_count * self.tokens_per_branch


def build_branch_layout(
    *,
    prefix_tokens: int,
    tokens_per_branch: int,
    cfg_enabled: bool,
    device: torch.device | str,
) -> BranchLayout:
    if prefix_tokens < 0:
        raise ValueError(f"prefix_tokens must be nonnegative, got {prefix_tokens}")
    if tokens_per_branch <= 0:
        raise ValueError(f"tokens_per_branch must be positive, got {tokens_per_branch}")

    branch_count = 2 if cfg_enabled else 1
    semantic_suffix = torch.arange(prefix_tokens, prefix_tokens + tokens_per_branch, device=device)
    input_pos = semantic_suffix.repeat(branch_count)
    cache_pos = torch.arange(prefix_tokens, prefix_tokens + branch_count * tokens_per_branch, device=device)
    return BranchLayout(
        branch_count=branch_count,
        prefix_tokens=prefix_tokens,
        tokens_per_branch=tokens_per_branch,
        input_pos=input_pos,
        cache_pos=cache_pos,
        cache_seq_length=prefix_tokens + branch_count * tokens_per_branch,
    )


def pack_branch_inputs(
    layout: BranchLayout,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    augments_pos_ref_augment: torch.Tensor,
    ref_augment_from_augments_euler: torch.Tensor,
    pose_mask: torch.Tensor,
    fidxs: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    inputs = (x, timesteps, augments_pos_ref_augment, ref_augment_from_augments_euler, pose_mask, fidxs)
    if layout.branch_count == 1:
        return inputs
    unconditional = (
        x, timesteps, torch.zeros_like(augments_pos_ref_augment),
        torch.zeros_like(ref_augment_from_augments_euler), torch.ones_like(pose_mask), fidxs,
    )
    return tuple(torch.cat(pair, dim=1) for pair in zip(unconditional, inputs))


def branch_mask_predicate(
    mask_fn: Callable | None,
    layout: BranchLayout,
    prefill: bool,
    b: torch.Tensor,
    h: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
) -> torch.Tensor:
    if prefill:
        logical = True if mask_fn is None else mask_fn(b, h, q_idx, kv_idx)
        return (kv_idx < layout.prefix_tokens) & logical

    prefix_tokens = layout.prefix_tokens
    tokens_per_branch = layout.tokens_per_branch
    semantic_q_idx = prefix_tokens + q_idx % tokens_per_branch
    suffix_kv_idx = kv_idx - prefix_tokens
    is_prefix = kv_idx < prefix_tokens
    semantic_kv_idx = torch.where(
        is_prefix,
        kv_idx,
        prefix_tokens + suffix_kv_idx % tokens_per_branch,
    )
    same_branch = q_idx // tokens_per_branch == suffix_kv_idx // tokens_per_branch
    branch_visible = is_prefix | (same_branch & (suffix_kv_idx >= 0))
    logical = True if mask_fn is None else mask_fn(b, h, semantic_q_idx, semantic_kv_idx)
    return branch_visible & logical


class KVCache(nn.Module):
    def __init__(
        self,
        max_batch_size: int,
        max_seq_length: int,
        n_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        self.dtype = dtype
        cache_shape = (max_batch_size, max_seq_length, n_heads, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype, device=device), persistent=False)
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype, device=device), persistent=False)

    def cache(
        self,
        cache_pos: torch.Tensor,
        k_value: torch.Tensor,
        v_value: torch.Tensor,
        cache_seq_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = k_value.shape[0]
        self.k_cache[:batch, cache_pos] = k_value.to(dtype=self.dtype)
        self.v_cache[:batch, cache_pos] = v_value.to(dtype=self.dtype)
        return self.k_cache[:batch, :cache_seq_length], self.v_cache[:batch, :cache_seq_length]


def _fp8_score_mod(
    score: torch.Tensor,
    b: torch.Tensor,
    h: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
) -> torch.Tensor:
    del b, h, q_idx, kv_idx
    return score.to(torch.float32)


class InferenceSelfAttention(SelfAttention):
    def upcast_kv(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if k.dtype != q.dtype:
            return k.to(q.dtype), v.to(q.dtype)
        return k, v

    def score_mod(self) -> Any:
        return _fp8_score_mod

    def forward(
        self,
        x: torch.Tensor,
        cache_pos: torch.Tensor | None = None,
        cache_seq_length: int | None = None,
        input_mask: TensorOrMask | None = None,
    ) -> torch.Tensor:
        if cache_pos is None:
            return super().forward(x, input_mask=input_mask)
        if cache_seq_length is None:
            raise ValueError("cache_seq_length is required when cache_pos is provided")

        batch, seq_len, emb_dim = x.shape
        qkv = self.c_attn(self.layer_norm(x)).view(batch, seq_len, 3, self.config.n_head, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k = _cast_if_autocast_enabled(self.q_norm(q)), _cast_if_autocast_enabled(self.k_norm(k))

        if self.training:
            raise RuntimeError("KV cache is only supported for inference")
        if self.kv_cache is None:
            raise RuntimeError("KV cache must be initialized before using cache_pos")
        k, v = self.kv_cache.cache(cache_pos, k, v, cache_seq_length)

        if self.config.attention_impl == "FLEX":
            assert self.flex_attention is not None
            if k.dtype == torch.float8_e4m3fn:
                y = self.flex_attention(
                    q.to(torch.float8_e4m3fn),
                    k,
                    v,
                    attention_masks=input_mask,
                    score_mod=self.score_mod(),
                    scale=1.0 / math.sqrt(self.head_dim),
                ).to(q.dtype)
            else:
                k, v = self.upcast_kv(q, k, v)
                y = self.flex_attention(q, k, v, attention_masks=input_mask, scale=1.0 / math.sqrt(self.head_dim))
        elif self.config.attention_impl == "SDPA":
            k, v = self.upcast_kv(q, k, v)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=input_mask,
                dropout_p=self.config.attn_pdrop if self.training else 0.0,
                scale=1.0 / math.sqrt(self.head_dim),
            )
        else:
            raise ValueError(f"unknown attention_impl {self.config.attention_impl}")

        if self.config.attention_impl == "SDPA":
            y = y.transpose(1, 2)
        return self.dropout(self.c_proj(y.reshape(batch, seq_len, emb_dim)))


@dataclass(frozen=True, slots=True)
class ModelIO:
    in_shape: dict[str, torch.Size]
    in_dtype: dict[str, torch.dtype]
    out_shape: dict[str, torch.Size]
    out_dtype: dict[str, torch.dtype]

    def to_dict(self) -> dict[str, dict[str, torch.Size] | dict[str, torch.dtype]]:
        return {
            "in_shape": self.in_shape,
            "in_dtype": self.in_dtype,
            "out_shape": self.out_shape,
            "out_dtype": self.out_dtype,
        }


class WorldModelForInference(WorldModel):
    def __init__(self, config: WorldModel.Config):
        super().__init__(config)
        for block in self.blocks:
            block.attn.__class__ = InferenceSelfAttention
        self.inference_masks: dict[tuple[str, int, int, int], tuple[TensorOrMask | None, TensorOrMask | None]] = {}
        self.max_batch_size = -1
        self.max_seq_length = -1
        self.cache_dtype: torch.dtype | None = None

    @staticmethod
    def input_shapes(config: WorldModel.Config, batch_size: int = 1) -> dict[str, tuple[int, ...]]:
        frames, height, width = config.input_size
        pose_size = config.pose_size // 2
        return {
            "latents": (batch_size, frames, config.in_channels, height, width),
            "augments_pos_ref_augment": (batch_size, frames, pose_size),
            "ref_augment_from_augments_euler": (batch_size, frames, pose_size),
            "pose_mask": (batch_size, frames),
            "fidxs": (batch_size, frames),
        }

    @staticmethod
    def input_dtypes(dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.dtype]:
        return {
            "latents": dtype,
            "augments_pos_ref_augment": dtype,
            "ref_augment_from_augments_euler": dtype,
            "pose_mask": torch.int64,
            "fidxs": torch.int64,
        }

    @classmethod
    def example_inputs(
        cls,
        config: WorldModel.Config,
        *,
        batch_size: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "meta",
    ) -> dict[str, torch.Tensor]:
        dtypes = cls.input_dtypes(dtype)
        return {
            name: (
                torch.randn(shape, dtype=dtypes[name], device=device)
                if dtypes[name].is_floating_point
                else torch.ones(shape, dtype=dtypes[name], device=device)
            )
            for name, shape in cls.input_shapes(config, batch_size=batch_size).items()
        }

    def get_model_io(
        self,
        *,
        batch_size: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        steps: int = 1,
        num_prefill_frames: int = 14,
        cfg: float = 0.0,
    ) -> dict[str, dict[str, torch.Size] | dict[str, torch.dtype]]:
        inputs = self.example_inputs(self.config, batch_size=batch_size, dtype=dtype, device=self.pos_embed.device)
        outputs = self.generate(
            **inputs,
            dtype=dtype,
            steps=steps,
            num_prefill_frames=num_prefill_frames,
            cfg=cfg,
        )
        return ModelIO(
            in_shape={key: value.shape for key, value in inputs.items()},
            in_dtype={key: value.dtype for key, value in inputs.items()},
            out_shape={key: value.shape for key, value in outputs.items()},
            out_dtype={key: value.dtype for key, value in outputs.items()},
        ).to_dict()

    def compile_for_inference(self) -> None:
        for block in self.blocks:
            block.compile(mode="max-autotune-no-cudagraphs")

    @torch.no_grad()
    def setup_inference_attention_attrs(
        self,
        device: torch.device,
        layout: BranchLayout,
    ) -> tuple[TensorOrMask | None, TensorOrMask | None]:
        key = (str(device), layout.prefix_tokens, layout.tokens_per_branch, layout.branch_count)
        if key in self.inference_masks:
            return self.inference_masks[key]

        mask_fn = _mask_fn(self.config.transformer)
        prefill_mask_fn = partial(branch_mask_predicate, mask_fn, layout, True)
        decode_mask_fn = partial(branch_mask_predicate, mask_fn, layout, False)

        if self.config.transformer.attention_impl == "FLEX":
            create_mask = _create_block_mask_fn(device)
            prefill_mask = None
            if layout.prefix_tokens:
                prefill_mask = create_mask(
                    prefill_mask_fn,
                    B=None,
                    H=None,
                    Q_LEN=layout.prefix_tokens,
                    KV_LEN=layout.cache_seq_length,
                    device=device,
                )
                prefill_mask.device = device
            decode_mask = create_mask(
                decode_mask_fn,
                B=None,
                H=None,
                Q_LEN=layout.packed_tokens,
                KV_LEN=layout.cache_seq_length,
                device=device,
            )
            decode_mask.device = device
        elif self.config.transformer.attention_impl == "SDPA":
            prefill_mask = None
            if layout.prefix_tokens:
                prefill_mask = _dense_mask(
                    prefill_mask_fn,
                    layout.prefix_tokens,
                    layout.cache_seq_length,
                )[None, None].to(device=device, dtype=torch.bool)
            decode_mask = _dense_mask(
                decode_mask_fn,
                layout.packed_tokens,
                layout.cache_seq_length,
            )[None, None].to(device=device, dtype=torch.bool)
            if not decode_mask.is_meta and bool(decode_mask.all()):
                decode_mask = None
        else:
            raise ValueError(f"unknown attention_impl {self.config.transformer.attention_impl}")

        masks = (prefill_mask, decode_mask)
        self.inference_masks[key] = masks
        return masks

    def cleanup_inference_attention_attrs(self) -> None:
        self.inference_masks = {}

    def _has_compatible_caches(
        self,
        max_batch_size: int,
        max_seq_length: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> bool:
        if self.max_seq_length < max_seq_length or self.max_batch_size < max_batch_size or self.cache_dtype != dtype:
            return False
        return all(
            (cache := block.attn.kv_cache) is not None and cache.dtype == dtype and cache.k_cache.device == device
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
        if self._has_compatible_caches(max_batch_size, max_seq_length, dtype=dtype, device=device):
            return

        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        self.cache_dtype = dtype
        for block in self.blocks:
            block.attn.kv_cache = KVCache(
                max_batch_size,
                max_seq_length,
                self.config.transformer.n_head,
                self.config.transformer.n_embd // self.config.transformer.n_head,
                dtype,
                device,
            )

    def cleanup_caches(self) -> None:
        for block in self.blocks:
            block.attn.kv_cache = None
        self.max_batch_size = -1
        self.max_seq_length = -1
        self.cache_dtype = None

    @torch.inference_mode()
    def forward_n_steps(
        self,
        x: torch.Tensor,
        augments_pos_ref_augment: torch.Tensor,
        ref_augment_from_augments_euler: torch.Tensor,
        pose_mask: torch.Tensor,
        fidxs: torch.Tensor,
        input_pos_mask_pair: InputPosMaskPair,
        layout: BranchLayout,
        cfg: float,
        scheduler: RFScheduler,
        steps: int,
        *,
        return_trajectory: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor | None]:
        device = x.device
        batch, frames = x.shape[:2]
        trajectory = [x.clone()] if return_trajectory else None

        dummy_timestep = torch.ones(batch, frames, device=device, dtype=torch.float32)
        model_output: dict[str, torch.Tensor] = {}
        for step_idx in range(steps):
            timesteps = dummy_timestep * scheduler.timesteps[step_idx]
            packed_inputs = pack_branch_inputs(
                layout,
                x,
                timesteps,
                augments_pos_ref_augment,
                ref_augment_from_augments_euler,
                pose_mask,
                fidxs,
            )
            packed_output = self(
                *packed_inputs,
                input_pos_mask_pair=input_pos_mask_pair,
            )
            velocity = packed_output["sample"]
            if layout.branch_count == 2:
                unconditional, conditional = velocity.chunk(2, dim=1)
                velocity = unconditional + cfg * (conditional - unconditional)
            packed_output["sample"] = velocity
            model_output = packed_output
            x = scheduler.step(velocity, step_idx, x).to(x.dtype)
            if trajectory is not None:
                trajectory.append(x.clone())

        return x, model_output, torch.stack(trajectory, dim=1) if trajectory is not None else None

    @torch.inference_mode()
    def generate(
        self,
        latents: torch.Tensor,
        augments_pos_ref_augment: torch.Tensor,
        ref_augment_from_augments_euler: torch.Tensor,
        pose_mask: torch.Tensor,
        fidxs: torch.Tensor,
        *,
        steps: int = 15,
        num_prefill_frames: int | None = None,
        num_conditioning_frames: int | None = None,
        dtype: torch.dtype = torch.bfloat16,
        inference_schedule: str = "linear",
        cfg: float = 0.0,
        return_trajectory: bool = False,
        **scheduler_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        if num_prefill_frames is None:
            num_prefill_frames = 14 if num_conditioning_frames is None else num_conditioning_frames
        elif num_conditioning_frames is not None and num_conditioning_frames != num_prefill_frames:
            raise ValueError("num_prefill_frames and legacy num_conditioning_frames must match when both are provided")
        assert self.config.transformer.attention_mask != "NONE" or num_prefill_frames == 0, (
            "prefill and decode masks only make sense if we have some causal attention masking"
        )

        batch, frames = latents.shape[:2]
        if not 0 <= num_prefill_frames <= frames:
            raise ValueError(f"num_prefill_frames={num_prefill_frames} must be in [0, {frames}]")

        latents = latents.to(dtype=dtype)
        device = latents.device
        is_meta = latents.is_meta

        if steps <= 0:
            self.cleanup_caches()
            self.setup_attention_attrs(device)
            timesteps = torch.full(
                (batch, frames),
                RFScheduler.no_noise_timestep_value,
                device=device,
                dtype=torch.float32,
            )
            latents = self.scale_latents(latents)
            model_output = self(
                latents,
                timesteps,
                augments_pos_ref_augment,
                ref_augment_from_augments_euler,
                pose_mask,
                fidxs,
                input_pos_mask_pair=None,
            )
            start = max(0, num_prefill_frames - 1)
            output_latents = self.unscale_latents(latents[:, start:])
            outputs = {"latents": output_latents}
            if "plan" in model_output:
                outputs["plan"] = model_output["plan"]
            if return_trajectory:
                outputs["trajectory"] = output_latents.unsqueeze(1)
            return outputs

        if self.final_layer is None:
            raise ValueError("diffusion sampling requires model.out_channels > 0")
        if num_prefill_frames == frames:
            raise ValueError("diffusion sampling requires at least one frame after the prefill")

        scheduler = RFScheduler(
            steps=steps, inference_schedule=inference_schedule, **scheduler_kwargs
        ).to(device=device)
        layout = build_branch_layout(
            prefix_tokens=num_prefill_frames * self.config.num_spatial_patches,
            tokens_per_branch=((frames - num_prefill_frames) * self.config.num_spatial_patches),
            cfg_enabled=cfg > 0.0,
            device=device,
        )
        self.setup_caches(
            batch,
            layout.cache_seq_length,
            dtype=_cache_dtype_for_device(device),
            device=device,
        )
        prefill_mask, decode_mask = self.setup_inference_attention_attrs(device, layout)

        latents[:, :num_prefill_frames] = self.scale_latents(latents[:, :num_prefill_frames])
        self._prefill(
            latents=latents,
            augments_pos_ref_augment=augments_pos_ref_augment,
            ref_augment_from_augments_euler=ref_augment_from_augments_euler,
            pose_mask=pose_mask,
            fidxs=fidxs,
            scheduler=scheduler,
            num_prefill_frames=num_prefill_frames,
            layout=layout,
            prefill_mask=prefill_mask,
        )

        decode_frames = latents[:, num_prefill_frames:]
        decode_frames, model_output, trajectory = self._decode(
            decode_frames=decode_frames,
            augments_pos_ref_augment=augments_pos_ref_augment,
            ref_augment_from_augments_euler=ref_augment_from_augments_euler,
            pose_mask=pose_mask,
            fidxs=fidxs,
            scheduler=scheduler,
            num_prefill_frames=num_prefill_frames,
            layout=layout,
            decode_mask=decode_mask,
            cfg=cfg,
            steps=steps,
            return_trajectory=return_trajectory,
        )

        if not is_meta and not all(torch.isfinite(value).all() for value in model_output.values()):
            self.cleanup_caches()
            raise ValueError("model outputs contain inf/nan")

        outputs = {"latents": self.unscale_latents(decode_frames)}
        if "plan" in model_output:
            outputs["plan"] = model_output["plan"]
        if trajectory is not None:
            outputs["trajectory"] = self.unscale_latents(trajectory)
        return outputs

    def _prefill(
        self,
        *,
        latents: torch.Tensor,
        augments_pos_ref_augment: torch.Tensor,
        ref_augment_from_augments_euler: torch.Tensor,
        pose_mask: torch.Tensor,
        fidxs: torch.Tensor,
        scheduler: RFScheduler,
        num_prefill_frames: int,
        layout: BranchLayout,
        prefill_mask: TensorOrMask | None,
    ) -> dict[str, torch.Tensor]:
        if num_prefill_frames <= 0:
            return {}

        batch = latents.shape[0]
        device = latents.device
        input_pos = torch.arange(0, num_prefill_frames * self.config.num_spatial_patches, device=device)
        timesteps = torch.ones(
            (batch, num_prefill_frames), device=device, dtype=torch.float32
        ) * scheduler.no_noise_timestep
        input_pos_mask_pair = InputPosMaskPair(
            input_pos=input_pos,
            cache_pos=input_pos,
            cache_seq_length=layout.cache_seq_length,
            input_mask=prefill_mask,
        )
        return self(
            latents[:, :num_prefill_frames],
            timesteps,
            augments_pos_ref_augment[:, :num_prefill_frames],
            ref_augment_from_augments_euler[:, :num_prefill_frames],
            pose_mask[:, :num_prefill_frames],
            fidxs[:, :num_prefill_frames],
            input_pos_mask_pair=input_pos_mask_pair,
        )

    def _decode(
        self,
        *,
        decode_frames: torch.Tensor,
        augments_pos_ref_augment: torch.Tensor,
        ref_augment_from_augments_euler: torch.Tensor,
        pose_mask: torch.Tensor,
        fidxs: torch.Tensor,
        scheduler: RFScheduler,
        num_prefill_frames: int,
        layout: BranchLayout,
        decode_mask: TensorOrMask | None,
        cfg: float,
        steps: int,
        return_trajectory: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor | None]:
        input_pos_mask_pair = InputPosMaskPair(
            input_pos=layout.input_pos,
            cache_pos=layout.cache_pos,
            cache_seq_length=layout.cache_seq_length,
            input_mask=decode_mask,
        )
        return self.forward_n_steps(
            decode_frames,
            augments_pos_ref_augment[:, num_prefill_frames:],
            ref_augment_from_augments_euler[:, num_prefill_frames:],
            pose_mask[:, num_prefill_frames:],
            fidxs[:, num_prefill_frames:],
            input_pos_mask_pair,
            layout,
            cfg,
            scheduler,
            steps,
            return_trajectory=return_trajectory,
        )


def main() -> None:
    import argparse

    from torchtitan.experiments.worldmodel.model_config import model_registry

    parser = argparse.ArgumentParser(description="Run a small worldmodel inference smoke test.")
    parser.add_argument("--flavor", default="debugmodel")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = model_registry(args.flavor).model
    model = WorldModelForInference(config).to(device=device, dtype=torch.bfloat16).eval()
    inputs = WorldModelForInference.example_inputs(
        config,
        batch_size=2,
        dtype=torch.bfloat16,
        device=device,
    )
    outputs = model.generate(
        **inputs,
        dtype=torch.bfloat16,
        steps=2,
        num_prefill_frames=14,
        cfg=2.0
    )
    print(
        {
            "device": str(device),
            "cache_dtype": str(model.cache_dtype),
            "inputs": {key: (tuple(value.shape), str(value.dtype)) for key, value in inputs.items()},
            "outputs": {key: (tuple(value.shape), str(value.dtype)) for key, value in outputs.items()},
        }
    )


if __name__ == "__main__":
    main()
