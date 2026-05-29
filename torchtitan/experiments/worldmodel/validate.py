import base64
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
import html
import io
import os
import socket
import time
from typing import Any, cast

import av
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.pipelining.schedules import _PipelineSchedule

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.loss import LossFunction
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.components.validate import ValidationContext, Validator
from torchtitan.config import ParallelismConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.experiments.worldmodel.compressor import (
    MAX_UINT8,
    load_compressor_decoder,
    load_compressor_encoder,
)
from torchtitan.experiments.worldmodel.dataloader import WorldModelDataLoader
from torchtitan.experiments.worldmodel.loss import WorldModelLoss
from torchtitan.experiments.worldmodel.model import WorldModel
from torchtitan.experiments.worldmodel.schedulers import RFScheduler
from torchtitan.experiments.worldmodel.step import (
    _floating_model_dtype,
    prepare_worldmodel_batch,
)
from torchtitan.tools.logging import logger

WorldModelBatch = tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]


def _clone_batch_to_cpu(batch: WorldModelBatch) -> WorldModelBatch:
    input_dict, targets = batch
    return (
        {key: value.detach().cpu().clone() for key, value in input_dict.items()},
        {key: value.detach().cpu().clone() for key, value in targets.items()},
    )


def _distributed_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _distributed_world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _rank_context() -> str:
    return (
        f"rank={_distributed_rank()}/{_distributed_world_size()} "
        f"local_rank={os.environ.get('LOCAL_RANK', '?')} host={socket.gethostname()}"
    )


def _all_gather_object(value: Any) -> list[Any]:
    if not dist.is_available() or not dist.is_initialized():
        return [value]
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


def encode_to_mp4(frames: np.ndarray, fps: int) -> bytes:
    height, width = frames.shape[1:3]
    if height % 2 or width % 2:
        frames = np.pad(
            frames,
            ((0, 0), (0, height % 2), (0, width % 2), (0, 0)),
            mode="edge",
        )
        height, width = frames.shape[1:3]

    frames = np.ascontiguousarray(frames, dtype=np.uint8)
    output = io.BytesIO()
    with av.open(output, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for frame_array in frames:
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return output.getvalue()


def _decoder_output_to_video_frames(decoded: torch.Tensor) -> np.ndarray:
    if decoded.dtype != torch.uint8:
        decoded = decoded.add(1).mul(MAX_UINT8 / 2).clamp(0, MAX_UINT8).to(torch.uint8)

    if decoded.shape[1] in (1, 3, 6):
        if decoded.shape[1] == 6:
            decoded = torch.cat([decoded[:, :3], decoded[:, 3:]], dim=-1)
        elif decoded.shape[1] == 1:
            decoded = decoded.repeat(1, 3, 1, 1)
        decoded = decoded.permute(0, 2, 3, 1)
    elif decoded.shape[-1] == 6:
        decoded = torch.cat([decoded[..., :3], decoded[..., 3:]], dim=-2)
    elif decoded.shape[-1] == 1:
        decoded = decoded.repeat(1, 1, 1, 3)

    return decoded.contiguous().cpu().numpy()


def _video_html(video: bytes, *, title: str) -> str:
    encoded = base64.b64encode(video).decode("ascii")
    return (
        f"<figure style='margin:0 0 16px 0'>"
        f"<figcaption>{html.escape(title)}</figcaption>"
        f"<video controls loop muted src='data:video/mp4;base64,{encoded}' "
        f"style='max-width:100%; height:auto'></video>"
        f"</figure>"
    )


class WorldModelValidator(Validator):
    @dataclass(kw_only=True, slots=True)
    class Config(Validator.Config):
        dataloader: BaseDataLoader.Config = field(
            default_factory=lambda: WorldModelDataLoader.Config(
                mock_data=True,
                infinite=False,
            )
        )
        pose_dropout: float = 0.0
        noise_scheduler_steps: int = 10
        cache_data: bool = True
        extra_methods: list[str] = field(default_factory=list)

        def __post_init__(self) -> None:
            if self.steps < 0:
                raise ValueError("worldmodel validation steps must be >= 0")

    def __init__(
        self,
        config: Config,
        *,
        parallelism: ParallelismConfig,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        parallel_dims: ParallelDims,
        loss_fn: LossFunction,
        validation_context: ValidationContext,
        local_batch_size: int,
        metrics_processor: MetricsProcessor,
        seq_len: int | None = None,
        pp_schedule: _PipelineSchedule | None = None,
        pp_has_first_stage: bool | None = None,
        pp_has_last_stage: bool | None = None,
        **kwargs,
    ):
        del parallelism, pp_schedule, pp_has_first_stage, pp_has_last_stage
        del kwargs
        self.config = config
        self.loss_fn = cast(WorldModelLoss, loss_fn)
        self.tokenizer = tokenizer
        self.parallel_dims = parallel_dims
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.local_batch_size = local_batch_size
        self.seq_len = seq_len
        self.validation_context = validation_context
        self.metrics_processor = metrics_processor
        self.dl_config = replace(config.dataloader, infinite=False)
        self.compressor_encoder: torch.nn.Module | None = None
        self.compressor_decoder: torch.nn.Module | None = None
        self._dtype: torch.dtype | None = None
        self._cached_batches: list[WorldModelBatch] = []

    def _load_compressor_encoder(
        self,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.nn.Module:
        if self.compressor_encoder is not None:
            return self.compressor_encoder
        self.compressor_encoder = load_compressor_encoder(
            compressor_model=self.dl_config.compressor_model,
            device=device,
            dtype=dtype,
        )
        return self.compressor_encoder

    def _load_compressor_decoder(
        self,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.nn.Module:
        if self.compressor_decoder is not None:
            return self.compressor_decoder
        self.compressor_decoder = load_compressor_decoder(
            compressor_model=self.dl_config.compressor_model,
            device=device,
            dtype=dtype,
        )
        return self.compressor_decoder

    def _fill_validation_cache(self) -> None:
        validation_dataloader = self.dl_config.build(
            dp_world_size=self.dp_world_size,
            dp_rank=self.dp_rank,
            tokenizer=self.tokenizer,
            local_batch_size=self.local_batch_size,
            val=True,
        )
        validation_iterator = iter(validation_dataloader)
        start_time = time.perf_counter()
        try:
            for step_idx, batch in zip(range(self.config.steps), validation_iterator):
                batch = _clone_batch_to_cpu(batch)
                self._cached_batches.append(batch)
                logger.info(
                    "worldmodel validation cached CPU batch "
                    f"{len(self._cached_batches)}/{self.config.steps}: "
                    f"{_rank_context()}, elapsed_s={time.perf_counter() - start_time:.1f}"
                )
        finally:
            del validation_iterator
            del validation_dataloader

    def _iter_validation_batches(self) -> Iterator[WorldModelBatch]:
        if self.config.steps == 0:
            return
        if self.config.cache_data:
            if self._cached_batches:
                logger.info(
                    "worldmodel validation reusing cached CPU batches: "
                    f"{_rank_context()}, batches={len(self._cached_batches)}"
                )
            else:
                self._fill_validation_cache()
            for batch in self._cached_batches[: self.config.steps]:
                yield batch
            return

        validation_dataloader = self.dl_config.build(
            dp_world_size=self.dp_world_size,
            dp_rank=self.dp_rank,
            tokenizer=self.tokenizer,
            local_batch_size=self.local_batch_size,
            val=True,
        )
        validation_iterator = iter(validation_dataloader)
        num_steps = 0
        try:
            for step_idx, batch in zip(range(self.config.steps), validation_iterator):
                num_steps = step_idx + 1
                yield batch
        finally:
            del validation_iterator
            del validation_dataloader
            logger.info(
                "worldmodel validation dataloader released: "
                f"{_rank_context()}, batches={num_steps}"
            )

    def _run_metrics_validation(
        self,
        model: WorldModel,
        step: int,
    ) -> tuple[float, dict[str, Any]] | None:
        if self.config.steps == 0:
            return None

        scheduler = RFScheduler(steps=self.config.noise_scheduler_steps).to(
            device=next(model.parameters()).device
        )
        discrete_timesteps = scheduler.timesteps[:-1]

        term_sums: dict[str, torch.Tensor] = {}
        num_steps = 0
        val_tokens = 0
        device = next(model.parameters()).device
        dtype = self._dtype or _floating_model_dtype(model)
        compressor_encoder = self._load_compressor_encoder(dtype, device)
        logger.info(
            "worldmodel validation metrics start: "
            f"{_rank_context()}, step={step}, target_batches={self.config.steps}"
        )
        for input_dict, targets in self._iter_validation_batches():
            model_inputs, targets = prepare_worldmodel_batch(
                model,
                input_dict,
                targets,
                device=device,
                scheduler=scheduler,
                discrete_timesteps=discrete_timesteps,
                compressor_encoder=compressor_encoder,
                pose_dropout=self.config.pose_dropout,
                train=False,
                dtype=dtype,
            )
            with self.validation_context():
                outputs = model(**model_inputs)
                _per_sample_loss, terms = self.loss_fn(outputs, targets)
            for name, term in terms.items():
                term_sums[name] = term_sums.get(
                    name,
                    torch.zeros((), device=device),
                ) + term.mean().detach()
            bsz = next(iter(input_dict.values())).shape[0]
            val_tokens += bsz * (self.seq_len or 1)
            num_steps += 1

        if num_steps == 0:
            return None

        metrics = {name: value / num_steps for name, value in term_sums.items()}
        loss = metrics["loss"]
        if self.parallel_dims.dp_cp_enabled:
            loss_value = dist_utils.dist_mean(
                loss,
                self.parallel_dims.get_optional_mesh("loss"),
            )
            global_val_tokens = dist_utils.dist_sum(
                torch.tensor(val_tokens, dtype=torch.int64, device=device),
                self.parallel_dims.get_optional_mesh("loss"),
            )
        else:
            loss_value = float(loss.item())
            global_val_tokens = val_tokens

        extra_metrics: dict[str, Any] = {
            "data/n_tokens_seen/val": global_val_tokens,
            "dataloader/val_cache_size/": len(self._cached_batches),
        }
        self.metrics_processor.ntokens_since_last_log += val_tokens
        for name, term in metrics.items():
            if name == "loss":
                continue
            if self.parallel_dims.dp_cp_enabled:
                term_value = dist_utils.dist_mean(
                    term,
                    self.parallel_dims.get_optional_mesh("loss"),
                )
            else:
                term_value = float(term.item())
            extra_metrics[f"worldmodel/{name}/val"] = term_value

        self.metrics_processor.log_validation(
            loss=loss_value,
            step=step,
            extra_metrics=extra_metrics,
        )
        return loss_value, extra_metrics

    def _generate_random_video(
        self,
        model: WorldModel,
        decoder: torch.nn.Module,
        step: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> bytes:
        rank = _distributed_rank()
        generator = torch.Generator(device=device).manual_seed(17_000 + step * 1_000 + rank)
        batch_size = 1
        num_frames, height, width = model.config.input_size
        logger.info(
            "worldmodel html report generating video: "
            f"rank={rank}, frames={num_frames}, latent_size=({height}, {width}), "
            f"steps={self.config.noise_scheduler_steps}"
        )
        latents = torch.randn(
            batch_size,
            num_frames,
            model.config.in_channels,
            height,
            width,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        augments_pos_ref_augment = torch.zeros(
            batch_size,
            num_frames,
            model.config.pose_size // 2,
            device=device,
            dtype=dtype,
        )
        ref_augment_from_augments_euler = torch.zeros_like(augments_pos_ref_augment)
        pose_mask = torch.ones(batch_size, num_frames, device=device, dtype=torch.int64)
        fidx = torch.arange(num_frames, device=device, dtype=torch.int64).expand(batch_size, -1)

        scheduler = RFScheduler(steps=self.config.noise_scheduler_steps).to(device=device)
        timestep_shape = (batch_size, num_frames)
        for timestep_idx in range(self.config.noise_scheduler_steps):
            timesteps = torch.ones(timestep_shape, device=device, dtype=torch.float32)
            timesteps = timesteps * scheduler.timesteps[timestep_idx]
            with self.validation_context():
                outputs = model(
                    latents,
                    timesteps,
                    augments_pos_ref_augment,
                    ref_augment_from_augments_euler,
                    pose_mask,
                    fidx,
                    return_plan=False,
                )
            latents = scheduler.step(outputs["sample"].detach(), timestep_idx, latents).to(dtype=dtype).detach()

        latents = latents * model.config.compressor_std + model.config.compressor_mean
        decoded = decoder(latents.flatten(0, 1).to(dtype=dtype))
        if isinstance(decoded, tuple):
            decoded = decoded[0]
        frames = _decoder_output_to_video_frames(decoded)
        video = encode_to_mp4(frames, fps=self.dl_config.fps)
        logger.info(
            "worldmodel html report generated video: "
            f"rank={rank}, frames={len(frames)}, bytes={len(video)}"
        )
        return video

    def _run_html_report(
        self,
        model: WorldModel,
        step: int,
        result: tuple[float, dict[str, Any]] | None,
    ) -> None:
        device = next(model.parameters()).device
        dtype = self._dtype or _floating_model_dtype(model)
        logger.info(f"worldmodel html report start: step={step}, rank={_distributed_rank()}")
        decoder = self._load_compressor_decoder(dtype, device)
        rank_video = {
            "rank": _distributed_rank(),
            "video": self._generate_random_video(
                model,
                decoder,
                step,
                device=device,
                dtype=dtype,
            ),
        }
        logger.info(f"worldmodel html report gathering videos: rank={_distributed_rank()}")
        gathered_videos = _all_gather_object(rank_video)
        if _distributed_rank() != 0:
            logger.info(f"worldmodel html report gathered on nonzero rank: rank={_distributed_rank()}")
            return

        metric_rows = ""
        if result is not None:
            loss, extra_metrics = result
            metric_rows = f"<p>val_loss={loss:.4f}</p>"
            metric_rows += "<ul>" + "".join(
                f"<li>{html.escape(name)}={value:.4f}</li>"
                for name, value in sorted(extra_metrics.items())
                if isinstance(value, float)
            ) + "</ul>"
        videos = "".join(
            _video_html(item["video"], title=f"rank {item['rank']}")
            for item in gathered_videos
        )
        report = f"<h1>Epoch {step}</h1>{metric_rows}<h2>Random latent samples</h2>{videos}"
        self.metrics_processor.write_report(report, step, "report", "html")
        logger.info(
            "worldmodel html report written: "
            f"step={step}, videos={len(gathered_videos)}, bytes={sum(len(item['video']) for item in gathered_videos)}"
        )

    @torch.no_grad()
    def validate(self, model_parts: list[nn.Module], step: int) -> None:
        model = cast(WorldModel, model_parts[0])
        model.eval()
        logger.info(
            "worldmodel validation start: "
            f"{_rank_context()}, step={step}, steps={self.config.steps}, cache_data={self.config.cache_data}, "
            f"cached_batches={len(self._cached_batches)}, extra_methods={self.config.extra_methods}"
        )
        try:
            result = self._run_metrics_validation(model, step)

            if "html_report" in self.config.extra_methods:
                self._run_html_report(model, step, result)
        finally:
            model.train()
            logger.info(f"worldmodel validation finished: {_rank_context()}, step={step}")
