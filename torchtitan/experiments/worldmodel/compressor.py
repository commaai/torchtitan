import io
import einops
import torch

COMPRESSOR_IN_CHANNELS = 6
MAX_UINT8 = 255.0


def load_compressor_encoder(*, compressor_model: str, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    from xx.training.lib.checkpoint import Checkpoint
    compressor = torch.jit.load(io.BytesIO(Checkpoint(compressor_model)["encoder.jit"]), map_location="cpu")
    return compressor.to(device=device, dtype=dtype).eval()


def load_compressor_decoder(*, compressor_model: str, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    from xx.training.lib.checkpoint import Checkpoint
    compressor = torch.jit.load(io.BytesIO(Checkpoint(compressor_model)["decoder.jit"]), map_location="cpu")
    return compressor.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def images_to_latents(compressor: torch.nn.Module, imgs: torch.Tensor, big_imgs: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    batch, timesteps = imgs.shape[:2]
    compressor_inputs = einops.rearrange(
        [imgs, big_imgs],
        "nc b t h w c -> (b t) (nc c) h w",
        nc=2,
        b=batch,
        t=timesteps,
    )
    compressor_inputs = compressor_inputs.to(device=device, dtype=dtype)
    compressor_inputs = compressor_inputs.div(MAX_UINT8).mul(2).sub(1).clamp(-1, 1)

    latents = compressor(compressor_inputs)
    if isinstance(latents, tuple):
        latents = latents[0]
    return einops.rearrange(
        latents,
        "(b t) (nc c) h w -> b t (nc c) h w",
        nc=2,
        b=batch,
        t=timesteps,
    )
