import torch


class RFScheduler(torch.nn.Module):
    no_noise_timestep_value: float = 0.0
    timesteps: torch.Tensor
    no_noise_timestep: torch.Tensor

    def __init__(self, steps: int = 15):
        super().__init__()
        self.num_timesteps = steps + 1
        timesteps = torch.linspace(1, 0, self.num_timesteps, dtype=torch.float32)
        self.register_buffer("timesteps", timesteps)
        self.register_buffer("dt", -torch.diff(timesteps))
        self.register_buffer("no_noise_timestep", torch.tensor(self.no_noise_timestep_value, dtype=torch.float32))

    def sample_timestep(self, shape: tuple[int, ...]):
        nt = torch.randn(shape, device=self.timesteps.device)
        return torch.sigmoid(nt)

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor):
        while len(timesteps.shape) < len(original_samples.shape):
            timesteps = timesteps.unsqueeze(-1)
        return (1 - timesteps) * original_samples + timesteps * noise

    def step(self, model_output: torch.Tensor, timestep_idx: int, sample: torch.Tensor):
        return sample + self.dt[timestep_idx] * model_output
