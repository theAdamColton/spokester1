from typing import Callable
import math

import torch
import torch.nn.functional as F

from src.net.torch_utils import random_uniform, unsqueeze_trailing


class FlowMatchingHelper:
    def __init__(
        self,
        time_shift_dim: float = 16**2 * 768,
        time_shift_base: float = 4096,
        pred_mode: str = "clean_input",
        loss_mode: str = "velocity",
        eps: float = 5e-2,
    ):
        self.time_shift_dim = time_shift_dim
        self.time_shift_base = time_shift_base
        self.prediction_mode = pred_mode
        self.loss_mode = loss_mode
        self.eps = eps

    def _get_timestep_interval(self):
        t0 = 0.0
        t1 = 1 - 1 / 1000
        return t0, t1

    def _shift_timesteps(self, t: torch.Tensor):
        shift = math.sqrt(self.time_shift_dim / self.time_shift_base)
        # Shifts timesteps to be smaller (assuming shift > 1)
        t = t / (t - shift * t + shift)
        return t

    def forward_noise(self, x_1: torch.Tensor, timesteps_shape=None, torch_rng=None):
        batch_size, *_ = x_1.shape
        device = x_1.device

        if timesteps_shape is None:
            timesteps_shape = (batch_size,)

        x_0 = torch.randn(
            x_1.shape, device=x_1.device, dtype=x_1.dtype, generator=torch_rng
        )

        t_0, t_1 = self._get_timestep_interval()

        t = random_uniform(
            timesteps_shape, t_0, t_1, device=device, torch_rng=torch_rng
        )

        t = self._shift_timesteps(t)

        # Noisy sample
        x_t = unsqueeze_trailing(t, x_1) * x_1 + unsqueeze_trailing(1 - t, x_0) * x_0

        # Target velocity
        v_t = (x_1 - x_t) / (1.0 - unsqueeze_trailing(t, x_1)).clip(self.eps)

        return t, x_t, v_t, x_0

    def compute_velocity(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        denoiser: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        model_prediction: torch.Tensor | None = None,
        sample_clipper: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        assert (denoiser is not None) ^ (model_prediction is not None)

        if denoiser is not None:
            model_prediction = denoiser(x_t, timesteps)

        og_dtype = model_prediction.dtype

        if self.prediction_mode == "velocity":
            v_t = model_prediction
        elif self.prediction_mode == "clean_input":
            x_1_pred = model_prediction

            if sample_clipper is not None:
                x_1_pred = sample_clipper(x_1_pred)

            with torch.autocast(x_t.device.type, enabled=False):
                x_t = x_t.float()
                x_1_pred = x_1_pred.float()
                timesteps = timesteps.float()
                timesteps = unsqueeze_trailing(timesteps, x_t)

                v_t = (x_1_pred - x_t) / (1.0 - timesteps).clip(self.eps)

            v_t = v_t.to(og_dtype)
        else:
            raise ValueError(self.prediction_mode)

        return v_t

    def compute_loss(
        self,
        x_1: torch.Tensor,
        denoiser: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        timesteps_shape=None,
        torch_rng=None,
    ):
        t, x_t, v_t, x_0 = self.forward_noise(
            x_1, timesteps_shape=timesteps_shape, torch_rng=torch_rng
        )

        if self.loss_mode != "velocity":
            raise NotImplementedError()

        v_t_hat = self.compute_velocity(x_t, t, denoiser)

        with torch.autocast(x_1.device.type, enabled=False):
            loss = F.mse_loss(v_t_hat.float(), v_t.float(), reduction="none")

        return loss

    def get_inference_timesteps(self, num_steps: int = 50, device=torch.device("cpu")):
        t0, t1 = self._get_timestep_interval()
        timesteps = torch.linspace(t0, t1, num_steps + 1, device=device)
        timesteps = self._shift_timesteps(timesteps)

        # Determine the sequence of time points and step sizes
        ts = timesteps[:-1]  # Start times for each step (t_i)
        next_ts = timesteps[1:]  # End times for each step (t_{i+1})
        dts = next_ts - ts  # Step sizes (h = t_{i+1} - t_i)

        return ts, dts

    def sample_euler(
        self,
        *,
        noise: torch.Tensor,
        denoiser: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        sample_clipper: Callable[[torch.Tensor], torch.Tensor] | None = None,
        num_steps: int = 50,
        trunc_num_steps: int | None = None,
    ) -> torch.Tensor:
        device = noise.device

        ts, dts = self.get_inference_timesteps(num_steps, device=device)

        sample = noise

        # special case - skip the velocity computation
        if num_steps == 1 and self.prediction_mode == "clean_input":
            sample = denoiser(sample, ts[0])
            if sample_clipper is not None:
                sample = sample_clipper(sample)
            return sample

        for t_idx, t in enumerate(ts):
            dt = dts[t_idx]

            v_t = self.compute_velocity(
                sample, t, denoiser, sample_clipper=sample_clipper
            )

            # Potentially exit early
            should_truncate = (trunc_num_steps is not None) and (
                t_idx + 1 >= trunc_num_steps
            )
            if should_truncate:
                sample = sample + (1 - t) * v_t
                return sample

            sample = sample + dt * v_t

            if sample_clipper is not None:
                sample = sample_clipper(sample)

        return sample
