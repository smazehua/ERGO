"""Paired partial DDIM inversion and deterministic denoising schedules."""

import math
from dataclasses import dataclass

import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler


@dataclass
class DDIMPlan:
    forward: DDIMScheduler
    inverse: DDIMInverseScheduler
    inversion_timesteps: torch.Tensor
    denoising_timesteps: torch.Tensor
    inversion_model_timesteps: torch.Tensor


def make_ddim_plan(config, steps, strength, device):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not math.isfinite(strength) or not 0 < strength <= 1:
        raise ValueError("strength must be in (0, 1]")
    count = int(steps * strength)
    if count < 1:
        raise ValueError("steps * strength must select at least one step")
    # Leading spacing gives matching ascending/descending endpoints. Disabling
    # clipping avoids destroying latent information during the inverse path.
    forward = DDIMScheduler.from_config(
        config,
        timestep_spacing="leading",
        steps_offset=0,
        clip_sample=False,
        thresholding=False,
        set_alpha_to_one=True,
    )
    inverse_config = dict(forward.config)
    inverse_config.pop("thresholding", None)
    inverse = DDIMInverseScheduler.from_config(inverse_config)
    forward.set_timesteps(steps, device=device)
    inverse.set_timesteps(steps, device=device)
    inversion_timesteps = inverse.timesteps[:count]
    denoising_timesteps = forward.timesteps[-count:]
    if not torch.equal(inversion_timesteps.flip(0), denoising_timesteps):
        raise ValueError("DDIM inversion and denoising schedules do not match")
    # inverse.step(output, target_t, x_source) uses alpha(target_t - stride).
    # The UNet must see that source time too. Its first supported time is 0;
    # clamp only the initial clean-boundary evaluation to that time.
    stride = inverse.config.num_train_timesteps // steps
    model_timesteps = (inversion_timesteps - stride).clamp(min=0)
    return DDIMPlan(
        forward, inverse, inversion_timesteps, denoising_timesteps, model_timesteps
    )


@torch.no_grad()
def invert_latents(clean_latents, plan, predict_noise):
    """Invert latents with the paired DDIM inverse schedule."""
    latents = clean_latents.float()
    for timestep, model_timestep in zip(
        plan.inversion_timesteps, plan.inversion_model_timesteps
    ):
        prediction = predict_noise(latents, model_timestep).float()
        latents = plan.inverse.step(prediction, timestep, latents, return_dict=False)[0]
        if not torch.isfinite(latents).all():
            raise FloatingPointError("nonfinite DDIM inversion latent")
    return latents
