"""Coarse reconstruction with multiview MSE mask supervision."""

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from cameras import fixed_cameras, orbit_camera, render_camera
from gs_renderer import Renderer
from guidance.zero123 import Zero123
from runtime import (
    seed_everything,
    load_coarse_images,
    setup_gaussians,
    density_step,
    save_grid,
)


def train(config, views, model_root, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seed_everything(config.seed)
    phase = config.coarse
    images, masks = load_coarse_images(views, config.ref_size)
    poses, cameras, _, centers = fixed_cameras(config, "cuda")
    renderer = Renderer(sh_degree=0)
    renderer.initialize(num_pts=config.num_pts)
    optimizer = setup_gaussians(renderer, phase)
    guidance = Zero123(Path(model_root) / "zero123-xl-diffusers", images)
    for step in range(1, phase.iters + 1):
        ratio = min(1, step / phase.iters)
        renderer.gaussians.update_learning_rate(step)
        front = renderer.render(cameras[0])
        rgb_loss = F.mse_loss(front["image"][None], images[0])
        predicted_masks = [front["alpha"][None]] + [
            renderer.render(c)["alpha"][None] for c in cameras[1:]
        ]
        mask_loss = F.mse_loss(torch.cat(predicted_masks), torch.cat(masks))
        elevation = int(np.random.randint(config.min_ver, config.max_ver))
        azimuth = int(np.random.randint(-180, 180))
        pose = orbit_camera(elevation, azimuth, config.radius)
        size = 128 if ratio < 0.3 else 256 if ratio < 0.6 else 512
        background = torch.tensor(
            (
                [1.0, 1.0, 1.0]
                if np.random.rand() > config.invert_bg_prob
                else [0.0, 0.0, 0.0]
            ),
            device="cuda",
        )
        sampled = renderer.render(
            render_camera(pose, size, config.fovy), bg_color=background
        )
        sample_center = torch.as_tensor(pose[:3, 3], device="cuda")
        nearest = int(
            F.cosine_similarity(sample_center[None], centers, dim=1)
            .topk(1)
            .indices.item()
        )
        # The map averages 4 latent channels; native coarse SDS sums them.
        sds = (
            guidance.loss_maps(
                sampled["image"][None], elevation, azimuth, ratio, [nearest]
            )[0].sum()
            * 4
        )
        loss = (
            ratio * (phase.lambda_pixel * rgb_loss + phase.lambda_mask * mask_loss)
            + phase.lambda_sds * sds
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite coarse loss at {step}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        density_step(renderer.gaussians, sampled, phase, step)
        if step % phase.opacity_reset_interval == 0 and step != phase.iters:
            renderer.gaussians.reset_opacity()
    model = output / "point_cloud_model.ply"
    renderer.gaussians.save_ply(str(model))
    save_grid(renderer, config, output / "grid.png")
    return model
