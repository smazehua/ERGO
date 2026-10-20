"""ERGO training from the coarse model, without opacity resets."""

from pathlib import Path
import numpy as np
import torch
from cameras import fixed_cameras, orbit_camera, render_camera
from gs_renderer import Renderer
from guidance.zero123 import Zero123
from runtime import (
    seed_everything,
    load_images,
    setup_gaussians,
    density_step,
    save_grid,
    view_tensors,
)
from ergo.geometry import local_supervision, geometry_loss
from ergo.texture import camera_proximity_weights, texture_aware_terms
from ergo.objective import AdaptiveWeights
from ergo.regularization import constrain_log_scales_


def train(config, views, coarse_model, model_root, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seed_everything(config.seed)
    rng = np.random.default_rng(config.seed)
    phase, ec = config.final, config.ergo
    images, _ = load_images(views, config.ref_size)
    _, cameras, projections, centers = fixed_cameras(config, "cuda")
    coarse = Renderer(sh_degree=0)
    coarse.initialize(str(coarse_model))
    with torch.no_grad():
        targets = [view_tensors(coarse.render(c)) for c in cameras]
        depths = [x[1] for x in targets]
        masks = [(x[2] >= 0.5).to(x[2]) for x in targets]
        images = [i * m + (1 - m) for i, m in zip(images, masks)]
        visibility, consistency = local_supervision(
            images,
            depths,
            masks,
            projections,
            ec.visibility_dilation_kernel,
            ec.discrepancy_sigma,
        )
    del coarse
    torch.cuda.empty_cache()
    renderer = Renderer(sh_degree=0)
    renderer.initialize(str(coarse_model))
    optimizer = setup_gaussians(renderer, phase)
    guidance = Zero123(
        Path(model_root) / "zero123-xl-diffusers",
        load_images(views, config.ref_size)[0],
    )
    weights = AdaptiveWeights(7, ec, "cuda")
    for step in range(1, phase.iters + 1):
        renderer.gaussians.update_learning_rate(step)
        outputs = [renderer.render(c) for c in cameras]
        rendered = [view_tensors(o) for o in outputs]
        geometry = [
            geometry_loss(*r, i, d, m, v, w, ec)
            for r, i, d, m, v, w in zip(
                rendered, images, depths, masks, visibility, consistency
            )
        ]
        elevation = int(rng.integers(config.min_ver, config.max_ver + 1))
        azimuth = int(rng.integers(-180, 180))
        pose = orbit_camera(elevation, azimuth, config.radius)
        size = (
            128
            if step / phase.iters < 0.3
            else 256 if step / phase.iters < 0.6 else 512
        )
        bg = torch.tensor(
            (
                [1.0, 1.0, 1.0]
                if rng.random() > config.invert_bg_prob
                else [0.0, 0.0, 0.0]
            ),
            device="cuda",
        )
        sampled = renderer.render(
            render_camera(pose, size, config.fovy), bg_color=bg
        )
        sample_image = sampled["image"][None]
        ratio = min(1 - phase.sds_min_timestep_ratio, step / phase.iters)
        maps = guidance.loss_maps(sample_image, elevation, azimuth, ratio, range(7))
        proximity = camera_proximity_weights(
            torch.as_tensor(pose[:3, 3], device="cuda"), centers, ec.eps
        )
        texture = texture_aware_terms(
            maps,
            [sample_image] * 7,
            proximity,
            ec.lambda_texture,
            ec.gaussian_kernel_size,
            ec.gaussian_sigma,
        )
        optimizer.zero_grad(set_to_none=True)
        parameters = [
            p
            for g in optimizer.param_groups
            for p in g["params"]
            if p.requires_grad
        ]
        loss = weights.loss(geometry, texture, parameters, step)
        # Risk probes also traverse rasterizer intermediates. Keep density
        # statistics from the final weighted backward only.
        for o in outputs + [sampled]:
            o["viewspace_points"].grad = None
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite ERGO loss at {step}")
        loss.backward()
        optimizer.step()
        density_step(renderer.gaussians, sampled, phase, step)
        if step % phase.remove_outliers_interval == 0:
            renderer.gaussians.remove_outliers(phase, step)
        constrain_log_scales_(
            renderer.gaussians._scaling,
            phase.max_gaussian_scale,
            phase.max_gaussian_anisotropy,
        )
    renderer.gaussians.prune(
        min_opacity=phase.final_prune_min_opacity, extent=1, max_screen_size=1
    )
    model = output / "point_cloud_model.ply"
    renderer.gaussians.save_ply(str(model))
    torch.save(
        dict(weights=weights.weights, risks=weights.risks), output / "ergo_weights.pt"
    )
    save_grid(renderer, config, output / "grid.png")
    return model
