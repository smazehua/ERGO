import random
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from cameras import fixed_cameras
from ergo.geometry import metric_depth


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_coarse_images(directory, size):
    images, masks = [], []
    for k in range(7):
        image = (
            np.asarray(
                Image.open(Path(directory) / f"{k}.png").convert("RGBA"),
                dtype=np.float32,
            )
            / 255
        )
        alpha = image[:, :, 3:4]
        rgb = image[:, :, :3] * alpha + (1 - alpha)
        images.append(
            F.interpolate(
                torch.from_numpy(rgb).permute(2, 0, 1)[None].cuda(),
                (size, size),
                mode="bilinear",
                align_corners=False,
            )
        )
        masks.append(
            F.interpolate(
                torch.from_numpy(alpha).permute(2, 0, 1)[None].cuda(),
                (size, size),
                mode="bilinear",
                align_corners=False,
            )
        )
    return images, masks


def load_images(directory, size):
    """Final-stage loader preserves the native uint8 INTER_AREA resize."""
    images, masks = [], []
    for k in range(7):
        path = Path(directory) / f"{k}.png"
        array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if array is None or array.ndim != 3 or array.shape[2] != 4:
            raise ValueError(f"Expected an RGBA reference: {path}")
        array = cv2.resize(array, (size, size), interpolation=cv2.INTER_AREA)
        mask = array[..., 3:4].astype(np.float32) / 255
        rgb = array[..., :3].astype(np.float32) / 255
        rgb = rgb[..., ::-1].copy() * mask + (1 - mask)
        images.append(torch.from_numpy(rgb).permute(2, 0, 1)[None].cuda())
        masks.append(torch.from_numpy(mask).permute(2, 0, 1)[None].cuda())
    return images, masks


def setup_gaussians(renderer, phase):
    # Native PLY loading leaves this at zero; final xyz learning must stay active.
    renderer.gaussians.spatial_lr_scale = 1.0
    options = phase
    renderer.gaussians.training_setup(options)
    renderer.gaussians.active_sh_degree = renderer.gaussians.max_sh_degree
    return renderer.gaussians.optimizer


def density_step(gaussians, output, phase, step):
    if phase.density_start_iter <= step <= phase.density_end_iter:
        visible = output["visibility_filter"]
        points = output["viewspace_points"]
        if points.grad is not None:
            gaussians.max_radii2D[visible] = torch.maximum(
                gaussians.max_radii2D[visible], output["radii"][visible]
            )
            gaussians.add_densification_stats(points, visible)
        if step % phase.densification_interval == 0:
            gaussians.densify_and_prune(
                phase.densify_grad_threshold,
                min_opacity=0.01,
                extent=4,
                max_screen_size=1,
            )


@torch.no_grad()
def save_grid(renderer, config, path):
    _, cameras, _, _ = fixed_cameras(config, "cuda")
    gray = torch.full((3,), 128 / 255, device="cuda")
    tiles = [
        (
            renderer.render(c, bg_color=gray)["image"].permute(1, 2, 0).cpu().numpy()
            * 255
        )
        .clip(0, 255)
        .astype(np.uint8)
        for c in cameras[1:]
    ]
    grid = np.concatenate(
        [np.concatenate(tiles[k : k + 2], axis=1) for k in (0, 2, 4)], axis=0
    )
    Image.fromarray(grid).save(path)


def view_tensors(output):
    return (
        output["image"][None],
        metric_depth(output["depth"][None], output["alpha"][None]),
        output["alpha"][None],
    )
