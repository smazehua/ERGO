"""Check runtime imports and CUDA extension contracts without loading weights."""

import argparse
import importlib
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check_cuda():
    import torch
    from simple_knn._C import distCUDA2
    from pytorch3d.ops import knn_points
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
    from cameras import orbit_camera, render_camera

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; use Linux with an NVIDIA CUDA GPU")
    points = torch.tensor(
        [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
        device="cuda",
        dtype=torch.float32,
    )
    distances = distCUDA2(points)
    assert distances.shape == (4,) and torch.isfinite(distances).all(), "simple_knn"
    neighbors = knn_points(points[None], points[None], K=2)
    assert torch.isfinite(neighbors.dists).all(), "pytorch3d.ops.knn_points"

    camera = render_camera(orbit_camera(0, 0, 2), 32, 49.1)
    settings = GaussianRasterizationSettings(
        image_height=32,
        image_width=32,
        tanfovx=math.tan(camera.FoVx / 2),
        tanfovy=math.tan(camera.FoVy / 2),
        bg=torch.zeros(3, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=0,
        campos=camera.camera_center,
        prefiltered=False,
        debug=False,
    )
    xyz = torch.zeros((1, 3), device="cuda", requires_grad=True)
    screen = torch.zeros_like(xyz, requires_grad=True)
    opacity = torch.tensor([[0.5]], device="cuda", requires_grad=True)
    scales = torch.full((1, 3), 0.1, device="cuda", requires_grad=True)
    colors = torch.tensor([[0.4, 0.6, 0.8]], device="cuda", requires_grad=True)
    rotation = torch.tensor([[1.0, 0, 0, 0]], device="cuda")
    outputs = GaussianRasterizer(raster_settings=settings)(
        means3D=xyz,
        means2D=screen,
        colors_precomp=colors,
        shs=None,
        opacities=opacity,
        scales=scales,
        rotations=rotation,
        cov3D_precomp=None,
    )
    assert len(outputs) == 4, "rasterizer must return image, radii, depth, alpha"
    image, radii, depth, alpha = outputs
    assert image.shape == (3, 32, 32), "image shape"
    assert depth.shape == alpha.shape == (1, 32, 32), "depth/alpha shape"
    assert radii.shape == (1,) and radii[0] > 0, "Gaussian not rasterized"
    assert all(torch.isfinite(t).all() for t in outputs), "non-finite raster output"
    foreground = alpha > 1e-4
    assert foreground.any(), "empty rasterized foreground"
    assert alpha.min() >= 0 and alpha.max() <= 1, "alpha outside [0, 1]"
    metric_z = depth[foreground] / alpha[foreground]
    assert torch.allclose(
        metric_z, torch.full_like(metric_z, 2.0), atol=1e-3
    ), "depth must be the alpha-weighted camera-z moment, not normalized depth"
    (image.mean() + depth.mean() + alpha.mean()).backward()
    for name, tensor in [
        ("xyz", xyz),
        ("screen", screen),
        ("opacity", opacity),
        ("scales", scales),
        ("colors", colors),
    ]:
        assert (
            tensor.grad is not None and torch.isfinite(tensor.grad).all()
        ), f"invalid rasterizer backward gradient: {name}"
    assert opacity.grad.abs().sum() > 0, "zero opacity gradient"
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--imports-only", action="store_true", help="skip CUDA execution"
    )
    args = parser.parse_args()
    failed = []
    for name in (
        "torch",
        "torchvision",
        "numpy",
        "PIL",
        "cv2",
        "yaml",
        "scipy",
        "plyfile",
        "packaging",
        "diffusers",
        "transformers",
        "accelerate",
        "rembg",
        "onnxruntime",
        "iopath",
        "simple_knn._C",
        "pytorch3d.ops",
        "diff_gaussian_rasterization",
        "views",
        "train_coarse",
        "train_ergo",
    ):
        try:
            importlib.import_module(name)
        except Exception as exc:
            failed.append(f"{name}: {type(exc).__name__}: {exc}")
    if failed:
        raise RuntimeError(
            "Missing or incompatible dependencies: "
            + "; ".join(failed)
            + ". Install or rebuild them following README sections 2–3."
        )
    from config import load_config
    import onnxruntime

    config = load_config()
    assert (config.coarse.iters, config.final.iters) == (800, 1500), "default steps"
    assert (
        "CPUExecutionProvider" in onnxruntime.get_available_providers()
    ), "ONNX CPU provider"
    if not args.imports_only:
        check_cuda()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
