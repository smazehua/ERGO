from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Camera:
    """Pinhole camera matrices using positive camera-space z depth."""

    intrinsics: torch.Tensor
    world_to_camera: torch.Tensor


def normalize_depth(
    depth: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-08
) -> torch.Tensor:
    """Normalize each depth map to [0, 1] over foreground pixels."""
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError("depth must have shape [B, 1, H, W]")
    valid = depth > 0 if mask is None else (mask > 0.5) & (depth > 0)
    flat = depth.flatten(2)
    vflat = valid.flatten(2)
    inf = torch.full_like(flat, torch.inf)
    minimum = torch.where(vflat, flat, inf).amin(dim=2, keepdim=True)
    maximum = torch.where(vflat, flat, -inf).amax(dim=2, keepdim=True)
    has_value = vflat.any(dim=2, keepdim=True)
    minimum = torch.where(has_value, minimum, torch.zeros_like(minimum))
    maximum = torch.where(has_value, maximum, torch.ones_like(maximum))
    result = (flat - minimum) / (maximum - minimum).clamp_min(eps)
    return torch.where(vflat, result, torch.zeros_like(result)).view_as(depth)


@torch.no_grad()
def forward_warp(
    source: torch.Tensor,
    source_depth: torch.Tensor,
    source_camera: Camera,
    target_camera: Camera,
    target_size: Optional[Tuple[int, int]] = None,
    depth_tolerance: float = 0.0001,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lift source pixels and splat them into a target view with nearest-depth filtering.

    Returns warped values, target-camera z depth, and a valid-pixel mask. This follows
    the depth-based correspondence construction used for Eq. (6) in ERGO.
    """
    if source.ndim != 4 or source_depth.ndim != 4:
        raise ValueError("source and source_depth must be BCHW tensors")
    if source.shape[0] != source_depth.shape[0] or source_depth.shape[1] != 1:
        raise ValueError("source/source_depth batch mismatch or invalid depth channels")
    batch, channels, height, width = source.shape
    target_height, target_width = target_size or (height, width)
    device, dtype = (source.device, source.dtype)

    def batched(matrix: torch.Tensor) -> torch.Tensor:
        matrix = matrix.to(device=device, dtype=dtype)
        return matrix.unsqueeze(0).expand(batch, -1, -1) if matrix.ndim == 2 else matrix

    k_source = batched(source_camera.intrinsics)
    k_target = batched(target_camera.intrinsics)
    w2c_source = batched(source_camera.world_to_camera)
    w2c_target = batched(target_camera.world_to_camera)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    pixels = (
        torch.stack((xx, yy, torch.ones_like(xx)), dim=0)
        .reshape(1, 3, -1)
        .expand(batch, -1, -1)
    )
    z_source = source_depth.reshape(batch, 1, -1)
    xyz_source = torch.linalg.solve(k_source, pixels) * z_source
    homogeneous = torch.cat(
        (xyz_source, torch.ones(batch, 1, height * width, device=device, dtype=dtype)),
        dim=1,
    )
    world = torch.linalg.solve(w2c_source, homogeneous)
    xyz_target = (w2c_target @ world)[:, :3]
    projected = k_target @ xyz_target
    z_target = xyz_target[:, 2]
    u = torch.round(projected[:, 0] / z_target.clamp_min(1e-08)).long()
    v = torch.round(projected[:, 1] / z_target.clamp_min(1e-08)).long()
    valid = (
        torch.isfinite(z_source[:, 0])
        & torch.isfinite(z_target)
        & (z_source[:, 0] > 0)
        & (z_target > 0)
        & (u >= 0)
        & (u < target_width)
        & (v >= 0)
        & (v < target_height)
    )
    pixel_count = target_height * target_width
    output = source.new_zeros((batch, channels, pixel_count))
    warped_depth = source.new_zeros((batch, 1, pixel_count))
    output_valid = torch.zeros((batch, 1, pixel_count), device=device, dtype=torch.bool)
    source_flat = source.flatten(2)
    for b in range(batch):
        if not valid[b].any():
            continue
        indices = v[b, valid[b]] * target_width + u[b, valid[b]]
        depths = z_target[b, valid[b]]
        nearest = torch.full((pixel_count,), torch.inf, device=device, dtype=dtype)
        nearest.scatter_reduce_(0, indices, depths, reduce="amin", include_self=True)
        keep = depths <= nearest[indices] + depth_tolerance
        indices = indices[keep]
        depths = depths[keep]
        values = source_flat[b, :, valid[b]][:, keep]
        counts = source.new_zeros(pixel_count)
        counts.scatter_add_(0, indices, torch.ones_like(depths))
        output[b].scatter_add_(1, indices.unsqueeze(0).expand(channels, -1), values)
        warped_depth[b, 0].scatter_add_(0, indices, depths)
        denom = counts.clamp_min(1)
        output[b] /= denom.unsqueeze(0)
        warped_depth[b, 0] /= denom
        output_valid[b, 0] = counts > 0
    shape = (batch, channels, target_height, target_width)
    return (
        output.view(shape),
        warped_depth.view(batch, 1, target_height, target_width),
        output_valid.view(batch, 1, target_height, target_width),
    )


def metric_depth(moment, alpha):
    """The native rasterizer returns sum(alpha_i*z_i), not camera z itself."""
    valid = (
        torch.isfinite(moment) & torch.isfinite(alpha) & (alpha > 1e-6) & (moment > 0)
    )
    return torch.where(valid, moment / alpha.clamp_min(1e-6), torch.zeros_like(moment))


@torch.no_grad()
def local_supervision(
    images, depths, masks, cameras, visibility_dilation_kernel, discrepancy_sigma
):
    """Adjacent reprojection with foreground-clipped V dilation and Gaussian W."""
    visibility = [torch.ones_like(masks[0])]
    consistency = [torch.ones_like(masks[0])]
    for k in range(1, len(images)):
        source_depth = depths[k - 1] * masks[k - 1]
        rgb, projected_depth, valid = forward_warp(
            images[k - 1], source_depth, cameras[k - 1], cameras[k]
        )
        valid = (
            valid
            & torch.isfinite(projected_depth)
            & (projected_depth > 0)
            & (depths[k] > 0)
            & (masks[k] >= 0.5)
        )
        before = (valid & (projected_depth <= depths[k])).to(depths[k])
        visibility.append(
            F.max_pool2d(
                before,
                visibility_dilation_kernel,
                stride=1,
                padding=visibility_dilation_kernel // 2,
            )
            * masks[k]
        )
        delta = (rgb - images[k]).abs().mean(dim=1, keepdim=True).clamp(0, 1)
        consistency.append(
            torch.exp(-delta.square() / (2.0 * discrepancy_sigma**2)) * valid.to(delta)
        )
    return visibility, consistency


def geometry_loss(
    image,
    depth,
    alpha,
    target_image,
    target_depth,
    target_mask,
    visibility,
    consistency,
    config,
):
    """Foreground RGB/depth MSE and full-frame silhouette MSE, Eqs. (3)-(7)."""
    denominator = target_mask.sum().clamp_min(config.eps)
    coefficient = visibility * consistency + (1 - visibility)
    rgb = (
        coefficient
        * (image - target_image).square().mean(dim=1, keepdim=True)
        * target_mask
    ).sum() / denominator
    pred_depth = normalize_depth(depth, target_mask, config.eps)
    ref_depth = normalize_depth(target_depth, target_mask, config.eps)
    dep = ((pred_depth - ref_depth).square() * target_mask).sum() / denominator
    silhouette = (alpha - target_mask).square().mean()
    return (
        config.lambda_rgb * rgb
        + config.lambda_depth * dep
        + config.lambda_mask * silhouette
    )
