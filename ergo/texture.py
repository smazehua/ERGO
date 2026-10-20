from typing import Sequence
import torch
import torch.nn.functional as F


def sds_surrogate_map(latents: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
    """Build a pixel-wise SDS surrogate whose gradient equals the SDS update.

    The detached target keeps diffusion inference outside autograd while retaining
    a differentiable loss map for ERGO's spatial texture weighting.
    """
    if latents.shape != gradient.shape:
        raise ValueError("latents and gradient must have the same shape")
    target = (latents - gradient).detach()
    return 0.5 * (latents.float() - target.float()).square().mean(dim=1, keepdim=True)


def _gaussian_kernel(
    kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    coordinates = (
        torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    )
    kernel = torch.exp(-coordinates.square() / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return torch.outer(kernel, kernel)


def texture_complexity_map(
    image: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0
) -> torch.Tensor:
    """Compute W^c = 1 + Gaussian(Sobel(I')) from Eq. (9)."""
    if image.ndim != 4:
        raise ValueError("image must have shape [B, C, H, W]")
    gray = image.mean(dim=1, keepdim=True)
    sobel_x = (
        image.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8.0
    )
    sobel_y = sobel_x.transpose(-1, -2)
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1e-12)
    kernel = _gaussian_kernel(kernel_size, sigma, image.device, image.dtype).view(
        1, 1, kernel_size, kernel_size
    )
    smoothed = F.conv2d(magnitude, kernel, padding=kernel_size // 2)
    return 1.0 + smoothed


def camera_proximity_weights(
    sample_center: torch.Tensor, reference_centers: torch.Tensor, eps: float = 1e-08
) -> torch.Tensor:
    """Return 1 - normalized camera distance from Eq. (8)."""
    if sample_center.shape[-1] != 3 or reference_centers.shape[-1] != 3:
        raise ValueError("camera centers must end in three coordinates")
    distances = torch.linalg.vector_norm(
        reference_centers - sample_center.unsqueeze(-2), dim=-1
    )
    return 1.0 - distances / distances.sum(dim=-1, keepdim=True).clamp_min(eps)


def texture_aware_terms(
    sds_loss_maps: Sequence[torch.Tensor],
    rendered_images: Sequence[torch.Tensor],
    proximity_weights: torch.Tensor,
    lambda_texture: float = 1.0,
    kernel_size: int = 5,
    sigma: float = 1.0,
):
    """Return the individual L_t^k components needed by Eq. (10)."""
    if len(sds_loss_maps) != len(rendered_images) or proximity_weights.numel() != len(
        sds_loss_maps
    ):
        raise ValueError("SDS maps, rendered images, and proximity weights must align")
    result = []
    for index, (loss_map, image) in enumerate(zip(sds_loss_maps, rendered_images)):
        complexity = texture_complexity_map(image, kernel_size, sigma)
        if loss_map.ndim > 0 and complexity.shape[-2:] != loss_map.shape[-2:]:
            complexity = F.interpolate(
                complexity,
                size=loss_map.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        value = (
            loss_map * complexity.mean()
            if loss_map.ndim == 0
            else (loss_map * complexity).sum()
        )
        result.append(lambda_texture * proximity_weights.reshape(-1)[index] * value)
    return result
