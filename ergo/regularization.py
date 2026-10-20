import math
import torch


@torch.no_grad()
def constrain_log_scales_(
    log_scales: torch.Tensor, max_scale: float = 0.1, max_anisotropy: float = 20.0
) -> None:
    """Clamp log scales in place to preserve Adam parameter identities and state."""
    if log_scales.ndim != 2 or log_scales.shape[1] != 3:
        raise ValueError("log_scales must have shape [N, 3]")
    if max_scale > 0:
        log_scales.clamp_(max=math.log(max_scale))
    if max_anisotropy > 0:
        minimum = log_scales.amin(dim=1, keepdim=True)
        log_scales.copy_(torch.minimum(log_scales, minimum + math.log(max_anisotropy)))
