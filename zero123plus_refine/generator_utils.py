from typing import Optional, Union

import torch


def normalize_generator(
    generator: Optional[torch.Generator],
    device: Union[str, torch.device],
) -> torch.Generator:
    """Reuse the supplied generator, or create one on the requested device."""
    if generator is not None:
        return generator
    return torch.Generator(device=device)
