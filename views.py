"""Zero123++ initial generation and DDIM refinement."""

import random
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import rembg
from zero123plus_refine.pipeline import Zero123PlusPipeline


def seed_generation(seed):
    # Keep generation cuDNN settings independent of training settings.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def background_session(model_root):
    import os

    os.environ["U2NET_HOME"] = str(Path(model_root) / "rembg-u2net")
    return rembg.new_session("u2net", providers=["CPUExecutionProvider"])


def prepare_anchor(image, session):
    image = image.convert("RGBA")
    if image.getextrema()[3] == (255, 255):
        image = rembg.remove(image, session=session)
    return image


def write_views(grid, anchor, directory, session):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    if grid.size != (640, 960):
        raise ValueError(f"Invalid multiview grid size: {grid.size}")
    grid.save(directory / "grid.png")
    if isinstance(anchor, Path):
        shutil.copyfile(anchor, directory / "0.png")
    else:
        anchor.save(directory / "0.png")
    for k in range(6):
        row, col = divmod(k, 2)
        view = grid.crop((col * 320, row * 320, (col + 1) * 320, (row + 1) * 320))
        view.save(directory / f"raw_{k+1}.png")
        rgba = rembg.remove(view.convert("RGBA"), session=session)
        alpha = np.asarray(rgba)[:, :, 3]
        if not np.any(alpha > 127):
            raise ValueError(f"Empty foreground in generated view {k+1}")
        rgba.save(directory / f"{k+1}.png")


def initial(config, image, model_root, output):
    seed_generation(config.seed)
    session = background_session(model_root)
    anchor = prepare_anchor(Image.open(image), session)
    pipeline = Zero123PlusPipeline.from_pretrained(
        str(Path(model_root) / "zero123plus-v1.1"),
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to("cuda")
    seed_generation(config.seed)
    grid = pipeline.generate(
        anchor,
        config.initial_steps,
        torch.Generator(device="cuda").manual_seed(config.seed),
    )
    write_views(grid, anchor, output, session)


def refine(config, anchor, coarse_grid, model_root, output):
    seed_generation(config.seed)
    pipeline = Zero123PlusPipeline.from_pretrained(
        str(Path(model_root) / "zero123plus-v1.1"),
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to("cuda")
    seed_generation(config.seed)
    grid = pipeline.refine(
        Image.open(anchor).convert("RGBA"),
        Image.open(coarse_grid),
        config.refinement,
        torch.Generator(device="cuda").manual_seed(config.seed),
    )
    write_views(grid, Path(anchor), output, background_session(model_root))
