from pathlib import Path
from types import SimpleNamespace
import math
import yaml

DEFAULT = Path(__file__).parent / "configs/ergo.yaml"


def _shape(value, schema, location="config"):
    if not isinstance(value, dict) or set(value) != set(schema):
        missing = set(schema) - set(value) if isinstance(value, dict) else set(schema)
        extra = set(value) - set(schema) if isinstance(value, dict) else set()
        raise ValueError(
            f"{location}: missing={sorted(missing)}, unknown={sorted(extra)}"
        )
    result = {}
    for key, expected in schema.items():
        item = value[key]
        if isinstance(expected, dict):
            result[key] = _shape(item, expected, f"{location}.{key}")
        else:
            if (
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(item)
            ):
                raise ValueError(f"{location}.{key} must be a finite number")
            if isinstance(expected, int) and not isinstance(item, int):
                raise ValueError(f"{location}.{key} must be an integer")
            result[key] = item
    return SimpleNamespace(**result)


def load_config(path=DEFAULT):
    schema = yaml.safe_load(DEFAULT.read_text())
    c = _shape(yaml.safe_load(Path(path).read_text()), schema)
    if c.ref_size != 320:
        raise ValueError("Zero123++ six-view checkpoint requires ref_size=320")
    if not 0 <= c.seed < 2**32 or not 0 <= c.invert_bg_prob <= 1:
        raise ValueError("Invalid seed/background probability")
    if (
        c.radius <= 0
        or not 0 < c.fovy < 180
        or not -80 <= c.min_ver < c.max_ver <= 80
        or c.num_pts < 2
    ):
        raise ValueError("Invalid camera or Gaussian initialization")
    for phase in (c.coarse, c.final):
        if (
            phase.iters < 1
            or phase.densification_interval < 1
            or phase.position_lr_max_steps < 1
        ):
            raise ValueError("Iterations and intervals must be positive")
        if (
            phase.position_lr_init <= 0
            or phase.position_lr_final <= 0
            or min(
                phase.feature_lr, phase.opacity_lr, phase.scaling_lr, phase.rotation_lr
            ) < 0
        ):
            raise ValueError(
                "Position learning rates must be positive; "
                "other learning rates must be nonnegative"
            )
        if (
            phase.density_start_iter < 0
            or phase.density_end_iter < phase.density_start_iter
            or phase.densify_grad_threshold < 0
            or phase.percent_dense < 0
        ):
            raise ValueError("Invalid densification settings")
    if (
        c.initial_steps < 1
        or c.coarse.opacity_reset_interval < 1
        or c.final.remove_outliers_interval < 1
    ):
        raise ValueError("Intervals must be positive")
    r, e = c.refinement, c.ergo
    if (
        not 0 <= c.final.sds_min_timestep_ratio < 1
        or not 0 <= c.final.final_prune_min_opacity < 1
        or c.final.max_gaussian_scale < 0
        or c.final.max_gaussian_anisotropy < 0
        or 0 < c.final.max_gaussian_anisotropy < 1
    ):
        raise ValueError("Invalid final-stage bounds")
    if (
        not 0 < r.strength <= 1
        or not 1 <= r.steps <= 1000
        or int(r.steps * r.strength) < 1
        or r.inversion_guidance_scale != 1
        or r.guidance_scale <= 1
    ):
        raise ValueError("Use conditional DDIM inversion and guided denoising")
    if (
        e.risk_update_interval < 1
        or e.hessian_damping <= 0
        or e.eps <= 0
        or e.discrepancy_sigma <= 0
        or e.visibility_dilation_kernel < 1
        or e.visibility_dilation_kernel % 2 != 1
        or e.gaussian_kernel_size < 1
        or e.gaussian_kernel_size % 2 != 1
        or e.gaussian_sigma <= 0
        or min(e.lambda_rgb, e.lambda_depth, e.lambda_mask, e.lambda_texture) < 0
        or min(c.coarse.lambda_pixel, c.coarse.lambda_mask, c.coarse.lambda_sds) < 0
    ):
        raise ValueError("Invalid ERGO numeric settings")
    if (
        min(e.min_geometry_weight_sum, e.min_texture_weight_sum) <= 0
        or e.min_geometry_weight_sum + e.min_texture_weight_sum > 1
    ):
        raise ValueError("Invalid AW group mass floors")
    return c
