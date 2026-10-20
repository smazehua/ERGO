# SPDX-License-Identifier: Apache-2.0
# Camera conventions adapted from kiuikit; modified for ERGO.
# See THIRD_PARTY_NOTICES.md and LICENSES/Kiuikit-Apache-2.0.txt.

"""One camera convention for the CUDA rasterizer, V, W and TAO."""

import math
import numpy as np
import torch
from ergo.geometry import Camera

ELEVATIONS = (0, -30, 20, -30, 20, -30, 20)
AZIMUTHS = (0, 30, 90, 150, -150, -90, -30)


def orbit_camera(elevation, azimuth, radius):
    e, a = np.deg2rad(elevation), np.deg2rad(azimuth)
    target = np.zeros(3, dtype=np.float32)
    center = (
        np.array(
            [
                radius * np.cos(e) * np.sin(a),
                -radius * np.sin(e),
                radius * np.cos(e) * np.cos(a),
            ]
        )
        + target
    )

    def normalize(value):
        return value / np.sqrt(
            np.maximum(np.sum(value * value, -1, keepdims=True), 1e-20)
        )

    forward = normalize(center - target)
    up = np.array([0, 1, 0], dtype=np.float32)
    right = normalize(np.cross(up, forward))
    up = normalize(np.cross(forward, right))
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.stack([right, up, forward], axis=1)
    pose[:3, 3] = center
    return pose


def projection_camera(pose, size, fovy, device):
    w2c = np.linalg.inv(pose)
    w2c[1:3, :3] *= -1
    w2c[:3, 3] *= -1
    focal = size / (2 * math.tan(math.radians(fovy) / 2))
    principal = (size - 1) / 2  # CUDA ndc2Pix convention
    return Camera(
        torch.tensor(
            [[focal, 0, principal], [0, focal, principal], [0, 0, 1]], device=device
        ),
        torch.as_tensor(w2c, device=device),
    )


def render_camera(pose, size, fovy):
    from gs_renderer import MiniCam

    angle = math.radians(fovy)
    return MiniCam(pose, size, size, angle, angle, 0.01, 100)


def fixed_cameras(config, device):
    poses = [orbit_camera(e, a, config.radius) for e, a in zip(ELEVATIONS, AZIMUTHS)]
    renders = [render_camera(p, config.ref_size, config.fovy) for p in poses]
    projections = [
        projection_camera(p, config.ref_size, config.fovy, device) for p in poses
    ]
    centers = torch.as_tensor(np.stack([p[:3, 3] for p in poses]), device=device)
    return poses, renders, projections, centers
