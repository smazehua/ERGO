#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# Copyright (c) 2023 dreamgaussian
# Adapted and modified for ERGO; see LICENSE.md and THIRD_PARTY_NOTICES.md.
# SPDX-License-Identifier: LicenseRef-Gaussian-Splatting AND MIT

C0 = 0.28209479177387814


def RGB2SH(rgb):
    return (rgb - 0.5) / C0
