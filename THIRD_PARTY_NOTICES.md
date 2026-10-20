# Third-party notices

ERGO includes adapted third-party code. These notices identify its sources and
preserve the applicable attribution; they do not expand the rights granted by
the corresponding licenses. License texts are provided in `LICENSES/`.

## Gaussian Splatting

Copyright (C) 2023, Inria. GRAPHDECO research group.
All rights reserved. Inria and MPII are identified as licensors in the license.

- Sources: [Gaussian model](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py) and [spherical harmonics utilities](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/sh_utils.py).
- Included adaptations: Gaussian parameter storage, optimizer maintenance, PLY I/O, densification/pruning, rotation/projection utilities, and RGB-to-SH conversion in `gs_renderer.py` and `sh_utils.py`.
- ERGO modifications include the reduced renderer interface, depth/alpha outputs, outlier removal, configuration integration, and removal of runtime logging and unused export paths.
- License: [Gaussian-Splatting License](LICENSES/Gaussian-Splatting.txt). Non-commercial research/evaluation restrictions and redistribution requirements apply.

The original copyright header, including its upstream contact information, is
retained in the adapted source files.

## DreamGaussian

Copyright (c) 2023 dreamgaussian.

- Source: [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian), particularly [`gs_renderer.py`](https://github.com/dreamgaussian/dreamgaussian/blob/main/gs_renderer.py) and [`sh_utils.py`](https://github.com/dreamgaussian/dreamgaussian/blob/main/sh_utils.py).
- Included adaptations: renderer, camera integration, and Gaussian model utilities in `gs_renderer.py` and `sh_utils.py`.
- License: [MIT](LICENSES/DreamGaussian-MIT.txt) for DreamGaussian contributions. This does not replace the Gaussian Splatting terms on inherited implementations.

## kiuikit camera conventions

Attribution: the kiuikit authors.

- Source: [kiuikit camera utilities](https://github.com/ashawkey/kiuikit/blob/main/kiui/cam.py).
- Included adaptation: orbit/look-at camera conventions in `cameras.py`, integrated with ERGO's fixed views, projection matrices, and rasterizer interface.
- License: [Apache-2.0](LICENSES/Kiuikit-Apache-2.0.txt).

## Hugging Face Diffusers / Zero123 component container

Copyright 2023 The HuggingFace Team.

- Sources: [Diffusers community Zero1-to-3 pipeline](https://github.com/huggingface/diffusers/blob/main/examples/community/zero1to3.py) and [Diffusers](https://github.com/huggingface/diffusers).
- Included adaptation: CLIP camera projection and pipeline component registration in `zero123.py`.
- ERGO retains the component container and compatibility handling, without the upstream generation interface.
- License: [Apache-2.0](LICENSES/Apache-2.0.txt).

## Zero123++

Copyright 2023 Hillbot.

- Source: [Zero123++ Diffusers pipeline](https://github.com/SUDO-AI-3D/zero123plus/blob/main/diffusers-support/pipeline.py).
- Included adaptation: reference attention, conditioned UNet, image/latent scaling, and pipeline components in `zero123plus_refine/pipeline.py`.
- ERGO modifications include paired partial DDIM inversion/refinement, fixed conditioning noise, generator handling, and removal of progress and metadata output.
- License: [Apache-2.0](LICENSES/Zero123Plus-Apache-2.0.txt), including the upstream Hillbot notice.

## Separately installed dependencies and model weights

The following are downloaded or installed separately and are not vendored here:

- [PyTorch3D](https://github.com/facebookresearch/pytorch3d), [simple-knn](https://gitlab.inria.fr/bkerbl/simple-knn), and the [four-output Gaussian rasterizer](https://github.com/ashawkey/diff-gaussian-rasterization/tree/8829d14f814fccdaf840b7b0f3021a616583c0a1). The rasterizer uses the Gaussian-Splatting License.
- [Zero123++ v1.1 weights](https://huggingface.co/sudo-ai/zero123plus-v1.1), [Zero123-XL weights](https://huggingface.co/ashawkey/zero123-xl-diffusers), and [rembg / U2-Net](https://github.com/danielgatis/rembg).
- Python packages declared in `requirements.txt` and their transitive dependencies.

Consult each upstream repository or model card for its own terms. A code license
does not automatically grant rights to model weights or training data.
