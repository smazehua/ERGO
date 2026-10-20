# ERGO

Single-image 3D reconstruction with Zero123++ initialization, coarse fitting, DDIM refinement, and ERGO optimization. Entry point: `run.py`; configuration: `configs/ergo.yaml`.

**Licensing:** This release has mixed licenses, including non-commercial research restrictions for Gaussian Splatting components. See [LICENSE.md](LICENSE.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

## 1. Requirements

- Linux with an NVIDIA CUDA GPU; CPU and macOS/MPS training are not supported. Each run uses one GPU.
- Tested with Python 3.10, PyTorch 2.7.1/CUDA 12.8, torchvision 0.22.1, and PyTorch3D 0.7.9 on one H100 80 GB. Minimum VRAM is not established.
- Building extensions requires **CUDA Toolkit 12.8 (`nvcc`)**, compatible GCC/G++, Git, Ninja, and an NVIDIA driver supporting the selected CUDA version. PyTorch wheels do not include `nvcc`.

## 2. Python environment

Run from the project root; the system-package command below assumes Ubuntu.

```bash
sudo apt-get update && sudo apt-get install -y build-essential git
conda create -n ergo python=3.10 -y
conda activate ergo
python -m pip install --upgrade pip setuptools wheel ninja
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Background removal uses `CPUExecutionProvider`. Use the CPU ONNX Runtime package in `requirements.txt`; do not install CPU and GPU variants together.

## 3. Native dependencies

```bash
mkdir -p ../ergo_deps

# Build PyTorch3D after installing PyTorch above.
git clone --branch v0.7.9 --depth 1 \
  https://github.com/facebookresearch/pytorch3d.git ../ergo_deps/pytorch3d
FORCE_CUDA=1 python -m pip install --no-build-isolation ../ergo_deps/pytorch3d

git clone https://gitlab.inria.fr/bkerbl/simple-knn.git ../ergo_deps/simple-knn
git -C ../ergo_deps/simple-knn checkout 86710c2d4b46680c02301765dd79e465819c8f19
python -m pip install --no-build-isolation ../ergo_deps/simple-knn

# Four-output rasterizer with differentiable depth and alpha.
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization.git \
  ../ergo_deps/diff-gaussian-rasterization
git -C ../ergo_deps/diff-gaussian-rasterization checkout 8829d14f814fccdaf840b7b0f3021a616583c0a1
git -C ../ergo_deps/diff-gaussian-rasterization submodule update --init --recursive
python -m pip install --no-build-isolation ../ergo_deps/diff-gaussian-rasterization
python -m pip check
```

Required rasterizer outputs: `(image, radii, depth, alpha)`, where `depth` is an unnormalized camera-z moment. Recover camera z as `depth / alpha`.

Installation has not been verified in a clean environment. Rebuild extensions when PyTorch, CUDA, or GPU architecture changes, then run:

```bash
python scripts/check_environment.py
```

This verifies imports and native CUDA forward/backward operations. Successful checks exit silently; failures raise errors.

Source-only configuration checks require no GPU: `python -m unittest discover -s tests`.

<details>
<summary>Troubleshooting</summary>

```bash
nvidia-smi
nvcc --version
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"
```

If `nvcc` is missing, set `CUDA_HOME` and PATH to the local CUDA Toolkit installation. Use a compiler supported by that Toolkit. When building without a visible GPU, set `TORCH_CUDA_ARCH_LIST` for the target GPU.

`python scripts/check_environment.py --imports-only` skips CUDA execution and cannot verify native extension compatibility.

</details>

Installation sources: [PyTorch](https://pytorch.org/get-started/previous-versions/), [PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/v0.7.9/INSTALL.md), and [four-output rasterizer](https://github.com/ashawkey/diff-gaussian-rasterization/tree/8829d14f814fccdaf840b7b0f3021a616583c0a1). Third-party dependencies remain subject to their original licenses.

## 4. Models

Weights are not included. Use complete, version-consistent snapshots, including submodel weights, configurations, tokenizers, and any custom components:

```text
models/
  zero123plus-v1.1/
    model_index.json
    ...
  zero123-xl-diffusers/
    model_index.json
    clip_camera_projection/
    ...
  rembg-u2net/
    u2net.onnx
```

Download models with network access during preparation; the training entry point loads them offline:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("sudo-ai/zero123plus-v1.1", local_dir="models/zero123plus-v1.1")
snapshot_download("ashawkey/zero123-xl-diffusers", local_dir="models/zero123-xl-diffusers")
PY

# Download U2-Net before training.
U2NET_HOME="$(pwd)/models/rembg-u2net" python - <<'PY'
import rembg
rembg.new_session("u2net", providers=["CPUExecutionProvider"])
PY
```

Model sources: [Zero123++ v1.1](https://huggingface.co/sudo-ai/zero123plus-v1.1), [Zero123-XL](https://huggingface.co/ashawkey/zero123-xl-diffusers), and [rembg](https://github.com/danielgatis/rembg). Record snapshot revisions and weight checksums for reproducibility.

`Zero123-XL` uses `trust_remote_code=True` for custom projection components. Load only verified local snapshots and keep model directories read-only during a run. See [security notes](SECURITY.md) for dependency and model-loading risks.

## 5. Run

Use a centered RGB/RGBA image containing the complete object. Existing alpha is preserved; opaque images undergo U2-Net background removal.

```bash
CUDA_VISIBLE_DEVICES=0 python run.py ./images/object.png \
  --output ./outputs/object \
  --model-root ./models
```

Use `--config PATH` to select a configuration; run `python run.py --help` for options. Defaults in `configs/ergo.yaml`: seed 42, 320×320 images, 800 coarse steps, and 1,500 ERGO steps.

<details>
<summary>Optional usage: model-root environment variable</summary>

```bash
export ERGO_MODEL_ROOT="$(pwd)/models"
CUDA_VISIBLE_DEVICES=0 python run.py ./images/object.png --output ./outputs/object
```

</details>

## 6. Outputs and repeated runs

Final Gaussian model: `ergo/point_cloud_model.ply`. Stage `grid.png` files provide six-view previews; mesh export is not implemented.

Completed stages are reused only after output-hash verification. Failed stages cannot resume from an intermediate step: inspect the exception, fix the issue, and use a new output directory. Also use a new output directory when inputs, configuration, source code, model paths, or weights change.

`run_manifest.json` records the absolute model directory path. Check model paths before sharing the manifest.

<details>
<summary>Output files</summary>

```text
outputs/object/
  initial/              # Initial views
  coarse/               # Coarse model and preview
  refined/              # Refined views
  ergo/
  run_manifest.json
  *_verified.json
```

View directories contain `0.png`–`6.png` and `grid.png`; generated auxiliary tiles are saved as `raw_*.png`. Coarse and ERGO outputs include `point_cloud_model.ply` and `grid.png`. The entry point does not export orbit videos or ground-truth evaluation metrics.

</details>

<details>
<summary>Implementation details</summary>

### Views and execution

The six auxiliary `(elevation, azimuth)` pairs in `cameras.py` are `(-30,30), (20,90), (-30,150), (20,-150), (-30,-90), (20,-30)`; the input view is `(0,0)`. By default, initial generation uses 75 Euler ancestral steps. View 0 retains the same file contents across refinement. The four stages run sequentially in separate subprocesses to release GPU memory between stages.

### Visibility and depth

V and W share adjacent-view projection from `k−1 → k`, with camera conventions, pixel centers, and scale matching CUDA rendering. The rasterizer returns the depth moment `Σ(T_i α_i z_i)`; divide it by alpha to recover camera z. Two-output rasterizers and forks returning normalized depth are incompatible.

Compute binary `V_k = M_k · 1[D_(k−1→k) ≤ D_k]`, set invalid projections to 0, apply a 5×5 square dilation, and clip to the coarse foreground mask. `V_0=W_0=1`. Depth normalization is used only for the depth loss.

### DDIM refinement

Partial inversion and paired deterministic denoising use source-time UNet evaluation, Zero123++ VAE scaling, and fixed conditioning noise. A 100-step schedule with strength 0.60 gives 60 inversion and 60 denoising steps. Inversion CFG is 1, denoising CFG is 4, and eta is 0.

### Mask loss

ERGO uses full-image MSE against binary pseudo-labels from coarse alpha ≥ 0.5. The coarse stage uses MSE mask supervision from the initial seven views.

### GAO

The RGB weight is `V·W+(1−V)`, with `W = exp(−δ²/(2σ²)) × projection_valid`, where δ is mean absolute RGB discrepancy and σ=0.1.

### TAO

Texture weighting uses Sobel/Gaussian complexity and camera distance, with Gaussian filter σ=1.0.

### AW

A damped diagonal proxy jointly normalizes 14 objectives and updates every 10 steps. GAO and TAO each retain a total weight of at least 0.1. Each update uses the current parameters, including changes from densification and pruning.

### Gaussian optimization

After loading the coarse PLY, spatial LR scale is restored to 1. With the default configuration, opacity resets at coarse step 500; ERGO does not reset opacity.

</details>
