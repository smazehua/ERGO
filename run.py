"""Run the four GPU stages in isolated subprocesses."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from config import DEFAULT, load_config

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_manifest():
    return {
        str(p.relative_to(ROOT)): digest(p)
        for p in sorted(ROOT.rglob("*.py"))
        if "__pycache__" not in p.parts
    }


def execute_stage(stage, config_path, input_path, output, model_root):
    config = load_config(config_path)
    if stage == "initial":
        from views import initial

        initial(config, input_path, model_root, output / "initial")
    elif stage == "coarse":
        from train_coarse import train

        train(config, output / "initial", model_root, output / "coarse")
    elif stage == "refine":
        from views import refine

        refine(
            config,
            output / "initial/0.png",
            output / "coarse/grid.png",
            model_root,
            output / "refined",
        )
    elif stage == "ergo":
        from train_ergo import train

        train(
            config,
            output / "refined",
            output / "coarse/point_cloud_model.ply",
            model_root,
            output / "ergo",
        )
    else:
        raise ValueError(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="centered RGB/RGBA object image")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=os.environ.get("ERGO_MODEL_ROOT"),
        required="ERGO_MODEL_ROOT" not in os.environ,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument(
        "--stage",
        choices=("initial", "coarse", "refine", "ergo"),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    config_path, input_path, output, model_root = [
        p.resolve() for p in (args.config, args.input, args.output, args.model_root)
    ]
    load_config(config_path)
    if not input_path.is_file():
        parser.error(f"Input not found: {input_path}")
    for relative in (
        "zero123plus-v1.1/model_index.json",
        "zero123-xl-diffusers/model_index.json",
        "rembg-u2net/u2net.onnx",
    ):
        if not (model_root / relative).is_file():
            parser.error(f"Missing local model: {model_root/relative}")
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        DIFFUSERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    if args.stage:
        execute_stage(args.stage, config_path, input_path, output, model_root)
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signature = dict(
            input_sha256=digest(input_path),
            config_sha256=digest(config_path),
            sources=source_manifest(),
            model_root=str(model_root),
        )
        contract = output / "run_manifest.json"
        if contract.exists():
            if json.loads(contract.read_text()) != signature:
                raise RuntimeError(
                    "Output belongs to a different input/config/source. Choose a new output directory."
                )
        else:
            contract.write_text(json.dumps(signature, indent=2) + "\n")
        for stage, folder in [
            ("initial", "initial"),
            ("coarse", "coarse"),
            ("refine", "refined"),
            ("ergo", "ergo"),
        ]:
            receipt = output / f"{stage}_verified.json"
            if receipt.exists():
                records = json.loads(receipt.read_text())
                if not records or any(
                    not (output / p).is_file() or digest(output / p) != sha
                    for p, sha in records.items()
                ):
                    raise RuntimeError(
                        f"{stage}: cached outputs changed or are missing"
                    )
                continue
            if (output / folder).exists():
                raise RuntimeError(
                    f"{stage}: unfinished output retained at {output/folder}; use a new output directory"
                )
            command = [
                sys.executable,
                str(ROOT / "run.py"),
                str(input_path),
                "--output",
                str(output),
                "--model-root",
                str(model_root),
                "--config",
                str(config_path),
                "--stage",
                stage,
            ]
            subprocess.run(command, check=True, cwd=ROOT)
            files = {
                str(p.relative_to(output)): digest(p)
                for p in sorted((output / folder).rglob("*"))
                if p.is_file()
            }
            receipt.write_text(json.dumps(files, indent=2) + "\n")


if __name__ == "__main__":
    main()
