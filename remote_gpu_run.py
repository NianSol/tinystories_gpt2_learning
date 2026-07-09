import argparse
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from urllib.request import urlretrieve


DATA_DIR = Path("data/tinystories")
TRAIN_FILE = DATA_DIR / "TinyStories-train.txt"
VALID_FILE = DATA_DIR / "TinyStories-valid.txt"
TRAIN_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
VALID_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
ARTIFACT_DIR = Path("artifacts")


SMOKE_ENV = {
    "RUN_DIR": "runs/gpu_smoke",
    "TRAIN_FRACTION": "0.00001",
    "MAX_STEPS": "2",
    "BATCH_SIZE": "1",
    "BLOCK_SIZE": "8",
    "D_MODEL": "16",
    "NUM_LAYERS": "1",
    "NUM_HEADS": "1",
    "MLP_HIDDEN_SIZE": "64",
    "GRAD_ACCUM_STEPS": "1",
    "EVAL_ITERS": "1",
    "EVAL_EVERY": "1",
    "LOG_EVERY": "1",
    "SAMPLE_EVERY": "1",
    "CHECKPOINT_EVERY": "1",
    "SAMPLE_TOKENS": "8",
    "COMPILE": "0",
    "MIXED_PRECISION": "0",
}


PRESETS = {
    "learning-1pct": {
        "RUN_DIR": "runs/tinystories_learning_1pct",
        "TRAIN_FRACTION": "0.01",
        "TARGET_VOCAB_SIZE": "256",
        "MAX_STEPS": "200",
        "BATCH_SIZE": "16",
        "BLOCK_SIZE": "128",
        "D_MODEL": "128",
        "NUM_LAYERS": "4",
        "NUM_HEADS": "4",
        "MLP_HIDDEN_SIZE": "512",
        "GRAD_ACCUM_STEPS": "4",
        "EVAL_EVERY": "50",
        "EVAL_ITERS": "20",
        "LOG_EVERY": "10",
        "CHECKPOINT_EVERY": "100",
        "SAMPLE_EVERY": "100",
        "COMPILE": "0",
        "MIXED_PRECISION": "1",
    },
    "gpu-30m": {
        "RUN_DIR": "runs/tinystories_gpt_30m",
        "TRAIN_FRACTION": "0.10",
        "TARGET_VOCAB_SIZE": "2048",
        "MAX_STEPS": "2000",
        "BATCH_SIZE": "16",
        "BLOCK_SIZE": "512",
        "D_MODEL": "512",
        "NUM_LAYERS": "8",
        "NUM_HEADS": "8",
        "MLP_HIDDEN_SIZE": "2048",
        "GRAD_ACCUM_STEPS": "8",
        "EVAL_EVERY": "100",
        "EVAL_ITERS": "30",
        "LOG_EVERY": "10",
        "CHECKPOINT_EVERY": "500",
        "SAMPLE_EVERY": "250",
        "MAX_LR": "0.0003",
        "MIN_LR": "0.00003",
        "WARMUP_STEPS": "100",
        "WEIGHT_DECAY": "0.1",
        "GRAD_CLIP": "1.0",
        "COMPILE": "1",
        "COMPILE_MODE": "reduce-overhead",
        "MIXED_PRECISION": "1",
    },
}


def merged_env(overrides):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.update(overrides)
    return env


def run_command(command, env=None, cwd=None):
    print()
    print(f"running: {' '.join(command)}", flush=True)
    result = subprocess.run(command, env=env, cwd=cwd)
    if result.returncode != 0:
        print()
        print(f"ERROR: command failed with exit code {result.returncode}")
        print(f"failed command: {' '.join(command)}")
        sys.exit(result.returncode)


def download_with_wget(url, output_dir):
    wget = shutil.which("wget")
    if wget is None:
        return False

    run_command([wget, "-c", "-P", str(output_dir), url])
    return True


def download_with_python(url, output_path):
    print(f"downloading with Python: {url}")
    urlretrieve(url, output_path)


def ensure_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    downloads = [
        (TRAIN_FILE, TRAIN_URL),
        (VALID_FILE, VALID_URL),
    ]

    for path, url in downloads:
        if path.exists() and path.stat().st_size > 0:
            print(f"dataset file exists: {path} ({path.stat().st_size:,} bytes)")
            continue

        print(f"dataset file missing: {path}")
        if not download_with_wget(url, DATA_DIR):
            download_with_python(url, path)


def check_torch():
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not installed in this environment.")
        print("Install PyTorch before running the GPU training scripts.")
        sys.exit(1)

    print(f"torch version: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda device: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA is not available. The scripts will run on CPU.")


def run_smoke_test():
    print()
    print("smoke test")
    run_command([sys.executable, "train_tinystories_gpt.py"], env=merged_env(SMOKE_ENV))


def parse_env_overrides(raw_overrides):
    overrides = {}
    for raw in raw_overrides:
        if "=" not in raw:
            raise ValueError(f"--set must look like KEY=VALUE, got {raw!r}")
        key, value = raw.split("=", 1)
        if not key:
            raise ValueError(f"--set must have a non-empty key, got {raw!r}")
        overrides[key] = value
    return overrides


def run_training(preset_name, run_dir_override, extra_overrides):
    preset = PRESETS[preset_name].copy()
    if run_dir_override:
        preset["RUN_DIR"] = run_dir_override
    preset.update(extra_overrides)

    print()
    print(f"training preset: {preset_name}")
    for key in sorted(preset):
        print(f"{key}={preset[key]}")

    run_command([sys.executable, "train_tinystories_gpt.py"], env=merged_env(preset))
    return Path(preset["RUN_DIR"])


def export_weights(run_dir):
    print()
    print("exporting weights-only artifact")
    run_command(
        [sys.executable, "export_tinystories_model.py"],
        env=merged_env({"RUN_DIR": str(run_dir)}),
    )
    return run_dir / "export"


def package_artifacts(export_dir, include_checkpoint, run_dir):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    package_path = ARTIFACT_DIR / f"{run_dir.name}_{timestamp}.tar.gz"

    with tarfile.open(package_path, "w:gz") as archive:
        archive.add(export_dir, arcname=export_dir.name)
        if include_checkpoint:
            checkpoint_dir = run_dir / "checkpoints"
            archive.add(checkpoint_dir, arcname="checkpoints")

    print()
    print(f"packaged artifact: {package_path}")
    print(f"artifact size: {package_path.stat().st_size:,} bytes")
    return package_path


def local_ip_guess():
    try:
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)
    except OSError:
        return "SERVER_IP"


def serve_artifacts(package_path, port, public_host):
    host = public_host or local_ip_guess()
    print()
    print("download server")
    print(f"file: {package_path}")
    print(f"try this URL: http://{host}:{port}/{package_path.name}")
    print("If this does not open, check the cloud firewall/security group for that port.")
    print("Press Ctrl+C to stop the server.")
    run_command(
        [sys.executable, "-m", "http.server", str(port), "--bind", "0.0.0.0"],
        cwd=package_path.parent,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download TinyStories, run a smoke test, train, export, and optionally serve artifacts."
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "train"],
        default="smoke",
        help="smoke only checks the pipeline; train runs smoke then the selected training preset.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="learning-1pct",
        help="training preset used with --mode train.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="override the preset RUN_DIR.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="do not download TinyStories even if files are missing.",
    )
    parser.add_argument(
        "--include-checkpoint",
        action="store_true",
        help="include full training checkpoints with optimizer state in the tarball.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="start an HTTP server for downloading the packaged artifact.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP server port used with --serve.",
    )
    parser.add_argument(
        "--public-host",
        default=os.environ.get("PUBLIC_HOST"),
        help="public IP/hostname to print in the download URL.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a training environment variable, for example --set MAX_STEPS=100.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    check_torch()

    if args.skip_download:
        print("dataset download skipped")
    else:
        ensure_dataset()

    run_smoke_test()

    if args.mode == "smoke":
        print()
        print("smoke test passed")
        print("No real training was run. Use --mode train when ready.")
        return

    extra_overrides = parse_env_overrides(args.set)
    run_dir = run_training(args.preset, args.run_dir, extra_overrides)
    export_dir = export_weights(run_dir)
    package_path = package_artifacts(export_dir, args.include_checkpoint, run_dir)

    print()
    print("done")
    print(f"download artifact: {package_path}")
    print("This tarball contains the weights-only model export, metadata, tokenizer, loss log, and README.")

    if args.serve:
        serve_artifacts(package_path, args.port, args.public_host)


if __name__ == "__main__":
    main()
