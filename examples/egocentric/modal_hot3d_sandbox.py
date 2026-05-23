from __future__ import annotations

import argparse
import json
from pathlib import Path

import modal


APP_NAME = "refiner-hot3d-hawor-benchmark"
VOLUME_NAME = "refiner-hot3d-hawor-cache"
SANDBOX_ID_PATH = Path("artifacts/modal-hot3d-hawor-sandbox.json")


def _image() -> modal.Image:
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.1-devel-ubuntu22.04",
            add_python="3.11",
        )
        .apt_install(
            "build-essential",
            "ca-certificates",
            "curl",
            "ffmpeg",
            "git",
            "libegl1",
            "libgl1",
            "libglib2.0-0",
            "libsm6",
            "libxext6",
            "libxrender1",
            "ninja-build",
            "wget",
        )
        .uv_pip_install(
            "huggingface-hub[hf-xet]>=1.4.1",
            "hf-transfer>=0.1.9",
            "imageio-ffmpeg",
            "natsort",
            "numpy",
            "opencv-python-headless",
            "requests",
            "tqdm",
        )
        .env(
            {
                "HF_HOME": "/cache/huggingface",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "HF_HUB_DOWNLOAD_TIMEOUT": "600",
                "PIP_CACHE_DIR": "/cache/pip",
                "TORCH_HOME": "/cache/torch",
                "TORCH_CUDA_ARCH_LIST": "9.0",
            }
        )
    )


def create_sandbox(*, timeout_hours: int) -> dict[str, str | int]:
    app = modal.App.lookup(APP_NAME, create_if_missing=True)
    cache = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    with modal.enable_output():
        sandbox = modal.Sandbox.create(
            app=app,
            image=_image(),
            gpu="H100",
            timeout=timeout_hours * 60 * 60,
            idle_timeout=timeout_hours * 60 * 60,
            volumes={"/cache": cache},
            name="hot3d-hawor-h100",
        )

    payload: dict[str, str | int] = {
        "app_name": APP_NAME,
        "volume_name": VOLUME_NAME,
        "sandbox_id": sandbox.object_id,
        "timeout_hours": timeout_hours,
    }
    SANDBOX_ID_PATH.parent.mkdir(parents=True, exist_ok=True)
    SANDBOX_ID_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def smoke_test(sandbox_id: str | None = None) -> int:
    sandbox_id = sandbox_id or json.loads(SANDBOX_ID_PATH.read_text())["sandbox_id"]
    sandbox = modal.Sandbox.from_id(sandbox_id)
    process = sandbox.exec(
        "bash",
        "-lc",
        "set -euo pipefail; "
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; "
        "nvcc --version | tail -n 1; "
        "python - <<'PY'\n"
        "import os\n"
        "print('HF_HOME=' + os.environ.get('HF_HOME', ''))\n"
        "print('cache_exists=' + str(os.path.exists('/cache')))\n"
        "PY",
        timeout=120,
    )
    print(process.stdout.read(), end="")
    stderr = process.stderr.read()
    if stderr:
        print(stderr, end="")
    process.wait()
    return int(process.returncode or 0)


def terminate(sandbox_id: str | None = None) -> None:
    sandbox_id = sandbox_id or json.loads(SANDBOX_ID_PATH.read_text())["sandbox_id"]
    modal.Sandbox.from_id(sandbox_id).terminate()


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--timeout-hours", type=int, default=12)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--sandbox-id")
    stop = sub.add_parser("terminate")
    stop.add_argument("--sandbox-id")
    args = parser.parse_args()

    if args.command == "create":
        print(json.dumps(create_sandbox(timeout_hours=args.timeout_hours), indent=2))
    elif args.command == "smoke":
        raise SystemExit(smoke_test(args.sandbox_id))
    elif args.command == "terminate":
        terminate(args.sandbox_id)


if __name__ == "__main__":
    main()
