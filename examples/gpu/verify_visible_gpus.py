from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from typing import Any

import refiner as mdr


def _probe_worker(task_rank: int, num_tasks: int) -> dict[str, Any]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    nvidia_smi = shutil.which("nvidia-smi")
    gpu_lines: list[str] = []
    nvidia_smi_error: str | None = None

    if nvidia_smi is not None:
        try:
            output = subprocess.check_output(
                [
                    nvidia_smi,
                    "--query-gpu=index,name,uuid",
                    "--format=csv,noheader",
                ],
                text=True,
            )
        except Exception as exc:  # noqa: BLE001
            nvidia_smi_error = f"{type(exc).__name__}: {exc}"
        else:
            gpu_lines = [line.strip() for line in output.splitlines() if line.strip()]

    probe = {
        "task_rank": task_rank,
        "num_tasks": num_tasks,
        "pid": os.getpid(),
        "hostname": os.uname().nodename,
        "cuda_visible_devices": visible_devices,
        "visible_gpu_count": len(
            [token for token in visible_devices.split(",") if token]
        ),
        "nvidia_smi_path": nvidia_smi,
        "nvidia_smi_gpus": gpu_lines,
        "nvidia_smi_error": nvidia_smi_error,
    }
    print(json.dumps(probe, sort_keys=True))
    return probe


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test Refiner GPU visibility")
    parser.add_argument("--launcher", choices=("local", "cloud"), default="local")
    parser.add_argument("--name", default="gpu-probe")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-tasks", type=int, default=1)
    parser.add_argument("--gpus-per-worker", type=int, default=1)
    parser.add_argument("--cpus-per-worker", type=int, default=None)
    parser.add_argument("--mem-mb-per-worker", type=int, default=None)
    parser.add_argument("--gpu-type", default=None)
    args = parser.parse_args()

    pipeline = mdr.task(_probe_worker, num_tasks=args.num_tasks)

    if args.launcher == "local":
        stats = pipeline.launch_local(
            name=args.name,
            num_workers=args.num_workers,
            cpus_per_worker=args.cpus_per_worker,
            gpus_per_worker=args.gpus_per_worker,
        )
        print(f"local launch complete: {stats}")
        print("worker probe results were logged by each worker process")
        return

    if not args.gpu_type:
        raise SystemExit("--gpu-type is required when --launcher=cloud")

    result = pipeline.launch_cloud(
        name=args.name,
        num_workers=args.num_workers,
        cpus_per_worker=args.cpus_per_worker,
        mem_mb_per_worker=args.mem_mb_per_worker,
        gpus_per_worker=args.gpus_per_worker,
        gpu_type=args.gpu_type,
    )
    print(f"cloud launch submitted: {result}")
    print("worker probe results will be emitted in worker logs")


if __name__ == "__main__":
    main()
