from __future__ import annotations

import subprocess

import refiner as mdr
from refiner.worker.context import logger


def log_nvidia_smi(_task_rank: int, _num_tasks: int) -> None:
    logger.info("{}", subprocess.check_output(["nvidia-smi"], text=True).rstrip())
    # Some GPU heavy job (running BERT, ViT or anything you need)


def main() -> None:
    pipeline = mdr.task(log_nvidia_smi, num_tasks=1)
    pipeline.launch_cloud(
        name="gpu-ranks",
        num_workers=1,
        gpu=mdr.GPU(type="h100", count=1),
    )


if __name__ == "__main__":
    main()
