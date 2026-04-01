from __future__ import annotations

import os
import shutil
import subprocess


def parse_gpu_ids(raw: str) -> list[str]:
    if not raw.strip():
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def _visible_gpu_ids_from_env() -> list[str] | None:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    return parse_gpu_ids(raw)


def available_gpu_ids() -> list[str]:
    env_gpu_ids = _visible_gpu_ids_from_env()
    if env_gpu_ids is not None:
        if not env_gpu_ids:
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES is set but does not expose any GPU devices"
            )
        return env_gpu_ids

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise RuntimeError(
            "Unable to determine available GPUs: set CUDA_VISIBLE_DEVICES or install nvidia-smi"
        )

    try:
        output = subprocess.check_output(
            [nvidia_smi, "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as err:
        stderr = (err.stderr or "").strip()
        message = f"nvidia-smi failed with exit code {err.returncode}"
        if stderr:
            message = f"{message}: {stderr}"
        raise RuntimeError(message) from err
    gpu_ids = [line.strip() for line in output.splitlines() if line.strip()]
    if not gpu_ids:
        raise RuntimeError("Unable to determine available GPUs from nvidia-smi")
    return gpu_ids


def build_gpu_sets(*, num_workers: int, gpus_per_worker: int) -> list[list[str]]:
    if gpus_per_worker <= 0:
        raise ValueError("gpus_per_worker must be > 0")
    gpu_ids = available_gpu_ids()
    needed = num_workers * gpus_per_worker
    if needed > len(gpu_ids):
        raise ValueError(
            f"Requested {needed} GPUs ({num_workers} workers x {gpus_per_worker}) but only {len(gpu_ids)} are available"
        )
    out: list[list[str]] = []
    for index in range(num_workers):
        start = index * gpus_per_worker
        out.append(gpu_ids[start : start + gpus_per_worker])
    return out


def set_visible_gpu_ids(gpu_ids: list[str]) -> None:
    if not gpu_ids:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)


__all__ = [
    "available_gpu_ids",
    "build_gpu_sets",
    "parse_gpu_ids",
    "set_visible_gpu_ids",
]
