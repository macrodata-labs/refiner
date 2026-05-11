from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

GPUType = Literal["h100"]
CUDAVersion = Literal["12.4", "12.6", "12.8"]

SUPPORTED_GPU_TYPES: tuple[str, ...] = get_args(GPUType)
SUPPORTED_CUDA_VERSIONS: tuple[str, ...] = get_args(CUDAVersion)


@dataclass(frozen=True, slots=True)
class GPU:
    count: int
    type: GPUType
    cuda_version: CUDAVersion | None = None

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("gpu.count must be > 0")
        if self.type not in SUPPORTED_GPU_TYPES:
            supported = ", ".join(SUPPORTED_GPU_TYPES)
            raise ValueError(f"gpu.type must be one of: {supported}")
        if (
            self.cuda_version is not None
            and self.cuda_version not in SUPPORTED_CUDA_VERSIONS
        ):
            supported = ", ".join(SUPPORTED_CUDA_VERSIONS)
            raise ValueError(f"gpu.cuda_version must be one of: {supported}")


__all__ = [
    "CUDAVersion",
    "GPU",
    "GPUType",
    "SUPPORTED_CUDA_VERSIONS",
    "SUPPORTED_GPU_TYPES",
]
