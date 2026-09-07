from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, get_args

GPUType = Literal["h100", "l40s", "a100", "a10", "l4", "t4", "any"]
GPUTypeRequest = GPUType | tuple[GPUType, ...] | list[GPUType]
CUDAVersion = Literal["12.4", "12.6", "12.8"]

SUPPORTED_GPU_TYPES: tuple[str, ...] = get_args(GPUType)
SUPPORTED_CUDA_VERSIONS: tuple[str, ...] = get_args(CUDAVersion)


@dataclass(frozen=True, slots=True)
class GPU:
    count: int
    type: GPUTypeRequest
    cuda_version: CUDAVersion | None = None

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("gpu.count must be > 0")
        if isinstance(self.type, str):
            choices = (self.type,)
        elif isinstance(self.type, (list, tuple)):
            choices = tuple(self.type)
            object.__setattr__(self, "type", choices)
        else:
            choices = ()
        if not 1 <= len(choices) <= 8:
            raise ValueError("gpu.type must contain between 1 and 8 choices")
        if any(choice not in SUPPORTED_GPU_TYPES for choice in choices):
            supported = ", ".join(SUPPORTED_GPU_TYPES)
            raise ValueError(f"gpu.type must be one of: {supported}")
        if len(set(choices)) != len(choices):
            raise ValueError("gpu.type choices must be unique")
        if "any" in choices[:-1]:
            raise ValueError("gpu.type 'any' must be the final choice")
        if (
            self.cuda_version is not None
            and self.cuda_version not in SUPPORTED_CUDA_VERSIONS
        ):
            supported = ", ".join(SUPPORTED_CUDA_VERSIONS)
            raise ValueError(f"gpu.cuda_version must be one of: {supported}")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "count": self.count,
            "type": list(self.type) if isinstance(self.type, tuple) else self.type,
        }
        if self.cuda_version is not None:
            payload["cuda_version"] = self.cuda_version
        return payload


__all__ = [
    "CUDAVersion",
    "GPU",
    "GPUType",
    "GPUTypeRequest",
    "SUPPORTED_CUDA_VERSIONS",
    "SUPPORTED_GPU_TYPES",
]
