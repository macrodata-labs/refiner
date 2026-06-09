from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_LAZY_ATTRS = {
    "Row": "refiner.pipeline.data.row",
    "Shard": "refiner.pipeline.data.shard",
    "RefinerPipeline": "refiner.pipeline.pipeline",
    "from_items": "refiner.pipeline.pipeline",
    "from_source": "refiner.pipeline.pipeline",
    "read_csv": "refiner.pipeline.pipeline",
    "read_files": "refiner.pipeline.pipeline",
    "read_hf_dataset": "refiner.pipeline.pipeline",
    "read_hdf5": "refiner.pipeline.pipeline",
    "read_json": "refiner.pipeline.pipeline",
    "read_jsonl": "refiner.pipeline.pipeline",
    "read_lerobot": "refiner.pipeline.pipeline",
    "read_mcap": "refiner.pipeline.pipeline",
    "read_parquet": "refiner.pipeline.pipeline",
    "read_tfds": "refiner.pipeline.pipeline",
    "read_tfrecords": "refiner.pipeline.pipeline",
    "read_videos": "refiner.pipeline.pipeline",
    "read_zarr": "refiner.pipeline.pipeline",
    "task": "refiner.pipeline.pipeline",
    "CUDAVersion": "refiner.pipeline.resources",
    "GPU": "refiner.pipeline.resources",
    "GPUType": "refiner.pipeline.resources",
    "SUPPORTED_CUDA_VERSIONS": "refiner.pipeline.resources",
    "SUPPORTED_GPU_TYPES": "refiner.pipeline.resources",
}

__all__ = [
    "CUDAVersion",
    "GPU",
    "GPUType",
    "RefinerPipeline",
    "Row",
    "Shard",
    "SUPPORTED_CUDA_VERSIONS",
    "SUPPORTED_GPU_TYPES",
    "read_csv",
    "read_files",
    "read_hf_dataset",
    "read_hdf5",
    "read_json",
    "read_jsonl",
    "read_lerobot",
    "read_mcap",
    "read_parquet",
    "read_tfds",
    "read_tfrecords",
    "read_videos",
    "read_zarr",
    "from_items",
    "from_source",
    "task",
]


def __getattr__(name: str) -> Any:
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_LAZY_ATTRS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from refiner.pipeline.data.row import Row
    from refiner.pipeline.data.shard import Shard
    from refiner.pipeline.pipeline import (
        RefinerPipeline,
        from_items,
        from_source,
        read_csv,
        read_files,
        read_hdf5,
        read_hf_dataset,
        read_json,
        read_jsonl,
        read_lerobot,
        read_mcap,
        read_parquet,
        read_tfds,
        read_tfrecords,
        read_videos,
        read_zarr,
        task,
    )
    from refiner.pipeline.resources import (
        CUDAVersion,
        GPU,
        GPUType,
        SUPPORTED_CUDA_VERSIONS,
        SUPPORTED_GPU_TYPES,
    )
