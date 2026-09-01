from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_LAZY_MODULES = {
    "inference": "refiner.inference",
    "io": "refiner.io",
    "pipeline": "refiner.pipeline",
    "robot": "refiner.robotics",
    "robotics": "refiner.robotics",
    "text": "refiner.text",
    "video": "refiner.video",
    "datatype": "refiner.pipeline.data.datatype",
}

_LAZY_ATTRS = {
    "AddColumns": "refiner.pipeline.sinks.lance",
    "Append": "refiner.pipeline.sinks.lance",
    "Create": "refiner.pipeline.sinks.lance",
    "Overwrite": "refiner.pipeline.sinks.lance",
    "CUDAVersion": "refiner.pipeline",
    "BlobAssetConfig": "refiner.pipeline.sinks.assets",
    "FileAssetConfig": "refiner.pipeline.sinks.assets",
    "GPU": "refiner.pipeline",
    "GPUType": "refiner.pipeline",
    "SUPPORTED_CUDA_VERSIONS": "refiner.pipeline",
    "SUPPORTED_GPU_TYPES": "refiner.pipeline",
    "ValidationContract": "refiner.pipeline",
    "ValidationError": "refiner.pipeline",
    "from_items": "refiner.pipeline",
    "from_source": "refiner.pipeline",
    "load_lance": "refiner.pipeline",
    "read_csv": "refiner.pipeline",
    "read_files": "refiner.pipeline",
    "read_hf_dataset": "refiner.pipeline",
    "read_hdf5": "refiner.pipeline",
    "read_json": "refiner.pipeline",
    "read_jsonl": "refiner.pipeline",
    "read_lerobot": "refiner.pipeline",
    "read_blob": "refiner.io",
    "read_mcap": "refiner.pipeline",
    "read_parquet": "refiner.pipeline",
    "read_tfds": "refiner.pipeline",
    "read_tfrecords": "refiner.pipeline",
    "read_videos": "refiner.pipeline",
    "read_zarr": "refiner.pipeline",
    "task": "refiner.pipeline",
    "coalesce": "refiner.pipeline.expressions",
    "col": "refiner.pipeline.expressions",
    "if_else": "refiner.pipeline.expressions",
    "lit": "refiner.pipeline.expressions",
    "Secrets": "refiner.launchers.secrets",
    "log_gauge": "refiner.worker.metrics.api",
    "log_gauges": "refiner.worker.metrics.api",
    "log_histogram": "refiner.worker.metrics.api",
    "log_throughput": "refiner.worker.metrics.api",
    "register_gauge": "refiner.worker.metrics.api",
    "logger": "refiner.worker.context",
}

__all__ = [
    # sources
    "AddColumns",
    "Append",
    "Create",
    "Overwrite",
    "CUDAVersion",
    "BlobAssetConfig",
    "FileAssetConfig",
    "GPU",
    "GPUType",
    "SUPPORTED_CUDA_VERSIONS",
    "SUPPORTED_GPU_TYPES",
    "ValidationContract",
    "ValidationError",
    "Secrets",
    "read_csv",
    "read_files",
    "read_hf_dataset",
    "read_hdf5",
    "read_json",
    "read_jsonl",
    "read_lerobot",
    "read_blob",
    "load_lance",
    "read_mcap",
    "read_parquet",
    "read_tfds",
    "read_tfrecords",
    "read_videos",
    "read_zarr",
    "from_items",
    "from_source",
    "task",
    # metrics
    "log_throughput",
    "log_gauge",
    "log_gauges",
    "log_histogram",
    "register_gauge",
    "logger",
    # expressions
    "col",
    "lit",
    "coalesce",
    "if_else",
    # submodules
    "inference",
    "io",
    "pipeline",
    "video",
    "robot",
    "robotics",
    "text",
    "datatype",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_MODULES:
        value = import_module(_LAZY_MODULES[name])
    elif name in _LAZY_ATTRS:
        module = import_module(_LAZY_ATTRS[name])
        value = getattr(module, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    import refiner.inference as inference
    import refiner.io as io
    import refiner.pipeline as pipeline
    import refiner.robotics as robot
    import refiner.robotics as robotics
    import refiner.text as text
    import refiner.video as video
    from refiner.launchers.secrets import Secrets
    from refiner.pipeline.sinks.lance import AddColumns, Append, Create, Overwrite
    from refiner.io import read_blob
    from refiner.pipeline import (
        CUDAVersion,
        GPU,
        GPUType,
        SUPPORTED_CUDA_VERSIONS,
        SUPPORTED_GPU_TYPES,
        ValidationContract,
        ValidationError,
        from_items,
        from_source,
        load_lance,
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
    from refiner.pipeline.data import datatype
    from refiner.pipeline.expressions import coalesce, col, if_else, lit
    from refiner.pipeline.sinks.assets import BlobAssetConfig, FileAssetConfig
    from refiner.worker.context import logger
    from refiner.worker.metrics.api import (
        log_gauge,
        log_gauges,
        log_histogram,
        log_throughput,
        register_gauge,
    )
