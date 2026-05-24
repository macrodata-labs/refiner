from __future__ import annotations

from collections.abc import Iterator
from typing import Literal
from typing import Any

from refiner.io.datafolder import DataFolder


def zarr_store(
    folder: DataFolder,
    path: str = "",
    *,
    mode: Literal["r", "r+", "w", "w-", "a"] = "r",
):
    import zarr

    create = mode in {"w", "w-", "a"}
    return zarr.storage.FSStore(
        folder._join(path),
        fs=folder.fs,
        mode=mode,
        create=create,
    )


def iter_zarr_array_paths(group: Any, prefix: str = "") -> Iterator[str]:
    items = group.items() if hasattr(group, "items") else group.members()
    for name, item in items:
        path = f"{prefix}/{name}" if prefix else name
        if hasattr(item, "shape"):
            yield path
        else:
            yield from iter_zarr_array_paths(item, path)


__all__ = ["iter_zarr_array_paths", "zarr_store"]
