from __future__ import annotations

from typing import Literal

from refiner.io.datafolder import DataFolder


def zarr_store(
    folder: DataFolder,
    path: str = "",
    *,
    mode: Literal["r", "w", "w-", "a"] = "r",
):
    import zarr

    create = mode in {"w", "w-", "a"}
    return zarr.storage.FSStore(
        folder._join(path),
        fs=folder.fs,
        mode=mode,
        create=create,
    )


__all__ = ["zarr_store"]
