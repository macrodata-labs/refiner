from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, cast

import numpy as np
import pyarrow as pa

from refiner.io.datafolder import DataFolder, DataFolderLike
from refiner.pipeline.data.block import Block
from refiner.pipeline.data.row import Row
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.sinks.base import BaseSink
from refiner.robotics.row import RoboticsRow
from refiner.utils import check_required_dependencies


class ZarrSink(BaseSink):
    def __init__(
        self,
        output: DataFolderLike,
        *,
        arrays: Mapping[str, str] | None = None,
        episode_ends_path: str | None = "meta/episode_ends",
        overwrite: bool = True,
    ):
        self.output = DataFolder.resolve(output)
        self.arrays = dict(arrays) if arrays is not None else None
        self.episode_ends_path = episode_ends_path
        self.overwrite = overwrite
        self._chunks: dict[str, list[np.ndarray]] = {}
        self._episode_ends: list[int] = []

    def write_shard_block(self, shard_id: str, block: Block) -> int:
        del shard_id
        rows = list(block) if isinstance(block, Tabular) else block
        for row in rows:
            self._write_row(row)
        return len(rows)

    def _write_row(self, row: Row) -> None:
        arrays = self.arrays or _default_robotics_arrays(row)
        lengths: list[int] = []
        for zarr_path, source_key in arrays.items():
            value = _row_value(row, source_key)
            if value is None:
                continue
            array = _as_array(value)
            if array.ndim == 0:
                array = array.reshape(1)
            lengths.append(int(array.shape[0]))
            self._chunks.setdefault(zarr_path, []).append(array)
        if lengths and self.episode_ends_path is not None:
            length = lengths[0]
            if any(item != length for item in lengths):
                raise ValueError("Zarr arrays for one row must have matching lengths")
            end = (self._episode_ends[-1] if self._episode_ends else 0) + length
            self._episode_ends.append(end)

    def close(self) -> None:
        if not self._chunks and not self._episode_ends:
            return
        check_required_dependencies("write_zarr", ["zarr"], dist="zarr")
        import zarr

        mode = "w" if self.overwrite else "w-"
        root = zarr.open_group(self.output.abs_path(), mode=mode)
        for path, chunks in self._chunks.items():
            root.create_dataset(path, data=np.concatenate(chunks, axis=0))
        if self.episode_ends_path is not None:
            root.create_dataset(
                self.episode_ends_path,
                data=np.asarray(self._episode_ends, dtype=np.int64),
            )

    def describe(self) -> tuple[str, str, dict[str, object]]:
        return (
            "write_zarr",
            "writer",
            {
                "path": self.output.abs_path(),
                "arrays": dict(self.arrays) if self.arrays is not None else None,
                "episode_ends_path": self.episode_ends_path,
                "overwrite": self.overwrite,
            },
        )


def _default_robotics_arrays(row: Row) -> dict[str, str]:
    if not isinstance(row, RoboticsRow):
        raise ValueError("write_zarr requires arrays=... for non-RoboticsRow inputs")
    arrays: dict[str, str] = {}
    if row.actions is not None:
        arrays["data/action"] = "action"
    if row.states is not None:
        arrays["data/observation.state"] = "observation.state"
    if row.timestamps is not None:
        arrays["data/timestamp"] = "timestamp"
    return arrays


def _row_value(row: Row, key: str) -> Any:
    if isinstance(row, RoboticsRow):
        if key == "action":
            return row.actions
        if key == "observation.state":
            return row.states
        if key == "timestamp":
            return row.timestamps
        if key.startswith("observation."):
            return row.observations(key)
    return row[key]


def _as_array(value: Any) -> np.ndarray:
    if isinstance(value, pa.ChunkedArray | pa.Array):
        return np.asarray(value.to_pylist())
    if isinstance(value, Iterable) and not isinstance(value, str | bytes | np.ndarray):
        return np.asarray(list(cast(Iterable[Any], value)))
    return np.asarray(value)


__all__ = ["ZarrSink"]
