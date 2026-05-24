from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from refiner.io.datafolder import DataFolder, DataFolderLike
from refiner.pipeline.data.block import Block
from refiner.pipeline.sinks.reducer.file import FileCleanupReducerSink
from refiner.pipeline.sinks.zarr import (
    _append_zarr_array,
    _batch_length,
    _render_store_relpath,
    _zarr_store,
)
from refiner.utils import check_required_dependencies
from refiner.worker.context import get_active_stage_index, get_finalized_workers
from refiner.worker.lifecycle import sort_finalized_workers


class ZarrReducerSink(FileCleanupReducerSink):
    def __init__(
        self,
        output: DataFolderLike,
        *,
        store_template: str,
        episode_ends_path: str | None = None,
        array_chunk_bytes: int = 8 * 1024 * 1024,
        reduce_to_single_store: bool = False,
    ) -> None:
        check_required_dependencies("write_zarr", ["zarr"], dist="zarr")
        super().__init__(
            output=output,
            filename_template=(
                f"_parts/{store_template}" if reduce_to_single_store else store_template
            ),
            reducer_name="write_zarr_reduce",
        )
        self.store_template = store_template
        self.episode_ends_path = episode_ends_path
        self.array_chunk_bytes = array_chunk_bytes
        self.reduce_to_single_store = reduce_to_single_store
        self._merged = False

    def write_shard_block(self, shard_id: str, block: Block) -> None:
        super().write_shard_block(shard_id, block)
        if self.reduce_to_single_store:
            self._merge()
            return

        stage_index = get_active_stage_index()
        if stage_index is None or stage_index <= 0:
            raise ValueError(
                "write_zarr_reduce requires an active reducer stage with a prior writer stage"
            )
        relpaths = [
            _render_store_relpath(
                self.store_template,
                shard_id=row.shard_id,
                worker_id=row.worker_token,
            )
            for row in sort_finalized_workers(
                get_finalized_workers(stage_index=stage_index - 1)
            )
        ]
        _validate_zarr_stores(self.output, relpaths)
        _remove_parts(self.output)
        self._clear_root_payload_except(relpaths)

    def _clear_root_payload_except(self, relpaths: Iterable[str]) -> None:
        import zarr

        keep_paths = set(relpaths)
        try:
            root = zarr.open_group(store=_zarr_store(self.output, "", mode="r+"))
        except Exception:
            return

        def clear_group(group: Any, prefix: str = "") -> None:
            group_keys = set(group.group_keys())
            for key in sorted({*group.array_keys(), *group_keys}):
                path = f"{prefix}/{key}" if prefix else key
                if path == "_refiner" or path.startswith("_refiner/"):
                    continue
                if path in keep_paths:
                    continue
                if any(keep_path.startswith(f"{path}/") for keep_path in keep_paths):
                    if key in group_keys:
                        clear_group(group[key], path)
                        continue
                del group[key]
            group.attrs.clear()

        clear_group(root)

    def describe(self) -> tuple[str, str, dict[str, object]]:
        return (
            "write_zarr_reduce",
            "writer",
            {
                "path": self.output.abs_path(),
                "store_template": self.store_template,
                "array_chunk_bytes": self.array_chunk_bytes,
                "reduce_to_single_store": self.reduce_to_single_store,
            },
        )

    def _merge(self) -> None:
        if self._merged:
            return

        stage_index = get_active_stage_index()
        if stage_index is None or stage_index <= 0:
            raise ValueError(
                "write_zarr_reduce requires an active reducer stage with a prior writer stage"
            )

        expected_parts = [
            "_parts/"
            + _render_store_relpath(
                self.store_template,
                shard_id=row.shard_id,
                worker_id=row.worker_token,
            )
            for row in sort_finalized_workers(
                get_finalized_workers(stage_index=stage_index - 1),
            )
        ]
        if not expected_parts:
            import zarr

            final = zarr.open_group(
                store=_zarr_store(self.output, "", mode="a"),
                mode="a",
            )
            _clear_final_group(final)
            self._merged = True
            return

        parts = self._collect_parts(expected_parts)

        import zarr

        final = zarr.open_group(
            store=_zarr_store(self.output, "", mode="a"),
            mode="a",
        )
        _clear_final_group(final)

        row_offset = 0
        arrays: dict[str, Any] = {}
        for relpath, paths in parts:
            source = zarr.open_group(
                store=_zarr_store(self.output, relpath, mode="r"),
                mode="r",
            )
            for path in sorted(paths):
                source_array = source[path]
                if path == self.episode_ends_path:
                    if source_array.shape[0] == 0:
                        continue
                    part_last = row_offset
                    batch_size = _batch_length(
                        source_array,
                        self.array_chunk_bytes,
                    )
                    for start in range(0, int(source_array.shape[0]), batch_size):
                        end = min(int(source_array.shape[0]), start + batch_size)
                        values = np.asarray(source_array[start:end], dtype=np.int64)
                        _append_zarr_array(
                            final,
                            arrays,
                            path,
                            values + row_offset,
                            chunks=getattr(source_array, "chunks", None),
                            compressor=getattr(source_array, "compressor", None),
                        )
                        part_last = int(values[-1])
                    row_offset += part_last
                    continue
                batch_size = _batch_length(source_array, self.array_chunk_bytes)
                if source_array.shape[0] == 0:
                    _append_zarr_array(
                        final,
                        arrays,
                        path,
                        np.asarray(source_array[:0]),
                        chunks=getattr(source_array, "chunks", None),
                        compressor=getattr(source_array, "compressor", None),
                    )
                    continue
                for start in range(0, int(source_array.shape[0]), batch_size):
                    end = min(int(source_array.shape[0]), start + batch_size)
                    _append_zarr_array(
                        final,
                        arrays,
                        path,
                        np.asarray(source_array[start:end]),
                        chunks=getattr(source_array, "chunks", None),
                        compressor=getattr(source_array, "compressor", None),
                    )
        self._merged = True

    def on_shard_finalized(self, shard_id: str) -> None:
        del shard_id
        if not self.reduce_to_single_store or not self._merged:
            return
        _remove_parts(self.output)
        try:
            if not self.output.ls("_parts"):
                self.output.rmdir("_parts")
        except (FileNotFoundError, OSError, ValueError):
            pass

    def _collect_parts(
        self, expected_parts: Iterable[str]
    ) -> list[tuple[str, set[str]]]:
        import zarr

        parts: list[tuple[str, set[str]]] = []
        payload_paths: set[str] | None = None
        schemas: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {}
        for relpath in expected_parts:
            if not self.output.exists(relpath):
                raise ValueError(f"Zarr part store is missing: {relpath}")
            source = zarr.open_group(
                store=_zarr_store(self.output, relpath, mode="r"),
                mode="r",
            )
            source_paths = set(_iter_array_paths(source))
            if not source_paths:
                continue
            source_payload_paths = {
                path for path in source_paths if path != self.episode_ends_path
            }
            if (
                self.episode_ends_path is not None
                and source_payload_paths
                and self.episode_ends_path not in source_paths
            ):
                raise ValueError(
                    f"Zarr part stores must contain {self.episode_ends_path!r}"
                )
            if payload_paths is None:
                payload_paths = source_payload_paths
            elif source_payload_paths != payload_paths:
                raise ValueError(
                    "Zarr part stores must contain the same payload arrays"
                )
            for path in source_paths:
                source_array = source[path]
                schema = (tuple(source_array.shape[1:]), np.dtype(source_array.dtype))
                previous = schemas.setdefault(path, schema)
                if previous != schema:
                    if previous[0] != schema[0]:
                        raise ValueError(
                            f"Zarr arrays for {path!r} must have matching trailing shapes"
                        )
                    raise ValueError(
                        f"Zarr arrays for {path!r} must have matching dtypes"
                    )
            parts.append((relpath, source_paths))
        return parts


def _iter_array_paths(group: Any, prefix: str = "") -> Iterable[str]:
    for name, item in group.items():
        path = f"{prefix}/{name}" if prefix else name
        if hasattr(item, "shape"):
            yield path
        else:
            yield from _iter_array_paths(item, path)


def _remove_parts(output: DataFolder) -> None:
    try:
        output.rm("_parts", recursive=True)
    except FileNotFoundError:
        pass


def _validate_zarr_stores(output: DataFolder, relpaths: Iterable[str]) -> None:
    import zarr

    payload_paths: set[str] | None = None
    schemas: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {}
    for relpath in relpaths:
        if not output.exists(relpath):
            raise ValueError(f"Zarr store is missing: {relpath}")
        source = zarr.open_group(
            store=_zarr_store(output, relpath, mode="r"),
            mode="r",
        )
        source_paths = set(_iter_array_paths(source))
        if not source_paths:
            continue
        if payload_paths is None:
            payload_paths = source_paths
        elif source_paths != payload_paths:
            raise ValueError("Zarr stores must contain the same arrays")
        for path in source_paths:
            source_array = source[path]
            schema = (tuple(source_array.shape[1:]), np.dtype(source_array.dtype))
            previous = schemas.setdefault(path, schema)
            if previous != schema:
                if previous[0] != schema[0]:
                    raise ValueError(
                        f"Zarr arrays for {path!r} must have matching trailing shapes"
                    )
                raise ValueError(f"Zarr arrays for {path!r} must have matching dtypes")


def _clear_final_group(group: Any) -> None:
    for key in sorted({*group.array_keys(), *group.group_keys()}):
        if key != "_parts":
            del group[key]
    group.attrs.clear()


__all__ = ["ZarrReducerSink"]
