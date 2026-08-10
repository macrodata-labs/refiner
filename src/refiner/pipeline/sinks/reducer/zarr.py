from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from refiner.io.datafolder import DataFolderLike
from refiner.pipeline.data.block import Block
from refiner.pipeline.sinks.reducer.file import FileCleanupReducerSink
from refiner.pipeline.sinks.zarr import (
    _DEFAULT_ARRAY_CHUNK_BYTES,
    _append_zarr_array,
    _batch_length,
    _render_store_relpath,
    _zarr_store,
)
from refiner.worker.context import get_active_stage_index, get_finalized_workers
from refiner.worker.lifecycle import sort_finalized_workers


class ZarrReducerSink(FileCleanupReducerSink):
    def __init__(
        self,
        output: DataFolderLike,
        *,
        store_template: str,
        episode_ends_path: str | None = None,
        array_chunk_bytes: int = _DEFAULT_ARRAY_CHUNK_BYTES,
        reduce_to_single_store: bool = True,
    ) -> None:
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

    def _declared_refiner_extras(self) -> tuple[str, ...]:
        return ("zarr",)

    def write_shard_block(self, shard_id: str, block: Block) -> None:
        self._run_cleanup()
        if self.reduce_to_single_store:
            self._merge()
            return

        relpaths = self._finalized_store_paths()
        self._collect_stores(relpaths, for_merge=False)
        try:
            self.output.rm("_parts", recursive=True)
        except FileNotFoundError:
            pass
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
                if key in group_keys and any(
                    keep_path.startswith(f"{path}/") for keep_path in keep_paths
                ):
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
        expected_parts = self._finalized_store_paths(prefix="_parts/")
        import zarr

        parts = self._collect_stores(expected_parts, for_merge=True)
        final_attrs: dict[str, Any] | None = None
        for relpath, _paths in parts:
            source = zarr.open_group(
                store=_zarr_store(self.output, relpath, mode="r"),
                mode="r",
            )
            source_attrs = dict(source.attrs)
            if final_attrs is None:
                final_attrs = source_attrs
            elif source_attrs != final_attrs:
                raise ValueError("Zarr part store attrs differ")

        final = zarr.open_group(
            store=_zarr_store(self.output, "", mode="a"),
            mode="a",
        )
        for key in sorted({*final.array_keys(), *final.group_keys()}):
            if key != "_parts":
                del final[key]
        final.attrs.clear()
        if final_attrs is not None:
            final.attrs.update(final_attrs)

        row_offset = 0
        arrays: dict[str, Any] = {}
        for relpath, paths in parts:
            source = zarr.open_group(
                store=_zarr_store(self.output, relpath, mode="r"),
                mode="r",
            )
            episode_path = self.episode_ends_path
            payload_paths = paths - (
                {episode_path} if episode_path is not None else set()
            )
            payload_lengths = {int(source[path].shape[0]) for path in payload_paths}
            if len(payload_lengths) > 1:
                raise ValueError(
                    "Zarr part store payload arrays must have matching lengths"
                )
            payload_rows = next(iter(payload_lengths), None)
            part_end = 0
            if episode_path is not None and episode_path in paths:
                episode_ends = source[episode_path]
                if episode_ends.shape[0] > 0:
                    part_end = int(np.asarray(episode_ends[-1]))
                if payload_rows is not None and part_end != payload_rows:
                    raise ValueError(
                        "Zarr part store episode_ends final value does not match "
                        "payload row count"
                    )
            for path in sorted(paths):
                source_array = source[path]
                chunks = getattr(source_array, "chunks", None)
                compressor = getattr(source_array, "compressor", None)
                if source_array.shape[0] == 0 and path == episode_path:
                    continue
                if source_array.shape[0] == 0:
                    _append_zarr_array(
                        final,
                        arrays,
                        path,
                        np.asarray(source_array[:0]),
                        chunks=chunks,
                        compressor=compressor,
                    )
                    continue

                batch_size = _batch_length(source_array, self.array_chunk_bytes)
                for start in range(0, int(source_array.shape[0]), batch_size):
                    end = min(int(source_array.shape[0]), start + batch_size)
                    values = np.asarray(source_array[start:end])
                    if path == episode_path:
                        values = np.asarray(values, dtype=np.int64)
                        values = values + row_offset
                    _append_zarr_array(
                        final,
                        arrays,
                        path,
                        values,
                        chunks=chunks,
                        compressor=compressor,
                    )
                if path == episode_path:
                    row_offset += part_end

    def on_shard_finalized(self, shard_id: str) -> None:
        del shard_id
        if not self.reduce_to_single_store:
            return
        try:
            self.output.rm("_parts", recursive=True)
        except FileNotFoundError:
            pass

    def _finalized_store_paths(self, prefix: str = "") -> list[str]:
        stage_index = get_active_stage_index()
        if stage_index is None or stage_index <= 0:
            raise ValueError(
                "write_zarr_reduce requires an active reducer stage with a prior writer stage"
            )
        return [
            prefix
            + _render_store_relpath(
                self.store_template,
                shard_id=row.shard_id,
                worker_id=row.worker_token,
            )
            for row in sort_finalized_workers(
                get_finalized_workers(stage_index=stage_index - 1)
            )
        ]

    def _collect_stores(
        self,
        relpaths: Iterable[str],
        *,
        for_merge: bool,
    ) -> list[tuple[str, set[str]]]:
        import zarr

        stores: list[tuple[str, set[str]]] = []
        payload_paths: set[str] | None = None
        schemas: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {}
        episode_path = self.episode_ends_path if for_merge else None
        for relpath in relpaths:
            if not self.output.exists(relpath):
                kind = "part store" if for_merge else "store"
                raise ValueError(f"Zarr {kind} is missing: {relpath}")
            source = zarr.open_group(
                store=_zarr_store(self.output, relpath, mode="r"),
                mode="r",
            )
            source_paths = set(_iter_array_paths(source))
            if not source_paths:
                continue
            source_payload_paths = (
                source_paths - {episode_path}
                if episode_path is not None
                else source_paths
            )
            if (
                episode_path is not None
                and source_payload_paths
                and episode_path not in source_paths
            ):
                raise ValueError(f"Zarr part stores must contain {episode_path!r}")
            if payload_paths is None:
                payload_paths = source_payload_paths
            elif source_payload_paths != payload_paths:
                kind = "part stores" if for_merge else "stores"
                payload = " payload" if for_merge else ""
                raise ValueError(f"Zarr {kind} must contain the same{payload} arrays")
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
            stores.append((relpath, source_paths))
        return stores


def _iter_array_paths(group: Any, prefix: str = "") -> Iterable[str]:
    for name, item in group.items():
        path = f"{prefix}/{name}" if prefix else name
        if hasattr(item, "shape"):
            yield path
        else:
            yield from _iter_array_paths(item, path)


__all__ = ["ZarrReducerSink"]
