from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import posixpath
import re
import tempfile
from typing import IO, Literal, TypeAlias, cast
from urllib.parse import unquote, urlsplit

import pyarrow as pa

from refiner.execution.asyncio.runtime import io_executor
from refiner.execution.asyncio.window import AsyncWindow
from refiner.io import DataFile
from refiner.io.datafolder import DataFolder
from refiner.io._s3fs import is_s3fs
from refiner.pipeline.data import datatype
from refiner.pipeline.data.row import Row
from refiner.pipeline.data.tabular import set_or_append_column
from refiner.worker.context import get_active_worker_token
from refiner.worker.metrics.api import log_throughput

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
MissingAssetPolicy = Literal["error", "drop_row", "set_null"]
AssetCopyResult = bool | list[bool]


def _validate_config(
    *,
    subdir: str,
    missing_policy: MissingAssetPolicy,
) -> None:
    if missing_policy not in {"error", "drop_row", "set_null"}:
        raise ValueError(
            "missing_policy must be one of: 'error', 'drop_row', 'set_null'"
        )
    if (
        not subdir
        or subdir.strip("/") != subdir
        or "\\" in subdir
        or "://" in subdir
        or any(part in {"", ".", ".."} for part in subdir.split("/"))
    ):
        raise ValueError("subdir must be a non-empty relative path")


@dataclass(frozen=True, slots=True)
class FileAssetConfig:
    subdir: str = "assets"
    max_in_flight: int = 16
    missing_policy: MissingAssetPolicy = "error"

    def __post_init__(self) -> None:
        _validate_config(subdir=self.subdir, missing_policy=self.missing_policy)
        if self.max_in_flight <= 0:
            raise ValueError("max_in_flight must be > 0")


@dataclass(frozen=True, slots=True)
class BlobAssetConfig:
    subdir: str = "assets"
    target_bytes: int = 1 << 30
    missing_policy: MissingAssetPolicy = "error"

    def __post_init__(self) -> None:
        _validate_config(subdir=self.subdir, missing_policy=self.missing_policy)
        if self.target_bytes <= 0:
            raise ValueError("target_bytes must be > 0")


AssetWriteConfig: TypeAlias = FileAssetConfig | BlobAssetConfig


def asset_config_to_plan(config: AssetWriteConfig) -> dict[str, object]:
    if isinstance(config, FileAssetConfig):
        return {
            "mode": "file",
            "subdir": config.subdir,
            "max_in_flight": config.max_in_flight,
            "missing_policy": config.missing_policy,
        }
    return {
        "mode": "blob",
        "subdir": config.subdir,
        "target_bytes": config.target_bytes,
        "missing_policy": config.missing_policy,
    }


class AssetUploadManager:
    def __init__(
        self,
        output: DataFolder,
        *,
        assets_subdir: str,
        filename_template: str,
        max_uploads_in_flight: int,
        missing_asset_policy: MissingAssetPolicy = "error",
    ) -> None:
        if missing_asset_policy not in {"error", "drop_row", "set_null"}:
            raise ValueError(
                "missing_asset_policy must be one of: 'error', 'drop_row', 'set_null'"
            )
        self.output = output
        # Assets live in a writer-managed subtree, so both the assets directory and
        # the sink's output template must stay relative and disjoint.
        if (
            not assets_subdir
            or assets_subdir.strip("/") != assets_subdir
            or "\\" in assets_subdir
            or "://" in assets_subdir
            or any(part in {"", ".", ".."} for part in assets_subdir.split("/"))
        ):
            raise ValueError("assets_subdir must be a non-empty relative path")
        self.assets_subdir = assets_subdir
        normalized_template = posixpath.normpath(filename_template)
        if (
            normalized_template == ".."
            or normalized_template.startswith("../")
            or normalized_template.startswith("/")
        ):
            raise ValueError("filename_template must be a relative path")
        if normalized_template == self.assets_subdir or normalized_template.startswith(
            f"{self.assets_subdir}/"
        ):
            raise ValueError("filename_template must not write into assets_subdir")
        self._window = AsyncWindow[AssetCopyResult](
            max_in_flight=max_uploads_in_flight,
            # Row-mode writers release rows as their asset-cell copies finish.
            # Preserve result order so a later row cannot be yielded before an
            # earlier row whose copies are still in flight.
            preserve_order=True,
        )
        self._next_row_index: dict[str, int] = {}
        self._asset_columns: dict[str, tuple[str, str, str]] = {}
        self._asset_column_segments: dict[str, str] = {}
        self._input_schema_set = False
        self.missing_asset_policy = missing_asset_policy

    def set_input_schema(self, schema: pa.Schema | None) -> None:
        self._set_asset_columns(_asset_columns_from_schema(schema))
        self._input_schema_set = True

    def _set_asset_columns(
        self,
        columns: dict[str, tuple[str, str, str]],
    ) -> None:
        self._asset_columns = columns
        self._asset_column_segments = {}
        used_segments: set[str] = set()
        for column_name in self._asset_columns:
            base_segment = _SAFE_NAME_RE.sub("_", column_name).strip("._-") or "column"
            segment = base_segment
            suffix = 2
            while segment in used_segments:
                segment = f"{base_segment}_{suffix}"
                suffix += 1
            self._asset_column_segments[column_name] = segment
            used_segments.add(segment)

    def require_input_schema(self) -> None:
        if not self._input_schema_set:
            raise ValueError(
                "Row asset upload requires an input schema. Mark asset columns with "
                "dtypes=... or cast(...), or call set_input_schema(...)."
            )

    def output_schema(self, schema: pa.Schema | None) -> pa.Schema | None:
        return _asset_output_schema(schema, mode="file")

    def close(self) -> None:
        self._window.cancel_pending()

    def on_shard_complete(self, shard_id: str) -> None:
        pass

    def rewrite_table(self, shard_id: str, table: pa.Table) -> pa.Table:
        start = self._next_row_index.get(shard_id, 0)
        out = table
        result_columns: list[tuple[str, pa.Field]] = []
        # Tables may already carry asset metadata, while row-derived tables rely on
        # the schema passed through set_input_schema().
        columns = dict(self._asset_columns)
        columns.update(_asset_columns_from_schema(table.schema))
        if columns != self._asset_columns:
            self._set_asset_columns(columns)
            columns = self._asset_columns
        for column_name, (kind, storage, asset_type) in columns.items():
            idx = out.schema.get_field_index(column_name)
            if idx < 0:
                continue
            field = out.schema.field(idx)
            column = out.column(idx)
            values = column.to_pylist()
            rewritten = []
            for row_offset, value in enumerate(values):
                if self.missing_asset_policy == "error" and value is None:
                    rewritten.append(None)
                    continue
                rewritten.append(
                    self._rewrite_path(
                        value,
                        shard_id=shard_id,
                        column_name=column_name,
                        row_index=start + row_offset,
                        list_items=kind == "list",
                        storage=storage,
                    )
                )
            output_field = _file_asset_output_field(field, kind, asset_type)
            rewritten_array = pa.array(rewritten, type=output_field.type)
            out = out.set_column(idx, output_field, rewritten_array)
            result_columns.append((column_name, output_field))
        # Do not expose output rows that point at assets until those copies have
        # completed; otherwise a later copy failure leaves dangling references.
        results = self._window.drain()
        self._next_row_index[shard_id] = start + table.num_rows
        if self.missing_asset_policy == "set_null":
            # Each result lines up with one rewritten table cell. Scalar cells
            # return bool; list cells return one bool per non-null path element.
            offset = 0
            for column_name, field in result_columns:
                column_results = results[offset : offset + table.num_rows]
                offset += table.num_rows
                idx = out.schema.get_field_index(column_name)
                column = out.column(idx)
                values = [
                    _set_null_value(value, result)
                    for value, result in zip(
                        column.to_pylist(),
                        column_results,
                        strict=True,
                    )
                ]
                out = set_or_append_column(
                    out,
                    column_name,
                    pa.array(values, type=field.type),
                )
        elif self.missing_asset_policy == "drop_row":
            # Drop-row only needs row-level validity, so list cell results collapse
            # with all(...): any failed path element drops the whole row.
            keep = [True] * table.num_rows
            offset = 0
            for _column_name, _field in result_columns:
                column_results = results[offset : offset + table.num_rows]
                offset += table.num_rows
                for row_offset, result in enumerate(column_results):
                    keep[row_offset] = keep[row_offset] and (
                        all(result) if isinstance(result, list) else result
                    )
            if not all(keep):
                rows_before_filter = out.num_rows
                out = out.filter(pa.array(keep, type=pa.bool_()))
                dropped = rows_before_filter - out.num_rows
                if dropped:
                    log_throughput(
                        "asset_rows_dropped",
                        dropped,
                        shard_id,
                        unit="rows",
                    )
        return out

    def rewrite_rows(
        self,
        shard_id: str,
        rows: Iterable[Row],
    ) -> Iterable[Row]:
        self.require_input_schema()
        if not self._asset_columns:
            yield from rows
            return

        start = self._next_row_index.get(shard_id, 0)
        # Results are ordered by the same nested loop used to build patches:
        # row, then asset column. Keep only rows whose copy results are still
        # pending instead of materializing the full input block.
        pending_rows: deque[tuple[Row, dict[str, object], list[str]]] = deque()
        results: deque[AssetCopyResult] = deque()
        dropped = 0
        row_count = 0

        def emit_ready() -> Iterable[Row]:
            nonlocal dropped
            while pending_rows:
                row, patch, result_columns = pending_rows[0]
                result_count = len(result_columns)
                if len(results) < result_count:
                    break
                pending_rows.popleft()
                row_results = [results.popleft() for _ in range(result_count)]
                if self.missing_asset_policy == "drop_row" and not all(
                    all(result) if isinstance(result, list) else result
                    for result in row_results
                ):
                    dropped += 1
                    continue
                if self.missing_asset_policy == "set_null":
                    for column_name, copied in zip(
                        result_columns,
                        row_results,
                        strict=True,
                    ):
                        patch[column_name] = _set_null_value(
                            patch[column_name],
                            copied,
                        )
                yield row.update(patch) if patch else row

        for row_index, row in enumerate(rows, start=start):
            row_count += 1
            patch: dict[str, object] = {}
            result_columns: list[str] = []
            for column_name, (
                kind,
                storage,
                _asset_type,
            ) in self._asset_columns.items():
                if column_name not in row:
                    continue
                if self.missing_asset_policy == "error" and row[column_name] is None:
                    continue
                value = self._rewrite_path(
                    row[column_name],
                    shard_id=shard_id,
                    column_name=column_name,
                    row_index=row_index,
                    list_items=kind == "list",
                    storage=storage,
                )
                patch[column_name] = value
                result_columns.append(column_name)

            pending_rows.append((row, patch, result_columns))
            # Error policy must remain block-atomic for row sinks: JSONL writes
            # rows as they are yielded, so do not expose any rewritten asset paths
            # until every copy in the block has succeeded.
            if self.missing_asset_policy != "error":
                # Non-error policies can stream bounded rows as soon as their
                # copy results are available.
                results.extend(self._window.take_completed())
                yield from emit_ready()

        results.extend(self._window.drain())
        yield from emit_ready()
        self._next_row_index[shard_id] = start + row_count
        if dropped:
            log_throughput("asset_rows_dropped", dropped, shard_id, unit="rows")

    def _asset_relpath(
        self,
        value: str,
        *,
        shard_id: str,
        column_name: str,
        row_index: int,
        item_index: int | None,
    ) -> str:
        basename = unquote(posixpath.basename(urlsplit(value).path.rstrip("/")))
        basename = _SAFE_NAME_RE.sub("_", basename.replace("\\", "_")).strip("._-")
        if not basename:
            basename = "asset"
        prefix = f"{row_index}" if item_index is None else f"{row_index}-{item_index}"
        column_segment = self._asset_column_segments[column_name]
        # Attempt directories are keyed by shard and worker so reducers can delete
        # whole failed attempts without inspecting individual asset files.
        attempt_dir = f"{shard_id}__w{get_active_worker_token()}"
        return (
            f"{self.assets_subdir}/{attempt_dir}/{column_segment}/{prefix}-{basename}"
        )

    def _copy_asset(
        self,
        value: object,
        storage: str,
        relpath: str,
        *,
        shard_id: str,
    ) -> bool:
        try:
            if storage == "path" and isinstance(value, str):
                DataFile.resolve(value).copy(self.output.file(relpath))
            else:
                source, offset, size = BlobAssetManager._source(value, storage)
                with self.output.open(relpath, mode="wb") as target:
                    if isinstance(source, bytes):
                        target.write(source)
                    else:
                        remaining = size
                        with source.open("rb") as stream:
                            stream.seek(offset)
                            while remaining:
                                chunk = stream.read(min(remaining, 2 * 1024 * 1024))
                                if not chunk:
                                    raise EOFError(
                                        "asset ended before its declared size"
                                    )
                                target.write(chunk)
                                remaining -= len(chunk)
        except Exception as e:
            message = str(e).lower()
            missing = isinstance(e, FileNotFoundError) or any(
                text in message for text in ("404", "entry not found", "no such file")
            )
            try:
                self.output.rm(relpath)
            except FileNotFoundError:
                pass
            if self.missing_asset_policy == "error" or not missing:
                raise
            log_throughput("asset_uploads_failed", 1, shard_id, unit="assets")
            return False
        log_throughput("assets_uploaded", 1, shard_id=shard_id, unit="assets")
        return True

    def _rewrite_path(
        self,
        value: object,
        *,
        shard_id: str,
        column_name: str,
        row_index: int,
        list_items: bool,
        storage: str,
    ) -> object:
        if value is None:
            self._window.submit_result(True)
            return None
        if list_items:
            if not isinstance(value, Sequence) or isinstance(
                value,
                (str, bytes, bytearray),
            ):
                raise TypeError(f"Asset column {column_name!r} expected list values")
            rewritten: list[object] = []
            copies: list[tuple[object, str]] | tuple[object, str] = []
            for item_index, item in enumerate(value):
                if item is None:
                    rewritten.append(None)
                    continue
                source_name = _asset_source_name(item, storage)
                relpath = self._asset_relpath(
                    source_name,
                    shard_id=shard_id,
                    column_name=column_name,
                    row_index=row_index,
                    item_index=item_index,
                )
                rewritten.append(self.output.abs_path(relpath))
                copies.append((item, relpath))
        else:
            source_name = _asset_source_name(value, storage)
            relpath = self._asset_relpath(
                source_name,
                shard_id=shard_id,
                column_name=column_name,
                row_index=row_index,
                item_index=None,
            )
            rewritten = self.output.abs_path(relpath)
            copies = (value, relpath)

        def copy_assets_sync() -> AssetCopyResult:
            if isinstance(copies, list):
                return [
                    self._copy_asset(item, storage, relpath, shard_id=shard_id)
                    for item, relpath in copies
                ]
            item, relpath = copies
            return self._copy_asset(item, storage, relpath, shard_id=shard_id)

        async def copy_assets() -> AssetCopyResult:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(io_executor(), copy_assets_sync)

        self._window.submit_blocking(copy_assets())
        return rewritten


@dataclass(slots=True)
class _BlobBlock:
    stream: IO[bytes]
    path: str
    size: int
    index: int


class BlobAssetManager:
    def __init__(
        self,
        output: DataFolder,
        *,
        config: BlobAssetConfig,
        filename_template: str,
    ) -> None:
        self.output = output
        self.config = config
        self.assets_subdir = config.subdir
        normalized_template = posixpath.normpath(filename_template)
        if normalized_template == self.assets_subdir or normalized_template.startswith(
            f"{self.assets_subdir}/"
        ):
            raise ValueError("filename_template must not write into asset subdir")
        self.missing_asset_policy = config.missing_policy
        self._asset_columns: dict[str, tuple[str, str, str]] = {}
        self._asset_column_segments: dict[str, str] = {}
        self._input_schema_set = False
        self._blocks: dict[tuple[str, str], _BlobBlock] = {}

    def set_input_schema(self, schema: pa.Schema | None) -> None:
        self._set_asset_columns(_asset_columns_from_schema(schema))
        self._input_schema_set = True

    def _set_asset_columns(
        self,
        columns: dict[str, tuple[str, str, str]],
    ) -> None:
        self._asset_columns = columns
        self._asset_column_segments = {}
        used_segments: set[str] = set()
        for column_name in columns:
            base = _SAFE_NAME_RE.sub("_", column_name).strip("._-") or "column"
            segment = base
            suffix = 2
            while segment in used_segments:
                segment = f"{base}_{suffix}"
                suffix += 1
            self._asset_column_segments[column_name] = segment
            used_segments.add(segment)

    def require_input_schema(self) -> None:
        if not self._input_schema_set:
            raise ValueError(
                "Row asset writing requires an input schema. Mark asset columns "
                "with dtypes=... or cast(...), or call set_input_schema(...)."
            )

    def output_schema(self, schema: pa.Schema | None) -> pa.Schema | None:
        return _asset_output_schema(schema, mode="blob")

    def _block_relpath(self, shard_id: str, column_name: str, index: int) -> str:
        attempt = f"{shard_id}__w{get_active_worker_token()}"
        segment = self._asset_column_segments[column_name]
        return f"{self.assets_subdir}/{attempt}/{segment}/{index:05d}.blob"

    def _block(
        self,
        shard_id: str,
        column_name: str,
        payload_size: int,
    ) -> _BlobBlock:
        key = (shard_id, column_name)
        block = self._blocks.get(key)
        if (
            block is not None
            and block.size > 0
            and block.size + payload_size > self.config.target_bytes
        ):
            block.stream.close()
            block = None
        if block is None:
            next_index = self._blocks[key].index + 1 if key in self._blocks else 0
            relpath = self._block_relpath(shard_id, column_name, next_index)
            open_kwargs = (
                {"size": max(self.config.target_bytes, payload_size)}
                if is_s3fs(self.output.fs)
                else {}
            )
            stream = self.output.open(relpath, mode="wb", **open_kwargs)
            block = _BlobBlock(
                stream=stream,
                path=self.output.abs_path(relpath),
                size=0,
                index=next_index,
            )
            self._blocks[key] = block
        return block

    @staticmethod
    def _source(value: object, storage: str) -> tuple[bytes | DataFile, int, int]:
        if storage == "bytes":
            if not isinstance(value, bytes):
                raise TypeError("bytes-backed asset value must be bytes")
            return value, 0, len(value)
        if storage == "bytes_with_path":
            if not isinstance(value, Mapping):
                raise TypeError("bytes_with_path asset value must be a mapping")
            mapping = cast(Mapping[str, object], value)
            data = mapping.get("bytes")
            path = mapping.get("path")
            if isinstance(data, bytes):
                return data, 0, len(data)
            if not isinstance(path, str) or not path:
                raise FileNotFoundError("asset has neither bytes nor a path")
            source = DataFile.resolve(path)
            return source, 0, int(source.fs.size(source.path))
        if storage == "blob_reference":
            if not isinstance(value, Mapping):
                raise TypeError("blob reference must be a mapping")
            mapping = cast(Mapping[str, object], value)
            path = mapping.get("path")
            offset = mapping.get("offset")
            size = mapping.get("size")
            if (
                not isinstance(path, str)
                or not path
                or not isinstance(offset, int)
                or isinstance(offset, bool)
                or offset < 0
                or not isinstance(size, int)
                or isinstance(size, bool)
                or size < 0
            ):
                raise ValueError("invalid blob reference")
            return DataFile.resolve(path), offset, size
        if storage == "path":
            if not isinstance(value, str) or not value:
                raise TypeError("path-backed asset value must be a path")
            source = DataFile.resolve(value)
            return source, 0, int(source.fs.size(source.path))
        raise TypeError(f"Asset storage {storage!r} cannot be written as a blob")

    def _append(
        self,
        value: object,
        *,
        shard_id: str,
        column_name: str,
        storage: str,
    ) -> tuple[object, bool]:
        if value is None:
            return None, True
        try:
            source, source_offset, size = self._source(value, storage)
            if isinstance(source, bytes):
                staged = None
            else:
                staged = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024)
                try:
                    remaining = size
                    with source.open("rb") as stream:
                        stream.seek(source_offset)
                        while remaining:
                            chunk = stream.read(min(remaining, 2 * 1024 * 1024))
                            if not chunk:
                                raise EOFError("asset ended before its declared size")
                            staged.write(chunk)
                            remaining -= len(chunk)
                    staged.seek(0)
                except Exception:
                    staged.close()
                    raise
            if staged is None:
                block = self._block(shard_id, column_name, size)
                output_offset = block.size
                block.stream.write(cast(bytes, source))
            else:
                try:
                    block = self._block(shard_id, column_name, size)
                    output_offset = block.size
                    while chunk := staged.read(2 * 1024 * 1024):
                        block.stream.write(chunk)
                finally:
                    staged.close()
            block.size += size
        except Exception as error:
            message = str(error).lower()
            missing = isinstance(error, FileNotFoundError) or any(
                text in message for text in ("404", "entry not found", "no such file")
            )
            if self.missing_asset_policy == "error" or not missing:
                raise
            log_throughput("asset_uploads_failed", 1, shard_id, unit="assets")
            return None, False
        log_throughput("assets_uploaded", 1, shard_id=shard_id, unit="assets")
        return {
            "path": block.path,
            "offset": output_offset,
            "size": size,
        }, True

    def _rewrite_value(
        self,
        value: object,
        *,
        shard_id: str,
        column_name: str,
        kind: str,
        storage: str,
    ) -> tuple[object, bool]:
        if kind == "scalar" or value is None:
            return self._append(
                value,
                shard_id=shard_id,
                column_name=column_name,
                storage=storage,
            )
        if not isinstance(value, Sequence) or isinstance(
            value, (str, bytes, bytearray)
        ):
            raise TypeError(f"Asset column {column_name!r} expected list values")
        rewritten: list[object] = []
        valid = True
        for item in value:
            next_value, copied = self._append(
                item,
                shard_id=shard_id,
                column_name=column_name,
                storage=storage,
            )
            rewritten.append(next_value)
            valid = valid and copied
        return rewritten, valid

    @staticmethod
    def _covers_entire_blob(
        references: Sequence[tuple[int, int]],
        source_size: int,
    ) -> bool:
        """Whether the union of byte references is exactly one source blob."""
        position = 0
        for offset, size in sorted(references):
            if (
                size < 0
                or offset < 0
                or offset > source_size
                or size > source_size - offset
            ):
                return False
            if offset > position:
                return False
            position = max(position, offset + size)
        return position == source_size

    def _coalesce_complete_blob_references(
        self,
        values: Sequence[object],
        *,
        shard_id: str,
        column_name: str,
    ) -> dict[int, object]:
        """Copy complete classic blob sources once and translate their offsets.

        This is deliberately an internal optimization of the existing blob
        reference writer.  A partial selection still uses the established
        per-range copier below, preserving its missing-asset policies.
        """
        if self.missing_asset_policy != "error":
            return {}
        grouped: dict[str, list[tuple[int, int, int]]] = {}
        for index, value in enumerate(values):
            if value is None or not isinstance(value, Mapping):
                continue
            path = value.get("path")
            offset = value.get("offset")
            size = value.get("size")
            if (
                isinstance(path, str)
                and path
                and isinstance(offset, int)
                and not isinstance(offset, bool)
                and isinstance(size, int)
                and not isinstance(size, bool)
            ):
                grouped.setdefault(path, []).append((index, offset, size))

        rewritten: dict[int, object] = {}
        for path, references in grouped.items():
            source = DataFile.resolve(path)
            source_size = int(source.fs.size(source.path))
            if not self._covers_entire_blob(
                [(offset, size) for _, offset, size in references], source_size
            ):
                continue
            block = self._block(shard_id, column_name, source_size)
            output_offset = block.size
            remaining = source_size
            with source.open("rb") as stream:
                while remaining:
                    chunk = stream.read(min(remaining, 8 * 1024 * 1024))
                    if not chunk:
                        raise EOFError("asset ended before its declared size")
                    block.stream.write(chunk)
                    remaining -= len(chunk)
            block.size += source_size
            for index, offset, size in references:
                rewritten[index] = {
                    "path": block.path,
                    "offset": output_offset + offset,
                    "size": size,
                }
            log_throughput("source_blobs_coalesced", 1, shard_id, unit="blobs")
            log_throughput("assets_uploaded", len(references), shard_id, unit="assets")
        return rewritten

    def rewrite_table(self, shard_id: str, table: pa.Table) -> pa.Table:
        columns = dict(self._asset_columns)
        columns.update(_asset_columns_from_schema(table.schema))
        if columns != self._asset_columns:
            self._set_asset_columns(columns)
        out = table
        keep = [True] * table.num_rows
        for column_name, (kind, storage, asset_type) in self._asset_columns.items():
            index = out.schema.get_field_index(column_name)
            if index < 0:
                continue
            original_values = out.column(index).to_pylist()
            coalesced = (
                self._coalesce_complete_blob_references(
                    original_values, shard_id=shard_id, column_name=column_name
                )
                if kind == "scalar" and storage == "blob_reference"
                else {}
            )
            values: list[object] = []
            for row_offset, value in enumerate(original_values):
                if row_offset in coalesced:
                    rewritten, valid = coalesced[row_offset], True
                else:
                    rewritten, valid = self._rewrite_value(
                        value,
                        shard_id=shard_id,
                        column_name=column_name,
                        kind=kind,
                        storage=storage,
                    )
                values.append(rewritten)
                if not valid and self.missing_asset_policy == "drop_row":
                    keep[row_offset] = False
            field = _asset_output_field(out.schema.field(index), kind, asset_type)
            out = out.set_column(index, field, pa.array(values, type=field.type))
        if self.missing_asset_policy == "drop_row" and not all(keep):
            out = out.filter(pa.array(keep, type=pa.bool_()))
        return out

    def rewrite_rows(self, shard_id: str, rows: Iterable[Row]) -> Iterable[Row]:
        self.require_input_schema()
        for row in rows:
            patch: dict[str, object] = {}
            valid = True
            for column_name, (
                kind,
                storage,
                _asset_type,
            ) in self._asset_columns.items():
                if column_name not in row:
                    continue
                rewritten, copied = self._rewrite_value(
                    row[column_name],
                    shard_id=shard_id,
                    column_name=column_name,
                    kind=kind,
                    storage=storage,
                )
                patch[column_name] = rewritten
                valid = valid and copied
            if not valid and self.missing_asset_policy == "drop_row":
                continue
            yield row.update(patch) if patch else row

    def close(self) -> None:
        for block in self._blocks.values():
            block.stream.close()
        self._blocks.clear()

    def on_shard_complete(self, shard_id: str) -> None:
        keys = [key for key in self._blocks if key[0] == shard_id]
        for key in keys:
            self._blocks.pop(key).stream.close()


ASSET_ATTEMPT_DIR_RE = re.compile(
    r"^(?P<shard_id>[0-9a-f]{12})__w(?P<worker_id>[0-9a-f]{12})$"
)


def _asset_columns_from_schema(
    schema: pa.Schema | None,
) -> dict[str, tuple[str, str, str]]:
    if schema is None:
        return {}
    columns: dict[str, tuple[str, str, str]] = {}
    for field in schema:
        storage = datatype.asset_storage(field)
        asset_type = datatype.asset_type(field)
        if storage is not None and asset_type is not None:
            columns[field.name] = ("scalar", storage, asset_type)
            continue
        field_type = field.type
        if not (
            pa.types.is_list(field_type)
            or pa.types.is_large_list(field_type)
            or pa.types.is_fixed_size_list(field_type)
        ):
            continue
        storage = datatype.asset_storage(field_type.value_field)
        asset_type = datatype.asset_type(field_type.value_field)
        if storage is not None and asset_type is not None:
            columns[field.name] = ("list", storage, asset_type)
    return columns


def _asset_source_name(value: object, storage: str) -> str:
    if storage == "path" and isinstance(value, str) and value:
        return value
    if storage in {"bytes_with_path", "blob_reference"} and isinstance(value, Mapping):
        path = cast(Mapping[str, object], value).get("path")
        if isinstance(path, str) and path:
            return path
    if storage == "bytes" and isinstance(value, bytes):
        return "asset"
    if storage == "bytes_with_path":
        return "asset"
    raise TypeError("asset value does not match its declared storage")


def _file_asset_output_field(field: pa.Field, kind: str, asset_type: str) -> pa.Field:
    value_field = datatype.asset_path(asset_type)
    if kind == "scalar":
        return pa.field(
            field.name,
            value_field.type,
            nullable=field.nullable,
            metadata=value_field.metadata,
        )
    field_type = field.type
    child = value_field.with_name(field_type.value_field.name)
    if pa.types.is_large_list(field_type):
        output_type = pa.large_list(child)
    elif pa.types.is_fixed_size_list(field_type):
        output_type = pa.list_(child, list_size=field_type.list_size)
    else:
        output_type = pa.list_(child)
    return pa.field(
        field.name,
        output_type,
        nullable=field.nullable,
        metadata=field.metadata,
    )


def _asset_output_field(field: pa.Field, kind: str, asset_type: str) -> pa.Field:
    value_field = datatype.blob_reference(asset_type)
    if kind == "scalar":
        return pa.field(
            field.name,
            value_field.type,
            nullable=field.nullable,
            metadata=value_field.metadata,
        )
    field_type = field.type
    child = value_field.with_name(field_type.value_field.name)
    if pa.types.is_large_list(field_type):
        output_type = pa.large_list(child)
    elif pa.types.is_fixed_size_list(field_type):
        output_type = pa.list_(child, list_size=field_type.list_size)
    else:
        output_type = pa.list_(child)
    return pa.field(
        field.name,
        output_type,
        nullable=field.nullable,
        metadata=field.metadata,
    )


def _asset_output_schema(
    schema: pa.Schema | None,
    *,
    mode: Literal["file", "blob"],
) -> pa.Schema | None:
    if schema is None:
        return None
    columns = _asset_columns_from_schema(schema)
    fields: list[pa.Field] = []
    for field in schema:
        column = columns.get(field.name)
        if column is None:
            fields.append(field)
            continue
        kind, _storage, asset_type = column
        fields.append(
            _file_asset_output_field(field, kind, asset_type)
            if mode == "file"
            else _asset_output_field(field, kind, asset_type)
        )
    return pa.schema(fields, metadata=schema.metadata)


def _set_null_value(value: object, result: AssetCopyResult) -> object:
    # Scalar copy result: failed copy nulls the whole cell.
    if isinstance(result, bool):
        return value if result else None
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError("Asset list result cannot be applied to non-list value")
    # List copy result: failed copy nulls only that path element. Existing None
    # values do not produce copy results, so they do not consume result entries.
    result_iter = iter(result)
    out: list[object] = []
    for item in value:
        if item is None:
            out.append(None)
            continue
        out.append(item if next(result_iter) else None)
    try:
        next(result_iter)
    except StopIteration:
        return out
    raise ValueError("Asset list result has more entries than list path values")


def cleanup_rejected_asset_attempts(
    output: DataFolder,
    assets_subdir: str,
    keep_pairs: set[tuple[str, str]],
) -> None:
    asset_prefix = f"{assets_subdir.rstrip('/')}/"
    try:
        asset_paths = output.find(assets_subdir)
    except FileNotFoundError:
        return
    attempt_dirs: set[str] = set()
    for rel_path in asset_paths:
        if not rel_path.startswith(asset_prefix):
            continue
        attempt_dir = rel_path[len(asset_prefix) :].split("/", maxsplit=1)[0]
        match = ASSET_ATTEMPT_DIR_RE.fullmatch(attempt_dir)
        if (
            match is not None
            and (
                match.group("shard_id"),
                match.group("worker_id"),
            )
            not in keep_pairs
        ):
            attempt_dirs.add(f"{asset_prefix}{attempt_dir}")
    for path in sorted(attempt_dirs):
        try:
            output.rm(path, recursive=True)
        except FileNotFoundError:
            continue


__all__ = [
    "AssetUploadManager",
    "AssetWriteConfig",
    "BlobAssetConfig",
    "BlobAssetManager",
    "FileAssetConfig",
    "MissingAssetPolicy",
    "asset_config_to_plan",
    "cleanup_rejected_asset_attempts",
]
