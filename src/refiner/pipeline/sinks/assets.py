from __future__ import annotations

import asyncio
from collections.abc import Iterable, Iterator, Sequence
import posixpath
import re
from typing import Literal
from urllib.parse import unquote, urlsplit

import pyarrow as pa
import pyarrow.compute as pc

from refiner.execution.asyncio.runtime import io_executor
from refiner.execution.asyncio.window import AsyncWindow
from refiner.io import DataFile
from refiner.io.datafolder import DataFolder
from refiner.pipeline.data import datatype
from refiner.pipeline.data.row import Row
from refiner.pipeline.data.tabular import set_or_append_column
from refiner.worker.context import get_active_worker_token
from refiner.worker.metrics.api import log_throughput

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
MissingAssetPolicy = Literal["error", "drop_row", "set_null"]


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
        self._window = AsyncWindow[bool](
            max_in_flight=max_uploads_in_flight,
            preserve_order=True,
        )
        self._next_row_index: dict[str, int] = {}
        self._asset_columns: dict[str, str] = {}
        self._asset_column_segments: dict[str, str] = {}
        self._input_schema_set = False
        self.missing_asset_policy = missing_asset_policy

    def set_input_schema(self, schema: pa.Schema | None) -> None:
        self._set_asset_columns(asset_columns_from_schema(schema))
        self._input_schema_set = True

    def _set_asset_columns(self, columns: dict[str, str]) -> None:
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

    def rewrite_table(self, shard_id: str, table: pa.Table) -> pa.Table:
        start = self._next_row_index.get(shard_id, 0)
        out = table
        pending: list[tuple[str, pa.Field, list[object]]] = []
        # Tables may already carry asset metadata, while row-derived tables rely on
        # the schema passed through set_input_schema().
        columns = dict(self._asset_columns)
        columns.update(asset_columns_from_schema(table.schema))
        if columns != self._asset_columns:
            self._set_asset_columns(columns)
            columns = self._asset_columns
        for column_name, kind in columns.items():
            idx = out.schema.get_field_index(column_name)
            if idx < 0:
                continue
            field = out.schema.field(idx)
            values = out.column(idx).to_pylist()
            rewritten = [
                self._rewrite_asset_path(
                    value,
                    shard_id=shard_id,
                    column_name=column_name,
                    row_index=start + row_offset,
                    list_items=kind == "list",
                )
                for row_offset, value in enumerate(values)
            ]
            pending.append((column_name, field, rewritten))

        upload_results = iter(self._window.flush())
        keep: pa.Array | None = None
        for column_name, field, rewritten in pending:
            finalized = [
                self._finalize_asset_value(value, upload_results) for value in rewritten
            ]
            rewritten_array = pa.array(finalized, type=field.type)
            out = set_or_append_column(
                out,
                column_name,
                rewritten_array,
            )
            if self.missing_asset_policy == "drop_row" and rewritten_array.null_count:
                valid = pc.call_function("is_valid", [rewritten_array])
                keep = valid if keep is None else pc.call_function("and", [keep, valid])
        self._next_row_index[shard_id] = start + table.num_rows
        if keep is not None:
            rows_before_filter = out.num_rows
            out = out.filter(keep)
            dropped = rows_before_filter - out.num_rows
            if dropped:
                log_throughput("asset_rows_dropped", dropped, shard_id, unit="rows")
        return out

    def rewrite_rows(
        self,
        shard_id: str,
        rows: Iterable[Row],
    ) -> Iterable[Row]:
        self.require_input_schema()
        if not self._asset_columns:
            return rows

        start = self._next_row_index.get(shard_id, 0)
        pending: list[tuple[Row, dict[str, object], dict[str, object]]] = []
        row_count = 0
        for row_index, row in enumerate(rows, start=start):
            row_count += 1
            patch: dict[str, object] = {}
            original: dict[str, object] = {}
            for column_name, kind in self._asset_columns.items():
                if column_name not in row:
                    continue
                original[column_name] = row[column_name]
                value = self._rewrite_asset_path(
                    row[column_name],
                    shard_id=shard_id,
                    column_name=column_name,
                    row_index=row_index,
                    list_items=kind == "list",
                )
                patch[column_name] = value
            pending.append((row, patch, original))
        upload_results = iter(self._window.flush())
        out: list[Row] = []
        dropped = 0
        for row, patch, original in pending:
            patch = {
                column_name: self._finalize_asset_value(value, upload_results)
                for column_name, value in patch.items()
            }
            drop_row = False
            for column_name, value in patch.items():
                if (
                    self.missing_asset_policy == "drop_row"
                    and original[column_name] is not None
                    and value is None
                ):
                    drop_row = True
                    break
            if drop_row:
                dropped += 1
                continue
            out.append(row.update(patch) if patch else row)
        self._next_row_index[shard_id] = start + row_count
        if dropped:
            log_throughput("asset_rows_dropped", dropped, shard_id, unit="rows")
        return out

    def flush(self) -> None:
        self._window.flush()

    def _finalize_asset_value(
        self,
        value: object,
        upload_results: Iterator[bool],
    ) -> object:
        if isinstance(value, str):
            return value if next(upload_results) else None
        if value is None:
            return None
        if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
            return value

        out: list[object] = []
        for item in value:
            if item is None:
                out.append(None)
                continue
            if isinstance(item, str) and not next(upload_results):
                # A missing list element drops the row for drop_row, but set_null
                # preserves the list and nulls only the missing item.
                if self.missing_asset_policy == "drop_row":
                    return None
                out.append(None)
            else:
                out.append(item)
        return out

    def _rewrite_asset_path(
        self,
        value: object,
        *,
        shard_id: str,
        column_name: str,
        row_index: int,
        list_items: bool,
    ) -> object:
        if value is None:
            return None
        if list_items:
            if not isinstance(value, Sequence) or isinstance(
                value,
                (str, bytes, bytearray),
            ):
                raise TypeError(f"Asset column {column_name!r} expected list values")
            out: list[object] = []
            for item_index, item in enumerate(value):
                if item is None:
                    out.append(None)
                    continue
                rewritten = self._rewrite_path(
                    item,
                    shard_id=shard_id,
                    column_name=column_name,
                    row_index=row_index,
                    item_index=item_index,
                )
                out.append(rewritten)
            return out
        return self._rewrite_path(
            value,
            shard_id=shard_id,
            column_name=column_name,
            row_index=row_index,
            item_index=None,
        )

    def _rewrite_path(
        self,
        value: object,
        *,
        shard_id: str,
        column_name: str,
        row_index: int,
        item_index: int | None,
    ) -> str:
        if not isinstance(value, str) or not value:
            raise TypeError(f"Asset value for column {column_name!r} must be a path")

        basename = unquote(posixpath.basename(urlsplit(value).path.rstrip("/")))
        basename = _SAFE_NAME_RE.sub("_", basename.replace("\\", "_")).strip("._-")
        if not basename:
            basename = "asset"
        prefix = f"{row_index}" if item_index is None else f"{row_index}-{item_index}"
        column_segment = self._asset_column_segments[column_name]
        # Attempt directories are keyed by shard and worker so reducers can delete
        # whole failed attempts without inspecting individual asset files.
        attempt_dir = f"{shard_id}__w{get_active_worker_token()}"
        relpath = (
            f"{self.assets_subdir}/{attempt_dir}/{column_segment}/{prefix}-{basename}"
        )

        output_path = self.output.abs_path(relpath)

        def copy_asset_sync() -> bool:
            try:
                DataFile.resolve(value).copy(self.output.file(relpath))
            except Exception as e:
                message = str(e).lower()
                missing = isinstance(e, FileNotFoundError) or any(
                    text in message
                    for text in ("404", "entry not found", "no such file")
                )
                if self.missing_asset_policy == "error" or not missing:
                    raise
                log_throughput("assets_uploads_failed", 1, shard_id, unit="assets")
                return False
            log_throughput("assets_uploaded", 1, shard_id=shard_id, unit="assets")
            return True

        async def copy_asset() -> bool:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(io_executor(), copy_asset_sync)

        self._window.submit_blocking(copy_asset())
        return output_path


ASSET_ATTEMPT_DIR_RE = re.compile(
    r"^(?P<shard_id>[0-9a-f]{12})__w(?P<worker_id>[0-9a-f]{12})$"
)


def asset_columns_from_schema(schema: pa.Schema | None) -> dict[str, str]:
    if schema is None:
        return {}
    columns: dict[str, str] = {}
    for field in schema:
        if datatype.is_asset_path_field(field):
            columns[field.name] = "scalar"
            continue
        field_type = field.type
        if (
            pa.types.is_list(field_type)
            or pa.types.is_large_list(field_type)
            or pa.types.is_fixed_size_list(field_type)
        ) and datatype.is_asset_path_field(field_type.value_field):
            columns[field.name] = "list"
    return columns


__all__ = ["AssetUploadManager", "MissingAssetPolicy"]
