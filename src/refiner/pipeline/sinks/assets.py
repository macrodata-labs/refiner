from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
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
        result_columns: list[tuple[str, pa.Field]] = []
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
            column = out.column(idx)
            values = column.to_pylist()
            rewritten = [
                self._rewrite_path(
                    value,
                    shard_id=shard_id,
                    column_name=column_name,
                    row_index=start + row_offset,
                    list_items=kind == "list",
                )
                for row_offset, value in enumerate(values)
            ]
            rewritten_array = pa.array(rewritten, type=field.type)
            out = set_or_append_column(
                out,
                column_name,
                rewritten_array,
            )
            result_columns.append((column_name, field))
        # Do not expose output rows that point at assets until those copies have
        # completed; otherwise a later copy failure leaves dangling references.
        results = self._window.drain()
        self._next_row_index[shard_id] = start + table.num_rows
        keep: pa.Array | None = None
        offset = 0
        for column_name, field in result_columns:
            column_results = results[offset : offset + table.num_rows]
            offset += table.num_rows
            if all(column_results):
                continue
            valid = pa.array(column_results, type=pa.bool_())
            if self.missing_asset_policy == "set_null":
                idx = out.schema.get_field_index(column_name)
                column = out.column(idx)
                out = set_or_append_column(
                    out,
                    column_name,
                    pc.call_function(
                        "if_else",
                        [valid, column, pa.nulls(out.num_rows, type=field.type)],
                    ),
                )
            elif self.missing_asset_policy == "drop_row":
                keep = valid if keep is None else pc.call_function("and", [keep, valid])
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
        pending_rows: list[tuple[Row, dict[str, object], list[str]]] = []
        row_count = 0
        for row_index, row in enumerate(rows, start=start):
            row_count += 1
            patch: dict[str, object] = {}
            result_columns: list[str] = []
            for column_name, kind in self._asset_columns.items():
                if column_name not in row:
                    continue
                value = self._rewrite_path(
                    row[column_name],
                    shard_id=shard_id,
                    column_name=column_name,
                    row_index=row_index,
                    list_items=kind == "list",
                )
                patch[column_name] = value
                result_columns.append(column_name)
            pending_rows.append((row, patch, result_columns))
        # Keep row-mode JSONL writes atomic at the block level for asset references.
        results = self._window.drain()
        self._next_row_index[shard_id] = start + row_count
        out: list[Row] = []
        dropped = 0
        offset = 0
        for row, patch, result_columns in pending_rows:
            row_results = results[offset : offset + len(result_columns)]
            offset += len(result_columns)
            if self.missing_asset_policy == "drop_row" and not all(row_results):
                dropped += 1
                continue
            if self.missing_asset_policy == "set_null":
                for column_name, copied in zip(
                    result_columns,
                    row_results,
                    strict=True,
                ):
                    if not copied:
                        patch[column_name] = None
            out.append(row.update(patch) if patch else row)
        if dropped:
            log_throughput("asset_rows_dropped", dropped, shard_id, unit="rows")
        return out

    def flush(self) -> None:
        self._window.drain()

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

    def _copy_asset(self, value: str, relpath: str, *, shard_id: str) -> bool:
        try:
            DataFile.resolve(value).copy(self.output.file(relpath))
        except Exception as e:
            message = str(e).lower()
            missing = isinstance(e, FileNotFoundError) or any(
                text in message for text in ("404", "entry not found", "no such file")
            )
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
            copies: list[tuple[str, str]] | tuple[str, str] = []
            for item_index, item in enumerate(value):
                if item is None:
                    rewritten.append(None)
                    continue
                if not isinstance(item, str) or not item:
                    raise TypeError(
                        f"Asset value for column {column_name!r} must be a path"
                    )
                relpath = self._asset_relpath(
                    item,
                    shard_id=shard_id,
                    column_name=column_name,
                    row_index=row_index,
                    item_index=item_index,
                )
                rewritten.append(self.output.abs_path(relpath))
                copies.append((item, relpath))
        else:
            if not isinstance(value, str) or not value:
                raise TypeError(
                    f"Asset value for column {column_name!r} must be a path"
                )
            relpath = self._asset_relpath(
                value,
                shard_id=shard_id,
                column_name=column_name,
                row_index=row_index,
                item_index=None,
            )
            rewritten = self.output.abs_path(relpath)
            copies = (value, relpath)

        def copy_assets_sync() -> bool:
            if isinstance(copies, list):
                return all(
                    self._copy_asset(item, relpath, shard_id=shard_id)
                    for item, relpath in copies
                )
            item, relpath = copies
            return self._copy_asset(item, relpath, shard_id=shard_id)

        async def copy_assets() -> bool:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(io_executor(), copy_assets_sync)

        self._window.submit_blocking(copy_assets())
        return rewritten


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
