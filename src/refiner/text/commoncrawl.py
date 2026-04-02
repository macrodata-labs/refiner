from __future__ import annotations

import asyncio
import gzip
import io
import threading
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Literal
from warcio.archiveiterator import ArchiveIterator

from refiner.execution.asyncio.runtime import io_executor
from refiner.execution.asyncio.window import AsyncWindow
from refiner.io import DataFile, DataFolder
from refiner.pipeline import RefinerPipeline
from refiner.pipeline.expressions import Expr, col
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.data.row import DictRow, Row
from refiner.pipeline.data.shard import FilePartsDescriptor, Shard
from refiner.pipeline.sources.base import BaseSource
from refiner.pipeline.sources.readers.parquet import ParquetReader
from refiner.pipeline.sources.readers.base import BaseReader
from refiner.pipeline.sources.readers.utils import DEFAULT_TARGET_SHARD_BYTES
from refiner.utils.imports import check_required_dependencies
from refiner.worker.metrics.api import log_throughput

_DEFAULT_S3_BASE_URL = "s3://commoncrawl"
_DEFAULT_HTTPS_BASE_URL = "https://data.commoncrawl.org"
_DEFAULT_FILE_PATH_COLUMN = object()
_WARC_INDEX_PATHS = "cc-index-table.paths.gz"
_WARC_INDEX_SUBSET = "subset=warc/"
_INDEX_COLUMNS = (
    "url",
    "warc_filename",
    "warc_record_offset",
    "warc_record_length",
)
_COMMONCRAWL_HTML_MIME_TYPES = ("text/html", "application/xhtml+xml")
_COMMONCRAWL_PDF_MIME_TYPES = (
    "application/pdf",
    "application/x-pdf",
    "application/acrobat",
    "applications/vnd.pdf",
    "text/pdf",
)
_DEFAULT_WARC_OUTPUT_FIELDS = (
    "WARC-Record-ID",
    "WARC-Target-URI",
    "WARC-Date",
    "WARC-Type",
    "WARC-Identified-Payload-Type",
    "Content-Type",
    "content_bytes",
)
_SPECIAL_WARC_OUTPUT_FIELDS = frozenset({"content_bytes"})

# HTML/XHTML pages, matching the Common Crawl index MIME metadata.
# This is fully pushdownable.
filter_html: Expr = col("content_mime_type").is_in(
    list(_COMMONCRAWL_HTML_MIME_TYPES)
) | (
    col("content_mime_type").is_null()
    & col("content_mime_detected").is_in(list(_COMMONCRAWL_HTML_MIME_TYPES))
)

# MIME-confirmed PDFs only.
# This is fully pushdownable.
filter_pdf: Expr = col("content_mime_detected").is_in(
    list(_COMMONCRAWL_PDF_MIME_TYPES)
) | col("content_mime_type").is_in(list(_COMMONCRAWL_PDF_MIME_TYPES))

# Rows marked as truncated in the Common Crawl index metadata.
# This is fully pushdownable.
filter_truncated: Expr = col("content_truncated").is_not_null()

# Rows not marked as truncated in the Common Crawl index metadata.
# This is fully pushdownable.
filter_not_truncated: Expr = col("content_truncated").is_null()


def filter_domain_suffixes(
    *suffixes: str,
) -> Expr:
    """Build an index filter on Common Crawl's host registry suffix column.

    Example:
        `filter_domain_suffixes("pt", "com.pt")`
        keeps pages whose hostname ends in `.pt` or `.com.pt`.
    """
    normalized = [suffix.strip().lstrip(".") for suffix in suffixes if suffix.strip()]
    if not normalized:
        raise ValueError("suffixes must contain at least one non-empty suffix")
    return col("url_host_registry_suffix").is_in(normalized)


def read_commoncrawl(
    dumps: str | Sequence[str],
    *,
    format: Literal["warc", "wet"] = "warc",
    output_fields: Literal["all"] | Sequence[str] = _DEFAULT_WARC_OUTPUT_FIELDS,
    segments: str | Sequence[str] | None = None,
    base_url: str | None = None,
    use_https: bool = False,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    num_shards: int | None = None,
    file_path_column: object = _DEFAULT_FILE_PATH_COLUMN,
) -> RefinerPipeline:
    """Create a file-backed pipeline over Common Crawl WARC or WET data.

    Resolution stays lazy: dump ids are turned into concrete file globs only when
    `list_shards()` is called.

    Args:
        dumps: One or more Common Crawl dump ids such as `CC-MAIN-2025-13`.
        format: Whether to read raw `warc` files or `wet` files.
        output_fields: WARC/HTTP fields to emit for each fetched record, or `"all"`
            to include all original WARC headers, all non-conflicting HTTP headers,
            and `content_bytes`. Common Crawl's index schema is documented at
            https://commoncrawl.org/columnar-index
        segments: Optional segment ids used to narrow the file globs.
        base_url: Optional Common Crawl root override.
        use_https: If True, default to the HTTPS mirror instead of `s3://commoncrawl`.
        target_shard_bytes: Approximate shard size used for file-backed shard planning.
        num_shards: Optional explicit number of planned shards.
        file_path_column: Output column name for the absolute source file path, or `None`.
    """
    source: BaseSource = CommonCrawlReader(
        dumps=dumps,
        format=format,
        segments=segments,
        output_fields=output_fields,
        base_url=base_url,
        use_https=use_https,
        target_shard_bytes=target_shard_bytes,
        num_shards=num_shards,
        file_path_column=file_path_column,
    )
    return RefinerPipeline(source=source)


def read_commoncrawl_from_index(
    dumps: str | Sequence[str],
    *,
    segments: str | Sequence[str] | None = None,
    filter: Expr | None = None,
    filter_fn: Callable[[Row], bool] | None = None,
    output_fields: Literal["all"] | Sequence[str] = _DEFAULT_WARC_OUTPUT_FIELDS,
    base_url: str | None = None,
    use_https: bool = False,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    num_shards: int | None = None,
    file_path_column: str | None = "warc_path",
    max_inflight: int = 4,
) -> RefinerPipeline:
    """Create a pipeline over Common Crawl WARC data via the parquet index.

    This path uses Common Crawl's columnar index parquet files to plan and filter
    WARC record fetches directly.

    Args:
        dumps: One or more Common Crawl dump ids such as `CC-MAIN-2025-13`.
        segments: Optional segment ids to keep after index rows are loaded.
        filter: Optional index-row expression filter passed to the parquet index reader.
        filter_fn: Optional Python predicate over index rows, applied before WARC fetches.
        output_fields: WARC/HTTP fields to emit for each fetched record, or `"all"`
            to include all original WARC headers, all non-conflicting HTTP headers,
            and `content_bytes`. Common Crawl's index schema is documented at
            https://commoncrawl.org/columnar-index
        base_url: Optional Common Crawl root override.
        use_https: If True, default to the HTTPS mirror instead of `s3://commoncrawl`.
        target_shard_bytes: Approximate shard size used for index parquet planning.
        num_shards: Optional explicit number of planned shards.
        file_path_column: Output column name for the absolute WARC file path, or `None`.
        max_inflight: Maximum number of WARC fetches to keep in flight per shard.
    """
    return RefinerPipeline(
        source=CommonCrawlWarcIndexSource(
            dumps=dumps,
            segments=segments,
            filter=filter,
            filter_fn=filter_fn,
            output_fields=output_fields,
            base_url=base_url,
            use_https=use_https,
            target_shard_bytes=target_shard_bytes,
            num_shards=num_shards,
            file_path_column=file_path_column,
            max_inflight=max_inflight,
        )
    )


class CommonCrawlReader(BaseReader):
    """Common Crawl file-backed reader over globbed WARC or WET files."""

    name = "read_commoncrawl"

    def __init__(
        self,
        *,
        dumps: str | Sequence[str],
        format: Literal["warc", "wet"] = "warc",
        segments: str | Sequence[str] | None = None,
        output_fields: Literal["all"] | Sequence[str] = _DEFAULT_WARC_OUTPUT_FIELDS,
        base_url: str | None = None,
        use_https: bool = False,
        target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
        num_shards: int | None = None,
        file_path_column: object = _DEFAULT_FILE_PATH_COLUMN,
    ) -> None:
        check_required_dependencies("read_commoncrawl", ["warcio"], dist="text")
        if format not in {"warc", "wet"}:
            raise ValueError("format must be 'warc' or 'wet'")
        if isinstance(dumps, str):
            self.dumps = (dumps,)
        else:
            self.dumps = tuple(str(dump) for dump in dumps)
        if not self.dumps:
            raise ValueError("dumps must contain at least one Common Crawl dump id")
        self.format = format
        self.segments = _normalize_segments(segments)
        if output_fields == "all":
            self.output_fields: Literal["all"] | tuple[str, ...] = "all"
        else:
            self.output_fields = tuple(
                str(field).strip() for field in output_fields if str(field).strip()
            )
            if not self.output_fields:
                raise ValueError("output_fields must contain at least one field")
        self.base_url = (
            str(base_url)
            if base_url is not None
            else (_DEFAULT_HTTPS_BASE_URL if use_https else _DEFAULT_S3_BASE_URL)
        )
        self.root = DataFolder.resolve(self.base_url)
        self._archive_iterator = ArchiveIterator
        if file_path_column is _DEFAULT_FILE_PATH_COLUMN:
            resolved_file_path_column: str | None = (
                "warc_path" if self.format == "warc" else "wet_path"
            )
        elif file_path_column is None or isinstance(file_path_column, str):
            resolved_file_path_column = file_path_column
        else:
            raise TypeError("file_path_column must be a string or None")
        super().__init__(
            self._source_globs(),
            target_shard_bytes=target_shard_bytes,
            num_shards=num_shards,
            file_path_column=resolved_file_path_column,
            split_by_bytes=False,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "dumps": list(self.dumps),
            "format": self.format,
            "segments": list(self.segments) if self.segments is not None else None,
            "output_fields": (
                "all" if self.output_fields == "all" else list(self.output_fields)
            ),
            "target_shard_bytes": self.target_shard_bytes,
            "num_shards": self.num_shards,
            "file_path_column": self.file_path_column,
            "base_url": self.base_url,
        }

    def read_shard(self, shard: Shard) -> Iterator[Row]:
        """Read one Common Crawl file shard by streaming each selected file in full."""
        descriptor = shard.descriptor
        if not isinstance(descriptor, FilePartsDescriptor):
            raise TypeError("Common Crawl reader requires file-backed shards")
        for part in descriptor.parts:
            if part.end != -1:
                raise ValueError("Common Crawl files are expected to be atomic")
            source = self.fileset.resolve_file(part.source_index, part.path)
            with source.open(mode="rb") as raw:
                stream: io.BufferedIOBase
                if source.path.lower().endswith(".gz"):
                    stream = gzip.GzipFile(fileobj=raw)
                else:
                    stream = raw
                for record in self._archive_iterator(stream):
                    payload = _warc_record_to_row(
                        record, output_fields=self.output_fields
                    )
                    if payload is None:
                        continue
                    yield DictRow(self._with_file_path(payload, source))

    def _source_globs(self) -> tuple[tuple[str, Any], ...]:
        """Build the dump/segment-specific WARC or WET glob inputs for BaseReader."""
        rel_globs: list[str] = []
        suffix = "warc/*.warc.gz" if self.format == "warc" else "wet/*.warc.wet.gz"
        for dump in self.dumps:
            if self.segments is None:
                rel_globs.append(f"crawl-data/{dump}/segments/*/{suffix}")
            else:
                for segment in self.segments:
                    rel_globs.append(f"crawl-data/{dump}/segments/{segment}/{suffix}")
        return tuple(
            (self.root.abs_path(rel_glob), self.root.fs) for rel_glob in rel_globs
        )


class CommonCrawlWarcIndexSource(BaseSource):
    """Common Crawl WARC source backed by the columnar index parquet files.

    Planning and index-row filtering happen on parquet metadata first; the
    surviving index rows are then used to fetch individual WARC records.
    """

    name = "read_commoncrawl_from_index"

    def __init__(
        self,
        *,
        dumps: str | Sequence[str],
        segments: str | Sequence[str] | None = None,
        filter: Expr | None = None,
        filter_fn: Callable[[Row], bool] | None = None,
        output_fields: Literal["all"] | Sequence[str] = _DEFAULT_WARC_OUTPUT_FIELDS,
        base_url: str | None = None,
        use_https: bool = False,
        target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
        num_shards: int | None = None,
        file_path_column: str | None = "warc_path",
        max_inflight: int = 4,
    ) -> None:
        check_required_dependencies(
            "read_commoncrawl_from_index", ["warcio"], dist="text"
        )
        if isinstance(dumps, str):
            self.dumps = (dumps,)
        else:
            self.dumps = tuple(str(dump) for dump in dumps)
        if not self.dumps:
            raise ValueError("dumps must contain at least one Common Crawl dump id")
        self.segments = _normalize_segments(segments)
        self.filter = filter
        self.filter_fn = filter_fn
        if output_fields == "all":
            self.output_fields: Literal["all"] | tuple[str, ...] = "all"
        else:
            self.output_fields = tuple(
                str(field).strip() for field in output_fields if str(field).strip()
            )
            if not self.output_fields:
                raise ValueError("output_fields must contain at least one field")
        self.base_url = (
            str(base_url)
            if base_url is not None
            else (_DEFAULT_HTTPS_BASE_URL if use_https else _DEFAULT_S3_BASE_URL)
        )
        self.root = DataFolder.resolve(self.base_url)
        self._archive_iterator = ArchiveIterator
        self.target_shard_bytes = target_shard_bytes
        self.num_shards = num_shards
        self.file_path_column = file_path_column
        self.max_inflight = max(1, int(max_inflight))
        self._index_reader: ParquetReader | None = None
        self._thread_local = threading.local()

    def describe(self) -> dict[str, Any]:
        return {
            "dumps": list(self.dumps),
            "format": "warc",
            "segments": list(self.segments) if self.segments is not None else None,
            "filter": self.filter.to_code() if self.filter is not None else None,
            "filter_fn": self.filter_fn is not None,
            "output_fields": (
                "all" if self.output_fields == "all" else list(self.output_fields)
            ),
            "target_shard_bytes": self.target_shard_bytes,
            "num_shards": self.num_shards,
            "file_path_column": self.file_path_column,
            "max_inflight": self.max_inflight,
            "base_url": self.base_url,
        }

    def list_shards(self) -> list[Shard]:
        return self._get_index_reader().list_shards()

    def read_shard(self, shard: Shard) -> Iterator[Row]:
        """Read one index-backed WARC shard and emit selected record fields."""
        if self.max_inflight == 1:
            for row in self._iter_index_rows(shard):
                payload = self._fetch_index_row_payload(row)
                if payload is not None:
                    yield DictRow(payload)
            return

        window = AsyncWindow[dict[str, Any] | None](
            max_in_flight=self.max_inflight,
            preserve_order=True,
        )
        for row in self._iter_index_rows(shard):
            window.submit_blocking(self._fetch_index_row_payload_async(row))
            for payload in window.poll():
                if payload is not None:
                    yield DictRow(payload)
        for payload in window.flush():
            if payload is not None:
                yield DictRow(payload)

    def _iter_index_rows(self, shard: Shard) -> Iterator[Row]:
        for unit in self._get_index_reader().read_shard(shard):
            if not isinstance(unit, Tabular):
                raise TypeError(
                    "Common Crawl WARC index reader requires tabular batches"
                )
            for row in unit.to_rows():
                if self.filter_fn is not None and not self.filter_fn(row):
                    log_throughput(
                        "filter_fn_rows_filtered",
                        1,
                        shard_id=shard.id,
                        unit="rows",
                    )
                    log_throughput(
                        "total_rows_filtered",
                        1,
                        shard_id=shard.id,
                        unit="rows",
                    )
                    continue
                yield row

    def _get_index_reader(self) -> ParquetReader:
        """Lazily build the parquet reader over Common Crawl index partitions."""
        if self._index_reader is not None:
            return self._index_reader
        # Common Crawl WARC sharding is delegated directly to ParquetReader
        # over the index parquet files.
        index_filter = self.filter
        if self.segments is not None:
            segment_filter = col("warc_segment").is_in(list(self.segments))
            index_filter = (
                segment_filter
                if index_filter is None
                else (index_filter & segment_filter)
            )
        self._index_reader = ParquetReader(
            self._resolve_index_files(),
            target_shard_bytes=self.target_shard_bytes,
            num_shards=self.num_shards,
            columns_to_read=None if self.filter_fn is not None else _INDEX_COLUMNS,
            filter=index_filter,
            file_path_column=None,
        )
        return self._index_reader

    def _resolve_index_files(self) -> tuple[DataFile, ...]:
        """Resolve `subset=warc` parquet index files for the requested dumps."""
        files: list[DataFile] = []
        for dump in self.dumps:
            with self.root.file(f"crawl-data/{dump}/{_WARC_INDEX_PATHS}").open(
                mode="rb"
            ) as raw:
                with gzip.GzipFile(fileobj=raw) as gz:
                    rel_paths = gz.read().decode("utf-8").splitlines()
            for rel_path in rel_paths:
                rel_path = rel_path.strip()
                if not rel_path:
                    continue
                if _WARC_INDEX_SUBSET not in rel_path:
                    continue
                files.append(self.root.file(rel_path.lstrip("/")))
        if not files:
            raise FileNotFoundError("No Common Crawl index parquet files resolved")
        return tuple(files)

    def _get_thread_file_handle(
        self, file: DataFile, *, mode: str = "rb", force_reopen: bool = False
    ) -> tuple[Any, bool]:
        current_file = getattr(self._thread_local, "open_file", None)
        current_fh = getattr(self._thread_local, "open_fh", None)
        if not force_reopen and current_file == file and current_fh is not None:
            return current_fh, False
        if current_fh is not None:
            try:
                current_fh.close()
            except Exception:
                pass
        fh = file.open(mode=mode)
        self._thread_local.open_file = file
        self._thread_local.open_fh = fh
        return fh, True

    def _read_indexed_record(self, *, warc_filename: str, offset: int, length: int):
        """Fetch one indexed WARC record from a compressed or uncompressed member span."""
        source = self.root.file(warc_filename.lstrip("/"))
        fh, _ = self._get_thread_file_handle(source, mode="rb")
        try:
            fh.seek(offset)
        except Exception:
            fh, _ = self._get_thread_file_handle(source, mode="rb", force_reopen=True)
            fh.seek(offset)
        member = fh.read(length)
        stream: io.BufferedIOBase
        if warc_filename.lower().endswith(".gz"):
            stream = gzip.GzipFile(fileobj=io.BytesIO(member))
        else:
            stream = io.BytesIO(member)
        iterator = self._archive_iterator(stream)
        return next(iterator)

    def _fetch_index_row_payload(self, row: Row) -> dict[str, Any] | None:
        warc_filename = str(row["warc_filename"])
        record = self._read_indexed_record(
            warc_filename=warc_filename,
            offset=int(row["warc_record_offset"]),
            length=int(row["warc_record_length"]),
        )
        payload = _warc_record_to_row(record, output_fields=self.output_fields)
        if payload is None:
            return None
        if self.file_path_column is not None:
            payload.setdefault(
                self.file_path_column,
                self.root.file(warc_filename.lstrip("/")).abs_path(),
            )
        return payload

    async def _fetch_index_row_payload_async(self, row: Row) -> dict[str, Any] | None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            io_executor(), self._fetch_index_row_payload, row
        )


def _normalize_segments(
    segments: str | Sequence[str] | None,
) -> tuple[str, ...] | None:
    """Normalize segment ids into a stripped tuple or `None`."""
    if segments is None:
        return None
    if isinstance(segments, str):
        values = (segments,)
    else:
        values = tuple(str(segment) for segment in segments)
    normalized = tuple(segment.strip() for segment in values if segment.strip())
    if not normalized:
        raise ValueError("segments must contain at least one non-empty segment id")
    return normalized


def _warc_record_to_row(
    record,
    *,
    output_fields: Literal["all"] | Sequence[str],
) -> dict[str, Any] | None:
    """Extract the requested fields from a WARC record."""
    rec_headers = record.rec_headers
    http_headers = record.http_headers
    if output_fields == "all":
        payload: dict[str, Any] = {}
        for field, value in rec_headers.headers:
            _add_header(payload, field, value)
        payload.setdefault("WARC-Type", record.rec_type)
        if http_headers is not None:
            for field, value in http_headers.headers:
                _add_header(payload, field, value)
        payload["content_bytes"] = record.content_stream().read()
        return payload

    payload: dict[str, Any] = {}
    if "content_bytes" in output_fields:
        payload["content_bytes"] = record.content_stream().read()

    for field in output_fields:
        if field in _SPECIAL_WARC_OUTPUT_FIELDS:
            continue
        value = rec_headers.get_header(field)
        if value is None and field == "WARC-Type":
            value = record.rec_type
        if value is None and http_headers is not None:
            value = http_headers.get_header(field)
        payload[field] = value
    return payload


def _add_header(payload: dict[str, Any], field: str, value: str) -> None:
    current = payload.get(field)
    if current is None:
        payload[field] = value
    elif isinstance(current, list):
        current.append(value)
    else:
        payload[field] = [current, value]


__all__ = [
    "CommonCrawlReader",
    "CommonCrawlWarcIndexSource",
    "filter_domain_suffixes",
    "filter_html",
    "filter_not_truncated",
    "filter_pdf",
    "filter_truncated",
    "read_commoncrawl",
    "read_commoncrawl_from_index",
]
