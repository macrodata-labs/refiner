from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any
from typing import cast
from urllib.parse import quote

import httpx
import pyarrow as pa
import pyarrow.compute as pc

from refiner.pipeline.data import datatype
from refiner.pipeline.data.datatype import DTypeMapping
from refiner.pipeline.data.shard import Shard
from refiner.pipeline.data.tabular import Tabular, filter_table, set_or_append_column
from refiner.pipeline.expressions import Expr
from refiner.pipeline.sources.base import BaseSource, SourceUnit
from refiner.pipeline.sources.readers.parquet import ParquetReader
from refiner.pipeline.sources.readers.utils import DEFAULT_TARGET_SHARD_BYTES

_HF_HUB = "https://huggingface.co"
_HF_DATASETS_SERVER = "https://datasets-server.huggingface.co"
_HF_DATASET_URI_PREFIX = "hf://datasets"


class HFDatasetReader(BaseSource):
    """Read Hugging Face datasets through the Hub's generated Parquet shards."""

    name = "read_hf_dataset"

    def __init__(
        self,
        repo: str,
        config: str | None = None,
        split: str = "train",
        *,
        resolve_filepaths: bool = True,
        dtypes: DTypeMapping | None = None,
        hf_token: str | None = None,
        timeout: float = 30.0,
        target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
        num_shards: int | None = None,
        arrow_batch_size: int = 65536,
        columns_to_read: Sequence[str] | None = None,
        filter: Expr | None = None,
        split_row_groups: bool = False,
        file_path_column: str | None = "file_path",
    ):
        self.repo = repo
        self.config = config
        self.split = split
        self.resolve_filepaths = resolve_filepaths
        self._explicit_dtypes = dtypes
        self.hf_token = hf_token
        self.timeout = float(timeout)
        self.target_shard_bytes = target_shard_bytes
        self.num_shards = num_shards
        self.arrow_batch_size = arrow_batch_size
        self.columns_to_read = tuple(columns_to_read) if columns_to_read else None

        # Refiner file dtypes, explicit or inferred from HF features, need
        # post-Parquet handling because HF media features are often structs;
        # normal Arrow dtypes can be pushed into ParquetReader.
        inferred_dtypes: dict[str, pa.Field] = {}
        url = f"{_HF_DATASETS_SERVER}/info?dataset={quote(self.repo, safe='/')}"
        payload = _get_json(url, hf_token=self.hf_token, timeout=self.timeout)
        payload = (
            cast(Mapping[str, Any], payload) if isinstance(payload, Mapping) else {}
        )
        dataset_info = payload.get("dataset_info", {})
        dataset_info = (
            cast(Mapping[str, Any], dataset_info)
            if isinstance(dataset_info, Mapping)
            else {}
        )
        info = dataset_info.get(self.config or "default")
        if info is None and self.config is None and len(dataset_info) == 1:
            info = next(iter(dataset_info.values()))
        features = info.get("features", {}) if isinstance(info, Mapping) else {}
        if isinstance(features, Mapping):
            for name, feature in features.items():
                if not isinstance(name, str) or not isinstance(feature, Mapping):
                    continue
                feature_type = feature.get("_type")
                if feature_type == "Image":
                    inferred_dtypes[name] = datatype.image_file()
                elif feature_type == "Audio":
                    inferred_dtypes[name] = datatype.audio_file()
                elif feature_type == "Video":
                    inferred_dtypes[name] = datatype.video_file()

        effective_dtypes = dict(inferred_dtypes)
        if dtypes:
            effective_dtypes.update(dtypes)
        self.dtypes = effective_dtypes or None
        file_dtypes = {
            name: dtype
            for name, dtype in effective_dtypes.items()
            if _is_file_dtype(dtype)
        }
        non_file_dtypes = {
            name: dtype
            for name, dtype in effective_dtypes.items()
            if not _is_file_dtype(dtype)
        }
        self._file_dtypes = file_dtypes or None
        self._non_file_dtypes = non_file_dtypes or None
        self._explicit_file_dtype_names = {
            name
            for name, dtype in (self._explicit_dtypes or {}).items()
            if _is_file_dtype(dtype)
        }

        # Filters on file columns must run after extracting the struct "path" field.
        # Other filters can be delegated to the Parquet reader for pushdown.
        filter_uses_file_dtype = (
            filter is not None
            and self._file_dtypes is not None
            and bool(self._file_dtypes.keys() & filter.referenced_columns())
        )
        self.filter = filter if filter_uses_file_dtype else None
        self._parquet_filter = (
            None if filter is None or filter_uses_file_dtype else filter
        )
        self.split_row_groups = split_row_groups
        self.file_path_column = file_path_column
        self._delegate: ParquetReader | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "config": self.config or "default",
            "split": self.split,
            "resolve_filepaths": self.resolve_filepaths,
            "dtypes": list(self.dtypes) if self.dtypes else None,
        }

    def list_shards(self) -> list[Shard]:
        return self._parquet_reader().list_shards()

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        for unit in self._parquet_reader().read_shard(shard):
            if not isinstance(unit, Tabular):
                yield unit
                continue
            table = unit.table
            if self._file_dtypes:
                extracted_file_dtypes: dict[str, object] = {}
                # Convert HF media structs with a "path" field into the string path
                # columns that Refiner file dtypes represent.
                for name, dtype in self._file_dtypes.items():
                    idx = table.schema.get_field_index(name)
                    if idx < 0:
                        continue
                    field = table.schema.field(idx)
                    if (
                        pa.types.is_struct(field.type)
                        and field.type.get_field_index("path") >= 0
                    ):
                        path = pc.call_function(
                            "struct_field",
                            [table.column(idx)],
                            options=pc.StructFieldOptions(["path"]),
                        )
                        table = set_or_append_column(table, name, path)
                        extracted_file_dtypes[name] = dtype
                        continue
                    if pa.types.is_string(field.type) or pa.types.is_large_string(
                        field.type
                    ):
                        extracted_file_dtypes[name] = dtype
                        continue
                    if name in self._explicit_file_dtype_names:
                        extracted_file_dtypes[name] = dtype
                if extracted_file_dtypes:
                    table = datatype.apply_dtypes_to_table(
                        table,
                        extracted_file_dtypes,
                        strict=False,
                    )
            if self.filter is not None:
                table = filter_table(table, self.filter)
            if self.columns_to_read is not None:
                # Keep ParquetReader's synthetic source path column when present so
                # downstream sinks can still trace rows back to their shard file.
                columns = list(self.columns_to_read)
                if (
                    self.file_path_column is not None
                    and self.file_path_column in table.column_names
                ):
                    columns.append(self.file_path_column)
                table = table.select(columns)
            if self.resolve_filepaths:
                table = resolve_hf_filepaths(table, self.repo)
            yield unit.with_table(table)

    @property
    def schema(self) -> pa.Schema | None:
        return datatype.schema_with_dtypes(None, self.dtypes)

    def _parquet_reader(self) -> ParquetReader:
        if self._delegate is not None:
            return self._delegate

        # HF exposes generated Parquet files through a small metadata endpoint; the
        # ParquetReader handles actual scanning, sharding, projection, and pushdown.
        urls = _list_parquet_urls(
            self.repo,
            self.config,
            self.split,
            hf_token=self.hf_token,
            timeout=self.timeout,
        )
        columns_to_read = self.columns_to_read
        if self.filter is not None and columns_to_read is not None:
            # Columns referenced by a delayed HF-side filter still need to be loaded,
            # even if the final projection would otherwise exclude them.
            extra_columns = sorted(
                self.filter.referenced_columns().difference(columns_to_read)
            )
            columns_to_read = (*columns_to_read, *extra_columns)
        self._delegate = ParquetReader(
            urls,
            target_shard_bytes=self.target_shard_bytes,
            num_shards=self.num_shards,
            arrow_batch_size=self.arrow_batch_size,
            columns_to_read=columns_to_read,
            filter=self._parquet_filter,
            split_row_groups=self.split_row_groups,
            file_path_column=self.file_path_column,
            dtypes=self._non_file_dtypes,
            storage_options=(
                {"headers": {"Authorization": f"Bearer {self.hf_token}"}}
                if self.hf_token is not None
                else None
            ),
        )
        return self._delegate


def _is_file_dtype(dtype: object) -> bool:
    return isinstance(dtype, pa.Field) and datatype.is_file_field(dtype)


def resolve_hf_filepaths(table: pa.Table, repo: str) -> pa.Table:
    """Resolve relative file columns against the HF dataset repository root."""

    out = table
    for idx, field in enumerate(table.schema):
        if not datatype.is_file_field(field):
            continue
        column = table.column(idx)

        # Treat local absolute paths and any scheme:// URI as already resolved.
        # find_substring is intentionally used instead of regex for the hot path.
        local_absolute = pc.call_function(
            "starts_with",
            [column],
            options=pc.MatchSubstringOptions("/"),
        )
        protocol_position = pc.call_function(
            "find_substring",
            [column],
            options=pc.MatchSubstringOptions("://"),
        )
        remote_absolute = pc.call_function(
            "greater_equal",
            [protocol_position, pa.scalar(0, type=pa.int32())],
        )
        relative = pc.call_function(
            "and",
            [
                pc.fill_null(pc.call_function("invert", [local_absolute]), False),
                pc.fill_null(pc.call_function("invert", [remote_absolute]), False),
            ],
        )
        if not bool(pc.call_function("any", [relative]).as_py()):
            continue

        # HF file feature paths sometimes start with ./; normalize those before
        # constructing the hf://datasets/{repo}/... reference.
        stripped = pc.call_function(
            "replace_substring_regex",
            [column],
            options=pc.ReplaceSubstringOptions(pattern=r"^(?:\./)+", replacement=""),
        )
        resolved = pc.call_function(
            "binary_join_element_wise",
            [
                pa.scalar(f"{_HF_DATASET_URI_PREFIX}/{repo}/"),
                stripped,
                pa.scalar(""),
            ],
        )
        out = set_or_append_column(
            out,
            field.name,
            pc.call_function("if_else", [relative, resolved, column]),
        )
    return out


def _list_parquet_urls(
    repo: str,
    config: str | None,
    split: str,
    *,
    hf_token: str | None,
    timeout: float,
) -> list[str]:
    """Return generated Parquet shard URLs for one HF dataset config/split."""

    url = f"{_HF_DATASETS_SERVER}/parquet?dataset={quote(repo, safe='/')}&config={quote(config or 'default', safe='')}&split={quote(split, safe='')}"
    payload = _get_json(url, hf_token=hf_token, timeout=timeout)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Unexpected Hugging Face parquet response for {repo!r}")
    payload = cast(Mapping[str, object], payload)
    parquet_files = payload.get("parquet_files")
    if not isinstance(parquet_files, list):
        raise ValueError(f"Unexpected Hugging Face parquet response for {repo!r}")
    urls: list[str] = []
    for item in parquet_files:
        if not isinstance(item, Mapping):
            continue
        item = cast(Mapping[str, object], item)
        url = item.get("url")
        if (
            item.get("config") == (config or "default")
            and item.get("split") == split
            and isinstance(url, str)
        ):
            urls.append(url)
    if not urls:
        urls = _list_parquet_urls_from_repo_tree(
            repo,
            config,
            split,
            hf_token=hf_token,
            timeout=timeout,
        )
    if not urls:
        urls = _list_parquet_urls_from_convert_tree(
            repo,
            config,
            split,
            hf_token=hf_token,
            timeout=timeout,
        )
    if not urls:
        raise FileNotFoundError(
            f"No Hugging Face parquet shards for {repo!r} config={config or 'default'!r} split={split!r}"
        )
    return urls


def _list_parquet_urls_from_repo_tree(
    repo: str,
    config: str | None,
    split: str,
    *,
    hf_token: str | None,
    timeout: float,
) -> list[str]:
    config_part = quote(config or "default", safe="")
    url = f"{_HF_HUB}/api/datasets/{quote(repo, safe='/')}/tree/main/{config_part}?recursive=false"
    payload = _get_json(url, hf_token=hf_token, timeout=timeout)
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected Hugging Face tree response for {repo!r}")

    urls: list[str] = []
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        item = cast(Mapping[str, object], item)
        path = item.get("path")
        if (
            item.get("type") == "file"
            and isinstance(path, str)
            and path.rsplit("/", maxsplit=1)[-1].startswith(f"{split}-")
            and path.endswith(".parquet")
        ):
            urls.append(
                f"{_HF_HUB}/datasets/{quote(repo, safe='/')}/resolve/main/{quote(path, safe='/')}"
            )
    return urls


def _list_parquet_urls_from_convert_tree(
    repo: str,
    config: str | None,
    split: str,
    *,
    hf_token: str | None,
    timeout: float,
) -> list[str]:
    config_part = quote(config or "default", safe="")
    split_part = quote(split, safe="")
    url = f"{_HF_HUB}/api/datasets/{quote(repo, safe='/')}/tree/refs%2Fconvert%2Fparquet/{config_part}/{split_part}?recursive=true"
    payload = _get_json(url, hf_token=hf_token, timeout=timeout)
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected Hugging Face tree response for {repo!r}")

    urls: list[str] = []
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        item = cast(Mapping[str, object], item)
        path = item.get("path")
        if item.get("type") == "file" and isinstance(path, str):
            urls.append(
                f"{_HF_HUB}/datasets/{quote(repo, safe='/')}/resolve/refs%2Fconvert%2Fparquet/{quote(path, safe='/')}"
            )
    return urls


def _get_json(url: str, *, hf_token: str | None, timeout: float) -> object:
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else None
    response = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    return response.json()


__all__ = [
    "HFDatasetReader",
]
