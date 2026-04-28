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
from refiner.pipeline.data.shard import RowRangeDescriptor, Shard
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
        self.split = split
        self.resolve_filepaths = resolve_filepaths
        self._explicit_dtypes = dtypes
        self.hf_token = hf_token
        self.timeout = float(timeout)
        self.target_shard_bytes = target_shard_bytes
        self.num_shards = num_shards
        self.arrow_batch_size = arrow_batch_size
        self.columns_to_read = tuple(columns_to_read) if columns_to_read else None

        try:
            from datasets import get_dataset_config_info
        except ImportError as e:
            raise ImportError(
                "read_hf_dataset requires the optional Hugging Face dependencies. "
                "Install with `macrodata-refiner[huggingface]`."
            ) from e

        info_kwargs: dict[str, Any] = {}
        if config is not None:
            info_kwargs["config_name"] = config
        if self.hf_token is not None:
            info_kwargs["token"] = self.hf_token
        info = get_dataset_config_info(self.repo, **info_kwargs)
        self.config: str = str(info.config_name or config or "default")

        file_feature_dtypes = {
            "Image": datatype.image_file,
            "Audio": datatype.audio_file,
            "Video": datatype.video_file,
            "Pdf": datatype.pdf_file,
        }
        inferred_dtypes: dict[str, pa.Field] = {}
        features = info.features or {}
        if isinstance(features, Mapping):
            for name, feature in features.items():
                dtype_factory = file_feature_dtypes.get(type(feature).__name__)
                if isinstance(name, str) and dtype_factory is not None:
                    inferred_dtypes[name] = dtype_factory()

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
        self._fallback_dataset: object | None = None
        self._fallback_num_shards: int | None = None
        self._parquet_shard_count: int | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "config": self.config,
            "split": self.split,
            "resolve_filepaths": self.resolve_filepaths,
            "dtypes": list(self.dtypes) if self.dtypes else None,
        }

    def list_shards(self) -> list[Shard]:
        if self._fallback_num_shards is not None:
            return self._fallback_shards(self._fallback_num_shards)
        try:
            shards = self._parquet_reader().list_shards()
        except Exception:
            return self._fallback_shards(self.num_shards)
        self._parquet_shard_count = len(shards)
        return shards

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        if isinstance(shard.descriptor, RowRangeDescriptor):
            yield from self._read_fallback_shard(shard)
            return

        try:
            for unit in self._parquet_reader().read_shard(shard):
                if isinstance(unit, Tabular):
                    yield unit.with_table(self._finish_table(unit.table))
                else:
                    yield unit
        except Exception:
            if shard.global_ordinal is None:
                raise
            fallback_count = self._parquet_shard_count or self.num_shards
            self._fallback_shards(fallback_count)
            yield from self._read_fallback_shard(
                Shard.from_row_range(
                    start=int(shard.global_ordinal),
                    end=int(shard.global_ordinal) + 1,
                    global_ordinal=shard.global_ordinal,
                )
            )

    @property
    def schema(self) -> pa.Schema | None:
        return datatype.schema_with_dtypes(None, self.dtypes)

    def _parquet_reader(self) -> ParquetReader:
        if self._delegate is not None:
            return self._delegate

        # URL discovery is intentionally separate from scanning; ParquetReader still
        # owns byte planning, projection, filtering, and Arrow conversion.
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
                col
                for col in self.filter.referenced_columns().difference(columns_to_read)
                if col != self.file_path_column
            )
            columns_to_read = (*columns_to_read, *extra_columns)
        if columns_to_read is not None and self.file_path_column is not None:
            columns_to_read = tuple(
                col for col in columns_to_read if col != self.file_path_column
            )
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

    def _fallback_shards(self, num_shards: int | None) -> list[Shard]:
        dataset = self._load_fallback_dataset()
        available = int(getattr(dataset, "num_shards", 1) or 1)
        wanted = int(num_shards or available)
        if wanted > available:
            raise ValueError(
                f"Hugging Face fallback for {self.repo!r} has only {available} "
                f"source shard(s), but {wanted} were requested."
            )
        self._fallback_num_shards = wanted
        return [
            Shard.from_row_range(start=i, end=i + 1, global_ordinal=i)
            for i in range(wanted)
        ]

    def _load_fallback_dataset(self) -> object:
        if self._fallback_dataset is None:
            from datasets import load_dataset

            kwargs: dict[str, Any] = {
                "split": self.split,
                "streaming": True,
            }
            if self.hf_token is not None:
                kwargs["token"] = self.hf_token
            self._fallback_dataset = cast(Any, load_dataset)(
                self.repo,
                self.config,
                **kwargs,
            )
        return self._fallback_dataset

    def _read_fallback_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        descriptor = shard.descriptor
        if not isinstance(descriptor, RowRangeDescriptor):
            raise TypeError("Hugging Face fallback requires row-range shards")
        total = self._fallback_num_shards
        if total is None:
            total = len(self._fallback_shards(self.num_shards))
        dataset = cast(Any, self._load_fallback_dataset()).shard(
            num_shards=total,
            index=int(descriptor.start),
        )
        dataset = dataset.with_format("arrow")
        for batch in dataset.iter(batch_size=self.arrow_batch_size):
            if isinstance(batch, pa.Table):
                table = batch
            elif isinstance(batch, Mapping):
                table = pa.table(cast(Mapping[str, object], batch))
            else:
                raise TypeError(
                    "Hugging Face fallback expected Arrow batches from datasets"
                )
            table = self._finish_table(table)
            if table.num_rows > 0:
                yield Tabular(table)

    def _finish_table(self, table: pa.Table) -> pa.Table:
        if self._file_dtypes:
            extracted_file_dtypes: dict[str, object] = {}
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
            columns = list(dict.fromkeys(self.columns_to_read))
            if (
                self.file_path_column is not None
                and self.file_path_column in table.column_names
                and self.file_path_column not in columns
            ):
                columns.append(self.file_path_column)
            table = table.select([col for col in columns if col in table.column_names])
        if self.resolve_filepaths:
            table = resolve_hf_filepaths(table, self.repo)
        return table


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
    config: str,
    split: str,
    *,
    hf_token: str | None,
    timeout: float,
) -> list[str]:
    """Return generated Parquet shard URLs for one HF dataset config/split."""

    split_names = (split, f"partial-{split}")
    repo_quoted = quote(repo, safe="/")
    config_quoted = quote(config, safe="")

    try:
        payload: object | None = _get_json(
            f"{_HF_HUB}/api/datasets/{repo_quoted}/parquet",
            hf_token=hf_token,
            timeout=timeout,
        )
    except httpx.HTTPError:
        payload = None
    if isinstance(payload, Mapping):
        config_payload = cast(Mapping[object, object], payload).get(config)
        if isinstance(config_payload, Mapping):
            for split_name in split_names:
                values = cast(Mapping[object, object], config_payload).get(split_name)
                if isinstance(values, list):
                    urls = [url for url in values if isinstance(url, str)]
                    if urls:
                        return urls

    for split_name in split_names:
        split_quoted = quote(split_name, safe="")
        try:
            payload = _get_json(
                f"{_HF_HUB}/api/datasets/"
                f"{repo_quoted}/parquet/{config_quoted}/{split_quoted}",
                hf_token=hf_token,
                timeout=timeout,
            )
        except httpx.HTTPError:
            continue
        if isinstance(payload, list):
            urls = [url for url in payload if isinstance(url, str)]
            if urls:
                return urls

    try:
        payload = _get_json(
            f"{_HF_DATASETS_SERVER}/parquet?dataset={repo_quoted}"
            f"&config={config_quoted}&split={quote(split, safe='')}",
            hf_token=hf_token,
            timeout=timeout,
        )
    except httpx.HTTPError:
        payload = None
    if isinstance(payload, Mapping):
        parquet_files = cast(Mapping[object, object], payload).get("parquet_files")
        urls = []
        if isinstance(parquet_files, list):
            for item in parquet_files:
                if not isinstance(item, Mapping):
                    continue
                item = cast(Mapping[str, object], item)
                url = item.get("url")
                if (
                    item.get("config") == config
                    and item.get("split") in split_names
                    and isinstance(url, str)
                ):
                    urls.append(url)
        if urls:
            return urls

    try:
        payload = _get_json(
            f"{_HF_HUB}/api/datasets/{repo_quoted}/tree/main/"
            f"{config_quoted}?recursive=false",
            hf_token=hf_token,
            timeout=timeout,
        )
    except httpx.HTTPError:
        payload = None
    if isinstance(payload, list):
        urls = []
        for item in payload:
            if not isinstance(item, Mapping):
                continue
            item = cast(Mapping[str, object], item)
            path = item.get("path")
            filename = path.rsplit("/", maxsplit=1)[-1] if isinstance(path, str) else ""
            if (
                item.get("type") == "file"
                and isinstance(path, str)
                and path.endswith(".parquet")
                and any(
                    filename.startswith(f"{split_name}-") for split_name in split_names
                )
            ):
                urls.append(
                    f"{_HF_HUB}/datasets/{repo_quoted}/resolve/main/"
                    f"{quote(path, safe='/')}"
                )
        if urls:
            return urls

    for split_name in split_names:
        split_quoted = quote(split_name, safe="")
        try:
            payload = _get_json(
                f"{_HF_HUB}/api/datasets/{repo_quoted}/tree/"
                f"refs%2Fconvert%2Fparquet/{config_quoted}/"
                f"{split_quoted}?recursive=true",
                hf_token=hf_token,
                timeout=timeout,
            )
        except httpx.HTTPError:
            continue
        urls = []
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, Mapping):
                    continue
                item = cast(Mapping[str, object], item)
                path = item.get("path")
                if item.get("type") != "file" or not isinstance(path, str):
                    continue
                urls.append(
                    f"{_HF_HUB}/datasets/{repo_quoted}/resolve/"
                    f"refs%2Fconvert%2Fparquet/{quote(path, safe='/')}"
                )
            if urls:
                return urls

    raise FileNotFoundError(
        f"No Hugging Face parquet shards for {repo!r} config={config!r} split={split!r}"
    )


def _get_json(url: str, *, hf_token: str | None, timeout: float) -> object:
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else None
    response = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    return response.json()


__all__ = [
    "HFDatasetReader",
]
