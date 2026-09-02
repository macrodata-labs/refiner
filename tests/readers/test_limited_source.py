from __future__ import annotations

import inspect
from collections.abc import Iterator, Sequence
from pathlib import Path

import cloudpickle
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from refiner.pipeline import (
    RefinerPipeline,
    load_lance,
    read_csv,
    read_files,
    read_hdf5,
    read_hf_dataset,
    read_json,
    read_jsonl,
    read_lerobot,
    read_mcap,
    read_parquet,
    read_tfds,
    read_tfrecords,
    read_videos,
    read_zarr,
)
from refiner.pipeline.data.row import DictRow
from refiner.pipeline.data.shard import RowRangeDescriptor, Shard
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.expressions import col
from refiner.pipeline.sources.base import BaseSource, SourceUnit
from refiner.pipeline.sources.limited import LimitedSource, limit_source
from refiner.pipeline.sources.readers.files import FilesReader
from refiner.pipeline.sources.readers.lerobot import LeRobotEpisodeReader
from refiner.pipeline.sources.readers.mcap import McapReader
from refiner.pipeline.sources.readers.parquet import ParquetReader
from refiner.pipeline.sources.readers.tfrecord import TfrecordReader


class _RecordingSource(BaseSource):
    name = "recording"

    def __init__(self) -> None:
        self.list_calls = 0
        self.read_starts: list[int] = []
        self.units_started: list[str] = []

    @property
    def schema(self) -> pa.Schema:
        return pa.schema([pa.field("x", pa.int64())])

    def required_refiner_extras(self) -> tuple[str, ...]:
        return ("s3",)

    def describe(self) -> dict[str, object]:
        return {"path": "recording://rows"}

    def list_shards(self) -> list[Shard]:
        self.list_calls += 1
        return [
            Shard.from_row_range(start=index, end=index + 1, global_ordinal=index)
            for index in range(3)
        ]

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        descriptor = shard.descriptor
        assert isinstance(descriptor, RowRangeDescriptor)
        self.read_starts.append(descriptor.start)
        if descriptor.start == 0:
            self.units_started.append("table-0")
            yield Tabular(pa.table({"x": [0, 1]}))
            return
        if descriptor.start == 1:
            self.units_started.append("row-2")
            yield DictRow({"x": 2})
            self.units_started.append("table-3")
            yield Tabular(pa.table({"x": [3, 4]}))
            return
        self.units_started.append("row-5")
        yield DictRow({"x": 5})


class _SideDataTabular(Tabular):
    def __init__(self, table: pa.Table, side_data: tuple[str, ...]) -> None:
        super().__init__(table)
        self.side_data = side_data

    @property
    def needs_row_indices(self) -> bool:
        return True

    def with_table(
        self,
        table: pa.Table,
        *,
        row_indices: Sequence[int] | None = None,
    ) -> "_SideDataTabular":
        if row_indices is None:
            if table.num_rows != len(self.side_data):
                raise ValueError("side data must stay row-aligned")
            side_data = self.side_data
        else:
            side_data = tuple(self.side_data[int(index)] for index in row_indices)
        return _SideDataTabular(table, side_data)


class _SideDataSource(BaseSource):
    name = "side-data"

    def list_shards(self) -> list[Shard]:
        return [Shard.from_row_range(start=0, end=1, global_ordinal=0)]

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        del shard
        yield _SideDataTabular(pa.table({"x": [0, 1]}), ("zero", "one"))


def _values(pipeline: RefinerPipeline) -> list[int]:
    return [int(row["x"]) for row in pipeline.iter_rows()]


def test_limited_source_applies_one_global_limit_across_source_shards() -> None:
    source = _RecordingSource()
    limited = LimitedSource(source, max_rows=3)

    shards = limited.list_shards()
    rows = list(RefinerPipeline(source=limited).iter_rows())

    assert len(shards) == 1
    assert [int(row["x"]) for row in rows] == [0, 1, 2]
    assert [row.source_row_id for row in rows] == [0, 1, 2]
    assert {row.shard_id for row in rows} == {shards[0].id}
    assert source.list_calls == 1
    assert source.read_starts == [0, 1]
    assert source.units_started == ["table-0", "row-2"]


def test_limited_source_claim_carries_plan_to_serialized_worker() -> None:
    pipeline = RefinerPipeline(source=LimitedSource(_RecordingSource(), max_rows=3))
    worker_payload = cloudpickle.dumps(pipeline)
    claimed_shard = Shard.from_dict(pipeline.list_shards()[0].to_dict())

    worker_pipeline = cloudpickle.loads(worker_payload)
    units = list(worker_pipeline.source.read_shard(claimed_shard))

    assert units[0].table.column("x").to_pylist() == [0, 1]
    assert units[1]["x"] == 2
    assert worker_pipeline.source.source.list_calls == 0


def test_limited_source_slices_only_the_final_tabular_unit() -> None:
    source = _RecordingSource()

    assert _values(RefinerPipeline(source=LimitedSource(source, max_rows=1))) == [0]
    assert source.units_started == ["table-0"]


def test_limited_source_preserves_row_aligned_tabular_side_data() -> None:
    limited = LimitedSource(_SideDataSource(), max_rows=1)

    unit = next(limited.read_shard(limited.list_shards()[0]))

    assert isinstance(unit, _SideDataTabular)
    assert unit.table.column("x").to_pylist() == [0]
    assert unit.side_data == ("zero",)


def test_limited_source_zero_avoids_underlying_planning_and_reads() -> None:
    source = _RecordingSource()
    limited = LimitedSource(source, max_rows=0)

    assert limited.list_shards() == []
    assert _values(RefinerPipeline(source=limited)) == []
    assert source.list_calls == 0
    assert source.read_starts == []


def test_limited_source_delegates_public_source_metadata() -> None:
    source = _RecordingSource()
    limited = LimitedSource(source, max_rows=2)

    assert limited.name == "recording"
    assert limited.schema == source.schema
    assert limited.required_refiner_extras() == ("s3",)
    assert limited.describe() == {
        "path": "recording://rows",
        "max_rows": 2,
    }


def test_limit_source_validates_and_avoids_a_wrapper_without_a_limit() -> None:
    source = _RecordingSource()

    assert limit_source(source, None) is source
    with pytest.raises(ValueError, match="max_rows must be >= 0"):
        limit_source(source, -1)


@pytest.mark.parametrize(
    "reader",
    [
        read_csv,
        read_json,
        read_jsonl,
        read_files,
        read_videos,
        read_hdf5,
        load_lance,
        read_zarr,
        read_mcap,
        read_parquet,
        read_hf_dataset,
        read_lerobot,
        read_tfrecords,
        read_tfds,
    ],
)
def test_builtin_reader_functions_expose_max_rows(reader) -> None:
    assert "max_rows" in inspect.signature(reader).parameters


def test_parquet_max_rows_is_global_and_applies_after_reader_filter(
    tmp_path: Path,
) -> None:
    paths = []
    for file_index in range(3):
        path = tmp_path / f"part-{file_index}.parquet"
        start = file_index * 4
        pq.write_table(pa.table({"x": list(range(start, start + 4))}), path)
        paths.append(path)

    pipeline = read_parquet(
        paths,
        num_shards=3,
        max_rows=3,
        filter=col("x") > 4,
        file_path_column=None,
    )

    assert _values(pipeline) == [5, 6, 7]
    assert len(pipeline.list_shards()) == 1
    assert isinstance(pipeline.source, LimitedSource)
    assert isinstance(pipeline.source.source, ParquetReader)
    assert pipeline.source.source.arrow_batch_size == 3


def test_csv_and_jsonl_max_rows_stop_across_planned_shards(tmp_path: Path) -> None:
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text("x\n" + "".join(f"{value}\n" for value in range(8)))
    jsonl_path = tmp_path / "rows.jsonl"
    jsonl_path.write_text("".join(f'{{"x": {value}}}\n' for value in range(8)))

    assert _values(read_csv(csv_path, target_shard_bytes=4, max_rows=3)) == [0, 1, 2]
    assert _values(read_jsonl(jsonl_path, target_shard_bytes=10, max_rows=3)) == [
        0,
        1,
        2,
    ]


def test_file_max_rows_bounds_content_read_concurrency(tmp_path: Path) -> None:
    paths = []
    for index in range(3):
        path = tmp_path / f"asset-{index}.bin"
        path.write_bytes(bytes([index]))
        paths.append(path)

    pipeline = read_files(
        paths,
        content_column="content",
        size_column=None,
        max_in_flight=8,
        max_rows=1,
    )

    rows = list(pipeline.iter_rows())
    assert len(rows) == 1
    assert rows[0]["content"] == b"\x00"
    assert isinstance(pipeline.source, LimitedSource)
    assert isinstance(pipeline.source.source, FilesReader)
    assert pipeline.source.source.max_in_flight == 1


def test_lerobot_max_rows_bounds_episode_hydration_batch(tmp_path: Path) -> None:
    pipeline = read_lerobot(tmp_path, max_rows=3)

    assert isinstance(pipeline.source, LimitedSource)
    assert isinstance(pipeline.source.source, LeRobotEpisodeReader)
    assert pipeline.source.source.arrow_batch_size == 3


def test_mcap_max_rows_streams_split_episodes(tmp_path: Path) -> None:
    pipeline = read_mcap(
        tmp_path / "unused.mcap",
        max_rows=1,
        episode_splitting={"time_gap_s": 0.5},
    )

    assert isinstance(pipeline.source, LimitedSource)
    assert isinstance(pipeline.source.source, McapReader)
    assert pipeline.source.source.stream_episodes is True


def test_mcap_max_rows_does_not_stream_single_episode(tmp_path: Path) -> None:
    pipeline = read_mcap(tmp_path / "unused.mcap", max_rows=1)

    assert isinstance(pipeline.source, LimitedSource)
    assert isinstance(pipeline.source.source, McapReader)
    assert pipeline.source.source.stream_episodes is False


def test_tfrecord_max_rows_bounds_eager_input_windows(tmp_path: Path) -> None:
    pipeline = read_tfrecords(
        tmp_path / "unused.tfrecord",
        features={},
        batch_size=1024,
        num_parallel_calls=8,
        prefetch=-1,
        max_rows=3,
    )

    assert isinstance(pipeline.source, LimitedSource)
    assert isinstance(pipeline.source.source, TfrecordReader)
    assert pipeline.source.source.batch_size == 3
    assert pipeline.source.source.num_parallel_calls == 1
    assert pipeline.source.source.prefetch == 1


def test_zero_max_rows_does_not_resolve_missing_file_glob(tmp_path: Path) -> None:
    pipeline = read_files(tmp_path / "missing-*.bin", max_rows=0)

    assert pipeline.list_shards() == []
    assert list(pipeline.iter_rows()) == []
