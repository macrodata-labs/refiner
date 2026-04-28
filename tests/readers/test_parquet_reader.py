import pyarrow as pa
import pyarrow.parquet as pq

from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.data import datatype
from refiner.pipeline.expressions import col
from refiner.pipeline.sources.readers import ParquetReader
from refiner.worker.context import set_active_run_context
from refiner.worker.metrics.emitter import UserMetricsEmitter


class _RecordingEmitter(UserMetricsEmitter):
    def __init__(self) -> None:
        self.counters: list[dict[str, object]] = []

    def emit_user_counter(self, **kwargs) -> None:
        self.counters.append(kwargs)

    def emit_user_gauge(self, **kwargs) -> None:
        del kwargs

    def register_user_gauge(self, **kwargs) -> None:
        del kwargs

    def emit_user_histogram(self, **kwargs) -> None:
        del kwargs

    def force_flush_user_metrics(self) -> None:
        return None

    def force_flush_resource_metrics(self) -> None:
        return None

    def force_flush_logs(self) -> None:
        return None


class _Runtime:
    def claim(self, previous=None):
        del previous
        return None

    def heartbeat(self, shards):
        del shards

    def complete(self, shard):
        del shard

    def fail(self, shard, error=None):
        del shard, error

    def finalized_workers(self, *, stage_index=None):
        del stage_index
        return []

    def shutdown(self) -> None:
        return None


def _counter_totals(emitter: _RecordingEmitter) -> dict[str, float]:
    totals: dict[str, float] = {}
    for counter in emitter.counters:
        label = counter["label"]
        value = counter["value"]
        assert isinstance(label, str)
        assert isinstance(value, (int, float))
        totals[label] = totals.get(label, 0.0) + float(value)
    return totals


def _write_parquet(tmp_path):
    p = tmp_path / "data.parquet"
    table = pa.table(
        {
            "id": pa.array(list(range(50)), type=pa.int64()),
            "x": pa.array([f"v{i}" for i in range(50)]),
        }
    )
    # Force multiple row groups
    pq.write_table(table, p, row_group_size=10)
    return p


def _rows_from_shard_units(units):
    for unit in units:
        if isinstance(unit, Tabular):
            yield from unit.to_rows()
        else:
            yield unit


def test_parquet_reads_all_rows(tmp_path):
    p = _write_parquet(tmp_path)
    r = ParquetReader(str(p), target_shard_bytes=200)
    shards = r.list_shards()
    assert len(shards) >= 1

    out = []
    for s in shards:
        out.extend(list(_rows_from_shard_units(r.read_shard(s))))

    ids = sorted(int(row["id"]) for row in out)
    assert ids == list(range(50))


def test_parquet_reader_schema_exposes_only_dtype_overrides(tmp_path):
    p = _write_parquet(tmp_path)
    reader = ParquetReader(
        str(p),
        columns_to_read=("x",),
        dtypes={"x": datatype.video_file()},
    )

    schema = reader.schema

    assert schema is not None
    assert schema.names == ["x"]
    assert schema.field("x").metadata == {b"asset_type": b"video"}


def test_parquet_filter_reads_only_matching_rows(tmp_path):
    p = _write_parquet(tmp_path)
    reader = ParquetReader(
        str(p),
        target_shard_bytes=200,
        filter=(col("id") >= 15) & (col("id") < 25),
    )

    out = []
    for shard in reader.list_shards():
        out.extend(list(_rows_from_shard_units(reader.read_shard(shard))))

    ids = [int(row["id"]) for row in out]
    assert ids == list(range(15, 25))


def test_parquet_filter_supports_residual_string_predicates(tmp_path):
    p = _write_parquet(tmp_path)
    reader = ParquetReader(
        str(p),
        target_shard_bytes=200,
        filter=col("x").str.endswith("7"),
    )

    out = []
    for shard in reader.list_shards():
        out.extend(list(_rows_from_shard_units(reader.read_shard(shard))))

    assert [int(row["id"]) for row in out] == [7, 17, 27, 37, 47]


def test_parquet_filter_runs_after_dtype_overrides(tmp_path):
    p = tmp_path / "string-ints.parquet"
    pq.write_table(
        pa.table(
            {
                "x": pa.array(["1", "2", "10", "20"]),
                "label": pa.array(["a", "b", "c", "d"]),
            }
        ),
        p,
        row_group_size=2,
    )

    reader = ParquetReader(
        str(p),
        filter=col("x") > 2,
        dtypes={"x": datatype.int64()},
    )

    out = []
    for shard in reader.list_shards():
        out.extend(list(_rows_from_shard_units(reader.read_shard(shard))))

    assert [row["x"] for row in out] == [10, 20]
    assert [row["label"] for row in out] == ["c", "d"]


def test_parquet_dtype_override_clears_stale_file_metadata(tmp_path):
    p = tmp_path / "metadata.parquet"
    table = datatype.apply_dtypes_to_table(
        pa.table({"frames": ["123"]}),
        {"frames": datatype.video_file()},
    )
    pq.write_table(table, p)

    reader = ParquetReader(
        str(p),
        file_path_column=None,
        dtypes={"frames": datatype.int64()},
    )
    out = [
        unit
        for shard in reader.list_shards()
        for unit in reader.read_shard(shard)
        if isinstance(unit, Tabular)
    ]

    assert len(out) == 1
    assert out[0].table["frames"].to_pylist() == [123]
    assert out[0].table.schema.field("frames").type == pa.int64()
    assert out[0].table.schema.field("frames").metadata is None


def test_parquet_filter_disables_pushdown_for_dtype_columns(tmp_path):
    p = tmp_path / "string-ints.parquet"
    pq.write_table(
        pa.table(
            {
                "x": pa.array(["100", "60", "5"]),
                "label": pa.array(["a", "b", "c"]),
            }
        ),
        p,
        row_group_size=1,
    )

    reader = ParquetReader(
        str(p),
        filter=col("x") > 50,
        dtypes={"x": datatype.int64()},
    )

    out = []
    for shard in reader.list_shards():
        out.extend(list(_rows_from_shard_units(reader.read_shard(shard))))

    assert [row["x"] for row in out] == [100, 60]
    assert [row["label"] for row in out] == ["a", "b"]


def test_parquet_can_split_inside_large_row_group(tmp_path):
    p = tmp_path / "large-row-group.parquet"
    table = pa.table(
        {
            "id": pa.array(list(range(10_000)), type=pa.int64()),
            "x": pa.array([f"{i:05d}-" + ("v" * 64) for i in range(10_000)]),
        }
    )
    pq.write_table(table, p, row_group_size=10_000)

    reader = ParquetReader(
        str(p),
        target_shard_bytes=64 * 1024,
        split_row_groups=True,
    )
    shards = reader.list_shards()

    assert len(shards) > 1

    out = []
    for shard in shards:
        out.extend(list(_rows_from_shard_units(reader.read_shard(shard))))

    ids = sorted(int(row["id"]) for row in out)
    assert ids == list(range(10_000))


def test_parquet_filter_preserves_split_row_groups_and_projection(tmp_path):
    p = tmp_path / "large-row-group.parquet"
    table = pa.table(
        {
            "id": pa.array(list(range(10_000)), type=pa.int64()),
            "x": pa.array([f"{i:05d}-" + ("v" * 64) for i in range(10_000)]),
            "y": pa.array([i % 7 for i in range(10_000)], type=pa.int64()),
        }
    )
    pq.write_table(table, p, row_group_size=10_000)

    reader = ParquetReader(
        str(p),
        target_shard_bytes=64 * 1024,
        columns_to_read=["x"],
        filter=(col("id") >= 9500) & (col("id") < 9510),
        split_row_groups=True,
    )

    out = []
    for shard in reader.list_shards():
        out.extend(list(_rows_from_shard_units(reader.read_shard(shard))))

    assert [row["x"] for row in out] == [
        f"{i:05d}-" + ("v" * 64) for i in range(9500, 9510)
    ]
    assert all(set(row.keys()) == {"x", "file_path"} for row in out)


def test_parquet_split_row_groups_has_no_gaps_or_overlaps(tmp_path):
    p = tmp_path / "large-row-group.parquet"
    table = pa.table(
        {
            "id": pa.array(list(range(10_000)), type=pa.int64()),
            "x": pa.array([f"{i:05d}-" + ("v" * 64) for i in range(10_000)]),
        }
    )
    pq.write_table(table, p, row_group_size=10_000)

    reader = ParquetReader(
        str(p),
        target_shard_bytes=64 * 1024,
        split_row_groups=True,
    )

    seen: list[int] = []
    for shard in reader.list_shards():
        seen.extend(
            int(row["id"]) for row in _rows_from_shard_units(reader.read_shard(shard))
        )

    assert seen == list(range(10_000))


def test_parquet_filter_preserves_split_row_group_bounds_when_middle_groups_are_pruned(
    tmp_path,
):
    p = tmp_path / "many-row-groups.parquet"
    table = pa.table(
        {
            "id": pa.array(list(range(200)), type=pa.int64()),
            "keep": pa.array([i < 50 or i >= 150 for i in range(200)]),
        }
    )
    pq.write_table(table, p, row_group_size=50)

    reader = ParquetReader(
        str(p),
        target_shard_bytes=80,
        split_row_groups=True,
        filter=col("keep"),
    )

    seen: list[int] = []
    for shard in reader.list_shards():
        seen.extend(
            int(row["id"]) for row in _rows_from_shard_units(reader.read_shard(shard))
        )

    assert seen == list(range(50)) + list(range(150, 200))


def test_parquet_logs_pushdown_and_total_filtered_metrics(tmp_path):
    p = tmp_path / "many-row-groups.parquet"
    table = pa.table(
        {
            "id": pa.array(list(range(20)), type=pa.int64()),
            "keep": pa.array([i < 10 for i in range(20)]),
        }
    )
    pq.write_table(table, p, row_group_size=10)

    reader = ParquetReader(
        str(p),
        target_shard_bytes=100_000,
        filter=col("keep"),
        dtypes={"keep": pa.field("keep", pa.bool_(), metadata={b"kind": b"flag"})},
    )

    emitter = _RecordingEmitter()
    with set_active_run_context(
        job_id="job-1",
        stage_index=0,
        worker_id="worker-1",
        worker_name=None,
        runtime_lifecycle=_Runtime(),
        user_metrics_emitter=emitter,
    ):
        out = []
        for shard in reader.list_shards():
            out.extend(list(_rows_from_shard_units(reader.iter_shard_units(shard))))

    assert [int(row["id"]) for row in out] == list(range(10))
    counters_by_label = _counter_totals(emitter)
    assert counters_by_label["pushdown_row_groups_filtered"] == 1.0
    assert counters_by_label["total_rows_filtered"] == 10.0
    assert counters_by_label["rows_read"] == 10.0


def test_parquet_logs_total_filtered_for_residual_in_memory_filter(tmp_path):
    p = _write_parquet(tmp_path)
    reader = ParquetReader(
        str(p),
        target_shard_bytes=200,
        filter=col("x").str.endswith("7"),
    )

    emitter = _RecordingEmitter()
    with set_active_run_context(
        job_id="job-1",
        stage_index=0,
        worker_id="worker-1",
        worker_name=None,
        runtime_lifecycle=_Runtime(),
        user_metrics_emitter=emitter,
    ):
        out = []
        for shard in reader.list_shards():
            out.extend(list(_rows_from_shard_units(reader.iter_shard_units(shard))))

    assert [int(row["id"]) for row in out] == [7, 17, 27, 37, 47]
    counters_by_label = _counter_totals(emitter)
    assert "pushdown_row_groups_filtered" not in counters_by_label
    assert counters_by_label["total_rows_filtered"] == 45.0
    assert counters_by_label["rows_read"] == 5.0


def test_parquet_does_not_log_pushdown_row_group_metrics_for_split_row_groups(tmp_path):
    p = tmp_path / "many-row-groups.parquet"
    table = pa.table(
        {
            "id": pa.array(list(range(200)), type=pa.int64()),
            "keep": pa.array([i < 50 or i >= 150 for i in range(200)]),
        }
    )
    pq.write_table(table, p, row_group_size=50)

    reader = ParquetReader(
        str(p),
        target_shard_bytes=80,
        split_row_groups=True,
        filter=col("keep"),
    )

    emitter = _RecordingEmitter()
    with set_active_run_context(
        job_id="job-1",
        stage_index=0,
        worker_id="worker-1",
        worker_name=None,
        runtime_lifecycle=_Runtime(),
        user_metrics_emitter=emitter,
    ):
        out = []
        for shard in reader.list_shards():
            out.extend(list(_rows_from_shard_units(reader.iter_shard_units(shard))))

    assert [int(row["id"]) for row in out] == list(range(50)) + list(range(150, 200))
    counters_by_label = _counter_totals(emitter)
    assert "pushdown_row_groups_filtered" not in counters_by_label
    assert counters_by_label["total_rows_filtered"] == 100.0
