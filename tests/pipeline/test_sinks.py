from __future__ import annotations

import json
from typing import cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fsspec.implementations.memory import MemoryFileSystem

from refiner import col
from refiner.pipeline.data import datatype
from refiner.pipeline.data.row import DictRow, Row
from refiner.pipeline.data.shard import SHARD_ID_COLUMN
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline import from_items, load_lance
from refiner.pipeline.sinks import JsonlSink
from refiner.pipeline.sinks.lance import (
    LanceDatasetCommitReducerSink,
    LanceDatasetSink,
    _fragment_data_paths,
    _relocate_fragment_files,
    _schema_to_base64,
)
from refiner.pipeline.sinks.parquet import ParquetSink
from refiner.pipeline.sinks.reducer.file import FileCleanupReducerSink
from refiner.pipeline.sources.lance import (
    LANCE_FRAGMENT_ID_COLUMN,
    LANCE_ROW_POSITION_COLUMN,
)
from refiner.worker.context import set_active_run_context
from refiner.worker.lifecycle import FinalizedShardWorker, RuntimeLifecycle
from refiner.worker.context import worker_token_for


class _FinalizedWorkersRuntime:
    def __init__(self, rows: list[FinalizedShardWorker]) -> None:
        self._rows = rows

    def finalized_workers(
        self, *, stage_index: int | None = None
    ) -> list[FinalizedShardWorker]:
        assert stage_index == 0
        return self._rows


def test_iter_rows_ignores_sink(tmp_path) -> None:
    pipeline = from_items([{"x": 1}, {"x": 2}], items_per_shard=1).write_jsonl(tmp_path)
    out = list(pipeline.iter_rows())
    assert [int(row["x"]) for row in out] == [1, 2]
    assert list(tmp_path.iterdir()) == []


def test_launch_local_writes_jsonl_per_shard(tmp_path) -> None:
    output_dir = tmp_path / "jsonl-output"
    pipeline = (
        from_items([{"x": 1}, {"x": 2}, {"x": 3}], items_per_shard=2)
        .map(lambda row: {"x": int(row["x"]) * 10})
        .write_jsonl(output_dir)
    )

    stats = pipeline.launch_local(
        name="jsonl-sink", num_workers=1, rundir=str(tmp_path / "run")
    )

    assert stats.claimed == 3
    assert stats.completed == 3
    assert stats.output_rows == 3
    written = sorted(path.name for path in output_dir.iterdir())
    assert len(written) == 2
    assert all("__w" in name for name in written)
    assert all(name.endswith(".jsonl") for name in written)


def test_launch_local_writes_parquet_per_shard(tmp_path) -> None:
    output_dir = tmp_path / "parquet-output"
    pipeline = (
        from_items([{"x": 1}, {"x": 2}, {"x": 3}], items_per_shard=2)
        .map(lambda row: {"x": int(row["x"]) * 10})
        .write_parquet(output_dir)
    )

    stats = pipeline.launch_local(
        name="parquet-sink", num_workers=1, rundir=str(tmp_path / "run")
    )

    assert stats.claimed == 3
    assert stats.completed == 3
    assert stats.output_rows == 3
    written = sorted(path for path in output_dir.iterdir() if path.suffix == ".parquet")
    assert len(written) == 2
    assert all("__w" in path.name for path in written)
    values = []
    for path in written:
        table = pq.read_table(path)
        values.extend(int(value) for value in table.column("x").to_pylist())
    assert sorted(values) == [10, 20, 30]


def test_write_parquet_dtypes_apply_to_row_blocks(tmp_path) -> None:
    output_dir = tmp_path / "parquet-dtypes"
    pipeline = from_items(
        [{"maybe": None}, {"maybe": "value"}],
        items_per_shard=2,
    ).write_parquet(output_dir, dtypes={"maybe": datatype.string()})

    pipeline.launch_local(
        name="parquet-dtypes",
        num_workers=1,
        rundir=str(tmp_path / "run"),
    )

    written = sorted(path for path in output_dir.iterdir() if path.suffix == ".parquet")
    assert len(written) == 1
    table = pq.read_table(written[0])
    assert table.schema.field("maybe").type == pa.string()
    assert table.column("maybe").to_pylist() == [None, "value"]


def test_write_parquet_dtypes_apply_to_tabular_blocks(tmp_path) -> None:
    output_dir = tmp_path / "parquet-tabular-dtypes"
    sink = ParquetSink(output_dir, dtypes={"image": datatype.image_path()})
    sink.write_shard_block(
        "abc",
        Tabular(pa.table({"image": ["s3://bucket/image.png"]})),
    )
    sink.on_shard_complete("abc")

    table = pq.read_table(next(output_dir.glob("*.parquet")))
    assert datatype.asset_type(table.schema.field("image")) == "image"
    assert datatype.asset_storage(table.schema.field("image")) == "path"


def test_parquet_sink_serializes_numpy_values(tmp_path) -> None:
    output_dir = tmp_path / "parquet-numpy"
    sink = ParquetSink(output_dir)
    sink.write_shard_block(
        "abc",
        [DictRow({"array": np.array([[1, 2], [3, 4]]), "scalar": np.int64(5)})],
    )
    sink.on_shard_complete("abc")

    table = pq.read_table(next(output_dir.glob("*.parquet")))
    assert table.schema.field("array").type == pa.list_(pa.list_(pa.int64()))
    assert table.column("array").to_pylist() == [[[1, 2], [3, 4]]]
    assert table.column("scalar").to_pylist() == [5]


def test_launch_local_writes_lance_files_per_shard(tmp_path) -> None:
    pytest.importorskip("lance")
    from lance.file import LanceFileReader

    output_dir = tmp_path / "lance-files-output"
    pipeline = (
        from_items([{"x": 1}, {"x": 2}, {"x": 3}], items_per_shard=2)
        .map(lambda row: {"x": int(row["x"]) * 10})
        .write_lance(output_dir)
    )

    stats = pipeline.launch_local(
        name="lance-sink", num_workers=1, rundir=str(tmp_path / "run")
    )

    assert stats.claimed == 3
    assert stats.completed == 3
    assert stats.output_rows == 3
    written = sorted(path for path in output_dir.iterdir() if path.suffix == ".lance")
    assert len(written) == 2
    assert all("__w" in path.name for path in written)
    values = []
    for path in written:
        table = LanceFileReader(str(path)).read_all().to_table()
        values.extend(int(value) for value in table.column("x").to_pylist())
    assert sorted(values) == [
        10,
        20,
        30,
    ]


def test_lance_source_hides_lineage_from_ordinary_sink_schema(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    input_uri = tmp_path / "schema-filter-input.lance"
    lance.write_dataset(pa.table({"x": [1]}), str(input_uri))

    pipeline = load_lance(input_uri).write_jsonl(tmp_path / "schema-filter-output")

    schema = pipeline._sink_input_schema()
    assert schema is not None
    assert schema.names == ["x"]


@pytest.mark.parametrize(
    "filename_template",
    [
        "../escaped/{shard_id}__w{worker_id}.lance",
        "/escaped/{shard_id}__w{worker_id}.lance",
        "nested\\{shard_id}__w{worker_id}.lance",
        "nested//{shard_id}__w{worker_id}.lance",
        "C:/{shard_id}__w{worker_id}.lance",
    ],
)
def test_lance_file_sink_rejects_unsafe_filename_template(
    tmp_path, filename_template: str
) -> None:
    with pytest.raises(ValueError, match="normalized relative path"):
        from_items([]).write_lance(
            tmp_path,
            filename_template=filename_template,
        )


def test_launch_local_writes_lance_dataset(tmp_path) -> None:
    lance = pytest.importorskip("lance")

    output_dir = tmp_path / "lance-output.lance"
    pipeline = (
        from_items([{"x": 1}, {"x": 2}, {"x": 3}], items_per_shard=2)
        .map(lambda row: {"x": int(row["x"]) * 10})
        .write_lance_dataset(output_dir)
    )

    stats = pipeline.launch_local(
        name="lance-dataset-sink", num_workers=1, rundir=str(tmp_path / "run")
    )

    assert stats.claimed == 3
    assert stats.completed == 3
    assert stats.output_rows == 3
    table = lance.dataset(str(output_dir)).to_table()
    fragments = lance.dataset(str(output_dir)).get_fragments()
    assert len(fragments) == 2
    assert sorted(int(value) for value in table.column("x").to_pylist()) == [
        10,
        20,
        30,
    ]


def test_lance_dataset_writer_handles_more_shards_than_io_threads(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    output_dir = tmp_path / "many-shards.lance"
    values = [{"x": index} for index in range(12)]

    from_items(values, items_per_shard=1).write_lance_dataset(output_dir).launch_local(
        name="lance-many-shards",
        num_workers=1,
        rundir=str(tmp_path / "many-shards-run"),
    )

    assert sorted(lance.dataset(str(output_dir)).to_table()["x"].to_pylist()) == list(
        range(12)
    )


def test_launch_local_adds_lance_columns_without_rewriting_base_files(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    dataset_uri = tmp_path / "evolved.lance"
    base = lance.write_dataset(
        pa.table({"x": [1, 2, 3, 4]}),
        str(dataset_uri),
        max_rows_per_file=2,
    )
    base_files = {
        file["path"]
        for fragment in base.get_fragments()
        for file in fragment.metadata.to_json()["files"]
    }

    pipeline = (
        load_lance(dataset_uri, version=base.version, columns=["x"], batch_size=1)
        .map(
            lambda row: {"y": int(row["x"]) * 10},
            dtypes={"y": datatype.int64()},
        )
        .write_lance_dataset(
            dataset_uri,
            mode="add_columns",
            columns=["y"],
        )
    )

    stats = pipeline.launch_local(
        name="lance-add-columns",
        num_workers=1,
        rundir=str(tmp_path / "run"),
    )

    evolved = lance.dataset(str(dataset_uri))
    evolved_files = {
        file["path"]
        for fragment in evolved.get_fragments()
        for file in fragment.metadata.to_json()["files"]
    }
    assert stats.completed == 3
    assert stats.output_rows == 4
    assert evolved.version == base.version + 1
    assert evolved.to_table().to_pydict() == {
        "x": [1, 2, 3, 4],
        "y": [10, 20, 30, 40],
    }
    assert base_files < evolved_files


def test_lance_add_columns_reorders_fragment_outputs(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    dataset_uri = tmp_path / "reordered.lance"
    base = lance.write_dataset(pa.table({"x": [1, 2, 3]}), str(dataset_uri))

    def reverse_batch(rows: list[Row]):
        for row in reversed(rows):
            yield row.update({"y": int(row["x"]) * 10})

    pipeline = (
        load_lance(dataset_uri, version=base.version, columns=["x"], batch_size=3)
        .batch_map(
            reverse_batch,
            batch_size=3,
            dtypes={"y": datatype.int64()},
        )
        .write_lance_dataset(
            dataset_uri,
            mode="add_columns",
            columns=["y"],
        )
    )

    pipeline.launch_local(
        name="lance-reorder-columns",
        num_workers=1,
        rundir=str(tmp_path / "run"),
    )

    assert lance.dataset(str(dataset_uri)).to_table().to_pydict() == {
        "x": [1, 2, 3],
        "y": [10, 20, 30],
    }


def test_lance_add_columns_preserves_existing_field_ids(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    dataset_uri = tmp_path / "field-ids.lance"
    dataset = lance.write_dataset(
        pa.table({"x": [1], "removed": [2], "z": [3]}),
        str(dataset_uri),
    )
    dataset.drop_columns(["removed"])
    base = lance.dataset(str(dataset_uri))

    pipeline = (
        load_lance(dataset_uri, version=base.version)
        .map(lambda row: {"w": 4}, dtypes={"w": datatype.int64()})
        .write_lance_dataset(
            dataset_uri,
            mode="add_columns",
            columns=["w"],
        )
    )
    pipeline.launch_local(
        name="lance-preserve-field-ids",
        num_workers=1,
        rundir=str(tmp_path / "run"),
    )

    assert lance.dataset(str(dataset_uri)).to_table().to_pydict() == {
        "x": [1],
        "z": [3],
        "w": [4],
    }


def test_lance_dataset_copy_strips_internal_columns(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    source_uri = tmp_path / "source-copy.lance"
    output_uri = tmp_path / "output-copy.lance"
    lance.write_dataset(pa.table({"x": [1, 2]}), str(source_uri))

    load_lance(source_uri).write_lance_dataset(output_uri).launch_local(
        name="lance-copy",
        num_workers=1,
        rundir=str(tmp_path / "copy-run"),
    )

    output = lance.dataset(str(output_uri))
    assert output.schema.names == ["x"]
    assert output.to_table().to_pydict() == {"x": [1, 2]}


def test_lance_map_table_can_drop_lineage_without_add_columns(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    source_uri = tmp_path / "map-table-public.lance"
    lance.write_dataset(pa.table({"x": [1, 2]}), str(source_uri))

    rows = list(
        load_lance(source_uri)
        .map_table(lambda table: table.select(["x", SHARD_ID_COLUMN]))
        .select("x")
        .iter_rows()
    )

    assert [row.to_dict() for row in rows] == [{"x": 1}, {"x": 2}]


def test_lance_add_columns_accepts_expression_created_column(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    dataset_uri = tmp_path / "expression-column.lance"
    base = lance.write_dataset(pa.table({"x": [1, 2]}), str(dataset_uri))

    (
        load_lance(dataset_uri, version=base.version)
        .with_column("y", col("x") + 1)
        .write_lance_dataset(dataset_uri, mode="add_columns", columns=["y"])
        .launch_local(
            name="lance-expression-column",
            num_workers=1,
            rundir=str(tmp_path / "expression-run"),
        )
    )

    assert lance.dataset(str(dataset_uri)).to_table().to_pydict() == {
        "x": [1, 2],
        "y": [2, 3],
    }


def test_lance_add_columns_preserves_internal_columns_across_replacement_row(
    tmp_path,
) -> None:
    lance = pytest.importorskip("lance")
    dataset_uri = tmp_path / "replacement-row.lance"
    base = lance.write_dataset(pa.table({"x": [1, 2]}), str(dataset_uri))
    pipeline = (
        load_lance(dataset_uri, version=base.version)
        .map(
            lambda row: DictRow(
                {"x": int(row["x"]), "y": int(row["x"]) + 1},
                shard_id=row.shard_id,
            ),
            dtypes={"y": datatype.int64()},
        )
        .write_lance_dataset(dataset_uri, mode="add_columns", columns=["y"])
    )

    pipeline.launch_local(
        name="lance-replacement-row",
        num_workers=1,
        rundir=str(tmp_path / "replacement-row-run"),
    )

    assert lance.dataset(str(dataset_uri)).to_table().to_pydict() == {
        "x": [1, 2],
        "y": [2, 3],
    }


def test_lance_add_columns_ignores_row_lineage_overwrites(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    dataset_uri = tmp_path / "lineage-overwrite.lance"
    base = lance.write_dataset(pa.table({"x": [1, 2]}), str(dataset_uri))
    pipeline = (
        load_lance(dataset_uri, version=base.version)
        .map(
            lambda row: {
                "y": int(row["x"]) * 10,
                LANCE_ROW_POSITION_COLUMN: 1 - int(row[LANCE_ROW_POSITION_COLUMN]),
            },
            dtypes={"y": datatype.int64()},
        )
        .write_lance_dataset(dataset_uri, mode="add_columns", columns=["y"])
    )

    pipeline.launch_local(
        name="lance-lineage-overwrite",
        num_workers=1,
        rundir=str(tmp_path / "lineage-overwrite-run"),
    )

    assert lance.dataset(str(dataset_uri)).to_table().to_pydict() == {
        "x": [1, 2],
        "y": [10, 20],
    }


def test_lance_add_columns_rejects_unaligned_replacement_batch(
    tmp_path,
) -> None:
    lance = pytest.importorskip("lance")
    dataset_uri = tmp_path / "replacement-batch.lance"
    base = lance.write_dataset(pa.table({"x": [1, 2]}), str(dataset_uri))
    pipeline = (
        load_lance(dataset_uri, version=base.version)
        .batch_map(
            lambda rows: [
                DictRow({"y": int(row["x"]) + 1}, shard_id=row.shard_id) for row in rows
            ],
            batch_size=2,
            dtypes={"y": datatype.int64()},
        )
        .write_lance_dataset(dataset_uri, mode="add_columns", columns=["y"])
    )

    with pytest.raises(RuntimeError, match="failed shard"):
        pipeline.launch_local(
            name="lance-replacement-batch",
            num_workers=1,
            rundir=str(tmp_path / "replacement-batch-run"),
        )

    assert lance.dataset(str(dataset_uri)).version == base.version


def test_non_lance_replacement_rows_do_not_preserve_lance_internal_columns() -> None:
    rows = list(
        from_items(
            [
                {
                    "x": 1,
                    LANCE_FRAGMENT_ID_COLUMN: 7,
                    LANCE_ROW_POSITION_COLUMN: 0,
                }
            ]
        )
        .map(lambda row: DictRow({"x": 2}, shard_id=row.shard_id))
        .iter_rows()
    )

    assert rows[0].to_dict() == {"x": 2}


def test_lance_empty_create_and_overwrite_preserve_schema(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    input_uri = tmp_path / "empty-input.lance"
    create_uri = tmp_path / "empty-create.lance"
    overwrite_uri = tmp_path / "empty-overwrite.lance"
    lance.write_dataset(pa.table({"x": [1]}), str(input_uri))
    lance.write_dataset(pa.table({"x": [1]}), str(overwrite_uri))

    for mode, output_uri in (("create", create_uri), ("overwrite", overwrite_uri)):
        (
            load_lance(input_uri)
            .filter(lambda _row: False)
            .write_lance_dataset(output_uri, mode=mode)
            .launch_local(
                name=f"lance-empty-{mode}",
                num_workers=1,
                rundir=str(tmp_path / f"empty-{mode}-run"),
            )
        )
        output = lance.dataset(str(output_uri))
        assert output.schema.names == ["x"]
        assert output.count_rows() == 0


def test_lance_empty_create_rejects_missing_selected_column(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    input_uri = tmp_path / "empty-select-input.lance"
    output_uri = tmp_path / "empty-select-output.lance"
    lance.write_dataset(pa.table({"x": pa.array([], type=pa.int64())}), str(input_uri))

    pipeline = (
        load_lance(input_uri).select("x", "missing").write_lance_dataset(output_uri)
    )

    with pytest.raises(RuntimeError):
        pipeline.launch_local(
            name="lance-empty-missing-select",
            num_workers=1,
            rundir=str(tmp_path / "empty-select-run"),
        )

    assert not output_uri.exists()


def test_lance_empty_create_preserves_valid_selected_schema(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    input_uri = tmp_path / "empty-valid-select-input.lance"
    output_uri = tmp_path / "empty-valid-select-output.lance"
    lance.write_dataset(
        pa.table(
            {
                "x": pa.array([], type=pa.int64()),
                "unused": pa.array([], type=pa.string()),
            }
        ),
        str(input_uri),
    )

    (
        load_lance(input_uri)
        .select("x")
        .write_lance_dataset(output_uri)
        .launch_local(
            name="lance-empty-valid-select",
            num_workers=1,
            rundir=str(tmp_path / "empty-valid-select-run"),
        )
    )

    output = lance.dataset(str(output_uri))
    assert output.schema == pa.schema([("x", pa.int64())])
    assert output.count_rows() == 0


def test_lance_empty_create_rejects_missing_cast_column(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    input_uri = tmp_path / "empty-cast-input.lance"
    output_uri = tmp_path / "empty-cast-output.lance"
    lance.write_dataset(pa.table({"x": pa.array([], type=pa.int64())}), str(input_uri))

    pipeline = (
        load_lance(input_uri)
        .cast(missing=datatype.int64())
        .write_lance_dataset(output_uri)
    )

    with pytest.raises(RuntimeError):
        pipeline.launch_local(
            name="lance-empty-missing-cast",
            num_workers=1,
            rundir=str(tmp_path / "empty-cast-run"),
        )

    assert not output_uri.exists()


@pytest.mark.parametrize("operation", ["drop", "filter"])
def test_lance_empty_create_rejects_missing_transform_column(
    tmp_path, operation: str
) -> None:
    lance = pytest.importorskip("lance")
    input_uri = tmp_path / f"empty-{operation}-input.lance"
    output_uri = tmp_path / f"empty-{operation}-output.lance"
    lance.write_dataset(pa.table({"x": pa.array([], type=pa.int64())}), str(input_uri))

    pipeline = load_lance(input_uri)
    pipeline = (
        pipeline.drop("missing")
        if operation == "drop"
        else pipeline.filter(col("missing") > 0)
    )
    pipeline = pipeline.write_lance_dataset(output_uri)

    with pytest.raises(RuntimeError):
        pipeline.launch_local(
            name=f"lance-empty-missing-{operation}",
            num_workers=1,
            rundir=str(tmp_path / f"empty-{operation}-run"),
        )

    assert not output_uri.exists()


def test_lance_fragment_relocation_cleans_partial_move_target(
    tmp_path, monkeypatch
) -> None:
    from refiner.io.datafolder import DataFolder

    output = DataFolder.resolve(tmp_path)
    output.makedirs("data", exist_ok=True)
    with output.open("data/source.lance", mode="wb") as file:
        file.write(b"fragment")

    def partial_move(source: str, target: str) -> None:
        output.file(source).copy(output.file(target))
        raise OSError("source deletion failed")

    monkeypatch.setattr(output, "mv", partial_move)
    fragment = json.dumps({"files": [{"path": "source.lance"}]})

    with pytest.raises(OSError, match="source deletion failed"):
        _relocate_fragment_files(output, fragment, attempt_prefix="attempt")

    assert output.exists("data/source.lance")
    assert output.find("data") == ["data/source.lance"]


def test_lance_add_columns_rejects_empty_dataset(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    dataset_uri = tmp_path / "empty-add-columns.lance"
    base = lance.write_dataset(
        pa.table({"x": pa.array([], type=pa.int64())}), str(dataset_uri)
    )
    pipeline = (
        load_lance(dataset_uri, version=base.version)
        .map(lambda _row: {"y": 1}, dtypes={"y": datatype.int64()})
        .write_lance_dataset(dataset_uri, mode="add_columns", columns=["y"])
    )

    with pytest.raises(RuntimeError):
        pipeline.launch_local(
            name="lance-empty-add-columns",
            num_workers=1,
            rundir=str(tmp_path / "empty-add-columns-run"),
        )
    assert lance.dataset(str(dataset_uri)).schema.names == ["x"]


def test_lance_empty_create_reducer_retry_is_idempotent(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    dataset_uri = tmp_path / "empty-create-retry.lance"
    runtime = cast(
        RuntimeLifecycle,
        _FinalizedWorkersRuntime(
            [FinalizedShardWorker(shard_id="input-shard", worker_id="worker-1")]
        ),
    )
    with set_active_run_context(
        job_id="job",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=runtime,
    ):
        for _ in range(2):
            reducer = LanceDatasetCommitReducerSink(
                dataset_uri,
                mode="create",
                planned_schema=pa.schema([("x", pa.int64())]),
            )
            reducer.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])

    assert lance.dataset(str(dataset_uri)).version == 1


def test_lance_empty_overwrite_retry_finds_historical_commit(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    dataset_uri = tmp_path / "empty-overwrite-retry.lance"
    lance.write_dataset(pa.table({"x": [1]}), str(dataset_uri))
    runtime = cast(
        RuntimeLifecycle,
        _FinalizedWorkersRuntime(
            [FinalizedShardWorker(shard_id="input-shard", worker_id="worker-1")]
        ),
    )
    with set_active_run_context(
        job_id="job",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=runtime,
    ):
        first = LanceDatasetCommitReducerSink(
            dataset_uri,
            mode="overwrite",
            source_version=1,
            planned_schema=pa.schema([("x", pa.int64())]),
        )
        first.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])
        lance.write_dataset(pa.table({"x": [99]}), str(dataset_uri), mode="append")
        concurrent_version = lance.dataset(str(dataset_uri)).version

        retry = LanceDatasetCommitReducerSink(
            dataset_uri,
            mode="overwrite",
            source_version=1,
            planned_schema=pa.schema([("x", pa.int64())]),
        )
        retry.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])

    latest = lance.dataset(str(dataset_uri))
    assert latest.version == concurrent_version
    assert latest.to_table().to_pydict() == {"x": [99]}


def test_lance_empty_overwrite_rejects_concurrent_update(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    dataset_uri = tmp_path / "empty-overwrite-conflict.lance"
    lance.write_dataset(pa.table({"x": [1]}), str(dataset_uri))
    pipeline = (
        load_lance(dataset_uri)
        .filter(col("x") < 0)
        .write_lance_dataset(dataset_uri, mode="overwrite")
    )
    sink = pipeline.sink
    assert sink is not None
    reducer = sink.build_reducer()
    assert isinstance(reducer, LanceDatasetCommitReducerSink)

    lance.write_dataset(pa.table({"x": [2]}), str(dataset_uri), mode="append")

    runtime = cast(
        RuntimeLifecycle,
        _FinalizedWorkersRuntime(
            [FinalizedShardWorker(shard_id="input-shard", worker_id="worker-1")]
        ),
    )
    with set_active_run_context(
        job_id="job",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=runtime,
    ):
        with pytest.raises(ValueError, match="dataset changed from version 1 to 2"):
            reducer.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])

    assert lance.dataset(str(dataset_uri)).to_table().to_pydict() == {"x": [1, 2]}


def test_lance_empty_create_rejects_partially_inferred_schema(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    input_uri = tmp_path / "partial-schema-input.lance"
    output_uri = tmp_path / "partial-schema-output.lance"
    lance.write_dataset(pa.table({"x": [1]}), str(input_uri))
    pipeline = (
        load_lance(input_uri)
        .with_column("y", col("x") + 1)
        .filter(lambda _row: False)
        .write_lance_dataset(output_uri)
    )

    with pytest.raises(RuntimeError):
        pipeline.launch_local(
            name="lance-empty-partial-schema",
            num_workers=1,
            rundir=str(tmp_path / "empty-partial-schema-run"),
        )
    assert not output_uri.exists()


def test_lance_empty_create_rejects_typed_replacement_schema(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    input_uri = tmp_path / "typed-replacement-input.lance"
    output_uri = tmp_path / "typed-replacement-output.lance"
    lance.write_dataset(pa.table({"x": [1]}), str(input_uri))
    pipeline = (
        load_lance(input_uri)
        .map(
            lambda row: DictRow({"y": 1}, shard_id=row.shard_id),
            dtypes={"y": datatype.int64()},
        )
        .filter(lambda _row: False)
        .write_lance_dataset(output_uri)
    )

    with pytest.raises(RuntimeError):
        pipeline.launch_local(
            name="lance-empty-typed-replacement-schema",
            num_workers=1,
            rundir=str(tmp_path / "typed-replacement-run"),
        )
    assert not output_uri.exists()


def test_lance_add_columns_rejects_concurrent_dataset_version(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    dataset_uri = tmp_path / "concurrent-version.lance"
    base = lance.write_dataset(pa.table({"x": [1]}), str(dataset_uri))
    pipeline = (
        load_lance(dataset_uri, version=base.version)
        .map(lambda row: {"y": 10}, dtypes={"y": datatype.int64()})
        .write_lance_dataset(
            dataset_uri,
            mode="add_columns",
            columns=["y"],
        )
    )
    appended = lance.write_dataset(
        pa.table({"x": [2]}),
        str(dataset_uri),
        mode="append",
    )

    with pytest.raises(RuntimeError):
        pipeline.launch_local(
            name="lance-version-conflict",
            num_workers=1,
            rundir=str(tmp_path / "run"),
        )

    latest = lance.dataset(str(dataset_uri))
    assert latest.version == appended.version
    assert latest.to_table().to_pydict() == {"x": [1, 2]}
    assert not list((dataset_uri / "data").glob("_refiner_lance_attempt_*"))
    assert list((dataset_uri / "_refiner_lance_fragments").glob("**/*.jsonl"))
    with pytest.raises(RuntimeError):
        pipeline.launch_local(
            name="lance-version-conflict",
            num_workers=1,
            rundir=str(tmp_path / "run"),
        )


def test_lance_append_rejects_concurrent_dataset_version(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    dataset_uri = tmp_path / "concurrent-append.lance"
    base = lance.write_dataset(pa.table({"x": [1]}), str(dataset_uri))
    shard_id = "0123456789ab"
    worker_id = "worker-1"
    runtime = cast(
        RuntimeLifecycle,
        _FinalizedWorkersRuntime(
            [FinalizedShardWorker(shard_id=shard_id, worker_id=worker_id)]
        ),
    )
    sink = LanceDatasetSink(dataset_uri, mode="append")
    with set_active_run_context(
        job_id="job",
        stage_index=0,
        worker_id=worker_id,
        worker_name=None,
        runtime_lifecycle=runtime,
    ):
        sink.write_block([DictRow({"x": 2}, shard_id=shard_id)])
        sink.on_shard_complete(shard_id)

    appended = lance.write_dataset(
        pa.table({"x": [3]}), str(dataset_uri), mode="append"
    )
    reducer = sink.build_reducer()
    assert isinstance(reducer, LanceDatasetCommitReducerSink)
    with set_active_run_context(
        job_id="job",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=runtime,
    ):
        with pytest.raises(ValueError, match="dataset changed"):
            reducer.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])

    latest = lance.dataset(str(dataset_uri))
    assert latest.version == appended.version
    assert latest.to_table().to_pydict() == {"x": [1, 3]}
    assert base.version < appended.version
    assert not list((dataset_uri / "data").glob("_refiner_lance_attempt_*"))
    assert list((dataset_uri / "_refiner_lance_fragments").glob("**/*.jsonl"))
    retry = sink.build_reducer()
    assert isinstance(retry, LanceDatasetCommitReducerSink)
    with set_active_run_context(
        job_id="job",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=runtime,
    ):
        with pytest.raises(ValueError, match="dataset changed"):
            retry.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])


def test_lance_add_columns_rejects_missing_rows(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    dataset_uri = tmp_path / "missing.lance"
    base = lance.write_dataset(pa.table({"x": [1, 2]}), str(dataset_uri))
    pipeline = (
        load_lance(dataset_uri, version=base.version)
        .filter(lambda row: int(row["x"]) == 1)
        .map(lambda row: {"y": 10}, dtypes={"y": datatype.int64()})
        .write_lance_dataset(
            dataset_uri,
            mode="add_columns",
            columns=["y"],
        )
    )

    with pytest.raises(RuntimeError, match="failed shard"):
        pipeline.launch_local(
            name="lance-missing-columns",
            num_workers=1,
            rundir=str(tmp_path / "run"),
        )

    assert lance.dataset(str(dataset_uri)).version == base.version


def test_lance_add_columns_reducer_rejects_missing_fragment(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    dataset_uri = tmp_path / "missing-fragment.lance"
    base = lance.write_dataset(
        pa.table({"x": [1, 2, 3, 4]}),
        str(dataset_uri),
        max_rows_per_file=2,
    )
    assert len(base.get_fragments()) == 2
    base_files = {
        dataset_uri / "data" / file["path"]
        for fragment in base.get_fragments()
        for file in fragment.metadata.to_json()["files"]
    }
    pipeline = (
        load_lance(dataset_uri, version=base.version)
        .filter(lambda row: int(row["x"]) <= 2)
        .map(
            lambda row: {"y": int(row["x"]) * 10},
            dtypes={"y": datatype.int64()},
        )
        .write_lance_dataset(
            dataset_uri,
            mode="add_columns",
            columns=["y"],
        )
    )

    with pytest.raises(RuntimeError, match="failed shard"):
        pipeline.launch_local(
            name="lance-missing-fragment",
            num_workers=1,
            rundir=str(tmp_path / "run"),
        )

    assert lance.dataset(str(dataset_uri)).version == base.version
    assert set((dataset_uri / "data").glob("*.lance")) == base_files


def test_lance_add_columns_reducer_cleans_only_rejected_new_files(
    tmp_path,
) -> None:
    lance = pytest.importorskip("lance")
    dataset_uri = tmp_path / "retry-cleanup.lance"
    base = lance.write_dataset(pa.table({"x": [1, 2]}), str(dataset_uri))
    shard = load_lance(dataset_uri, version=base.version).list_shards()[0]
    fragment_id = int(base.get_fragments()[0].fragment_id)
    worker_ids = ["worker-1", "worker-2"]
    created_by_worker: dict[str, set[str]] = {}

    for worker_id, values in zip(worker_ids, [[10, 20], [90, 99]], strict=True):
        sink = LanceDatasetSink(
            dataset_uri,
            mode="add_columns",
            columns=["y"],
            source_uri=str(dataset_uri),
            source_version=base.version,
        )
        sink.set_input_schema(
            pa.schema(
                [
                    pa.field("y", pa.int64()),
                    pa.field(LANCE_FRAGMENT_ID_COLUMN, pa.uint64()),
                    pa.field(LANCE_ROW_POSITION_COLUMN, pa.uint64()),
                ]
            )
        )
        with set_active_run_context(
            job_id="job",
            stage_index=0,
            worker_id=worker_id,
            worker_name=None,
            runtime_lifecycle=cast(
                RuntimeLifecycle,
                _FinalizedWorkersRuntime(
                    [FinalizedShardWorker(shard_id=shard.id, worker_id=worker_ids[1])]
                ),
            ),
        ):
            sink.write_shard_block(
                shard.id,
                Tabular(
                    pa.table(
                        {
                            "y": values,
                            LANCE_FRAGMENT_ID_COLUMN: pa.array(
                                [fragment_id, fragment_id], type=pa.uint64()
                            ),
                            LANCE_ROW_POSITION_COLUMN: pa.array(
                                [0, 1], type=pa.uint64()
                            ),
                        }
                    )
                ),
            )
            sink.on_shard_complete(shard.id)
            with sink.output.open(
                sink._relpath(shard.id), mode="rt", encoding="utf-8"
            ) as metadata_file:
                payload = json.load(metadata_file)
            created_by_worker[worker_id] = set(payload["created_files"])

    base_paths = {
        dataset_uri / "data" / file["path"]
        for fragment in base.get_fragments()
        for file in fragment.metadata.to_json()["files"]
    }
    reducer = LanceDatasetCommitReducerSink(
        dataset_uri,
        mode="add_columns",
        source_version=base.version,
    )
    with set_active_run_context(
        job_id="job",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=cast(
            RuntimeLifecycle,
            _FinalizedWorkersRuntime(
                [FinalizedShardWorker(shard_id=shard.id, worker_id=worker_ids[1])]
            ),
        ),
    ):
        reducer.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])
        reducer.on_shard_finalized("reduce")

    assert lance.dataset(str(dataset_uri)).to_table().to_pydict() == {
        "x": [1, 2],
        "y": [90, 99],
    }
    assert all(path.exists() for path in base_paths)
    assert all(
        not (dataset_uri / path).exists() for path in created_by_worker[worker_ids[0]]
    )
    assert all(
        (dataset_uri / path).exists() for path in created_by_worker[worker_ids[1]]
    )


@pytest.mark.parametrize(
    "unsafe_path",
    ["../victim", "data/attempt__x\\..\\..\\..\\victim"],
)
def test_lance_reducer_rejects_unsafe_created_file_paths(
    tmp_path, unsafe_path: str
) -> None:
    reducer = LanceDatasetCommitReducerSink(
        tmp_path / "unsafe-cleanup.lance",
        mode="add_columns",
        source_version=1,
    )
    rel_path = "_refiner_lance_fragments/job/0123456789ab__w0123456789ab.jsonl"
    with reducer.output.open(rel_path, mode="wt", encoding="utf-8") as metadata:
        json.dump(
            {
                "schema": _schema_to_base64(pa.schema([("y", pa.int64())])),
                "fragments": ["{}"],
                "created_files": [unsafe_path],
                "source_version": 1,
                "source_fragment_id": 0,
            },
            metadata,
        )

    with pytest.raises(ValueError, match="Invalid Lance created-file path"):
        reducer._read_metadata(rel_path)


def test_lance_reducer_rejects_backslashes_in_fragment_paths(tmp_path) -> None:
    fragment = json.dumps({"files": [{"path": "attempt__x\\..\\victim"}]})

    with pytest.raises(ValueError, match="Invalid Lance fragment file path"):
        _fragment_data_paths(fragment)


def test_lance_reducer_rejects_files_outside_worker_attempt(tmp_path) -> None:
    reducer = LanceDatasetCommitReducerSink(
        tmp_path / "unsafe-attempt.lance",
        mode="create",
    )
    rel_path = "_refiner_lance_fragments/job/0123456789ab__w0123456789ab.jsonl"
    fragment = json.dumps({"files": [{"path": "other/file.lance"}]})

    with pytest.raises(ValueError, match="outside its worker attempt"):
        reducer._verified_created_files(
            None,
            fragments=[fragment],
            created_files=["data/other/file.lance"],
            source_version=None,
            source_fragment_id=None,
            metadata_path=rel_path,
        )


def test_lance_reducer_cleans_files_after_schema_mismatch(tmp_path) -> None:
    pytest.importorskip("lance")
    output_dir = tmp_path / "schema-mismatch.lance"
    finalized = [
        FinalizedShardWorker(shard_id="0123456789ab", worker_id="worker-1"),
        FinalizedShardWorker(shard_id="abcdef012345", worker_id="worker-2"),
    ]
    runtime = cast(RuntimeLifecycle, _FinalizedWorkersRuntime(finalized))

    for finalized_worker, metadata_value in zip(
        finalized, [b"one", b"two"], strict=True
    ):
        sink = LanceDatasetSink(output_dir)
        with set_active_run_context(
            job_id="job",
            stage_index=0,
            worker_id=finalized_worker.worker_id,
            worker_name=None,
            runtime_lifecycle=runtime,
        ):
            sink.write_shard_block(
                finalized_worker.shard_id,
                Tabular(
                    pa.Table.from_arrays(
                        [pa.array([1], type=pa.int64())],
                        schema=pa.schema(
                            [
                                pa.field(
                                    "x", pa.int64(), metadata={b"kind": metadata_value}
                                )
                            ]
                        ),
                    )
                ),
            )
            sink.on_shard_complete(finalized_worker.shard_id)

    loser = LanceDatasetSink(output_dir)
    with set_active_run_context(
        job_id="job",
        stage_index=0,
        worker_id="worker-loser",
        worker_name=None,
        runtime_lifecycle=runtime,
    ):
        loser.write_block([DictRow({"x": 99}, shard_id=finalized[0].shard_id)])
        loser.on_shard_complete(finalized[0].shard_id)

    reducer = LanceDatasetCommitReducerSink(output_dir, mode="create")
    with set_active_run_context(
        job_id="job",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=runtime,
    ):
        with pytest.raises(ValueError, match="inconsistent schemas"):
            reducer.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])

    assert not list((output_dir / "data").glob("*.lance"))
    assert list((output_dir / "_refiner_lance_fragments").glob("**/*.jsonl"))


def test_launch_local_vectorized_filter_with_sink_completes_shards(tmp_path) -> None:
    output_dir = tmp_path / "vectorized-output"
    pipeline = (
        from_items([{"x": 1}, {"x": 2}, {"x": 3}], items_per_shard=2)
        .filter(col("x") > 1)
        .write_jsonl(output_dir)
    )

    stats = pipeline.launch_local(
        name="vectorized-jsonl-sink",
        num_workers=1,
        rundir=str(tmp_path / "run"),
    )

    assert stats.claimed == 3
    assert stats.completed == 3
    assert stats.output_rows == 2
    written = sorted(path for path in output_dir.iterdir() if path.suffix == ".jsonl")
    assert len(written) == 2
    assert all("__w" in path.name for path in written)


def test_jsonl_sink_uses_local_worker_suffix_outside_runtime(tmp_path) -> None:
    sink = JsonlSink(tmp_path)
    sink.write_block([DictRow({"x": 1}, shard_id="abc")])
    sink.on_shard_complete("abc")

    written = sorted(tmp_path.iterdir())
    assert [path.name for path in written] == [
        f"abc__w{worker_token_for('local')}.jsonl"
    ]
    assert json.loads(written[0].read_text(encoding="utf-8")) == {"x": 1}


def test_jsonl_sink_serializes_numpy_values(tmp_path) -> None:
    sink = JsonlSink(tmp_path)
    sink.write_shard_block(
        "abc",
        [DictRow({"array": np.array([[1, 2], [3, 4]]), "scalar": np.int64(5)})],
    )
    sink.on_shard_complete("abc")

    written = sorted(tmp_path.iterdir())
    assert json.loads(written[0].read_text(encoding="utf-8")) == {
        "array": [[1, 2], [3, 4]],
        "scalar": 5,
    }


def test_parquet_sink_uploads_asset_columns(tmp_path) -> None:
    source = tmp_path / "source clip.mp4"
    source.write_bytes(b"video-bytes")
    output_dir = tmp_path / "parquet-assets"
    shard_id = "0123456789ab"
    worker_id = "worker-1"
    table = datatype.apply_dtypes_to_table(
        pa.table({"video": [str(source), None], "label": ["keep", "none"]}),
        {"video": datatype.video_path()},
    )
    sink = ParquetSink(output_dir, upload_assets=True)

    with set_active_run_context(
        job_id="job",
        stage_index=0,
        worker_id=worker_id,
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, _FinalizedWorkersRuntime([])),
    ):
        sink.write_shard_block(shard_id, Tabular(table))
        sink.on_shard_complete(shard_id)

    worker = worker_token_for(worker_id)
    asset = (
        output_dir / "assets" / f"{shard_id}__w{worker}" / "video" / "0-source_clip.mp4"
    )
    written = output_dir / f"{shard_id}__w{worker}.parquet"
    assert asset.read_bytes() == b"video-bytes"
    out = pq.read_table(written)
    assert out.column("video").to_pylist() == [str(asset), None]
    assert out.schema.field("video").metadata == {b"asset_type": b"video"}


def test_parquet_sink_does_not_upload_embedded_assets(tmp_path) -> None:
    output_dir = tmp_path / "embedded-assets"
    shard_id = "0123456789ab"
    worker_id = "worker-1"
    field = datatype.image_bytes_with_path().with_name("image")
    table = pa.Table.from_arrays(
        [
            pa.array(
                [{"bytes": b"image-bytes", "path": "source.png"}],
                type=field.type,
            )
        ],
        schema=pa.schema([field]),
    )
    sink = ParquetSink(output_dir, upload_assets=True)

    with set_active_run_context(
        job_id="job",
        stage_index=0,
        worker_id=worker_id,
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, _FinalizedWorkersRuntime([])),
    ):
        sink.write_shard_block(shard_id, Tabular(table))
        sink.on_shard_complete(shard_id)

    worker = worker_token_for(worker_id)
    written = output_dir / f"{shard_id}__w{worker}.parquet"
    out = pq.read_table(written)
    assert not (output_dir / "assets").exists()
    assert out.column("image").to_pylist() == [
        {"bytes": b"image-bytes", "path": "source.png"}
    ]
    assert datatype.asset_storage(out.schema.field("image")) == "bytes_with_path"


def test_parquet_sink_can_drop_rows_with_missing_assets(tmp_path) -> None:
    source = tmp_path / "source.png"
    missing = tmp_path / "missing.png"
    source.write_bytes(b"image")
    output_dir = tmp_path / "parquet-drop-missing-assets"
    shard_id = "0123456789ab"
    worker_id = "worker-1"
    table = datatype.apply_dtypes_to_table(
        pa.table(
            {
                "image": [str(source), str(source), str(missing)],
                "images": [[str(source)], [str(source), str(missing)], [str(source)]],
                "label": ["keep", "drop-list", "drop-scalar"],
            }
        ),
        {
            "image": datatype.image_path(),
            "images": datatype.list(datatype.image_path()),
        },
    )
    sink = ParquetSink(output_dir, upload_assets=True, missing_asset_policy="drop_row")

    with set_active_run_context(
        job_id="job",
        stage_index=0,
        worker_id=worker_id,
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, _FinalizedWorkersRuntime([])),
    ):
        sink.write_shard_block(shard_id, Tabular(table))
        sink.on_shard_complete(shard_id)

    worker = worker_token_for(worker_id)
    asset = output_dir / "assets" / f"{shard_id}__w{worker}" / "image" / "0-source.png"
    list_asset = (
        output_dir / "assets" / f"{shard_id}__w{worker}" / "images" / "0-0-source.png"
    )
    written = output_dir / f"{shard_id}__w{worker}.parquet"
    out = pq.read_table(written)
    assert asset.read_bytes() == b"image"
    assert list_asset.read_bytes() == b"image"
    assert out.column("image").to_pylist() == [str(asset)]
    assert out.column("images").to_pylist() == [[str(list_asset)]]
    assert out.column("label").to_pylist() == ["keep"]


def test_parquet_sink_can_set_missing_list_assets_to_null(tmp_path) -> None:
    source = tmp_path / "source.png"
    missing = tmp_path / "missing.png"
    source.write_bytes(b"image")
    output_dir = tmp_path / "parquet-null-missing-assets"
    shard_id = "0123456789ab"
    worker_id = "worker-1"
    table = datatype.apply_dtypes_to_table(
        pa.table(
            {
                "images": [[str(source), str(missing)], [str(missing)]],
                "label": ["partial", "missing"],
            }
        ),
        {"images": datatype.list(datatype.image_path())},
    )
    sink = ParquetSink(output_dir, upload_assets=True, missing_asset_policy="set_null")

    with set_active_run_context(
        job_id="job",
        stage_index=0,
        worker_id=worker_id,
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, _FinalizedWorkersRuntime([])),
    ):
        sink.write_shard_block(shard_id, Tabular(table))
        sink.on_shard_complete(shard_id)

    worker = worker_token_for(worker_id)
    asset = (
        output_dir / "assets" / f"{shard_id}__w{worker}" / "images" / "0-0-source.png"
    )
    written = output_dir / f"{shard_id}__w{worker}.parquet"
    out = pq.read_table(written)
    assert asset.read_bytes() == b"image"
    assert out.column("images").to_pylist() == [[str(asset), None], [None]]
    assert out.column("label").to_pylist() == ["partial", "missing"]


def test_jsonl_sink_can_set_missing_assets_to_null(tmp_path) -> None:
    source = tmp_path / "source.png"
    missing = tmp_path / "missing.png"
    source.write_bytes(b"image")
    output_dir = tmp_path / "jsonl-null-missing-assets"
    shard_id = "0123456789ab"
    worker_id = "worker-1"
    sink = JsonlSink(output_dir, upload_assets=True, missing_asset_policy="set_null")
    sink.set_input_schema(
        pa.schema(
            [
                datatype.image_path().with_name("image"),
                pa.field("images", pa.list_(datatype.image_path())),
            ]
        )
    )

    with set_active_run_context(
        job_id="job",
        stage_index=0,
        worker_id=worker_id,
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, _FinalizedWorkersRuntime([])),
    ):
        sink.write_shard_block(
            shard_id,
            [
                DictRow(
                    {
                        "image": str(source),
                        "images": [str(source), str(missing)],
                        "label": "keep",
                    }
                ),
                DictRow(
                    {
                        "image": str(missing),
                        "images": [str(missing)],
                        "label": "null",
                    }
                ),
            ],
        )
        sink.on_shard_complete(shard_id)

    worker = worker_token_for(worker_id)
    asset = output_dir / "assets" / f"{shard_id}__w{worker}" / "image" / "0-source.png"
    list_asset = (
        output_dir / "assets" / f"{shard_id}__w{worker}" / "images" / "0-0-source.png"
    )
    jsonl = output_dir / f"{shard_id}__w{worker}.jsonl"
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert asset.read_bytes() == b"image"
    assert list_asset.read_bytes() == b"image"
    assert rows == [
        {"image": str(asset), "images": [str(list_asset), None], "label": "keep"},
        {"image": None, "images": [None], "label": "null"},
    ]


def test_jsonl_sink_error_policy_raises_on_later_failed_asset(tmp_path) -> None:
    source = tmp_path / "source.png"
    missing = tmp_path / "missing.png"
    source.write_bytes(b"image")
    output_dir = tmp_path / "jsonl-error-missing-assets"
    shard_id = "0123456789ab"
    worker_id = "worker-1"
    sink = JsonlSink(output_dir, upload_assets=True)
    sink.set_input_schema(pa.schema([datatype.image_path().with_name("image")]))

    with set_active_run_context(
        job_id="job",
        stage_index=0,
        worker_id=worker_id,
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, _FinalizedWorkersRuntime([])),
    ):
        with pytest.raises(FileNotFoundError):
            sink.write_shard_block(
                shard_id,
                [
                    DictRow({"image": str(source), "label": "valid"}),
                    DictRow({"image": str(missing), "label": "missing"}),
                ],
            )


def test_asset_upload_rejects_unsafe_assets_subdir(tmp_path) -> None:
    with pytest.raises(ValueError, match="assets_subdir"):
        JsonlSink(tmp_path / "jsonl-assets", upload_assets=True, assets_subdir="../x")

    with pytest.raises(ValueError, match="assets_subdir"):
        ParquetSink(
            tmp_path / "parquet-assets",
            upload_assets=True,
            assets_subdir="a/../x",
        )


def test_asset_upload_sanitizes_column_path_segment(tmp_path) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    output_dir = tmp_path / "column-segment-assets"
    shard_id = "0123456789ab"
    worker_id = "worker-1"
    field = datatype.image_path().with_name("../image")
    table = pa.Table.from_arrays(
        [pa.array([str(source)], type=field.type)],
        schema=pa.schema([field]),
    )
    sink = ParquetSink(output_dir, upload_assets=True)

    with set_active_run_context(
        job_id="job",
        stage_index=0,
        worker_id=worker_id,
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, _FinalizedWorkersRuntime([])),
    ):
        sink.write_shard_block(shard_id, Tabular(table))
        sink.on_shard_complete(shard_id)

    uploaded = list((output_dir / "assets").glob("**/0-source.png"))
    assert len(uploaded) == 1
    assert uploaded[0].read_bytes() == b"image"
    assert tmp_path.joinpath("image", "0-source.png").exists() is False


def test_asset_upload_disambiguates_sanitized_column_segments(tmp_path) -> None:
    first = tmp_path / "first" / "asset.png"
    second = tmp_path / "second" / "asset.png"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    output_dir = tmp_path / "column-collision-assets"
    shard_id = "0123456789ab"
    worker_id = "worker-1"
    schema = pa.schema(
        [
            datatype.image_path().with_name("a/b"),
            datatype.image_path().with_name("a?b"),
        ]
    )
    table = pa.Table.from_pydict(
        {"a/b": [str(first)], "a?b": [str(second)]},
        schema=schema,
    )
    sink = ParquetSink(output_dir, upload_assets=True)

    with set_active_run_context(
        job_id="job",
        stage_index=0,
        worker_id=worker_id,
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, _FinalizedWorkersRuntime([])),
    ):
        sink.write_shard_block(shard_id, Tabular(table))
        sink.on_shard_complete(shard_id)

    worker = worker_token_for(worker_id)
    first_asset = (
        output_dir / "assets" / f"{shard_id}__w{worker}" / "a_b" / "0-asset.png"
    )
    second_asset = (
        output_dir / "assets" / f"{shard_id}__w{worker}" / "a_b_2" / "0-asset.png"
    )
    written = output_dir / f"{shard_id}__w{worker}.parquet"
    assert first_asset.read_bytes() == b"first"
    assert second_asset.read_bytes() == b"second"
    out = pq.read_table(written)
    assert out.column("a/b").to_pylist() == [str(first_asset)]
    assert out.column("a?b").to_pylist() == [str(second_asset)]


def test_jsonl_sink_uploads_assets_with_shard_local_row_indexes(tmp_path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    output_dir = tmp_path / "jsonl-assets"
    shard_id = "0123456789ab"
    worker_id = "worker-1"
    sink = JsonlSink(output_dir, upload_assets=True)

    with set_active_run_context(
        job_id="job",
        stage_index=0,
        worker_id=worker_id,
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, _FinalizedWorkersRuntime([])),
    ):
        for path in [first, second]:
            table = datatype.apply_dtypes_to_table(
                pa.table({"image": [str(path)]}),
                {"image": datatype.image_path()},
            )
            sink.write_shard_block(shard_id, Tabular(table))
        sink.on_shard_complete(shard_id)

    worker = worker_token_for(worker_id)
    asset_dir = output_dir / "assets" / f"{shard_id}__w{worker}" / "image"
    assert (asset_dir / "0-first.png").read_bytes() == b"first"
    assert (asset_dir / "1-second.png").read_bytes() == b"second"
    jsonl = output_dir / f"{shard_id}__w{worker}.jsonl"
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"image": str(asset_dir / "0-first.png")},
        {"image": str(asset_dir / "1-second.png")},
    ]


def test_jsonl_sink_uploads_assets_from_row_blocks_without_tabularizing(
    tmp_path,
) -> None:
    source = tmp_path / "frame.png"
    source.write_bytes(b"frame")
    output_dir = tmp_path / "jsonl-row-assets"
    shard_id = "0123456789ab"
    worker_id = "worker-1"
    sink = JsonlSink(output_dir, upload_assets=True)
    sink.set_input_schema(pa.schema([datatype.image_path().with_name("image")]))

    with set_active_run_context(
        job_id="job",
        stage_index=0,
        worker_id=worker_id,
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, _FinalizedWorkersRuntime([])),
    ):
        sink.write_shard_block(shard_id, [DictRow({"image": str(source)})])
        sink.on_shard_complete(shard_id)

    worker = worker_token_for(worker_id)
    asset = output_dir / "assets" / f"{shard_id}__w{worker}" / "image" / "0-frame.png"
    jsonl = output_dir / f"{shard_id}__w{worker}.jsonl"
    assert asset.read_bytes() == b"frame"
    assert json.loads(jsonl.read_text(encoding="utf-8")) == {"image": str(asset)}


def test_row_asset_upload_requires_input_schema(tmp_path) -> None:
    row: list[Row] = [DictRow({"image": str(tmp_path / "frame.png")})]

    jsonl = JsonlSink(tmp_path / "jsonl-row-assets", upload_assets=True)
    with pytest.raises(ValueError, match="input schema"):
        jsonl.write_shard_block("0123456789ab", row)

    parquet = ParquetSink(tmp_path / "parquet-row-assets", upload_assets=True)
    with pytest.raises(ValueError, match="input schema"):
        parquet.write_shard_block("0123456789ab", row)


def test_jsonl_pipeline_uploads_row_assets_from_dtype_schema(tmp_path) -> None:
    source = tmp_path / "frame.png"
    source.write_bytes(b"frame")
    output_dir = tmp_path / "jsonl-pipeline-assets"
    pipeline = (
        from_items([{"image": str(source)}])
        .map(
            lambda row: {"image": row["image"]},
            dtypes={"image": datatype.image_path()},
        )
        .write_jsonl(output_dir, upload_assets=True)
    )

    stats = pipeline.launch_local(
        name="jsonl-row-asset-upload",
        num_workers=1,
        rundir=str(tmp_path / "run"),
    )

    asset = next((output_dir / "assets").glob("*/image/0-frame.png"))
    written = next(output_dir.glob("*.jsonl"))
    assert stats.output_rows == 1
    assert asset.read_bytes() == b"frame"
    assert json.loads(written.read_text(encoding="utf-8")) == {"image": str(asset)}


def test_jsonl_pipeline_counts_rows_after_missing_asset_drop(tmp_path) -> None:
    source = tmp_path / "frame.png"
    missing = tmp_path / "missing.png"
    source.write_bytes(b"frame")
    output_dir = tmp_path / "jsonl-pipeline-drop-assets"
    pipeline = (
        from_items([{"image": str(source)}, {"image": str(missing)}])
        .map(
            lambda row: {"image": row["image"]},
            dtypes={"image": datatype.image_path()},
        )
        .write_jsonl(
            output_dir,
            upload_assets=True,
            missing_asset_policy="drop_row",
        )
    )

    stats = pipeline.launch_local(
        name="jsonl-drop-asset-counts",
        num_workers=1,
        rundir=str(tmp_path / "run"),
    )

    written = next(output_dir.glob("*.jsonl"))
    rows = [
        json.loads(line) for line in written.read_text(encoding="utf-8").splitlines()
    ]
    assert stats.output_rows == 1
    assert len(rows) == 1


def test_parquet_pipeline_uploads_row_assets_from_dtype_schema(tmp_path) -> None:
    source = tmp_path / "frame.png"
    source.write_bytes(b"frame")
    output_dir = tmp_path / "parquet-pipeline-assets"
    pipeline = (
        from_items([{"image": str(source)}])
        .map(
            lambda row: {"image": row["image"]},
            dtypes={"image": datatype.image_path()},
        )
        .write_parquet(output_dir, upload_assets=True)
    )

    stats = pipeline.launch_local(
        name="parquet-row-asset-upload",
        num_workers=1,
        rundir=str(tmp_path / "run"),
    )

    asset = next((output_dir / "assets").glob("*/image/0-frame.png"))
    written = next(output_dir.glob("*.parquet"))
    table = pq.read_table(written)
    assert stats.output_rows == 1
    assert asset.read_bytes() == b"frame"
    assert table.column("image").to_pylist() == [str(asset)]
    assert table.schema.field("image").metadata == {b"asset_type": b"image"}


def test_parquet_sink_uploads_list_asset_columns(tmp_path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    output_dir = tmp_path / "parquet-list-assets"
    shard_id = "0123456789ab"
    worker_id = "worker-1"
    field = pa.field("images", pa.list_(datatype.image_path()))
    table = pa.Table.from_arrays(
        [pa.array([[str(first), str(second)], None], type=field.type)],
        schema=pa.schema([field]),
    )
    sink = ParquetSink(output_dir, upload_assets=True)

    with set_active_run_context(
        job_id="job",
        stage_index=0,
        worker_id=worker_id,
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, _FinalizedWorkersRuntime([])),
    ):
        sink.write_shard_block(shard_id, Tabular(table))
        sink.on_shard_complete(shard_id)

    worker = worker_token_for(worker_id)
    asset_dir = output_dir / "assets" / f"{shard_id}__w{worker}" / "images"
    assert (asset_dir / "0-0-first.png").read_bytes() == b"first"
    assert (asset_dir / "0-1-second.png").read_bytes() == b"second"
    written = output_dir / f"{shard_id}__w{worker}.parquet"
    out = pq.read_table(written)
    assert out.column("images").to_pylist() == [
        [
            str(asset_dir / "0-0-first.png"),
            str(asset_dir / "0-1-second.png"),
        ],
        None,
    ]


def test_jsonl_sink_uploads_tuple_asset_columns(tmp_path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    output_dir = tmp_path / "jsonl-tuple-assets"
    shard_id = "0123456789ab"
    worker_id = "worker-1"
    sink = JsonlSink(output_dir, upload_assets=True)
    sink.set_input_schema(
        pa.schema([pa.field("images", pa.list_(datatype.image_path()))])
    )

    with set_active_run_context(
        job_id="job",
        stage_index=0,
        worker_id=worker_id,
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, _FinalizedWorkersRuntime([])),
    ):
        sink.write_shard_block(
            shard_id,
            [DictRow({"images": (str(first), str(second))})],
        )
        sink.on_shard_complete(shard_id)

    worker = worker_token_for(worker_id)
    asset_dir = output_dir / "assets" / f"{shard_id}__w{worker}" / "images"
    assert (asset_dir / "0-0-first.png").read_bytes() == b"first"
    assert (asset_dir / "0-1-second.png").read_bytes() == b"second"
    jsonl = output_dir / f"{shard_id}__w{worker}.jsonl"
    assert json.loads(jsonl.read_text(encoding="utf-8")) == {
        "images": [
            str(asset_dir / "0-0-first.png"),
            str(asset_dir / "0-1-second.png"),
        ]
    }


def test_jsonl_reducer_keeps_only_finalized_worker_outputs(tmp_path) -> None:
    output_dir = tmp_path / "jsonl-cleanup"
    shard_id = "0123456789ab"
    worker_ids = ["worker-1", "worker-2"]

    for worker_id, value in zip(worker_ids, [1, 9], strict=True):
        sink = JsonlSink(output_dir)
        with set_active_run_context(
            job_id="job",
            stage_index=0,
            worker_id=worker_id,
            worker_name=None,
            runtime_lifecycle=cast(
                RuntimeLifecycle,
                _FinalizedWorkersRuntime(
                    [FinalizedShardWorker(shard_id=shard_id, worker_id=worker_ids[1])]
                ),
            ),
        ):
            sink.write_block([DictRow({"x": value}, shard_id=shard_id)])
            sink.on_shard_complete(shard_id)

    reducer = JsonlSink(output_dir).build_reducer()
    assert reducer is not None
    with set_active_run_context(
        job_id="job",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=cast(
            RuntimeLifecycle,
            _FinalizedWorkersRuntime(
                [FinalizedShardWorker(shard_id=shard_id, worker_id=worker_ids[1])]
            ),
        ),
    ):
        reducer.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])
        reducer.on_shard_finalized("reduce")

    kept = output_dir / f"{shard_id}__w{worker_token_for(worker_ids[1])}.jsonl"
    deleted = output_dir / f"{shard_id}__w{worker_token_for(worker_ids[0])}.jsonl"
    assert kept.exists()
    assert not deleted.exists()
    assert json.loads(kept.read_text(encoding="utf-8")) == {"x": 9}


def test_parquet_reducer_keeps_only_finalized_worker_outputs(tmp_path) -> None:
    output_dir = tmp_path / "parquet-cleanup"
    shard_id = "0123456789ab"
    worker_ids = ["worker-1", "worker-2"]

    for worker_id, value in zip(worker_ids, [1, 9], strict=True):
        sink = ParquetSink(output_dir)
        with set_active_run_context(
            job_id="job",
            stage_index=0,
            worker_id=worker_id,
            worker_name=None,
            runtime_lifecycle=cast(
                RuntimeLifecycle,
                _FinalizedWorkersRuntime(
                    [FinalizedShardWorker(shard_id=shard_id, worker_id=worker_ids[1])]
                ),
            ),
        ):
            sink.write_block([DictRow({"x": value}, shard_id=shard_id)])
            sink.on_shard_complete(shard_id)

    reducer = ParquetSink(output_dir).build_reducer()
    assert reducer is not None
    with set_active_run_context(
        job_id="job",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=cast(
            RuntimeLifecycle,
            _FinalizedWorkersRuntime(
                [FinalizedShardWorker(shard_id=shard_id, worker_id=worker_ids[1])]
            ),
        ),
    ):
        reducer.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])

    kept = output_dir / f"{shard_id}__w{worker_token_for(worker_ids[1])}.parquet"
    deleted = output_dir / f"{shard_id}__w{worker_token_for(worker_ids[0])}.parquet"
    assert kept.exists()
    assert not deleted.exists()
    assert pq.read_table(kept).column("x").to_pylist() == [9]


def test_lance_reducer_keeps_only_finalized_worker_outputs(tmp_path) -> None:
    pytest.importorskip("lance")
    from lance.file import LanceFileReader

    output_dir = tmp_path / "lance-file-cleanup"
    shard_id = "0123456789ab"
    worker_ids = ["worker-1", "worker-2"]

    for worker_id, value in zip(worker_ids, [1, 9], strict=True):
        sink = from_items([]).write_lance(output_dir).sink
        assert sink is not None
        with set_active_run_context(
            job_id="job",
            stage_index=0,
            worker_id=worker_id,
            worker_name=None,
            runtime_lifecycle=cast(
                RuntimeLifecycle,
                _FinalizedWorkersRuntime(
                    [FinalizedShardWorker(shard_id=shard_id, worker_id=worker_ids[1])]
                ),
            ),
        ):
            sink.write_block([DictRow({"x": value}, shard_id=shard_id)])
            sink.on_shard_complete(shard_id)

    reducer = from_items([]).write_lance(output_dir).sink
    assert reducer is not None
    reducer = reducer.build_reducer()
    assert reducer is not None
    with set_active_run_context(
        job_id="job",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=cast(
            RuntimeLifecycle,
            _FinalizedWorkersRuntime(
                [FinalizedShardWorker(shard_id=shard_id, worker_id=worker_ids[1])]
            ),
        ),
    ):
        reducer.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])

    kept = output_dir / f"{shard_id}__w{worker_token_for(worker_ids[1])}.lance"
    deleted = output_dir / f"{shard_id}__w{worker_token_for(worker_ids[0])}.lance"
    assert kept.exists()
    assert not deleted.exists()
    assert LanceFileReader(str(kept)).read_all().to_table().column("x").to_pylist() == [
        9
    ]


def test_file_cleanup_reducer_removes_non_finalized_asset_attempt_dirs(
    tmp_path,
) -> None:
    output_dir = tmp_path / "asset-cleanup"
    shard_id = "0123456789ab"
    winner = worker_token_for("winner")
    loser = worker_token_for("loser")
    keep_asset = output_dir / "assets" / f"{shard_id}__w{winner}" / "video" / "0-a.mp4"
    drop_asset = output_dir / "assets" / f"{shard_id}__w{loser}" / "video" / "0-a.mp4"
    keep_asset.parent.mkdir(parents=True)
    drop_asset.parent.mkdir(parents=True)
    keep_asset.write_bytes(b"keep")
    drop_asset.write_bytes(b"drop")
    unmanaged = output_dir / "assets" / "manual" / "keep.txt"
    unmanaged.parent.mkdir(parents=True)
    unmanaged.write_text("keep", encoding="utf-8")

    reducer = FileCleanupReducerSink(
        output_dir,
        filename_template="{shard_id}__w{worker_id}.jsonl",
        reducer_name="cleanup_jsonl",
        assets_subdir="assets",
    )

    with set_active_run_context(
        job_id="job",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=cast(
            RuntimeLifecycle,
            _FinalizedWorkersRuntime(
                [FinalizedShardWorker(shard_id=shard_id, worker_id="winner")]
            ),
        ),
    ):
        reducer.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])

    assert keep_asset.exists()
    assert unmanaged.exists()
    assert not drop_asset.exists()


def test_lance_dataset_reducer_commits_only_finalized_worker_outputs(
    tmp_path, monkeypatch
) -> None:
    lance = pytest.importorskip("lance")

    output_dir = tmp_path / "lance-cleanup.lance"
    shard_id = "0123456789ab"
    worker_ids = ["worker-1", "worker-2"]

    for worker_id, value in zip(worker_ids, [1, 9], strict=True):
        sink = LanceDatasetSink(output_dir)
        with set_active_run_context(
            job_id="job",
            stage_index=0,
            worker_id=worker_id,
            worker_name=None,
            runtime_lifecycle=cast(
                RuntimeLifecycle,
                _FinalizedWorkersRuntime(
                    [FinalizedShardWorker(shard_id=shard_id, worker_id=worker_ids[1])]
                ),
            ),
        ):
            sink.write_block([DictRow({"x": value}, shard_id=shard_id)])
            sink.on_shard_complete(shard_id)

    reducer = LanceDatasetSink(output_dir).build_reducer()
    assert isinstance(reducer, LanceDatasetCommitReducerSink)
    listed_prefixes: list[str] = []
    original_find = reducer.output.find

    def _recording_find(path: str):
        listed_prefixes.append(path)
        paths = original_find(path)
        return [*paths, *paths]

    monkeypatch.setattr(reducer.output, "find", _recording_find)
    with set_active_run_context(
        job_id="job",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=cast(
            RuntimeLifecycle,
            _FinalizedWorkersRuntime(
                [FinalizedShardWorker(shard_id=shard_id, worker_id=worker_ids[1])]
            ),
        ),
    ):
        reducer.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])
        reducer.on_shard_finalized("reduce")

    table = lance.dataset(str(output_dir)).to_table()
    assert table.column("x").to_pylist() == [9]
    assert len(list((output_dir / "data").glob("*.lance"))) == 1
    assert listed_prefixes == ["_refiner_lance_fragments/job"]
    assert not any((output_dir / "_refiner_lance_fragments" / "job").glob("*.jsonl"))


def test_lance_dataset_reducer_finds_finalized_metadata_from_resumed_job(
    tmp_path,
) -> None:
    lance = pytest.importorskip("lance")
    output_dir = tmp_path / "lance-resume.lance"
    shard_id = "0123456789ab"
    worker_ids = ["worker-1", "worker-2"]
    runtime = cast(
        RuntimeLifecycle,
        _FinalizedWorkersRuntime(
            [
                FinalizedShardWorker(
                    shard_id=shard_id,
                    worker_id=worker_ids[1],
                    job_id="original-job",
                )
            ]
        ),
    )
    for worker_id, value in zip(worker_ids, [1, 9], strict=True):
        sink = LanceDatasetSink(output_dir)
        with set_active_run_context(
            job_id="original-job",
            stage_index=0,
            worker_id=worker_id,
            worker_name=None,
            runtime_lifecycle=runtime,
        ):
            sink.write_block([DictRow({"x": value}, shard_id=shard_id)])
            sink.on_shard_complete(shard_id)

    reducer = sink.build_reducer()
    assert isinstance(reducer, LanceDatasetCommitReducerSink)
    with set_active_run_context(
        job_id="resumed-job",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=runtime,
    ):
        reducer.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])
        reducer.on_shard_finalized("reduce")

    assert lance.dataset(str(output_dir)).to_table().to_pydict() == {"x": [9]}
    assert len(list((output_dir / "data").glob("*.lance"))) == 1
    assert not any(
        (output_dir / "_refiner_lance_fragments" / "original-job").glob("*.jsonl")
    )


def test_lance_dataset_reducer_prefers_current_job_metadata(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    output_dir = tmp_path / "lance-current-job.lance"
    shard_id = "0123456789ab"
    worker_id = "worker-1"
    runtime = cast(
        RuntimeLifecycle,
        _FinalizedWorkersRuntime(
            [FinalizedShardWorker(shard_id=shard_id, worker_id=worker_id)]
        ),
    )
    for job_id, value in (("old-job", 1), ("current-job", 9)):
        sink = LanceDatasetSink(output_dir)
        with set_active_run_context(
            job_id=job_id,
            stage_index=0,
            worker_id=worker_id,
            worker_name=None,
            runtime_lifecycle=runtime,
        ):
            sink.write_block([DictRow({"x": value}, shard_id=shard_id)])
            sink.on_shard_complete(shard_id)

    reducer = sink.build_reducer()
    assert isinstance(reducer, LanceDatasetCommitReducerSink)
    with set_active_run_context(
        job_id="current-job",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=runtime,
    ):
        reducer.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])

    assert lance.dataset(str(output_dir)).to_table().to_pydict() == {"x": [9]}


def test_lance_dataset_reducer_retry_is_idempotent(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    output_dir = tmp_path / "lance-idempotent-retry.lance"
    shard_id = "0123456789ab"
    worker_id = "worker-1"
    runtime = cast(
        RuntimeLifecycle,
        _FinalizedWorkersRuntime(
            [FinalizedShardWorker(shard_id=shard_id, worker_id=worker_id)]
        ),
    )
    sink = LanceDatasetSink(output_dir)
    with set_active_run_context(
        job_id="job",
        stage_index=0,
        worker_id=worker_id,
        worker_name=None,
        runtime_lifecycle=runtime,
    ):
        sink.write_block([DictRow({"x": 9}, shard_id=shard_id)])
        sink.on_shard_complete(shard_id)

    with set_active_run_context(
        job_id="job",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=runtime,
    ):
        first = sink.build_reducer()
        assert isinstance(first, LanceDatasetCommitReducerSink)
        first.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])
        committed_version = lance.dataset(str(output_dir)).version

        retry = sink.build_reducer()
        assert isinstance(retry, LanceDatasetCommitReducerSink)
        retry.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])
        assert lance.dataset(str(output_dir)).version == committed_version
        retry.on_shard_finalized("reduce")

    assert not any((output_dir / "_refiner_lance_fragments").glob("**/*.jsonl"))


def test_lance_overwrite_retry_finds_historical_commit(tmp_path) -> None:
    lance = pytest.importorskip("lance")
    output_dir = tmp_path / "lance-overwrite-retry.lance"
    lance.write_dataset(pa.table({"x": [0]}), str(output_dir))
    shard_id = "0123456789ab"
    worker_id = "worker-1"
    runtime = cast(
        RuntimeLifecycle,
        _FinalizedWorkersRuntime(
            [FinalizedShardWorker(shard_id=shard_id, worker_id=worker_id)]
        ),
    )
    sink = LanceDatasetSink(output_dir, mode="overwrite")
    with set_active_run_context(
        job_id="job",
        stage_index=0,
        worker_id=worker_id,
        worker_name=None,
        runtime_lifecycle=runtime,
    ):
        sink.write_block([DictRow({"x": 9}, shard_id=shard_id)])
        sink.on_shard_complete(shard_id)

    with set_active_run_context(
        job_id="job",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=runtime,
    ):
        first = sink.build_reducer()
        assert isinstance(first, LanceDatasetCommitReducerSink)
        first.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])
        lance.write_dataset(pa.table({"x": [10]}), str(output_dir), mode="append")
        concurrent_version = lance.dataset(str(output_dir)).version

        retry = sink.build_reducer()
        assert isinstance(retry, LanceDatasetCommitReducerSink)
        retry.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])

    latest = lance.dataset(str(output_dir))
    assert latest.version == concurrent_version
    assert latest.to_table().to_pydict() == {"x": [9, 10]}


def test_lance_dataset_reducer_rejects_ambiguous_resume_metadata(tmp_path) -> None:
    pytest.importorskip("lance")
    output_dir = tmp_path / "lance-ambiguous-resume.lance"
    shard_id = "0123456789ab"
    worker_id = "worker-1"
    runtime = cast(
        RuntimeLifecycle,
        _FinalizedWorkersRuntime(
            [FinalizedShardWorker(shard_id=shard_id, worker_id=worker_id)]
        ),
    )
    for job_id, value in (("old-job-1", 1), ("old-job-2", 2)):
        sink = LanceDatasetSink(output_dir)
        with set_active_run_context(
            job_id=job_id,
            stage_index=0,
            worker_id=worker_id,
            worker_name=None,
            runtime_lifecycle=runtime,
        ):
            sink.write_block([DictRow({"x": value}, shard_id=shard_id)])
            sink.on_shard_complete(shard_id)

    reducer = sink.build_reducer()
    assert isinstance(reducer, LanceDatasetCommitReducerSink)
    with set_active_run_context(
        job_id="resumed-job",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=runtime,
    ):
        with pytest.raises(ValueError, match="Ambiguous resumed Lance metadata"):
            reducer.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])


def test_lance_sinks_reject_configured_fsspec_handles() -> None:
    configured_output = ("bucket/output", MemoryFileSystem())

    with pytest.raises(ValueError, match="configured fsspec handles"):
        from_items([]).write_lance(configured_output)
    with pytest.raises(ValueError, match="configured fsspec handles"):
        from_items([]).write_lance_dataset(configured_output)


@pytest.mark.parametrize(
    "uri",
    [
        "s3://user:password@bucket/dataset.lance",
        "s3://bucket/data.lance?token=x",
        "s3://bucket/data.lance#token=x",
        "simplecache::s3://user:password@bucket/dataset.lance",
    ],
)
def test_lance_sinks_reject_secret_bearing_uris(uri: str) -> None:
    with pytest.raises(ValueError, match="must not contain credentials"):
        from_items([]).write_lance(uri)
    with pytest.raises(ValueError, match="must not contain credentials"):
        from_items([]).write_lance_dataset(uri)


def test_lance_add_columns_rejects_internal_column_names(tmp_path) -> None:
    with pytest.raises(ValueError, match="internal column"):
        LanceDatasetSink(
            tmp_path / "reserved-columns.lance",
            mode="add_columns",
            columns=[LANCE_ROW_POSITION_COLUMN],
            source_uri=str(tmp_path / "reserved-columns.lance"),
            source_version=1,
        )


def test_lance_dataset_post_commit_metadata_cleanup_is_best_effort(
    tmp_path, monkeypatch
) -> None:
    pytest.importorskip("lance")
    output_dir = tmp_path / "cleanup-failure.lance"
    shard_id = "0123456789ab"
    worker_id = "worker-1"
    runtime = cast(
        RuntimeLifecycle,
        _FinalizedWorkersRuntime(
            [FinalizedShardWorker(shard_id=shard_id, worker_id=worker_id)]
        ),
    )
    sink = LanceDatasetSink(output_dir)
    with set_active_run_context(
        job_id="job",
        stage_index=0,
        worker_id=worker_id,
        worker_name=None,
        runtime_lifecycle=runtime,
    ):
        sink.write_block([DictRow({"x": 1}, shard_id=shard_id)])
        sink.on_shard_complete(shard_id)

    reducer = sink.build_reducer()
    assert isinstance(reducer, LanceDatasetCommitReducerSink)

    def _failed_rm(_path: str) -> None:
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(reducer.output, "rm", _failed_rm)
    with set_active_run_context(
        job_id="job",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=runtime,
    ):
        reducer.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])

    assert LanceDatasetSink(output_dir)._load_existing_schema().names == ["x"]


def test_lance_dataset_sink_close_removes_unfinished_fragment_data(tmp_path) -> None:
    pytest.importorskip("lance")

    output_dir = tmp_path / "lance-unfinished-cleanup.lance"
    shard_id = "0123456789ab"
    sink = LanceDatasetSink(output_dir)
    with set_active_run_context(
        job_id="job",
        stage_index=0,
        worker_id="worker-1",
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, _FinalizedWorkersRuntime([])),
    ):
        sink.write_block([DictRow({"x": 1}, shard_id=shard_id)])
        sink.close()

    assert not any((output_dir / "data").glob("*.lance"))
    assert not any((output_dir / "_refiner_lance_fragments").glob("**/*.jsonl"))


def test_file_cleanup_reducer_ignores_extra_template_fields(tmp_path) -> None:
    output_dir = tmp_path / "jsonl-cleanup-extra"
    shard_id = "0123456789ab"
    winner_worker_id = "worker-2"
    loser_worker_id = "worker-1"
    winner_token = worker_token_for(winner_worker_id)
    loser_token = worker_token_for(loser_worker_id)

    winner_files = [
        output_dir / f"{shard_id}__w{winner_token}__part0.jsonl",
        output_dir / f"{shard_id}__w{winner_token}__part1.jsonl",
    ]
    loser_file = output_dir / f"{shard_id}__w{loser_token}__part0.jsonl"
    unmanaged_file = output_dir / "notes.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in winner_files + [loser_file]:
        path.write_text("{}", encoding="utf-8")
    unmanaged_file.write_text("keep me", encoding="utf-8")

    reducer = FileCleanupReducerSink(
        output_dir,
        filename_template="{shard_id}__w{worker_id}__{part}.jsonl",
        reducer_name="cleanup_jsonl",
    )
    with set_active_run_context(
        job_id="job",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=cast(
            RuntimeLifecycle,
            _FinalizedWorkersRuntime(
                [FinalizedShardWorker(shard_id=shard_id, worker_id=winner_worker_id)]
            ),
        ),
    ):
        reducer.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])

    assert all(path.exists() for path in winner_files)
    assert not loser_file.exists()
    assert unmanaged_file.exists()


def test_file_cleanup_reducer_removes_non_finalized_directories(tmp_path) -> None:
    output_dir = tmp_path / "zarr-cleanup"
    shard_id = "0123456789ab"
    winner_worker_id = "worker-2"
    loser_worker_id = "worker-1"
    winner_dir = output_dir / f"{shard_id}__w{worker_token_for(winner_worker_id)}.zarr"
    loser_dir = output_dir / f"{shard_id}__w{worker_token_for(loser_worker_id)}.zarr"
    (winner_dir / "data").mkdir(parents=True)
    (loser_dir / "data").mkdir(parents=True)
    (winner_dir / "data" / "0").write_bytes(b"keep")
    (loser_dir / "data" / "0").write_bytes(b"drop")

    reducer = FileCleanupReducerSink(
        output_dir,
        filename_template="{shard_id}__w{worker_id}.zarr",
        reducer_name="cleanup_zarr",
    )
    with set_active_run_context(
        job_id="job",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=cast(
            RuntimeLifecycle,
            _FinalizedWorkersRuntime(
                [FinalizedShardWorker(shard_id=shard_id, worker_id=winner_worker_id)]
            ),
        ),
    ):
        reducer.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])

    assert winner_dir.exists()
    assert not loser_dir.exists()


def test_file_cleanup_reducer_removes_non_finalized_nested_directories(
    tmp_path,
) -> None:
    output_dir = tmp_path / "zarr-cleanup-nested"
    shard_id = "0123456789ab"
    winner_worker_id = "worker-2"
    loser_worker_id = "worker-1"
    winner_dir = (
        output_dir / "split" / f"{shard_id}__w{worker_token_for(winner_worker_id)}.zarr"
    )
    loser_dir = (
        output_dir / "split" / f"{shard_id}__w{worker_token_for(loser_worker_id)}.zarr"
    )
    (winner_dir / "data").mkdir(parents=True)
    (loser_dir / "data").mkdir(parents=True)
    (winner_dir / "data" / "0").write_bytes(b"keep")
    (loser_dir / "data" / "0").write_bytes(b"drop")

    reducer = FileCleanupReducerSink(
        output_dir,
        filename_template="split/{shard_id}__w{worker_id}.zarr",
        reducer_name="cleanup_zarr",
    )
    with set_active_run_context(
        job_id="job",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=cast(
            RuntimeLifecycle,
            _FinalizedWorkersRuntime(
                [FinalizedShardWorker(shard_id=shard_id, worker_id=winner_worker_id)]
            ),
        ),
    ):
        reducer.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])

    assert winner_dir.exists()
    assert not loser_dir.exists()


def test_file_cleanup_reducer_removes_dynamic_nested_directories(tmp_path) -> None:
    output_dir = tmp_path / "zarr-cleanup-dynamic-nested"
    shard_id = "0123456789ab"
    winner_worker_id = "worker-2"
    loser_worker_id = "worker-1"
    winner_dir = (
        output_dir / "split" / shard_id / f"{worker_token_for(winner_worker_id)}.zarr"
    )
    loser_dir = (
        output_dir / "split" / shard_id / f"{worker_token_for(loser_worker_id)}.zarr"
    )
    (winner_dir / "data").mkdir(parents=True)
    (loser_dir / "data").mkdir(parents=True)
    (winner_dir / "data" / "0").write_bytes(b"keep")
    (loser_dir / "data" / "0").write_bytes(b"drop")

    reducer = FileCleanupReducerSink(
        output_dir,
        filename_template="split/{shard_id}/{worker_id}.zarr",
        reducer_name="cleanup_zarr",
    )
    with set_active_run_context(
        job_id="job",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=cast(
            RuntimeLifecycle,
            _FinalizedWorkersRuntime(
                [FinalizedShardWorker(shard_id=shard_id, worker_id=winner_worker_id)]
            ),
        ),
    ):
        reducer.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])

    assert winner_dir.exists()
    assert not loser_dir.exists()


def test_file_cleanup_reducer_ignores_files_during_template_listing(
    tmp_path,
) -> None:
    output_dir = tmp_path / "zarr-cleanup-mixed"
    shard_id = "0123456789ab"
    winner_worker_id = "worker-2"
    loser_worker_id = "worker-1"
    winner_dir = (
        output_dir / "split" / shard_id / f"{worker_token_for(winner_worker_id)}.zarr"
    )
    loser_dir = (
        output_dir / "split" / shard_id / f"{worker_token_for(loser_worker_id)}.zarr"
    )
    (winner_dir / "data").mkdir(parents=True)
    (loser_dir / "data").mkdir(parents=True)
    (winner_dir / "data" / "0").write_bytes(b"keep")
    (loser_dir / "data" / "0").write_bytes(b"drop")
    (output_dir / "split" / "README.txt").write_text("notes", encoding="utf-8")

    reducer = FileCleanupReducerSink(
        output_dir,
        filename_template="split/{shard_id}/{worker_id}.zarr",
        reducer_name="cleanup_zarr",
    )
    with set_active_run_context(
        job_id="job",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=cast(
            RuntimeLifecycle,
            _FinalizedWorkersRuntime(
                [FinalizedShardWorker(shard_id=shard_id, worker_id=winner_worker_id)]
            ),
        ),
    ):
        reducer.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])

    assert winner_dir.exists()
    assert not loser_dir.exists()
    assert (output_dir / "split" / "README.txt").read_text(encoding="utf-8") == "notes"


def test_file_cleanup_reducer_propagates_template_listing_errors(
    tmp_path, monkeypatch
) -> None:
    output_dir = tmp_path / "zarr-cleanup-list-error"
    output_dir.mkdir()
    reducer = FileCleanupReducerSink(
        output_dir,
        filename_template="{shard_id}__w{worker_id}.zarr",
        reducer_name="cleanup_zarr",
    )

    def fail_ls(*_args, **_kwargs):
        raise OSError("list failed")

    monkeypatch.setattr(reducer.output, "ls", fail_ls)

    with set_active_run_context(
        job_id="job",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=cast(RuntimeLifecycle, _FinalizedWorkersRuntime([])),
    ):
        with pytest.raises(OSError, match="list failed"):
            reducer.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])


def test_file_cleanup_reducer_tolerates_duplicate_listed_paths(
    tmp_path, monkeypatch
) -> None:
    output_dir = tmp_path / "jsonl-cleanup-duplicates"
    shard_id = "0123456789ab"
    winner_worker_id = "worker-2"
    loser_worker_id = "worker-1"
    winner_path = (
        output_dir / f"{shard_id}__w{worker_token_for(winner_worker_id)}.jsonl"
    )
    loser_path = output_dir / f"{shard_id}__w{worker_token_for(loser_worker_id)}.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    winner_path.write_text("{}", encoding="utf-8")
    loser_path.write_text("{}", encoding="utf-8")

    reducer = FileCleanupReducerSink(
        output_dir,
        filename_template="{shard_id}__w{worker_id}.jsonl",
        reducer_name="cleanup_jsonl",
    )
    monkeypatch.setattr(
        reducer.output,
        "ls",
        lambda *_args, **_kwargs: [
            winner_path.name,
            winner_path.name,
            loser_path.name,
        ],
    )

    with set_active_run_context(
        job_id="job",
        stage_index=1,
        worker_id="reducer",
        worker_name=None,
        runtime_lifecycle=cast(
            RuntimeLifecycle,
            _FinalizedWorkersRuntime(
                [FinalizedShardWorker(shard_id=shard_id, worker_id=winner_worker_id)]
            ),
        ),
    ):
        reducer.write_block([DictRow({"task_rank": 0}, shard_id="reduce")])

    assert winner_path.exists()
    assert not loser_path.exists()


def test_jsonl_sink_rejects_unsupported_cleanup_filename_template(tmp_path) -> None:
    sink = JsonlSink(
        tmp_path / "jsonl-custom",
        filename_template="{shard_id}.jsonl",
    )

    with pytest.raises(ValueError, match="requires fields"):
        sink.build_reducer()


def test_jsonl_sink_rejects_asset_subdir_filename_template(tmp_path) -> None:
    with pytest.raises(ValueError, match="assets_subdir"):
        JsonlSink(
            tmp_path / "jsonl-custom",
            filename_template="assets/{shard_id}__w{worker_id}.jsonl",
            upload_assets=True,
        )

    with pytest.raises(ValueError, match="assets_subdir"):
        JsonlSink(
            tmp_path / "jsonl-custom",
            filename_template="tmp/../assets/{shard_id}__w{worker_id}.jsonl",
            upload_assets=True,
        )


def test_parquet_sink_rejects_unsupported_cleanup_filename_template(tmp_path) -> None:
    sink = ParquetSink(
        tmp_path / "parquet-custom",
        filename_template="{shard_id:>12}.parquet",
    )

    with pytest.raises(ValueError, match="without conversion or format specifiers"):
        sink.build_reducer()
