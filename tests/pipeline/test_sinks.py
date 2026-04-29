from __future__ import annotations

import json
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from refiner import col
from refiner.pipeline.data import datatype
from refiner.pipeline.data.row import DictRow, Row
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline import from_items
from refiner.pipeline.sinks import JsonlSink
from refiner.pipeline.sinks.parquet import ParquetSink
from refiner.pipeline.sinks.reducer.file import FileCleanupReducerSink
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
        "find",
        lambda _: [winner_path.name, winner_path.name, loser_path.name],
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
