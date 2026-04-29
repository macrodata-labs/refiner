from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

import refiner as rf
from refiner.execution.engine import RowSegment, VectorSegment, execute_segments
from refiner.pipeline import from_items
from refiner.pipeline.data import datatype
from refiner.pipeline.data.row import DictRow, Row
from refiner.pipeline.data.tabular import Tabular, set_or_append_column
from refiner.pipeline.sinks.parquet import ParquetSink
from refiner.pipeline.sources.readers import ParquetReader
from refiner.pipeline.steps import FilterExprStep, FnRowStep


def test_datatype_constructors_produce_arrow_types_and_metadata() -> None:
    assert datatype.int64() == pa.int64()
    assert datatype.list("uint8") == pa.list_(pa.uint8())
    assert datatype.video_path().type == pa.string()
    assert datatype.video_path().metadata == {b"asset_type": b"video"}
    assert datatype.file_path().metadata == {b"asset_type": b"file"}
    assert datatype.video_path().type == pa.string()
    assert datatype.video_bytes().type == pa.binary()
    assert datatype.video_bytes_with_path().type == pa.struct(
        [pa.field("bytes", pa.binary()), pa.field("path", pa.string())]
    )
    assert datatype.asset_storage(datatype.video_path()) == "path"
    assert datatype.asset_storage(datatype.video_bytes()) == "bytes"
    assert datatype.asset_storage(datatype.video_bytes_with_path()) == "bytes_with_path"
    assert datatype.list(datatype.uint8()) == pa.list_(pa.uint8())
    assert datatype.list(datatype.uint8(), size=3) == pa.list_(pa.uint8(), list_size=3)
    assert datatype.struct({"x": datatype.float32()}) == pa.struct(
        [pa.field("x", pa.float32())]
    )
    assert datatype.map(datatype.string(), datatype.int32()) == pa.map_(
        pa.string(), pa.int32()
    )


def test_schema_and_table_dtypes_preserve_unrelated_metadata() -> None:
    table = pa.table(
        {
            "frames": pa.array(["clip.mp4"]),
            "label": pa.array(["pick"]),
        },
        schema=pa.schema(
            [
                pa.field("frames", pa.string(), metadata={b"old": b"kept"}),
                pa.field("label", pa.string(), metadata={b"label": b"kept"}),
            ]
        ),
    )

    schema = datatype.schema_with_dtypes(
        table.schema,
        {"frames": datatype.video_path(), "label": pa.large_string()},
    )
    out = datatype.apply_dtypes_to_table(
        table,
        {"frames": datatype.video_path(), "label": pa.large_string()},
    )

    assert schema is not None
    assert schema.field("frames").metadata == {
        b"old": b"kept",
        b"asset_type": b"video",
    }
    assert schema.field("label").metadata == {b"label": b"kept"}
    assert out.schema.field("frames").metadata == {
        b"old": b"kept",
        b"asset_type": b"video",
    }
    assert out.schema.field("frames").type == pa.string()
    assert out.schema.field("label").metadata == {b"label": b"kept"}
    assert out.schema.field("label").type == pa.large_string()


def test_dict_rows_with_schema_preserve_field_metadata() -> None:
    schema = pa.schema(
        [pa.field("frames", pa.string(), metadata={b"asset_type": b"video"})]
    )

    table = Tabular.from_rows(
        [
            DictRow({"frames": "a.mp4"}),
            DictRow({"frames": "b.mp4"}),
        ],
        schema=schema,
    ).table

    assert table.schema.field("frames").metadata == {b"asset_type": b"video"}


def test_arrow_row_fast_path_preserves_field_metadata() -> None:
    table = datatype.apply_dtypes_to_table(
        pa.table({"frames": ["a.mp4", "b.mp4"]}),
        {"frames": datatype.video_path()},
    )
    rows = Tabular(table).to_rows()

    out = Tabular.from_rows(rows).table

    assert out.schema.field("frames").metadata == {b"asset_type": b"video"}


def test_arrow_row_patch_preserves_base_type_without_schema_override() -> None:
    table = pa.table({"x": pa.array([1, 2], type=pa.uint8())})
    rows = Tabular(table).to_rows()
    rows[0] = rows[0].update({"x": None})

    out = Tabular.from_rows(rows).table

    assert out.schema.field("x").type == pa.uint8()
    assert out["x"].to_pylist() == [None, 2]


def test_arrow_row_fast_path_applies_carried_schema_to_unchanged_columns() -> None:
    table = pa.table({"frames": ["a.mp4", "b.mp4"]})
    rows = Tabular(table).to_rows()
    schema = datatype.schema_with_dtypes(
        table.schema,
        {"frames": datatype.video_path()},
    )

    out = Tabular.from_rows(rows, schema=schema).table

    assert out.schema.field("frames").metadata == {b"asset_type": b"video"}


def test_set_or_append_column_keeps_metadata_only_for_same_type() -> None:
    table = pa.table(
        {"frames": ["a.mp4"]},
        schema=pa.schema(
            [
                pa.field(
                    "frames",
                    pa.string(),
                    nullable=False,
                    metadata={b"asset_type": b"video"},
                )
            ]
        ),
    )

    same_type = set_or_append_column(table, "frames", pa.array(["b.mp4"]))
    same_type_with_null = set_or_append_column(
        table,
        "frames",
        pa.array([None], type=pa.string()),
    )
    different_type = set_or_append_column(table, "frames", pa.array([123]))

    assert same_type.schema.field("frames").metadata == {b"asset_type": b"video"}
    assert not same_type.schema.field("frames").nullable
    assert same_type_with_null.schema.field("frames").metadata == {
        b"asset_type": b"video"
    }
    assert same_type_with_null.schema.field("frames").nullable
    assert different_type.schema.field("frames").type == pa.int64()
    assert different_type.schema.field("frames").metadata is None


def test_row_segment_dtypes_apply_to_unchanged_arrow_rows() -> None:
    blocks = list(
        execute_segments(
            [Tabular(pa.table({"frames": ["clip.mp4"]}))],
            [
                RowSegment(
                    steps=(
                        FnRowStep(
                            fn=lambda row: row,
                            index=1,
                            dtypes={"frames": rf.datatype.video_path()},
                        ),
                    )
                ),
                VectorSegment(
                    ops=(
                        FilterExprStep(
                            predicate=rf.col("frames").is_not_null(),
                            index=2,
                        ),
                    )
                ),
            ],
        )
    )

    assert isinstance(blocks[0], Tabular)
    assert blocks[0].table.schema.field("frames").metadata == {b"asset_type": b"video"}


def test_map_dtypes_do_not_attach_schema_to_row_outputs() -> None:
    pipeline = from_items([{"id": 1}]).map(
        lambda row: {"frames": "clip.mp4"},
        dtypes={"frames": rf.datatype.video_path()},
    )

    blocks = list(pipeline.execute(pipeline.source.read()))

    assert len(blocks) == 1
    assert isinstance(blocks[0], list)
    row = blocks[0][0]
    assert isinstance(row, Row)
    assert not hasattr(row, "schema")


def test_map_dtypes_do_not_attach_declared_schema_overrides_to_rows() -> None:
    pipeline = from_items([{"id": 1}]).map(
        lambda row: {"frames": "clip.mp4"},
        dtypes={
            "frames": rf.datatype.video_path(),
            "missing": rf.datatype.video_path(),
        },
    )

    blocks = list(pipeline.execute(pipeline.source.read()))

    assert isinstance(blocks[0], list)
    row = blocks[0][0]
    assert isinstance(row, Row)
    assert not hasattr(row, "schema")


def test_map_dtypes_accumulate_across_row_steps() -> None:
    pipeline = (
        from_items([{"id": 1}])
        .map(
            lambda row: row.update({"frames": "clip.mp4"}),
            dtypes={"frames": rf.datatype.video_path()},
        )
        .map(
            lambda row: row.update({"audio": "clip.wav"}),
            dtypes={"audio": rf.datatype.audio_path()},
        )
        .filter(rf.col("frames").is_not_null())
    )

    blocks = list(pipeline.execute(pipeline.source.read()))

    assert isinstance(blocks[0], Tabular)
    assert blocks[0].table.schema.field("frames").metadata == {b"asset_type": b"video"}
    assert blocks[0].table.schema.field("audio").metadata == {b"asset_type": b"audio"}


def test_row_drop_removes_schema_override() -> None:
    schema = pa.schema(
        [pa.field("frames", pa.string(), metadata={b"asset_type": b"video"})]
    )
    row = DictRow({"frames": "clip.mp4"})

    out = row.drop("frames")
    table = Tabular.from_rows([out], schema=schema).table

    assert "frames" not in out
    assert not hasattr(out, "schema")
    assert "frames" not in table.schema.names


def test_arrow_row_drop_removes_column_on_fast_path() -> None:
    table = pa.table({"frames": ["a.mp4", "b.mp4"], "x": [1, 2]})
    rows = [row.drop("frames") for row in Tabular(table)]
    schema = pa.schema(
        [pa.field("frames", pa.string(), metadata={b"asset_type": b"video"})]
    )

    without_schema = Tabular.from_rows(rows).table
    with_schema = Tabular.from_rows(rows, schema=schema).table

    assert without_schema.to_pydict() == {"x": [1, 2]}
    assert with_schema.to_pydict() == {"x": [1, 2]}


def test_untyped_row_overwrite_does_not_attach_schema_to_rows() -> None:
    pipeline = (
        from_items([{"id": 1}])
        .map(
            lambda row: row.update({"frames": "clip.mp4"}),
            dtypes={"frames": rf.datatype.video_path()},
        )
        .map(lambda row: {"frames": "label-not-video"})
    )

    blocks = list(pipeline.execute(pipeline.source.read()))

    assert isinstance(blocks[0], list)
    row = blocks[0][0]
    assert isinstance(row, Row)
    assert row["frames"] == "label-not-video"
    assert not hasattr(row, "schema")


def test_incompatible_row_value_clears_schema_field_on_table_conversion() -> None:
    pipeline = (
        from_items([{"id": 1}])
        .map(
            lambda row: row.update({"frames": "clip.mp4"}),
            dtypes={"frames": rf.datatype.video_path()},
        )
        .map(lambda row: {"frames": 123})
        .filter(rf.col("frames").is_not_null())
    )

    blocks = list(pipeline.execute(pipeline.source.read()))

    assert isinstance(blocks[0], Tabular)
    assert blocks[0].table["frames"].to_pylist() == [123]
    assert blocks[0].table.schema.field("frames").type == pa.int64()
    assert blocks[0].table.schema.field("frames").metadata is None


def test_incompatible_arrow_row_patch_clears_field_metadata() -> None:
    table = datatype.apply_dtypes_to_table(
        pa.table({"frames": ["a.mp4", "b.mp4"]}),
        {"frames": datatype.video_path()},
    )
    rows = [row.update({"frames": idx}) for idx, row in enumerate(Tabular(table))]

    out = Tabular.from_rows(rows).table

    assert out["frames"].to_pylist() == [0, 1]
    assert out.schema.field("frames").type == pa.int64()
    assert out.schema.field("frames").metadata is None


def test_row_step_schemas_flow_through_untyped_steps() -> None:
    pipeline = (
        from_items([{"id": 1}])
        .map(
            lambda row: row.update({"frames": "clip.mp4"}),
            dtypes={"frames": rf.datatype.video_path()},
        )
        .map(lambda row: row.update({"label": "keep"}))
        .filter(rf.col("frames").is_not_null())
    )

    blocks = list(pipeline.execute(pipeline.source.read()))

    assert isinstance(blocks[0], Tabular)
    assert blocks[0].table.schema.field("frames").metadata == {b"asset_type": b"video"}


def test_vectorized_schema_flows_into_later_row_segment() -> None:
    pipeline = (
        from_items([{"id": 1}])
        .map(
            lambda row: row.update({"frames": "clip.mp4"}),
            dtypes={"frames": rf.datatype.video_path()},
        )
        .filter(rf.col("frames").is_not_null())
        .map(lambda row: row.update({"frames": row["frames"]}))
        .filter(rf.col("frames").is_not_null())
    )

    blocks = list(pipeline.execute(pipeline.source.read()))

    assert isinstance(blocks[0], Tabular)
    assert blocks[0].table.schema.field("frames").metadata == {b"asset_type": b"video"}


def test_vectorized_assignment_clears_previous_schema_override() -> None:
    pipeline = (
        from_items([{"id": 1}])
        .map(
            lambda row: row.update({"frames": "clip.mp4"}),
            dtypes={"frames": rf.datatype.video_path()},
        )
        .filter(rf.col("frames").is_not_null())
        .with_column("frames", "label")
        .filter(rf.col("frames").is_not_null())
    )

    blocks = list(pipeline.execute(pipeline.source.read()))

    assert isinstance(blocks[0], Tabular)
    assert blocks[0].table["frames"].to_pylist() == ["label"]
    assert blocks[0].table.schema.field("frames").metadata is None


def test_vectorized_cast_becomes_global_schema_override() -> None:
    pipeline = (
        from_items([{"x": 1}])
        .cast(x=rf.datatype.float32())
        .map(lambda row: row.update({"x": row["x"]}))
        .filter(rf.col("x").is_not_null())
    )

    blocks = list(pipeline.execute(pipeline.source.read()))

    assert isinstance(blocks[0], Tabular)
    assert blocks[0].table.schema.field("x").type == pa.float32()


def test_vectorized_cast_clears_previous_file_metadata() -> None:
    pipeline = (
        from_items([{"id": 1}])
        .map(
            lambda row: row.update({"frames": "123"}),
            dtypes={"frames": rf.datatype.video_path()},
        )
        .filter(rf.col("frames").is_not_null())
        .cast(frames="int64")
        .map(lambda row: row.update({"frames": row["frames"]}))
        .filter(rf.col("frames").is_not_null())
    )

    blocks = list(pipeline.execute(pipeline.source.read()))

    assert isinstance(blocks[0], Tabular)
    assert blocks[0].table.schema.field("frames").type == pa.int64()
    assert blocks[0].table.schema.field("frames").metadata is None


def test_row_dtype_redefinition_clears_previous_file_metadata() -> None:
    pipeline = (
        from_items([{"id": 1}])
        .map(
            lambda row: row.update({"frames": "123"}),
            dtypes={"frames": rf.datatype.video_path()},
        )
        .map(
            lambda row: row.update({"frames": row["frames"]}),
            dtypes={"frames": "int64"},
        )
        .filter(rf.col("frames") > 100)
    )

    blocks = list(pipeline.execute(pipeline.source.read()))

    assert isinstance(blocks[0], Tabular)
    assert blocks[0].table["frames"].to_pylist() == [123]
    assert blocks[0].table.schema.field("frames").type == pa.int64()
    assert blocks[0].table.schema.field("frames").metadata is None


def test_vectorized_cast_accepts_file_dtype_metadata() -> None:
    pipeline = (
        from_items([{"frames": "clip.mp4"}])
        .cast(frames=rf.datatype.video_path())
        .map(lambda row: row.update({"frames": row["frames"]}))
        .filter(rf.col("frames").is_not_null())
    )

    blocks = list(pipeline.execute(pipeline.source.read()))

    assert isinstance(blocks[0], Tabular)
    assert blocks[0].table.schema.field("frames").type == pa.string()
    assert blocks[0].table.schema.field("frames").metadata == {b"asset_type": b"video"}


def test_vectorized_cast_after_map_table_reestablishes_schema_override() -> None:
    pipeline = (
        from_items([{"x": 1}])
        .map_table(lambda table: table)
        .cast(x="float32")
        .map(lambda row: row.update({"x": row["x"]}))
        .filter(rf.col("x").is_not_null())
    )

    blocks = list(pipeline.execute(pipeline.source.read()))

    assert isinstance(blocks[0], Tabular)
    assert blocks[0].table.schema.field("x").type == pa.float32()


def test_pipeline_exposes_final_row_schema_for_sink(tmp_path) -> None:
    pipeline = (
        from_items([{"id": 1}])
        .map(
            lambda row: row.update({"frames": "clip.mp4"}),
            dtypes={"frames": rf.datatype.video_path()},
        )
        .write_parquet(tmp_path)
    )

    blocks = list(pipeline.execute(pipeline.source.read()))

    assert isinstance(blocks[0], list)
    schema = pipeline.output_schema()
    assert schema is not None
    assert schema.field("frames").metadata == {b"asset_type": b"video"}


def test_tabular_schema_flows_through_row_segment() -> None:
    table = datatype.apply_dtypes_to_table(
        pa.table({"frames": ["clip.mp4"]}),
        {"frames": datatype.video_path()},
    )

    blocks = list(
        execute_segments(
            [Tabular(table)],
            [
                RowSegment(
                    steps=(FnRowStep(fn=lambda row: {"label": "keep"}, index=1),)
                ),
                VectorSegment(
                    ops=(
                        FilterExprStep(
                            predicate=rf.col("frames").is_not_null(),
                            index=2,
                        ),
                    )
                ),
            ],
        )
    )

    assert isinstance(blocks[0], Tabular)
    assert blocks[0].table.schema.field("frames").metadata == {b"asset_type": b"video"}


def test_map_dtypes_attach_declared_schema_even_when_column_is_absent() -> None:
    pipeline = from_items([{"id": 1}]).map(
        lambda row: {"frames": "clip.mp4"},
        dtypes={"missing": rf.datatype.video_path()},
    )

    blocks = list(pipeline.execute(pipeline.source.read()))

    assert isinstance(blocks[0], list)
    row = blocks[0][0]
    assert isinstance(row, Row)
    assert not hasattr(row, "schema")


def test_map_dtypes_apply_to_downstream_tabular_blocks() -> None:
    pipeline = (
        from_items([{"id": 1}])
        .map(
            lambda row: {"frames": "clip.mp4"},
            dtypes={"frames": rf.datatype.video_path()},
        )
        .filter(rf.col("frames").is_not_null())
    )

    blocks = list(pipeline.execute(pipeline.source.read()))

    assert len(blocks) == 1
    assert isinstance(blocks[0], Tabular)
    assert blocks[0].table.schema.field("frames").metadata == {b"asset_type": b"video"}


def test_parquet_sink_and_reader_preserve_field_metadata(tmp_path) -> None:
    table = datatype.apply_dtypes_to_table(
        pa.table({"frames": ["clip.mp4"]}),
        {"frames": datatype.video_path()},
    )
    sink = ParquetSink(tmp_path)
    sink.write_shard_block("train", Tabular(table))
    sink.on_shard_complete("train")

    path = next(tmp_path.glob("*.parquet"))
    read_table = pq.read_table(path)
    assert read_table.schema.field("frames").metadata == {b"asset_type": b"video"}

    reader = ParquetReader(str(path), file_path_column=None)
    units = [
        unit
        for shard in reader.list_shards()
        for unit in reader.read_shard(shard)
        if isinstance(unit, Tabular)
    ]
    assert units
    assert units[0].table.schema.field("frames").metadata == {b"asset_type": b"video"}
