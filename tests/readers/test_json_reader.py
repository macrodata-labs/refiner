import orjson

from refiner.pipeline import read_json, read_jsonl
from refiner.pipeline.data import datatype
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.sources.readers import JsonReader


def _rows_from_shard_units(units):
    for unit in units:
        if isinstance(unit, Tabular):
            yield from unit.to_rows()
        else:
            yield unit


def test_json_lines_bytes_lazy_reads_all_objects(tmp_path):
    p = tmp_path / "data.jsonl"
    # Make file large enough to exceed the min shard size clamp (16MiB) so we get >1 shard.
    n = 5000
    payload = "x" * 4096
    p.write_bytes(
        b"".join(orjson.dumps({"id": i, "x": payload}) + b"\n" for i in range(n))
    )

    r = JsonReader(str(p), target_shard_bytes=16 * 1024 * 1024, lines=True)
    shards = r.list_shards()
    assert len(shards) > 1

    ids = set()
    count = 0
    for s in shards:
        for row in _rows_from_shard_units(r.read_shard(s)):
            ids.add(int(row["id"]))
            count += 1

    assert count == n
    assert ids == set(range(n))


def test_json_lines_reader_applies_dtypes(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_bytes(orjson.dumps({"video": "clip.mp4"}) + b"\n")

    reader = JsonReader(str(p), dtypes={"video": datatype.video_path()}, lines=True)
    unit = next(iter(reader.read_shard(reader.list_shards()[0])))

    assert isinstance(unit, Tabular)
    assert unit.table.schema.field("video").metadata == {b"asset_type": b"video"}


def test_json_reader_schema_exposes_dtype_overrides(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_bytes(orjson.dumps({"video": "clip.mp4"}) + b"\n")

    reader = JsonReader(str(p), dtypes={"video": datatype.video_path()}, lines=True)

    assert reader.schema is not None
    assert reader.schema.field("video").metadata == {b"asset_type": b"video"}


def test_read_json_defaults_to_whole_json_file(tmp_path):
    p = tmp_path / "data.json"
    p.write_bytes(orjson.dumps({"x": 1}))

    assert read_json(str(p)).take(1)[0]["x"] == 1


def test_read_json_default_handles_pretty_printed_multiline_file(tmp_path):
    p = tmp_path / "data.json"
    p.write_text('{\n  "x": 1,\n  "nested": {\n    "y": 2\n  }\n}\n')

    row = read_json(str(p)).take(1)[0].to_dict()

    assert row == {"x": 1, "nested": {"y": 2}, "file_path": str(p)}


def test_read_jsonl_alias_reads_json_lines(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_bytes(orjson.dumps({"x": 1}) + b"\n")

    assert read_jsonl(str(p)).take(1)[0]["x"] == 1


def test_read_json_file_object_emits_one_row_with_keys(tmp_path):
    p = tmp_path / "data.json"
    p.write_bytes(orjson.dumps({"x": 1, "name": "demo"}))

    row = read_json(str(p)).take(1)[0].to_dict()

    assert row == {"x": 1, "name": "demo", "file_path": str(p)}


def test_read_json_file_array_emits_one_value_row(tmp_path):
    p = tmp_path / "data.json"
    p.write_bytes(orjson.dumps([{"x": 1}, {"x": 2}]))

    row = read_json(str(p)).take(1)[0].to_dict()

    assert row == {"value": [{"x": 1}, {"x": 2}], "file_path": str(p)}


def test_read_json_file_preserves_existing_file_path_column(tmp_path):
    p = tmp_path / "data.json"
    p.write_bytes(orjson.dumps({"x": 1, "file_path": "inside.json"}))

    row = read_json(str(p)).take(1)[0].to_dict()

    assert row == {"x": 1, "file_path": "inside.json"}


def test_read_json_file_can_disable_file_path_column(tmp_path):
    p = tmp_path / "data.json"
    p.write_bytes(orjson.dumps({"x": 1}))

    row = read_json(str(p), file_path_column=None).take(1)[0].to_dict()

    assert row == {"x": 1}


def test_read_json_empty_object_without_dtypes_still_emits_row(tmp_path):
    p = tmp_path / "data.json"
    p.write_bytes(orjson.dumps({}))

    rows = read_json(str(p), file_path_column=None).take(2)

    assert len(rows) == 1
    assert rows[0].to_dict() == {}


def test_read_json_empty_object_uses_dtypes_for_arrow_schema(tmp_path):
    p = tmp_path / "data.json"
    p.write_bytes(orjson.dumps({}))

    reader = JsonReader(str(p), file_path_column=None, dtypes={"x": datatype.int64()})
    unit = next(iter(reader.read_shard(reader.list_shards()[0])))

    assert isinstance(unit, Tabular)
    assert unit.table.schema.field("x").type == datatype.int64()
    assert unit.to_rows()[0].to_dict() == {"x": None}


def test_read_json_file_applies_dtypes(tmp_path):
    p = tmp_path / "data.json"
    p.write_bytes(orjson.dumps({"video": "clip.mp4"}))

    reader = JsonReader(str(p), dtypes={"video": datatype.video_path()})
    unit = next(iter(reader.read_shard(reader.list_shards()[0])))

    assert isinstance(unit, Tabular)
    assert unit.table.schema.field("video").metadata == {b"asset_type": b"video"}
