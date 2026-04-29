import orjson

from refiner.pipeline.data import datatype
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.sources.readers import JsonlReader


def _rows_from_shard_units(units):
    for unit in units:
        if isinstance(unit, Tabular):
            yield from unit.to_rows()
        else:
            yield unit


def test_jsonl_bytes_lazy_reads_all_objects(tmp_path):
    p = tmp_path / "data.jsonl"
    # Make file large enough to exceed the min shard size clamp (16MiB) so we get >1 shard.
    n = 5000
    payload = "x" * 4096
    p.write_bytes(
        b"".join(orjson.dumps({"id": i, "x": payload}) + b"\n" for i in range(n))
    )

    r = JsonlReader(str(p), target_shard_bytes=16 * 1024 * 1024)
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


def test_jsonl_reader_applies_dtypes(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_bytes(orjson.dumps({"video": "clip.mp4"}) + b"\n")

    reader = JsonlReader(str(p), dtypes={"video": datatype.video_path()})
    unit = next(iter(reader.read_shard(reader.list_shards()[0])))

    assert isinstance(unit, Tabular)
    assert unit.table.schema.field("video").metadata == {b"asset_type": b"video"}


def test_jsonl_reader_schema_exposes_dtype_overrides(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_bytes(orjson.dumps({"video": "clip.mp4"}) + b"\n")

    reader = JsonlReader(str(p), dtypes={"video": datatype.video_path()})

    assert reader.schema is not None
    assert reader.schema.field("video").metadata == {b"asset_type": b"video"}
