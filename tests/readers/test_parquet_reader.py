import pyarrow as pa
import pyarrow.parquet as pq

from refiner.pipeline.sources.readers import ParquetReader


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
        if isinstance(unit, pa.RecordBatch):
            tbl = pa.Table.from_batches([unit])
            yield from tbl.to_pylist()
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
