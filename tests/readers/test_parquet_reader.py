import pyarrow as pa
import pyarrow.parquet as pq

from forklift.readers import ParquetReader


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


def test_parquet_rowgroups_reads_all_rows(tmp_path):
    p = _write_parquet(tmp_path)
    r = ParquetReader(str(p), sharding_mode="rowgroups", target_shard_bytes=1)
    shards = r.list_shards()
    assert len(shards) >= 1

    out = []
    for s in shards:
        out.extend(list(r.read_shard(s)))

    ids = sorted(int(row["id"]) for row in out)
    assert ids == list(range(50))


def test_parquet_bytes_lazy_reads_all_rows(tmp_path):
    p = _write_parquet(tmp_path)
    # Force multiple byte shards
    r = ParquetReader(str(p), sharding_mode="bytes_lazy", target_shard_bytes=200)
    shards = r.list_shards()
    assert len(shards) >= 1

    out = []
    for s in shards:
        out.extend(list(r.read_shard(s)))

    ids = sorted(int(row["id"]) for row in out)
    assert ids == list(range(50))
