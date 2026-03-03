from refiner.sources.readers import CsvReader


def test_csv_bytes_lazy_reads_all_rows_exactly_once(tmp_path):
    p = tmp_path / "data.csv"
    # Make this file large enough to exceed the min shard size clamp (16MiB) so we get >1 shard.
    n = 2500
    payload = "x" * 8192
    with p.open("wt") as f:
        f.write("id,val\n")
        for i in range(n):
            f.write(f"{i},{payload}\n")

    r = CsvReader(
        str(p), target_shard_bytes=16 * 1024 * 1024, sharding_mode="bytes_lazy"
    )
    shards = r.list_shards()
    assert len(shards) > 1

    ids = set()
    count = 0
    for s in shards:
        for row in r.read_shard(s):
            ids.add(int(row["id"]))
            count += 1

    assert count == n
    assert ids == set(range(n))


def test_csv_multiline_forces_scan_and_parses_embedded_newline(tmp_path):
    p = tmp_path / "multi.csv"
    # second row contains an embedded newline in a quoted field
    p.write_text('id,text\n0,"hello\nworld"\n1,"ok"\n')

    r = CsvReader(
        str(p), target_shard_bytes=8, multiline_rows=True, sharding_mode="bytes_lazy"
    )
    # bytes_lazy should auto-fallback to scan when multiline_rows=True
    shards = r.list_shards()
    assert len(shards) >= 1

    rows = []
    for s in shards:
        rows.extend(list(r.read_shard(s)))

    assert len(rows) == 2
    assert rows[0]["id"] == "0"
    assert rows[0]["text"] == "hello\nworld"
    assert rows[1]["id"] == "1"
    assert rows[1]["text"] == "ok"
