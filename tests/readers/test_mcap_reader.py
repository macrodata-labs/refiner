from __future__ import annotations

from pathlib import Path

import pytest

from refiner.pipeline import read_mcap
from refiner.pipeline.sources.readers import McapReader

mcap_writer = pytest.importorskip("mcap.writer")


def _write_mcap(path: Path) -> None:
    with path.open("wb") as stream:
        writer = mcap_writer.Writer(stream)
        writer.start()
        schema_id = writer.register_schema(
            name="demo.Json",
            encoding="jsonschema",
            data=b'{"type":"object"}',
        )
        joint_channel = writer.register_channel(
            topic="/joint_states",
            message_encoding="json",
            schema_id=schema_id,
        )
        image_channel = writer.register_channel(
            topic="/image",
            message_encoding="json",
            schema_id=schema_id,
        )
        writer.add_message(
            channel_id=joint_channel,
            log_time=100,
            publish_time=90,
            sequence=1,
            data=b'{"q":[1,2]}',
        )
        writer.add_message(
            channel_id=image_channel,
            log_time=120,
            publish_time=110,
            sequence=2,
            data=b'{"frame":0}',
        )
        writer.finish()


def test_mcap_reader_reads_raw_message_rows(tmp_path: Path) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    rows = read_mcap(str(path)).materialize()

    assert [row["topic"] for row in rows] == ["/joint_states", "/image"]
    assert rows[0]["log_time"] == 100
    assert rows[0]["publish_time"] == 90
    assert rows[0]["message_encoding"] == "json"
    assert rows[0]["schema_name"] == "demo.Json"
    assert rows[0]["data"] == b'{"q":[1,2]}'
    assert rows[0]["file_path"] == str(path)


def test_mcap_reader_filters_topics(tmp_path: Path) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    rows = read_mcap(str(path), topics=["/image"], data_column="payload").materialize()

    assert len(rows) == 1
    assert rows[0]["topic"] == "/image"
    assert rows[0]["payload"] == b'{"frame":0}'


def test_mcap_reader_plans_files_atomically(tmp_path: Path) -> None:
    first = tmp_path / "first.mcap"
    second = tmp_path / "second.mcap"
    _write_mcap(first)
    _write_mcap(second)

    reader = McapReader([str(first), str(second)], target_shard_bytes=1)

    assert len(reader.list_shards()) == 2
