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


def test_mcap_reader_reads_one_row_per_file_with_message_table(tmp_path: Path) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    rows = read_mcap(str(path)).materialize()

    assert len(rows) == 1
    assert rows[0]["file_path"] == str(path)
    assert rows[0]["message_count"] == 2
    assert rows[0]["topics"] == ["/image", "/joint_states"]

    messages = rows[0]["messages"]
    assert messages.table.column_names == [
        "topic",
        "log_time",
        "publish_time",
        "sequence",
        "message_encoding",
        "schema_id",
        "schema_name",
        "schema_encoding",
        "schema_data",
        "data",
    ]
    assert messages.table.column("topic").to_pylist() == ["/joint_states", "/image"]
    first = messages.to_rows()[0]
    assert first["log_time"] == 100
    assert first["publish_time"] == 90
    assert first["message_encoding"] == "json"
    assert first["schema_name"] == "demo.Json"
    assert first["data"] == b'{"q":[1,2]}'


def test_mcap_reader_filters_topics(tmp_path: Path) -> None:
    path = tmp_path / "demo.mcap"
    _write_mcap(path)

    rows = read_mcap(
        str(path),
        topics=["/image"],
        messages_column="mcap_messages",
        data_column="payload",
    ).materialize()

    assert len(rows) == 1
    assert rows[0]["message_count"] == 1
    assert rows[0]["topics"] == ["/image"]
    messages = rows[0]["mcap_messages"]
    assert messages.table.column_names[-1] == "payload"
    assert messages.to_rows()[0]["topic"] == "/image"
    assert messages.to_rows()[0]["payload"] == b'{"frame":0}'


def test_mcap_reader_plans_files_atomically(tmp_path: Path) -> None:
    first = tmp_path / "first.mcap"
    second = tmp_path / "second.mcap"
    _write_mcap(first)
    _write_mcap(second)

    reader = McapReader([str(first), str(second)], target_shard_bytes=1)

    assert len(reader.list_shards()) == 2
