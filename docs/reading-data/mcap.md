---
title: "MCAP Reader"
description: "Read MCAP robotics logs as one row per file"
---

# MCAP Reader

Use `read_mcap` for robotics or autonomy logs stored as MCAP files.

```python
import refiner as mdr

pipeline = mdr.read_mcap(
    "/data/logs/*.mcap",
    topics=["/joint_states", "/camera/image"],
)
```

Install `macrodata-refiner[mcap]` to use this reader.

## Output Rows

`read_mcap` emits one row per MCAP file. It does not synchronize topics into
episodes and does not decode ROS, protobuf, or JSON payloads into domain-specific
columns. The raw logged messages are exposed as an Arrow-backed nested table.

Each row includes:

| Column | Meaning |
| --- | --- |
| `file_path` | Source MCAP file path. |
| `message_count` | Number of selected messages in the file. |
| `topics` | Sorted topic names present after filtering. |
| `messages` | `Tabular` message table. Use `row["messages"].table` for the `pyarrow.Table`. |

The nested `messages` table includes:

| Column | Meaning |
| --- | --- |
| `topic` | MCAP channel topic, such as `/joint_states`. |
| `log_time` | Message log timestamp in nanoseconds. |
| `publish_time` | Message publish timestamp in nanoseconds. |
| `sequence` | Message sequence number. |
| `message_encoding` | Channel message encoding. |
| `schema_id` | Channel schema id. |
| `schema_name` | Schema name, if present. |
| `schema_encoding` | Schema encoding, if present. |
| `schema_data` | Raw schema bytes, if present. |
| `data` | Raw message payload bytes. |

## Filtering Topics

```python
pipeline = mdr.read_mcap(
    "/data/logs/*.mcap",
    topics=["/joint_states"],
    messages_column="mcap_messages",
    data_column="payload",
)
```

`topics` limits the messages included in each file row. `messages_column`
renames the nested table column. `data_column` renames the raw payload column
inside that nested table.

## Related Pages

- [Reader Model](reader-model.md)
- [Files and Videos](files-and-videos.md)
- [Converting to Robot Rows](../episode-data/converting-to-robot-rows.md)
