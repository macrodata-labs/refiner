---
title: "MCAP Reader"
description: "Read MCAP robotics logs as raw message rows"
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

`read_mcap` emits one row per logged message. It does not synchronize topics
into episodes and does not decode ROS, protobuf, or JSON payloads into structured
columns.

Each row includes:

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
| `file_path` | Source MCAP file path. |

## Filtering Topics

```python
pipeline = mdr.read_mcap(
    "/data/logs/*.mcap",
    topics=["/joint_states"],
    data_column="payload",
)
```

`topics` limits the emitted messages. `data_column` renames the raw payload
column when `data` would collide with another pipeline convention.

## Related Pages

- [Reader Model](reader-model.md)
- [Files and Videos](files-and-videos.md)
- [Converting to Robot Rows](../episode-data/converting-to-robot-rows.md)
