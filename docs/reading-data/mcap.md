---
title: "MCAP Reader"
description: "Read MCAP robotics logs as episode rows"
---

# MCAP Reader

Use `read_mcap` for robotics or autonomy logs stored as MCAP files.

```python
import refiner as mdr

pipeline = (
    mdr.read_mcap(
        "/data/logs/*.mcap",
        fields={
            "state": "/joint_states.position",
            "action": "/joint_states.velocity",
        },
        videos={"front": "/camera/image/compressed"},
        sync_primary="state",
        fps=30,
    )
    .to_robot_rows(
        nested_frames_key="records",
        state_key="state",
        action_key="action",
        video_keys={"observation.images.front": "videos/front"},
        fps_key="fps",
    )
)
```

Install `macrodata-refiner[mcap]` to use this reader.

Directory inputs are filtered to paths ending in `.mcap`. Explicit file paths
and glob patterns are honored as written.

## Output Rows

`read_mcap` emits one row per episode. By default, each input file is one
episode. Use `episode_splitting` to split a file into multiple episodes; the
next section describes the supported splitting modes.

Each row includes:

| Column | Meaning |
| --- | --- |
| `file_path` | Source MCAP file path. |
| `episode_index` | Episode number within the file. |
| `records` | `Tabular` records table. |
| `videos` | Mapping of selected video names to `VideoFrameArray` values, when `videos` is set. |
| `fps` | Explicit fps, or inferred fps when possible. |

## Episode Splitting

`read_mcap` always emits episode rows. By default, each input MCAP file becomes
one episode:

```python
mdr.read_mcap("run.mcap")
```

Use `episode_splitting={"time_gap_s": seconds}` when episodes are separated by a
long pause in the log:

```python
mdr.read_mcap(
    "run.mcap",
    episode_splitting={"time_gap_s": 5.0},
)
```

The reader gathers selected message timestamps, sorts them by log time, and
starts a new episode whenever the gap between consecutive timestamps is greater
than the threshold. The split is based on MCAP log timestamps, not file order.

Use `episode_splitting={"marker_topic": topic}` when the log contains explicit
episode markers:

```python
mdr.read_mcap(
    "run.mcap",
    fields={"state": "/state.q"},
    episode_splitting={"marker_topic": "/episode_start"},
)
```

Each marker timestamp starts an episode. The episode ends at the next marker, or
at the end of the file for the final marker. Marker messages are used only for
splitting and are not included as default record columns.

If there are no messages, no time gaps, or no marker messages, the reader falls
back to one episode for the file.

## Selecting Fields

`fields` maps output records table columns to MCAP sources:

```python
mdr.read_mcap(
    "run.mcap",
    fields={
        "state": "/joint_states.position",
        "target": "/cmd.target",
    },
)
```

A source can be a whole topic, such as `"/joint_states"`, or a decoded subfield,
such as `"/joint_states.position"`. The reader first checks whether the exact
source string is a topic. If not, it treats the longest matching topic prefix as
the topic and the remainder as a dotted field path.

If `fields` is omitted, decoded object messages are expanded into default
columns like `"/joint_states.position"`. Selected video topics and marker topics
are excluded from those default record columns.

When `fields` is set, the reader loads the topics needed by `fields`, `videos`,
`sync_primary`, and marker-based episode splitting. When `fields` is omitted, it reads
all topics and builds default record columns from decoded non-video messages.

## Decoding

`read_mcap` decodes JSON messages, ROS2 messages, and protobuf messages when the
matching optional decoder is available. Decoded object messages can be selected
with dotted field paths like `"/joint_states.position"`.

Unknown encodings are preserved as raw bytes. Field paths cannot be applied to
raw bytes; select the whole topic or decode the payload before using subfields.

## Synchronization

MCAP logs usually contain multiple topics with different rates: robot state,
actions, camera frames, commands, diagnostics, and events may not share exactly
the same timestamps. `read_mcap` supports these synchronization choices:

| Mode | How to select it | Output rows | Best for |
| --- | --- | --- | --- |
| Unsynchronized | Leave `sync_primary=None` | One records table row for each selected field message timestamp. Missing fields are null. Selected videos stay separate in `videos`. | Event logs, debugging, preserving each topic's original timing. |
| Field sync-primary aligned | Set `sync_primary` to a selected field name, topic, or dotted source path. | One records table row for each sync-primary field timestamp. Other fields and videos are aligned with `sync_method`. | Robot state or action streams that define the trajectory clock. |
| Video sync-primary aligned | Set `sync_primary` to a selected video name, video topic, or dotted video source path. | One records table row for each sync-primary video frame timestamp. Fields and other videos are aligned with `sync_method`. | Camera-driven datasets where image frames define the training samples. |

### Unsynchronized Mode

Use unsynchronized mode by leaving `sync_primary` unset:

```python
mdr.read_mcap(
    "run.mcap",
    fields={
        "state": "/joint_states.position",
        "command": "/cmd.target",
    },
)
```

This creates one records table over the union of selected message
timestamps. For example, if state arrives at `0.0` and `1.0` seconds, while
commands arrive at `0.5` seconds, the output is:

| timestamp | state | command |
| --- | --- | --- |
| `0.0` | `[1, 2]` | null |
| `0.5` | null | `[10]` |
| `1.0` | `[3, 4]` | null |

If two selected messages have the exact same timestamp, they become separate
rows so no message is dropped:

| timestamp | state |
| --- | --- |
| `1.0` | `[1, 2]` |
| `1.0` | `[3, 4]` |

### Sync-Primary-Aligned Mode

Use sync-primary-aligned mode by setting `sync_primary` to the source that
should define the output frame rate:

```python
mdr.read_mcap(
    "run.mcap",
    fields={
        "state": "/joint_states.position",
        "command": "/cmd.target",
    },
    sync_primary="state",
)
```

`sync_primary` can be an output field name (`"state"`), a video name, a topic,
or a dotted source path. Its timestamps define the output rows. In field
sync-primary and video sync-primary mode, every non-sync-primary field and
selected video is aligned to those rows with `sync_method`. In unsynchronized
mode, no alignment is applied.

`sync_method` applies globally to all non-sync-primary fields and videos:

| Method | Behavior |
| --- | --- |
| `"nearest"` | Use the closest source timestamp before or after the sync-primary timestamp. |
| `"hold"` | Use the most recent source value at or before the sync-primary timestamp. |
| `"interpolate"` | Linearly interpolate numeric scalar/list/array values. Non-numeric values and videos fall back to nearest. |

For example, if state is the sync-primary source and command messages arrive
slightly after each state sample:

| state timestamp | nearest command timestamp | state | command |
| --- | --- | --- | --- |
| `0.0` | `0.1` | `[1, 2]` | `[10]` |
| `1.0` | `0.9` | `[3, 4]` | `[20]` |
| `2.0` | `0.9` | `[5, 6]` | `[20]` |

If `include_skew=True`, `read_mcap` also adds columns such as
`mcap.command.timestamp` and `mcap.command.skew_ms` so you can inspect how far
each non-primary aligned sample was from the sync-primary timestamp. Set
`include_skew=False` to omit those diagnostic columns.

The reader does not assume raw MCAP events arrive in timestamp order. It sorts
timestamps where ordering affects semantics: splitting, unsynchronized record
rows, sync-primary alignment, and fps inference.

## FPS

Pass `fps=...` when you already know the intended episode frame rate:

```python
mdr.read_mcap("run.mcap", sync_primary="state", fps=30)
```

If `fps` is omitted and `sync_primary` is set, the reader infers fps from the
median gap between sync-primary timestamps. If neither explicit fps nor inferred fps is
available, the row has no `fps` column. Selected videos still need an fps value,
so video frame arrays fall back to `30` when no better value is available. MCAP
videos require an integer fps because `VideoFrameArray` stores integer-rate frame
arrays.

## Videos

`videos` maps video names to MCAP sources:

```python
mdr.read_mcap(
    "run.mcap",
    videos={"front": "/camera/image/compressed"},
    sync_primary="front",
    fps=30,
)
```

Each selected video becomes a `VideoFrameArray` under the row's `videos` column.
When you later call `to_robot_rows`, map those values into semantic video keys:

```python
pipeline = (
    mdr.read_mcap(
        "run.mcap",
        fields={"state": "/joint_states.position"},
        videos={"front": "/camera/image/compressed"},
        sync_primary="state",
        fps=30,
    )
    .to_robot_rows(
        nested_frames_key="records",
        state_key="state",
        action_key=None,
        timestamp_key="timestamp",
        video_keys={"observation.images.front": "videos/front"},
        fps_key="fps",
    )
)
```

Video sources must resolve to image-like frame payloads: ROS compressed image
messages, raw ROS image messages, decoded image arrays, or base64 image bytes.
The reader does not decode encoded video packet streams such as H.264 byte
streams. For those logs, read the packet metadata as fields, or decode the video
stream before converting it to robotics rows.

When `sync_primary` is set, videos are aligned to the sync-primary timestamps with
`sync_method`, except `"interpolate"` falls back to nearest because video frames
are not numeric arrays. When `sync_primary` is omitted, each video keeps the frames
from its own topic timestamps.

Using a video as `sync_primary` is useful when camera frames should define the output
rows:

```python
mdr.read_mcap(
    "run.mcap",
    fields={"state": "/joint_states.position"},
    videos={"front": "/camera/image/compressed"},
    sync_primary="front",
    fps=30,
)
```

## Performance trade-offs

By default, `read_mcap` buffers decoded messages for the whole MCAP file before
emitting episode rows. This favors sequential reads and is usually best for
throughput, especially on cloud object storage, but peak memory grows with the
selected decoded messages and video frames.

The `stream_episodes` option is the low-memory path for split episodes. When
`stream_episodes=True` and `episode_splitting` is not `"single"`, the reader
uses MCAP log-time ordering and buffers one episode at a time. This lowers peak
application memory, but it can be slower: seekable indexed MCAPs may require
chunk seeks, and on S3-like storage those seeks can become ranged requests.

For non-seekable streams, `stream_episodes=True` falls back to buffered reading
because MCAP log-time ordering would require buffering and sorting the stream.
When `episode_splitting="single"`, `stream_episodes` is ignored because the
single episode is the whole file.

## Conversion Examples

Convert the Franka MCAP sample from Hugging Face to LeRobot and Zarr. The
camera topic is the sync primary, so each output row corresponds to one decoded
image frame and the robot state/action fields are aligned to that frame clock:

```python
source = (
    mdr.read_mcap(
        "hf://datasets/SLAI-scientific-embodied-2026/"
        "Franka1_mcap_short_task1_0520/"
        "task101-bag_20260517_195933/"
        "task101-bag_20260517_195933_0.mcap",
        fields={
            "state": "/joint_states.position",
            "action": "/joint_states.velocity",
        },
        videos={"front": "/cam1/realsense_camera/color/image_raw/compressed"},
        sync_primary="front",
        fps=30,
    )
    .to_robot_rows(
        nested_frames_key="records",
        state_key="state",
        action_key="action",
        timestamp_key="timestamp",
        video_keys={"observation.images.front": "videos/front"},
        fps_key="fps",
        robot_type="franka",
    )
)

source.write_lerobot("s3://bucket/slai-franka-lerobot")
source.write_zarr("s3://bucket/slai-franka.zarr")
```

Convert MCAP robot logs to LeRobot:

```python
(
    mdr.read_mcap(
        "run.mcap",
        fields={
            "state": "/joint_states.position",
            "action": "/joint_states.velocity",
        },
        videos={"front": "/camera/image/compressed"},
        sync_primary="state",
        fps=30,
    )
    .to_robot_rows(
        nested_frames_key="records",
        state_key="state",
        action_key="action",
        timestamp_key="timestamp",
        video_keys={"observation.images.front": "videos/front"},
        fps_key="fps",
        robot_type="franka",
    )
    .write_lerobot("s3://bucket/robot-dataset")
)
```

Convert MCAP robot logs to Zarr:

```python
(
    mdr.read_mcap(
        "run.mcap",
        fields={"state": "/joint_states.position"},
        videos={"front": "/camera/image/compressed"},
        sync_primary="state",
        fps=30,
    )
    .to_robot_rows(
        nested_frames_key="records",
        state_key="state",
        action_key=None,
        timestamp_key="timestamp",
        video_keys={"observation.images.front": "videos/front"},
        fps_key="fps",
    )
    .write_zarr("s3://bucket/robot-dataset.zarr")
)
```

For non-robotics event logs, write the record fields directly:

```python
(
    mdr.read_mcap(
        "events.mcap",
        fields={
            "mouse_x": "mouse/state.x",
            "key": "keyboard.vk",
        },
    )
    .flat_map(
        lambda row: [
            frame.update(
                {
                    "file_path": row["file_path"],
                    "episode_index": row["episode_index"],
                }
            )
            for frame in row["records"]
        ]
    )
    .write_parquet("s3://bucket/mcap-events")
)
```

## Limitations

- There is no max-skew cutoff or automatic row dropping. Use the generated
  `mcap.<field>.skew_ms` columns to filter after reading.
- Encoded video packet streams, such as H.264 payloads, are not decoded into
  frames.
- MCAP files are read as atomic files. They are not split by byte range across
  workers.
- Folder discovery only includes files with the `.mcap` extension. Explicit
  file paths and glob patterns are honored as written.

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `inputs` | required | MCAP file, glob, folder, or list of inputs. |
| `fs` | `None` | Optional fsspec filesystem for string inputs. |
| `storage_options` | `None` | Optional fsspec options when constructing a filesystem. |
| `recursive` | `False` | Recursively list folder inputs. |
| `target_shard_bytes` | `128 MiB` | Target file-shard planning size. MCAP files remain atomic. |
| `num_shards` | `None` | Optional target number of planned shards. |
| `file_path_column` | `"file_path"` | Source file column name. Set to `None` to omit it. Cannot collide with `records`, `episode_index`, `videos`, or `fps`. |
| `episode_splitting` | `"single"` | One file per episode, `{"time_gap_s": seconds}`, or `{"marker_topic": topic}`. |
| `stream_episodes` | `False` | Buffer one split episode at a time for seekable indexed MCAPs. Ignored for single-episode reads. |
| `fields` | `None` | Mapping, sequence, or string selecting record fields. |
| `videos` | `None` | Mapping, sequence, or string selecting image-like video frame sources. |
| `sync_primary` | `None` | Source used for sync-primary-aligned synchronization. |
| `sync_method` | `"nearest"` | Alignment method for sync-primary-aligned mode: `"nearest"`, `"hold"`, or `"interpolate"`. |
| `include_skew` | `True` | Add alignment timestamp/skew columns in sync-primary-aligned mode. |
| `fps` | `None` | Explicit frame rate. Overrides inferred fps. |

## Related Pages

- [Reader Model](reader-model.md)
- [Files and Videos](files-and-videos.md)
- [Converting to Robot Rows](../episode-data/converting-to-robot-rows.md)
