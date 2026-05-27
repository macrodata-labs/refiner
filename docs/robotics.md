---
title: "Robotics"
description: "Robotics data workflows in Refiner"
---

Refiner already includes robotics-specific support through the LeRobot reader,
writer, and robotics transforms.

## What You Can Do Today

- read LeRobot datasets with `read_lerobot(...)`
- transform episode rows with normal pipeline ops like `map(...)`, `filter(...)`, and expression-backed transforms where applicable
- run robotics-specific transforms under `mdr.robotics.*`
- write LeRobot-compatible output datasets with `write_lerobot(...)`
- merge compatible LeRobot datasets into one output dataset

## Quick Toc

- [reading datasets](#reading-datasets)
- [working with lerobotrow](#working-with-lerobotrow)
  - [frames](#frames)
  - [videos](#videos)
  - [stats](#stats)
  - [metadata](#metadata)
  - [updating rows](#updating-rows)
- [transforming rows](#transforming-rows)
- [writing datasets](#writing-datasets)
  - [stage-1 writes and stage-2 reduction](#stage-1-writes-and-stage-2-reduction)
  - [performance notes](#lerobot-performance-notes)
- [motion trimming](#motion-trimming)
- [egocentric hand tracking](#egocentric-hand-tracking)
- [reward scoring](#reward-scoring)
- [subtask annotation](#subtask-annotation)
- [merging datasets](#merging-datasets)

## Reading Datasets

Read a single LeRobot dataset:

```python
import refiner as mdr

pipeline = mdr.read_lerobot("hf://datasets/macrodata/aloha_static_battery_ep005_009")
```

Read multiple compatible datasets together:

```python
import refiner as mdr

pipeline = mdr.read_lerobot(
    [
        "hf://datasets/macrodata/aloha_static_battery_ep005_009",
        "hf://datasets/macrodata/aloha_static_battery_ep000_004",
    ]
)
```

## Working With `LeRobotRow`

`read_lerobot(...)` yields `LeRobotRow` objects, so you usually do not need to
manually unpack the raw LeRobot transport fields.

`LeRobotRow` is a view over the underlying episode row, not a disconnected copy.
That means you get LeRobot-specific helpers on top, but you can still access and
patch the original row columns directly through the normal row API.

Typical things you will touch:

- `row.frames`
- `row.videos`
- `row.stats`
- `row.metadata`
- `row.update(...)`

Example:

```python
import refiner as mdr

def inspect_episode(row):
    assert isinstance(row, mdr.robotics.LeRobotRow)

    episode_index = row.episode_index
    length = row.length
    task_names = row.tasks
    fps = row.metadata.info.fps

    frames = row.frames
    first_frame = next(iter(frames))
    first_timestamp = first_frame["timestamp"]

    video_spans = {
        key: (video.from_timestamp_s, video.to_timestamp_s)
        for key, video in row.videos.items()
    }
    available_stats = list(row.stats)

    return row.update(
        debug_summary={
            "episode_index": episode_index,
            "length": length,
            "tasks": task_names,
            "fps": fps,
            "first_timestamp": first_timestamp,
            "videos": video_spans,
            "stats": available_stats,
        }
    )
```

The important unit here is:

- one pipeline row = one episode

not:

- one pipeline row = one frame

So transforms like `map(...)` and `motion_trim(...)` operate on an episode row
whose `frames` field contains that episode's frame table.

### Frames

`row.frames` is the episode frame payload.

In practice it is a normal `Tabular`, so you can:

- iterate it frame-by-frame
- inspect `row.frames.num_rows`
- access `row.frames.table` when you want the Arrow table
- replace it with `row.update(frames=...)`

Example:

```python
def keep_first_ten_frames(row):
    frames = row.frames
    kept = frames.with_table(frames.table.slice(0, 10))
    return row.update(frames=kept)
```

### Videos

`row.videos` is a mapping from video feature key to `VideoSource`.

For LeRobot rows, each source is a clipped `VideoFile`, so you can access:

- `video.uri`
- `video.from_timestamp_s`
- `video.to_timestamp_s`

All video sources expose:

- `video.iter_frames()`
- `video.iter_frame_windows(offsets=[...], stride=...)`
- `video.clipped(from_timestamp_s=..., to_timestamp_s=...)`

Example:

```python
def shift_videos_by_half_second(row):
    for key, video in row.videos.items():
        row = row.with_video(
            key,
            video.clipped(from_timestamp_s=0.5),
        )
    return row
```

Decode frames lazily:

```python
import asyncio
import refiner as mdr

async def inspect_video(video):
    async for frame in video.iter_frames():
        print(frame.index, frame.timestamp_s, frame.width, frame.height)

    async for window in video.iter_frame_windows(
        offsets=[-1, 0, 1],
        stride=4,
        drop_incomplete=False,
    ):
        print(window.anchor.index, window.offsets)
```

### Stats

`row.stats` is a LeRobot-aware mapping over `stats/<feature>/...` fields.

You can:

- iterate available feature names with `for feature in row.stats`
- read one feature with `row.stats["observation.images.main"]`
- drop stale stats with `row.stats.drop(feature)`

Example:

```python
def invalidate_video_stats(row):
    for key in row.videos:
        row = row.stats.drop(key)
    return row
```

### Metadata

`row.metadata` is a `LeRobotMetadata` dataclass carrying dataset-level state:

- `row.metadata.info`
- `row.metadata.stats`
- `row.metadata.tasks`

That is the right place to look for canonical dataset facts such as:

- `fps`
- `robot_type`
- merged task mapping

### Updating rows

Use:

- `row.update(...)` for normal row patches
- `row.with_video(...)` for video placement or timestamp patches
- `row.with_stats(...)` when you want to write one feature's stats back
- `row.drop(...)` to hide ordinary row fields

Those helpers return a new `LeRobotRow`; they do not mutate the input row.

## Transforming Rows

Once read, LeRobot data is manipulated through the same row pipeline model as
everything else in Refiner.

Example:

```python
pipeline = pipeline.map(lambda row: row.update(source_dataset="aloha_static_battery"))
```

Because `map(...)` patches rows by default, this is a convenient way to add
episode-level metadata or derived fields before writing.

Because episode rows carry a `tasks` list, you can also remove episodes by task
on the vectorized path:

```python
import refiner as mdr

pipeline = (
    mdr.read_lerobot("hf://datasets/macrodata/aloha_static_battery_ep005_009")
    .filter(~mdr.col("tasks").is_in(["pick"]))
)
```

For list-valued columns like `tasks`, `col("tasks").is_in(["pick"])` means
"does this episode contain any task in that set?"

### Adapting Generic Robotics Rows

Use `to_robot_rows(...)` when your data is already organized as generic
robotics episodes or frame rows and you want the common `RoboticsRow` view:

```python
pipeline = pipeline.to_robot_rows(
    episode_id_key="episode_id",
    fps=30.0,
    robot_type="aloha",
    nested_frames_key="frames",
    timestamp_key="time_s",
    action_key="actions",
    state_key="qpos",
    extra_observation_keys={"images.main": "camera"},
    video_keys={"images.front": "front_video"},
)
```

By default `episode_id_key`, `fps`, and `robot_type` are unset. Pass
`episode_id_key=` when source rows carry a stable episode id. Pass literal
`fps=` and `robot_type=` values when they are dataset constants, or pass
`fps_key=` and `robot_type_key=` when each source row carries those values in
columns.

For `layout="episode_rows"`, `nested_frames_key=` names the source column that
contains nested per-frame rows. For `layout="frame_rows"`, it names the grouped
per-episode frame table written by the adapter. When omitted in `frame_rows`
mode, Refiner uses an internal hidden key, so unclaimed source columns stay in
the nested frame table and do not appear as episode metadata unless listed in
`episode_metadata_keys=`.

Video fields are detected from schema asset metadata, such as
`dtypes={"front_video": mdr.datatype.video_path()}` on readers that accept
`dtypes`. For raw rows without asset metadata, pass `video_keys=` to declare
which source keys are video streams. `extra_observation_keys=` and `video_keys=`
both accept either a mapping for renames or an iterable of source keys when no
rename is needed.

## Writing Datasets

Use `write_lerobot(...)` to write a LeRobot-compatible output dataset:

```python
pipeline = pipeline.write_lerobot("hf://buckets/macrodata/my_robotics_output")
```

Inputs must be `LeRobotRow` values, such as rows from `read_lerobot(...)`, or
generic `RoboticsRow` values. For raw robotics rows, call `to_robot_rows(...)`
before `write_lerobot(...)`.

This is more than a generic file writer. The LeRobot writer handles:

- LeRobot-compatible dataset layout
- episode/frame materialization
- video handling
- dataset metadata reduction and finalization across stages

Current writer tuning is passed directly on `write_lerobot(...)`, including:

- `data_files_size_in_mb`
  - approximate rollover target for frame parquet files
- `video_files_size_in_mb`
  - approximate rollover target for emitted video files
- `max_video_prepare_in_flight`
  - upper bound on concurrent episode-level video preparation work inside one worker
- `codec`
  - requested output video codec when transcoding is needed
- `pix_fmt`
  - requested output pixel format for transcoded videos
- `transencoding_threads`
  - total per-worker transcode thread budget, divided across concurrent video streams on the same row
- `encoder_options`
  - extra codec-specific options passed to the encoder
- `quantile_bins`
  - quantile resolution used when computing LeRobot stats files
- `force_recompute_video_stats`
  - force decoded-frame video stats recomputation even when compatible source stats already exist

### Stage-1 Writes And Stage-2 Reduction

`write_lerobot(...)` is a two-stage write.

Stage 1 is shard-local:

- each worker writes episode rows, frame parquet files, and any emitted videos
- each worker also writes shard-local metadata under `meta/chunk-*`
- this stage keeps writes incremental and avoids cross-worker coordination while
  rows are still in flight

Stage 2 reduces the finalized shard outputs into the final dataset metadata:

- `meta/episodes/chunk-000/file-000.parquet`
- `meta/tasks.parquet`
- `meta/info.json`
- `meta/stats.json`

During reduction, Refiner also:

- keeps only finalized `data/chunk-*` and `videos/.../chunk-*` payloads
- removes stage-1 `meta/chunk-*` metadata
- fixes duplicated `episode_index` values when merging datasets that overlap

This is why the writer can stay fast during stage 1 while still producing a
single normal LeRobot dataset layout at the end.

### LeRobot Performance Notes

Current LeRobot output is optimized for:

- incremental frame parquet writes
- asynchronous per-episode video preparation
- cheap metadata reduction after shard-local stage-1 output

Key decisions:

- remux is preferred when source packets and boundaries are compatible
- transcode is used when compatibility or stats recomputation requires decoded frames
- `max_video_prepare_in_flight` bounds concurrent episode video work inside one worker
- `transencoding_threads` is treated as a worker budget and divided across simultaneous video streams in the same row

The practical consequence is:

- frame-heavy no-video datasets mostly behave like a parquet writer
- video-heavy datasets are dominated by source probing, remux/transcode work, and the quality of source clip alignment

## Motion Trimming

Motion trimming is currently available through `mdr.robotics.motion_trim(...)`.

Example:

```python
import refiner as mdr

(
    mdr.read_lerobot("hf://datasets/macrodata/aloha_static_battery_ep005_009")
    .map(
        mdr.robotics.motion_trim(
            threshold=0.001,
            pad_frames=5,
        )
    )
    .write_lerobot("hf://buckets/macrodata/test_bucket/aloha_motion")
    .launch_cloud(
        name="motion_trim",
        num_workers=4,
    )
)
```

`motion_trim(...)` assumes LeRobot episode rows:

- it expects a `LeRobotRow`
- it trims the episode frame table directly
- it updates video timestamps on the row itself
- when a video span changes, the corresponding `stats/<video_key>/...` fields are dropped so the writer recomputes them later

## Egocentric Hand Tracking

Use `track_hands(...)` inside `batch_map(...)` if you want to enhance
your video with vision-based hand tracking. This is especially useful when you
want to derive actions from egocentric videos.

```python
import refiner as mdr

pipeline = (
    mdr.from_items(rows)
    .to_robot_rows(video_keys={"video": "video"})
    .batch_map(
        mdr.robotics.track_hands(
            video_key="video",
            output_key="hand_tracking",
        ),
        batch_size=4,
    )
)
```

The output column contains one hand-tracking result per input row with:

- `camera_trajectory` (estimated camera pose per frame)
- `intrinsics` (camera projection parameters)
- `hands_camera` (reconstructed hands in the camera frame)
- `hands_world` (hands transformed with the estimated camera trajectory)
- `relative_actions` (frame-to-frame wrist/hand motion deltas)
- `prediction` (lower-level model output)
- `diagnostics` (debugging metadata)

## Reward Scoring

Use `mdr.robotics.reward_score(...)` to score LeRobot episodes with Robometer.

```python
import refiner as mdr

pipeline = (
    mdr.read_lerobot("hf://datasets/lerobot/aloha_sim_transfer_cube_human")
    .map_async(
        mdr.robotics.reward_score(
            model="aliangdw/Robometer-4B",
            video_key="observation.images.top",
            max_frames=8,
        ),
        max_in_flight=256,
        preserve_order=False,
    )
    .write_lerobot("hf://buckets/acme/aloha_scored")
)
```

The transform writes `reward_score` and `robometer_success` columns. Each value
is a list aligned to the sampled frames, so `max_frames=8` produces up to eight
scores per episode.

## Subtask Annotation

Subtask annotation predicts timestamped subtasks for each robot episode.

```python
import refiner as mdr

annotate_subtasks = mdr.robotics.subtask_annotation(
    provider=mdr.inference.GoogleEndpointProvider(
        model="gemini-flash-latest",
    ),
    sample_sec=0.5,
    frame_width=224,
    frames_per_sheet=20,
    columns=5,
    min_segment_duration_sec=0.0,
    include_contact_sheet_manifest=False,
)

pipeline = (
    mdr.read_lerobot("hf://datasets/acme/robot_episodes")
    .map_async(
        annotate_subtasks,
        max_in_flight=16,
        preserve_order=False,
    )
    .write_lerobot("hf://buckets/acme/robot_subtask_annotations")
)

stats = pipeline.launch_local(
    name="robot-subtask-annotation",
    num_workers=8,
)
```

Output columns:

- `predicted_subtasks`: list of `{start_sec, end_sec, subtask}` objects

Install `macrodata-refiner[robotics]` and set the provider API key:
`GOOGLE_API_KEY`.

## Merging Datasets

You can merge compatible LeRobot datasets by reading multiple roots and writing
them back through `write_lerobot(...)`.

```python
import refiner as mdr

(
    mdr.read_lerobot(
        [
            "hf://datasets/macrodata/aloha_static_battery_ep005_009",
            "hf://datasets/macrodata/aloha_static_battery_ep000_004",
        ]
    )
    .write_lerobot("hf://buckets/macrodata/test_bucket/aloha_merge")
    .launch_local(
        name="merge_aloha",
        num_workers=2,
    )
)
```

## Related Pages

- [Reading and writing data](reading-and-writing.md)
- [Pipeline basics](pipeline-basics.md)
- [Transforms](transforms.md)
- [Launchers](launchers.md)
- [Observability](observability.md)
