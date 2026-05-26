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
- [reward scoring](#reward-scoring)
- [task segmentation](#task-segmentation)
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

`row.videos` is a mapping from video feature key to `LeRobotVideoRef`.

For each video ref, you can access:

- `video.uri`
- `video.from_timestamp_s`
- `video.to_timestamp_s`
- `video.video`
  - the underlying `VideoFile`

You can also decode a `VideoFile` lazily through methods on the handle itself:

- `video.iter_frames()`
- `video.iter_frame_windows(offsets=[...], stride=...)`
- `await video.export_clip()`

Example:

```python
def shift_videos_by_half_second(row):
    for key, video in row.videos.items():
        row = row.with_video(
            key,
            from_timestamp_s=(video.from_timestamp_s or 0.0) + 0.5,
            to_timestamp_s=(video.to_timestamp_s or 0.0) + 0.5,
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

## Writing Datasets

Use `write_lerobot(...)` to write a LeRobot-compatible output dataset:

```python
pipeline = pipeline.write_lerobot("hf://buckets/macrodata/my_robotics_output")
```

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

## Task Segmentation

Task segmentation turns one robot episode into a list of timestamped subtasks.
For multimodal models that can read video directly, use
`mdr.inference.generate_text(...)` with a video content part and a Pydantic
schema. Refiner handles the provider request shape, retries, response parsing,
and local schema validation.

```python
from pydantic import BaseModel
import refiner as mdr


class Segment(BaseModel):
    start_sec: float
    end_sec: float
    subtask: str


class Segmentation(BaseModel):
    segments: list[Segment]


PROMPT = """Reconstruct the sequence of manipulation events in this robot video.

Return only JSON with this shape:
{"segments":[{"start_sec":0.0,"end_sec":1.0,"subtask":"short action description"}]}

Rules:
- Treat each segment as one event that changes what is true about the world.
- Use boundaries where an object is held, released, moved, opened, closed, or
  contents move.
- Avoid idle time, camera motion, hesitation, and tiny hand adjustments.
"""


async def segment_episode(row, generate_text):
    video = next(iter(row.videos.values())).video
    video_bytes = await video.export_clip()
    instruction = " ".join(row.tasks)

    response = await generate_text(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{PROMPT}\nEpisode instruction: {instruction}\n",
                    },
                    {
                        "type": "file",
                        "mediaType": "video/mp4",
                        "data": video_bytes,
                    },
                ],
            }
        ],
        schema=Segmentation,
        temperature=0.1,
        providerOptions={
            "google": {
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE",
                    },
                ]
            }
        },
    )

    segments = [
        segment.model_dump()
        for segment in response.object.segments
        if segment.end_sec > segment.start_sec
    ]
    return row.update(
        predicted_subtasks=segments,
        raw_annotation_output=response.text,
        annotation_model="gemini-flash-latest",
    )


pipeline = (
    mdr.read_lerobot("hf://datasets/acme/robot_episodes")
    .map_async(
        mdr.inference.generate_text(
            fn=segment_episode,
            provider=mdr.inference.GoogleEndpointProvider(
                model="gemini-flash-latest",
            ),
        ),
        max_in_flight=16,
        preserve_order=False,
    )
    .write_lerobot("hf://buckets/acme/robot_task_segments")
)
```

For path-only or embedded-video datasets that are not LeRobot rows, pass the
video bytes from the row directly:

```python
{
    "type": "file",
    "mediaType": "video/mp4",
    "data": row["video"],
}
```

If the model performs better with contact sheets than raw video, you can still
avoid OpenCV. Refiner's video extra already uses PyAV, so contact sheets can be
built from `VideoFile.iter_frames()` and encoded as JPEG with PyAV. The
recommended prompt shape is to send the JPEG sheets as image parts and include a
text manifest that maps tile positions to timestamps, instead of drawing
timestamps into the pixels.

```python
import io
import numpy as np
import av


def encode_jpeg(rgb: np.ndarray) -> bytes:
    output = io.BytesIO()
    frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
    with av.open(output, mode="w", format="mjpeg") as container:
        stream = container.add_stream("mjpeg", rate=1)
        stream.width = frame.width
        stream.height = frame.height
        stream.pix_fmt = "yuvj420p"
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return output.getvalue()


async def sample_rgb_frames(video, every_sec: float) -> list[tuple[float, np.ndarray]]:
    sampled = []
    next_time = 0.0
    async for frame in video.iter_frames():
        timestamp = frame.timestamp_s
        if timestamp is None or timestamp + 1e-6 < next_time:
            continue
        sampled.append((timestamp, frame.frame.to_ndarray(format="rgb24")))
        next_time = timestamp + every_sec
    return sampled
```

The contact sheet helper above only needs `macrodata-refiner[video]`; it does
not require `opencv-python-headless`.

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
