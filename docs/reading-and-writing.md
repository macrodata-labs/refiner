---
title: "Reading and Writing Data"
description: "Sources, row shape, sharding, and sinks in Refiner"
---

Most Refiner pipelines follow the same shape:

1. read rows from a source
2. transform those rows
3. write them through a sink

This page covers the read and write side. See [Transforms](transforms.md) for what you can run in the middle.

## Reading data

Built-in readers:

| reader | what it reads | typical row shape |
| --- | --- | --- |
| `read_csv(...)` | CSV files | dict-like rows keyed by column name |
| `read_jsonl(...)` | JSON Lines files | dict-like rows from each JSON object |
| `read_parquet(...)` | Parquet datasets or files | row views backed by Arrow columns |
| `read_hf_dataset(...)` | Hugging Face datasets | rows from generated Parquet shards, with optional file path resolution |
| `read_lerobot(...)` | LeRobot robotics datasets | one row per episode, including frame/video metadata |
| `text.read_commoncrawl(...)` | Common Crawl WARC or WET files | one row per WARC/WET record with selected WARC/HTTP fields |
| `text.read_commoncrawl_from_index(...)` | Common Crawl WARC records via the parquet index | one row per fetched WARC record, planned from index rows |
| `from_items(...)` | in-memory Python values | rows from mappings directly, or `{"item": value}` for primitives |
| `from_source(...)` | a custom source object | whatever row shape your source emits |

Example:

```python
import refiner as mdr

pipeline = mdr.read_parquet("s3://my-bucket/documents/*.parquet")
```

## Hugging Face datasets

Use `read_hf_dataset(...)` for datasets hosted on the Hugging Face Hub. It
resolves the dataset config with the `datasets` package, reads generated Parquet
shards through Refiner's Parquet reader, and falls back to `datasets` streaming
when generated Parquet shards are unavailable.

```bash
uv add "macrodata-refiner[huggingface]"
```

```python
import refiner as mdr

pipeline = mdr.read_hf_dataset(
    "user/my-dataset",
    config="default",
    split="train",
    dtypes={"video": mdr.datatype.video_file()},
)
```

Hugging Face `Image`, `Audio`, and `Video` features are marked as file columns
automatically. Relative file paths in those columns are resolved to
`hf://datasets/...` references by default; pass `resolve_relative_paths=False` to
keep the raw values. Absolute paths and URI values are not rewritten.

## Common Crawl text readers

[Common Crawl](https://commoncrawl.org/) publishes large public web crawls.
Refiner supports two Common Crawl text-oriented inputs:

- `warc`
  - the raw crawl archive files, one WARC record per fetched response or metadata object
- `wet`
  - the derived text-extraction files that Common Crawl publishes separately

Use these readers when you want to process public web crawl data directly from a
Common Crawl dump such as `CC-MAIN-2025-13`.

Common Crawl readers live under the optional `macrodata-refiner[text]` extra because they
rely on `warcio`. HTTPS is the default transport. If you want to read directly
from `s3://commoncrawl`, install the separate `macrodata-refiner[s3]` extra.

```bash
uv add "macrodata-refiner[text]"
```

```bash
uv add "macrodata-refiner[s3]"
```

There are two distinct entrypoints:

- `mdr.text.read_commoncrawl(...)`
  - direct file-backed reader over Common Crawl `warc` or `wet` files
  - shards over whole files
  - uses the public HTTPS mirror by default
  - best for dense scans where you expect to read many records from the files you touch
- `mdr.text.read_commoncrawl_from_index(...)`
  - WARC-only reader backed by Common Crawl's parquet index
  - supports `filter=...` and `filter_fn=...` on index rows before WARC fetches
  - uses the public HTTPS mirror by default
  - best for sparse targeted retrieval where matching records are scattered across many WARC files

Use the direct reader for sequential scans:

```python
import refiner as mdr

pipeline = mdr.text.read_commoncrawl(
    "CC-MAIN-2025-13",
    format="warc",
    output_fields=(
        "WARC-Target-URI",
        "Content-Type",
        "content_bytes",
    ),
)
```

Use the index-backed reader when the hit rate is low and direct scans would waste
work, for example when fetching only PDFs:

```python
import refiner as mdr

pipeline = mdr.text.read_commoncrawl_from_index(
    "CC-MAIN-2025-13",
    filter=mdr.text.commoncrawl.filter_pdf,
    output_fields=(
        "WARC-Target-URI",
        "Content-Type",
        "content_bytes",
    ),
)
```

`output_fields` accepts either an explicit field list or `"all"`.

- Explicit fields use original WARC or HTTP header names, plus `content_bytes`
- `"all"` includes all original WARC headers, all non-conflicting HTTP headers,
  and `content_bytes`

Common Crawl index metadata and column names are documented by Common Crawl at:
<https://commoncrawl.org/columnar-index>

## Sharding

Readers also define how work is split into shards.

A shard is the unit of work claimed by a worker. Common shard metadata includes things like:

- file path
- byte range
- row-group range
- reader-specific ids

Reader behavior differs by format:

- CSV and JSONL usually shard by file and byte range
- Parquet shards by file and planned row-group or row ranges
- LeRobot shards by episode parquet metadata
- `from_items(...)` shards synthetic in-memory rows into planned chunks

You can often influence shard planning with options like:

- `target_shard_bytes=...`
  - approximate shard size target used by readers that plan work from file size or byte ranges
- `num_shards=...`
  - explicit shard-count target when you want to cap or force parallelism more directly than a byte budget

## Writing data

Built-in sinks:

| sink | what it writes | notes |
| --- | --- | --- |
| `.write_jsonl(output, ...)` | JSON Lines files | one output file per worker/shard according to the filename template |
| `.write_parquet(output, ...)` | Parquet files | columnar output with optional compression |
| `.write_lerobot(output, ...)` | LeRobot-compatible robotics datasets | materializes frame/video assets and dataset metadata |

Example:

```python
pipeline = pipeline.write_parquet("s3://my-bucket/clean-output/")
```

`write_jsonl(...)` and `write_parquet(...)` can also copy file-typed asset
columns into the same output folder:

```python
pipeline = pipeline.write_parquet(
    "s3://my-bucket/clean-output/",
    upload_assets=True,
)
```

When `upload_assets=True`, Refiner infers asset columns from file dtype metadata,
copies those files without decoding them, and rewrites the column values to the
copied asset paths. Assets are written under
`{output}/assets/{shard_id}__w{worker_id}/...` by default; use `assets_subdir` to
change the subfolder name and `max_asset_uploads_in_flight` to bound per-worker
copy concurrency. Missing or unreadable asset paths fail the shard.

Mark path columns as assets with `dtypes=...` on row transforms or with
`cast(...)`:

```python
pipeline = pipeline.map(
    lambda row: {"image": f"{row['image_dir']}/{row['image_name']}"},
    dtypes={"image": mdr.datatype.image_file()},
)

pipeline = pipeline.cast(video=mdr.datatype.video_file())
```

When you run a writer through `launch_local(...)` or `launch_cloud(...)`, some
sinks add a reducer stage after the main writer stage. For `write_jsonl(...)`
and `write_parquet(...)`, that reducer removes stale shard/worker files and
uploaded asset attempt folders, keeping only finalized outputs. The output
prefix should therefore be dedicated to Refiner-managed files.

## What Python Functions Actually See

Reader output eventually flows into Python UDFs as `Row` objects:

- `map(...)` gets one `Row`
- `map_async(...)` gets one `Row`
- `flat_map(...)` gets one `Row`
- Python `filter(...)` gets one `Row`
- `batch_map(...)` gets `list[Row]`
- `map_table(...)` gets one `pa.Table`

Even when the engine is using Arrow-backed `Tabular` blocks internally, ordinary
Python row UDFs do not receive them directly.

That means your custom Python code should assume:

- mapping-like row access (`row["col"]`)
- immutable row helpers such as `row.update(...)` and `row.drop(...)`

and should not assume:

- every row is a plain `dict`
- mutating a row in place is part of the API
- internal vectorized blocks are exposed directly to row UDFs

`map_table(...)` is the explicit exception when you want the fused underlying
Arrow table. It receives a `pa.Table` and must return a `pa.Table`.

## LeRobot-specific notes

`read_lerobot(...)` and `write_lerobot(...)` are specialized for robotics datasets.

`read_lerobot(...)` yields one row per episode. Those rows include:

- `frames`
- episode metadata
- video feature columns as handles
- dataset and episode stats metadata

In practice, LeRobot rows are `LeRobotRow` wrappers over a normal base row. The
important high-level fields/views are:

- `frames`
  - a normal `Tabular` for the episode frame rows
- `metadata`
  - a `LeRobotMetadata` dataclass with dataset-level info, stats, and canonical tasks
- `videos`
  - `VideoFile` handles plus timestamp helpers
- `stats`
  - a LeRobot-aware stats view over `stats/<feature>/...` row fields

`write_lerobot(...)` is more than a generic file writer. It handles:

- LeRobot dataset layout
- frame parquet output
- video materialization
- dataset metadata reduction across stages

Important writer assumptions:

- input rows are episode-shaped LeRobot rows
- `metadata` must be present on each row
- per-frame `task_index` is canonical; episode-level `tasks` are rebuilt from frames
- `fps` and `robot_type` must remain stable within a written shard

### LeRobot performance notes

The LeRobot writer is optimized around one pipeline row = one episode.

Within a shard:

- frame parquet writing is incremental and synchronous
- video preparation is asynchronous per episode
- shard-local metadata is buffered and flushed at shard completion

This split helps because frame parquet writes are cheap Arrow appends, while
video work is dominated by probing, file IO, and codec work.

#### `max_video_prepare_in_flight`

`max_video_prepare_in_flight` bounds how many episode-level video preparation
tasks can be in flight inside one worker.

It helps by:

- overlapping remote file opens and source probing
- letting frame parquet writes keep moving while video prep is outstanding
- preventing unbounded fan-out on video-heavy shards

#### Remux vs transcode

The writer prefers remux when it can, and falls back to transcode when it must.

Remux is used when:

- source and output container or codec details are compatible
- clip boundaries are safe to represent without decoding or re-encoding
- existing source video stats may be reused

Transcode is used when:

- the requested output codec or pixel format requires it
- clip boundaries are not remux-safe
- `force_recompute_video_stats=True`
- source stats are missing or were invalidated by an upstream transform

Why this matters:

- remux is usually much faster because it mostly moves packets
- transcode is more expensive, but it is what makes fresh decoded-frame stats possible

#### Threading assumptions

`transencoding_threads` is treated as a per-worker budget for transcode work.
When several video streams exist on the same row, the writer divides that budget
across them so separate stream writers do not all try to claim the full worker.

See [Robotics](robotics.md) for a fuller walkthrough.

## In-process vs launched execution

When you iterate a pipeline directly with `.take(...)` or `.materialize(...)`, Refiner returns rows and does not write sink output.

Writers matter when you launch the pipeline with:

- `.launch_local(...)`
- `.launch_cloud(...)`

## Related pages

- [Transforms](transforms.md)
- [Expressions](expressions.md)
- [Robotics](robotics.md)
- [Launchers](launchers.md)
