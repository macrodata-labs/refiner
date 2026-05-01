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
| `read_hdf5(...)` | HDF5 files | one row per selected HDF5 group |
| `read_jsonl(...)` | JSON Lines files | dict-like rows from each JSON object |
| `read_parquet(...)` | Parquet datasets or files | row views backed by Arrow columns |
| `read_webdataset(...)` | WebDataset tar archives | one row per sample, with member extensions as fields |
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
    dtypes={"video": mdr.datatype.video_path()},
)
```

Set `HF_TOKEN` in the environment when reading private or gated datasets, or when
you want authenticated Hugging Face rate limits. For cloud launches, pass
`HF_TOKEN` through `secrets`.

Hugging Face `Image`, `Audio`, and `Video` features are marked as embedded asset
columns automatically. Refiner leaves path values unchanged; use `map(...)` if a
dataset stores paths that need custom resolution.

## HDF5

HDF5 support lives behind the optional `macrodata-refiner[hdf5]` extra because
it depends on `h5py`.

```bash
uv add "macrodata-refiner[hdf5]"
```

`read_hdf5(...)` treats each selected HDF5 group as one output row. Dataset and
attribute paths are resolved relative to that group.

```python
import refiner as mdr

pipeline = mdr.read_hdf5(
    "robot-data/*.hdf5",
    groups="/data/demo_*",
    datasets={
        "actions": "actions",
        "frames": "obs/agentview_rgb",
    },
    attrs={"task": "task"},
)
```

For a file shaped like:

```text
demo.hdf5
└── data
    ├── demo_0
    │   ├── attrs: task="stack blocks"
    │   ├── actions
    │   └── obs
    │       └── agentview_rgb
    └── demo_1
        ├── attrs: task="place cup"
        ├── actions
        └── obs
            └── agentview_rgb
```

this emits rows with columns such as:

- `hdf5_group`: the selected group path, for example `/data/demo_0`
- `file_path`: the source HDF5 file path
- `actions`: the selected dataset value
- `frames`: the selected `obs/agentview_rgb` dataset value
- `task`: the selected group attribute

`datasets` and `attrs` can be:

- a mapping of output column name to HDF5 path
- a single HDF5 path string
- a sequence of HDF5 path strings, as long as the final path components are
  unique

Use an explicit mapping when paths would derive the same column name:

```python
pipeline = mdr.read_hdf5(
    "robot-data/*.hdf5",
    groups="/data/demo_*",
    datasets={
        "left_rgb": "left/rgb",
        "right_rgb": "right/rgb",
    },
)
```

HDF5 files are planned at file granularity. `list_shards()` does not open HDF5
files or inspect group metadata. This keeps planning cheap for remote storage,
but it also means one large HDF5 file is one shard. If you pass `num_shards`,
the reader uses the shared file-sharding planner and may emit fewer shards when
there are fewer files.

`groups` accepts `"/"`, one glob such as `"/data/demo_*"`, or a list of exact
group paths. Missing groups emit no rows for that file. Missing selected
datasets or attributes default to raising an error. Set
`missing_policy="drop_row"` to drop rows with missing selected values, or
`missing_policy="set_null"` to emit `None` for missing selected datasets or attributes.
If a selected column can be missing from every group in an input file, pass
`dtypes` for that column so the reader can emit a stable Arrow type.

## WebDataset

`read_webdataset(...)` reads `.tar`, `.tar.gz`, and `.tgz` archives using the
same fsspec-backed input handling as the other file readers. Inputs can be
archive paths, globs, directories, `DataFile` values, or mixed lists of those.

```python
import refiner as mdr

pipeline = mdr.read_webdataset(
    "s3://my-bucket/shards/*.tar",
    dtypes={"jpg": mdr.datatype.image_bytes()},
)
```

The reader streams each tar archive sequentially and does not load the full
archive into memory. Archives are planned as atomic files, so `num_shards`
cannot split one large archive across multiple workers. Members for a sample
must be contiguous in the archive, which is the standard WebDataset shard
layout.

Members are grouped by the path before the first dot. The suffix after that
first dot becomes the output field name:

```text
0001.jpg
0001.json
0002.jpg
0002.txt
```

emits rows like:

- `sample_key="0001"`, `jpg=<bytes>`, `json=<dict>`
- `sample_key="0002"`, `jpg=<bytes>`, `txt=<bytes>`

Dots in the basename start the field suffix, so sample keys should not contain dots.

The archive path is added as `file_path` by default. Set
`file_path_column=None` to omit it, or `sample_key_column=...` to rename the
sample key column. JSON members are parsed to Python values by default; pass
`parse_json=False` to keep `.json` members as raw bytes. All non-JSON payloads
are emitted as bytes. Members without a dot are skipped.

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
- HDF5 shards by file only; `num_shards` cannot exceed the input file count
- Parquet shards by file and planned row-group or row ranges
- WebDataset shards by archive file only; samples are streamed from each archive
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
copy concurrency.

Missing or unreadable asset paths are controlled by `missing_asset_policy`:

| policy | behavior |
| --- | --- |
| `"error"` | fail the shard on the first missing asset path |
| `"drop_row"` | skip rows with missing asset paths |
| `"set_null"` | replace missing asset paths with null values |

For list-typed asset columns, `"drop_row"` drops the row when any copied list
item is missing, while `"set_null"` nulls only the missing list items.

Mark path columns as assets with `dtypes=...` on row transforms or with
`cast(...)`:

```python
pipeline = pipeline.map(
    lambda row: {"image": f"{row['image_dir']}/{row['image_name']}"},
    dtypes={"image": mdr.datatype.image_path()},
)

pipeline = pipeline.cast(video=mdr.datatype.video_path())
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
