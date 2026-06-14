---
title: "Rerun writer"
description: "Write Rerun recording rows as distributed RRD files"
---

# Rerun writer

Use `write_rerun` to write rows containing a `RerunRecording` value, as emitted
by `read_rerun(output="recording")`.

```python
pipeline = (
    mdr.read_rerun(
        "s3://bucket/input/*.rrd",
        output="recording",
        contents=("/action/**", "/observation/**"),
    )
    .write_rerun("s3://bucket/output/rrd")
)
```

Install `macrodata-refiner[rerun]` to use this writer. Add storage extras such
as `s3` when writing remote paths.

## Output layout

`write_rerun` writes one RRD file per input recording row. The default file name
template is:

```text
{shard_id}__w{worker_id}/{row_index}.rrd
```

The template must include `{shard_id}` and `{worker_id}` so retry cleanup can
distinguish finalized worker outputs from abandoned attempt outputs. It must
also include `{row_index}` or `{segment_id}` so each input row writes a distinct
RRD file. When `{segment_id}` is present, the recording segment id must be a
single path segment; ids containing `/`, `\`, `.`, or `..` are rejected before
writing.

## Writer strategy

When the input row came from `read_rerun`, the writer uses Rerun's raw
`LazyChunkStream` path and writes the selected source chunks directly. This
preserves Rerun chunk metadata and avoids re-emitting large Arrow tables through
Python.

For pure copy jobs, use `read_rerun(..., materialize_tables=False)` before
`write_rerun(...)` to skip timeline/static table materialization while keeping
the raw source chunks available to the writer.

If a `RerunRecording` has no source file, the writer falls back to table
emission with `send_dataframe`. Static Rerun component columns are sent as
static data, and dynamic timeline tables are sent separately. The same fallback
is used when `write_footer=False`, because Rerun's raw chunk writer always
writes footer metadata. No-footer writes require materialized Rerun table data;
metadata-only rows from `materialize_tables=False` should use the default
`write_footer=True` raw chunk path.

## Reducer

The writer is distributed. Each worker writes deterministic shard-local files,
then a reducer stage removes files from non-finalized worker attempts. There is
no global merge step because the output is a directory of independent RRD
recordings.

Use `write_lerobot` instead when the goal is a single training-ready robotics
dataset with merged LeRobot metadata.
