---
title: "Lance"
description: "Write Lance files, datasets, and distributed schema evolution results"
---

# Lance

Lance support is optional:

```bash
pip install macrodata-refiner[lance]
```

## Standalone files

Use `write_lance(...)` to create one independent Lance file per finalized
Refiner shard:

```python
import refiner as mdr

pipeline = (
    mdr.read_parquet("s3://my-bucket/raw/*.parquet")
    .write_lance("s3://my-bucket/lance-files/")
)
```

## Lance datasets

Use `write_lance_dataset(...)` for committed Lance datasets:

```python
pipeline = (
    mdr.read_parquet("s3://my-bucket/raw/*.parquet")
    .write_lance_dataset("s3://my-bucket/clean.lance", mode=mdr.Create())
)
```

Asset output uses the same format-independent configurations as Parquet and
JSONL. Individual files use `FileAssetConfig`; packed block files use
`BlobAssetConfig` and are stored as `{path, offset, size}` columns:

```python
pipeline.write_lance_dataset(
    "s3://my-bucket/clean.lance",
    assets=mdr.BlobAssetConfig(target_bytes=1 << 30),
)
```

Here `1 << 30` selects a 1 GiB target block size. Refiner does not create
Lance-native blob columns when writing; it writes generic block files and typed
`{path, offset, size}` references that can be read with `mdr.read_blob(...)`.

Supported modes are `Create()`, `Overwrite()`, `Append()`, and `AddColumns()`.
The legacy strings remain accepted for backward compatibility.
Empty `Create()` and `Overwrite()` jobs fail explicitly. Empty `Append()` jobs
are no-ops.

Lance opens dataset URIs through its native storage layer. Configured fsspec
filesystem objects and `storage_options` are therefore rejected instead of
being silently ignored. Configure the endpoint and credentials through
provider-standard environment settings recognized by both Lance and fsspec;
the dataset writer uses Lance for fragments and fsspec for attempt handoff
metadata. Credential-bearing URIs and URI query parameters are rejected so
pipeline plans do not serialize secrets.

## Adding columns

Use `AddColumns()` for row-preserving enrichment such as model inference:

```python
pipeline = (
    mdr.load_lance(
        "s3://my-bucket/hands.lance",
        version=42,
        columns=["image"],
    )
    .map(
        detect_hands,
        dtypes={
            "hand_boxes": mdr.datatype.list(mdr.datatype.float32()),
            "detector_score": mdr.datatype.float32(),
        },
    )
    .write_lance_dataset(
        "s3://my-bucket/hands.lance",
        mode=mdr.AddColumns(),
        columns=["hand_boxes", "detector_score"],
    )
)
```

New asset columns can use either output layout in the same operation:

```python
pipeline.write_lance_dataset(
    "s3://my-bucket/hands.lance",
    mode=mdr.AddColumns(),
    columns=["crop"],
    assets=mdr.BlobAssetConfig(target_bytes=1 << 30),
)
```

The `columns` argument is required and only those columns are written. Existing
columns, including large blob columns, remain referenced by their original
files. Results may arrive out of order; Refiner restores fragment-local source
order before writing. Omitted results are filled with null by default, while
duplicate source row identities fail execution and do not create a new dataset
version. The legacy string `mode="add_columns"` retains its strict behavior and
fails if any result is missing. `AddColumns()` also fails explicitly for a
dataset with no rows because no fragment exists to receive the new column files.
Fragments containing deleted rows are currently rejected because their physical
row layout cannot yet be safely reconstructed by the column writer.

Every finalized shard emits coordination metadata, including shards that
produce no rows. The reducer requires complete metadata coverage before it
commits, so a missing worker result fails explicitly instead of silently
omitting rows.

Source row identity is carried in the hidden `__source_row_id` execution column,
alongside `__shard_id`. Ordinary readers receive a shard-local row ordinal;
Lance supplies its physical row address so `add_columns` can restore
fragment-local order after asynchronous or batched transforms. Both columns are
removed before user data is persisted.

### Filling rows outside a bounded or filtered source

Use `AddColumns` when inference should run on only part of the source while the
new columns still need to align with every row in the existing dataset:

```python
pipeline = (
    mdr.load_lance(
        "s3://my-bucket/hands.lance",
        columns=["image"],
        max_rows=200_000,
    )
    .map(
        embed_hand,
        dtypes={"embedding": mdr.datatype.list(mdr.datatype.float32())},
    )
    .write_lance_dataset(
        "s3://my-bucket/hands.lance",
        mode=mdr.AddColumns(),
        columns=["embedding"],
    )
)
```

Processed rows receive their computed values. Rows beyond `max_rows`, or rows
removed by a filter, receive null. Supply one Arrow-compatible scalar for every
column or a mapping for per-column defaults:

```python
mode=mdr.AddColumns(fill={"embedding": None, "status": "not_processed"})
```

Unknown fill-column names and values that cannot be converted to the declared
Arrow type fail before the dataset commit. Duplicate source row identities also
remain an error. Existing columns and data files are retained; the operation
commits only the new column files as a new Lance dataset version.
