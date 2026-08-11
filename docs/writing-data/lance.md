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
    .write_lance_dataset("s3://my-bucket/clean.lance", mode="create")
)
```

Supported modes are `create`, `overwrite`, `append`, and `add_columns`.
Empty `create` and `overwrite` jobs commit an empty dataset when Refiner can
determine the output Arrow schema statically; otherwise they fail explicitly.

Lance opens dataset URIs through its native storage layer. Configured fsspec
filesystem objects and `storage_options` are therefore rejected instead of
being silently ignored. Configure the endpoint and credentials through
provider-standard environment settings recognized by both Lance and fsspec;
the dataset writer uses Lance for fragments and fsspec for attempt handoff
metadata. Credential-bearing URIs and URI query parameters are rejected so
pipeline plans do not serialize secrets.

## Adding columns

Use `add_columns` for row-preserving enrichment such as model inference:

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
        mode="add_columns",
        columns=["hand_boxes", "detector_score"],
    )
)
```

The `columns` argument is required and only those columns are written. Existing
columns, including large blob columns, remain referenced by their original
files. Results may arrive out of order; Refiner restores fragment-local source
order before writing. Missing or duplicate results fail execution and do not
create a new dataset version. `add_columns` also fails explicitly for a dataset
with no rows because no fragment exists to receive the new column files.
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
