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
being silently ignored. Put the endpoint and credentials in the URI or Lance's
supported environment/configuration.

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

## Internal Notes

Workers buffer only the requested output columns plus internal ordering columns
as Arrow tables. At fragment completion, they reorder those tables with Arrow,
write uncommitted column files, and record replacement-fragment metadata. The
reducer commits finalized attempts once against the pinned read version with a
Lance merge operation. Before committing, it verifies that every non-empty
fragment in the pinned source version has exactly one finalized result. Cleanup
records only files created by each attempt, so rejected retries cannot delete
base dataset files.

This follows the worker-output/coordinator-commit pattern used by Spark,
Beam/Dataflow, Daft, and Ray Data. Refiner keeps Lance fragments as its work
unit because Lance schema evolution is fragment-based; repartitioning first
would require a keyed join or staging dataset. Hugging Face Datasets generally
materializes a new dataset revision instead of attaching column files to
existing fragments.
