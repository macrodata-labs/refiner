---
title: "Validation Contracts"
description: "Assert nullability, ranges, uniqueness, row counts, and custom data-quality rules"
---

# Validation Contracts

Add `validate(...)` wherever a pipeline should stop on invalid data. Validation
does not filter or modify rows: passing rows continue to the next transform or
writer unchanged, and the first violation raises `ValidationError` with the
contract name, failed rule, and source location when available.

```python
import refiner as mdr

pipeline = mdr.read_parquet("s3://training-data/episodes/*.parquet").validate(
    name="training_episodes",
    not_null=["episode_id", "score"],
    unique=["episode_id"],
    ranges={"score": (0.0, 1.0)},
    min_rows=1,
    predicates={
        "non_empty_label": lambda row: bool(str(row["label"]).strip()),
    },
)
```

The range endpoints are inclusive. Null values pass a range check; add the same
column to `not_null` when null is invalid. Nulls participate in uniqueness, so
two null values violate `unique` unless nulls are rejected earlier by
`not_null`.

## Available Rules

| Argument | Check |
| --- | --- |
| `not_null=["id"]` | Every listed column exists and contains no nulls. |
| `ranges={"score": (0, 1)}` | Non-null values fall inside an inclusive range. Use `None` for an open endpoint. |
| `predicates={"valid": fn}` | Every row makes each named Python predicate return true. |
| `unique=["id"]` | Each listed column is globally unique. |
| `unique_together=[("tenant", "id")]` | Each composite key is globally unique. |
| `min_rows=1` / `max_rows=...` | The dataset row count stays within the requested bound. |
| `exact_rows=...` | The dataset contains exactly that many rows. This cannot be combined with `min_rows` or `max_rows`. |

Rule checks happen at the position of the validation step. This lets you
validate raw inputs, transformed outputs, or both:

```python
pipeline = (
    mdr.read_parquet("episodes/*.parquet")
    .validate(name="raw", not_null=["episode_id"])
    .with_column("score", mdr.col("reward").clip(min_value=0.0, max_value=1.0))
    .validate(name="normalized", ranges={"score": (0.0, 1.0)})
)
```

## Reuse a Contract

Create `ValidationContract` once when several pipelines share the same rules:

```python
import refiner as mdr

episode_contract = mdr.ValidationContract(
    name="episode_identity",
    not_null=["dataset_id", "episode_id"],
    unique_together=[("dataset_id", "episode_id")],
    min_rows=1,
)

validated = mdr.read_parquet("episodes/*.parquet").validate(episode_contract)
```

Pass either a contract or inline rules to one `validate(...)` call, not both.

## Run the Validation

Pipelines remain lazy. A local inspection that exhausts the input runs every
rule:

```python
for _ in validated.iter_rows():
    pass
```

For production-sized inputs, attach the intended writer or launch the pipeline
with its null sink using `launch_local(...)` or `launch_cloud(...)`. `take(n)`
only reads a prefix, so it cannot prove a global row-count or uniqueness rule
over unseen rows.

## Parallelism and Global Checks

Null, range, and custom predicate checks are row-local. When the source exposes
a schema, they preserve its shard plan and worker count, and built-in checks
operate directly on Arrow blocks.

Uniqueness and row-count checks need exact dataset-wide state. Refiner currently
groups the physical source shards into one deterministic scheduling shard and
uses one worker for a pipeline containing either rule. The grouped claim carries
the physical plan to workers, so cloud execution and retries do not rediscover a
different source order. This guarantees exact results, including for empty
datasets, but it serializes the whole pipeline and stores observed uniqueness
keys in worker memory. Keep high-cardinality global checks separate from your
main production transform when that cost is material.

A required-column rule on a schema-less input also uses the grouped one-worker
path. Without schema metadata, an empty distributed input cannot otherwise be
distinguished from workers that happened to receive empty shards. Supplying a
reader schema keeps null and range validation shard-parallel.

## Failure Handling

Catch `ValidationError` when a caller should handle contract failures
programmatically:

```python
try:
    list(validated.iter_rows())
except mdr.ValidationError as error:
    print(error.contract_name, error.rule, error.location)
```

A named custom predicate that raises is wrapped in `ValidationError`, with the
original exception retained as its cause.

## Internal Notes

Spark, Daft, and Ray Data can implement global uniqueness through distributed
grouping or aggregation; Beam/Dataflow expresses the same work as keyed
combines with explicit window semantics. Hugging Face Datasets commonly runs
validation in one process or relies on a separate validation pass. Refiner does
not yet have a general shuffle/barrier transform, so this implementation keeps
schema-backed row-local rules parallel and deliberately serializes exact global
rules and schema-less column-existence checks instead of presenting
worker-local results as dataset-wide guarantees. A future partitioned
validation reducer can replace that execution strategy without changing
`ValidationContract` or `validate(...)`.
