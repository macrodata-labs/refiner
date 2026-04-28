---
title: "TensorFlow"
description: "Read TFRecord files and TensorFlow Datasets"
---

# TensorFlow

TensorFlow-backed readers are optional dependencies:

```bash
uv add "macrodata-refiner[tfds]"
```

Use `macrodata-refiner[tensorflow]` if you only need TFRecord files and do not
need the TensorFlow Datasets catalog.

## TFRecord Files

Use `read_tfrecords(...)` when you already have TFRecord files and know the
`tf.io` feature spec:

```python
import tensorflow as tf
import refiner as mdr

pipeline = mdr.read_tfrecords(
    "data/*.tfrecord",
    features={
        "id": tf.io.FixedLenFeature([], tf.int64),
        "image": tf.io.FixedLenFeature([], tf.string),
        "action": tf.io.FixedLenFeature([7], tf.float32),
    },
    batch_size=1024,
)
```

`read_tfrecords(...)` plans shards by whole files. Small files may be grouped
into one shard, but one large TFRecord file is not split internally. Directory
inputs are filtered to `.tfrecord`, `.tfrecords`, `.tfrec`, and their `.gz` or
`.zlib` variants.

`compression="auto"` detects gzip and zlib from the filename suffix. Pass
`compression=None`, `"gzip"`, or `"zlib"` to override it.

By default, Refiner adds a `file_path` column with the source TFRecord path. Set
`file_path_column=None` to omit it. If your parsed features already include the
same column name, Refiner leaves the parsed value unchanged.

## TensorFlow Datasets

Use `read_tfds(...)` for datasets available through TensorFlow Datasets, a
local TFDS `data_dir`, or a prepared TFDS builder directory:

```python
import refiner as mdr

pipeline = mdr.read_tfds(
    "mnist",
    split="train",
    batch_size=1024,
    examples_per_shard=10_000,
)
```

For RLDS datasets published as TFDS directories, pass the dataset version
directory:

```python
pipeline = mdr.read_tfds(
    builder_dir="data/libero_10_no_noops/1.0.0",
    split="train",
)
```

`read_tfds(...)` shards a plain split name by example ranges. Pass a split from
`builder.info.splits`, such as `"train"` or `"validation"`. Complex split
expressions are not sharded; use a prepared local TFDS split when you need a
custom subset.

TFDS decoding stays under TensorFlow Datasets. Pass `decoders`, `read_config`,
`shuffle_files`, or `as_supervised` when you need the same controls as
`builder.as_dataset(...)`.

## Performance Trade-Offs

- TFRecord files are read through `tf.data.TFRecordDataset`, batched, parsed, and
  converted to Arrow-backed `Tabular` blocks.
- TFRecord shard planning is file-granular. Many medium files parallelize well;
  one very large file runs as one source shard.
- TFDS shard planning is example-range based within one split. Use
  `examples_per_shard` or `num_shards` to control parallelism.
- RLDS-style TFDS datasets with dataset-valued `steps` are streamed one episode
  at a time because TensorFlow cannot batch nested datasets.
- For image-heavy TFDS/RLDS reads, pass TFDS `decoders` such as
  `tfds.decode.SkipDecoding()` when you want encoded image bytes instead of
  decoded pixel arrays.
- Increasing `batch_size` reduces Python overhead but raises peak memory during
  TensorFlow-to-Arrow conversion.
- `num_parallel_calls` and `prefetch` apply to TFRecord parsing. Higher values
  can improve throughput when parsing is CPU-bound, at the cost of more memory.

## Related Pages

- [Reader Model](reader-model.md)
- [Sharding](sharding.md)
- [Converting to Robot Rows](../episode-data/converting-to-robot-rows.md)
