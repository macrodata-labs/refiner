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
mdr.read_tfrecords(
    inputs,
    *,
    features,
    fs=None,
    storage_options=None,
    recursive=False,
    target_shard_bytes=128 * 1024 * 1024,
    num_shards=None,
    batch_size=1024,
    compression="auto",
    num_parallel_calls=None,
    prefetch=1,
    file_path_column="file_path",
)
```

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

### TFRecord Options

| Option | Default | Meaning |
| --- | --- | --- |
| `inputs` | required | TFRecord file, glob, directory, or sequence of inputs. |
| `features` | required | Mapping passed to `tf.io.parse_example`. |
| `fs` | `None` | Optional fsspec filesystem for string inputs. TensorFlow reads resolved local paths. |
| `storage_options` | `None` | Optional fsspec options when constructing a filesystem. |
| `recursive` | `False` | Recursively list directory inputs. |
| `target_shard_bytes` | `128 MiB` | Target file-shard planning size. TFRecord files remain atomic. |
| `num_shards` | `None` | Optional target number of planned file shards. |
| `batch_size` | `1024` | Serialized examples parsed per TensorFlow batch. |
| `compression` | `"auto"` | `None`, `"auto"`, `"gzip"`, or `"zlib"`. |
| `num_parallel_calls` | `None` | TensorFlow parse map parallelism. |
| `prefetch` | `1` | TensorFlow prefetch depth. Set to `None` to disable prefetching. |
| `file_path_column` | `"file_path"` | Source file column name. Set to `None` to omit it. Existing parsed feature names are not overwritten. |

## TensorFlow Datasets

Use `read_tfds(...)` for datasets available through TensorFlow Datasets, a
local TFDS `data_dir`, or a prepared TFDS directory:

```python
mdr.read_tfds(
    input,
    *,
    config=None,
    split="train",
    data_dir=None,
    download=False,
    batch_size=1024,
    examples_per_shard=10_000,
    num_shards=None,
    shuffle_files=False,
    read_config=None,
    decoders=None,
    as_supervised=False,
    videos=None,
    fps=30.0,
)
```

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
directory as the input:

```python
pipeline = mdr.read_tfds(
    "data/libero_10_no_noops/1.0.0",
    split="train",
    videos={"front": "steps/observation/image"},
    fps=30,
)
```

`read_tfds(...)` shards a plain split name by example ranges. Pass a split from
`builder.info.splits`, such as `"train"` or `"validation"`. Complex split
expressions are not sharded; use a prepared local TFDS split when you need a
custom subset.

TFDS decoding stays under TensorFlow Datasets. Pass `decoders`, `read_config`,
`shuffle_files`, or `as_supervised` when you need the same controls as
`builder.as_dataset(...)`.

For RLDS-style datasets, `videos` lifts image sequences from nested `steps`
datasets into lazy `VideoFrameSequence` values and removes those frame arrays
from the per-step table.

### TFDS Options

| Option | Default | Meaning |
| --- | --- | --- |
| `input` | required | TFDS dataset name or prepared TFDS directory. |
| `config` | `None` | Optional TFDS builder config. Not used with prepared TFDS directories. |
| `split` | `"train"` | Plain split name from `builder.info.splits`. |
| `data_dir` | `None` | Optional local TFDS data directory for catalog datasets. Not used with prepared TFDS directories. |
| `download` | `False` | Call `download_and_prepare()` for catalog datasets. Not used with prepared TFDS directories. |
| `batch_size` | `1024` | Decoded examples per emitted tabular batch when the dataset can be batched. |
| `examples_per_shard` | `10_000` | Target examples per planned shard when `num_shards` is omitted. |
| `num_shards` | `None` | Optional target number of planned row-range shards. |
| `shuffle_files` | `False` | Passed to `builder.as_dataset`. |
| `read_config` | `None` | Optional TFDS read config. |
| `decoders` | `None` | Optional TFDS feature decoders. |
| `as_supervised` | `False` | Read supervised `(input, target)` pairs. |
| `videos` | `None` | Video-name to nested dataset frame path mapping, such as `{"front": "steps/observation/image"}`. |
| `fps` | `30.0` | Frame rate used for `videos`. |

## Performance Trade-Offs

- TFRecord files are read through `tf.data.TFRecordDataset`, batched, parsed, and
  converted to Arrow-backed `Tabular` blocks.
- TFRecord shard planning is file-granular. Many medium files parallelize well;
  one very large file runs as one source shard.
- TFDS shard planning is example-range based within one split. Use
  `examples_per_shard` or `num_shards` to control parallelism.
- RLDS-style TFDS datasets with dataset-valued `steps` are streamed one episode
  at a time because TensorFlow cannot batch nested datasets.
- `videos` avoids keeping selected decoded image sequences in the row table, but
  writing those videos still decodes the underlying TFDS frames when the video is
  consumed.
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
