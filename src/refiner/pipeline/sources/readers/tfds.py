from __future__ import annotations

import importlib
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from refiner.pipeline.data.row import DictRow
from refiner.pipeline.data.shard import RowRangeDescriptor, Shard
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.sources.base import BaseSource, SourceUnit
from refiner.pipeline.sources.readers.utils import (
    PathSelection,
    path_selection_map,
    tensorflow_batch_to_table,
    tensorflow_value_to_python,
)
from refiner.utils import check_required_dependencies
from refiner.video import VideoFrameSequence

_DEFAULT_EXAMPLES_PER_SHARD = 10_000
_VideoPath = tuple[str, str, tuple[str, ...]]


class TfdsReader(BaseSource):
    """TensorFlow Datasets reader that preserves TFDS feature decoding.

    The reader plans row-range shards inside one plain TFDS split and emits
    Arrow-backed `Tabular` blocks converted from TensorFlow batches.
    """

    name = "read_tfds"

    def __init__(
        self,
        name: str | None = None,
        *,
        config: str | None = None,
        split: str = "train",
        data_dir: str | None = None,
        builder_dir: str | None = None,
        download: bool = False,
        batch_size: int = 1024,
        examples_per_shard: int = _DEFAULT_EXAMPLES_PER_SHARD,
        num_shards: int | None = None,
        shuffle_files: bool = False,
        read_config: Any | None = None,
        decoders: Mapping[str, Any] | None = None,
        as_supervised: bool = False,
        videos: PathSelection | None = None,
        fps: float = 30.0,
    ):
        """Create a TensorFlow Datasets reader.

        Args:
            name: TFDS dataset name. Required unless `builder_dir` is set.
            config: Optional TFDS builder config.
            split: Plain split name from `builder.info.splits`, such as
                `"train"` or `"validation"`.
            data_dir: Optional local TFDS data directory.
            builder_dir: Optional path to a prepared TFDS builder directory,
                such as a Hugging Face RLDS dataset version directory.
            download: Whether to call `download_and_prepare()`.
            batch_size: Number of examples converted per emitted batch.
            examples_per_shard: Target number of examples per planned shard
                when `num_shards` is omitted.
            num_shards: Optional explicit number of row-range shards.
            shuffle_files: Passed to `builder.as_dataset`.
            read_config: Optional TFDS read config.
            decoders: Optional TFDS feature decoders.
            as_supervised: Whether to read supervised `(input, target)` pairs.
            videos: Optional video-name to nested dataset frame path mapping,
                such as `{"front": "steps/observation/image"}`.
            fps: Frame rate used for `videos`.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if examples_per_shard <= 0:
            raise ValueError("examples_per_shard must be > 0")
        if num_shards is not None and num_shards <= 0:
            raise ValueError("num_shards must be > 0 when provided")
        check_required_dependencies(
            "read_tfds",
            [
                ("tensorflow", "tensorflow"),
                ("tensorflow_datasets", "tensorflow-datasets"),
            ],
            dist="tfds",
        )
        self.tf = importlib.import_module("tensorflow")
        self.tfds = importlib.import_module("tensorflow_datasets")
        if builder_dir is None and name is None:
            raise ValueError("name is required unless builder_dir is provided")
        if builder_dir is not None and any(
            value is not None for value in (config, data_dir)
        ):
            raise ValueError("config and data_dir cannot be used with builder_dir")
        if builder_dir is not None and download:
            raise ValueError("download cannot be used with builder_dir")
        self.config = config
        self.builder_dir = builder_dir
        self.split = split
        self.batch_size = int(batch_size)
        self.examples_per_shard = int(examples_per_shard)
        self.num_shards = num_shards
        self.shuffle_files = shuffle_files
        self.read_config = read_config
        self.decoders = dict(decoders) if decoders else None
        self.as_supervised = as_supervised
        video_map = path_selection_map(videos, format_name="TFDS")
        split_video_paths = {
            name: tuple(path.split("/")) for name, path in video_map.items()
        }
        if any(len(parts) < 2 for parts in split_video_paths.values()):
            raise ValueError("TFDS video paths must include a dataset and frame field")
        self.video_paths = tuple(
            (name, parts[0], parts[1:]) for name, parts in split_video_paths.items()
        )
        excluded_video_paths: dict[str, list[tuple[str, ...]]] = {}
        for _, dataset_key, frame_path in self.video_paths:
            excluded_video_paths.setdefault(dataset_key, []).append(frame_path)
        self.excluded_video_paths = {
            key: tuple(paths) for key, paths in excluded_video_paths.items()
        }
        self.fps = float(fps)
        if builder_dir is not None:
            self.builder = self.tfds.builder_from_directory(builder_dir)
        else:
            assert name is not None
            self.builder = self.tfds.builder(name, config=config, data_dir=data_dir)
        if download:
            self.builder.download_and_prepare()
        self.dataset_name = name or self.builder.info.name
        if split not in self.builder.info.splits:
            raise ValueError(
                "read_tfds currently shards plain split names only; pass a split "
                f"from builder.info.splits, got {split!r}"
            )
        self.num_examples = int(self.builder.info.splits[split].num_examples)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dataset": self.dataset_name,
            "config": self.config,
            "builder_dir": self.builder_dir,
            "split": self.split,
            "batch_size": self.batch_size,
            "examples_per_shard": self.examples_per_shard,
            "num_shards": self.num_shards,
            "videos": {
                name: "/".join((dataset_key, *frame_path))
                for name, dataset_key, frame_path in self.video_paths
            },
            "fps": self.fps,
        }

    def list_shards(self) -> list[Shard]:
        """Plan deterministic row ranges for the configured split."""
        shards: list[Shard] = []
        if self.num_shards is None:
            for start in range(0, self.num_examples, self.examples_per_shard):
                shards.append(
                    Shard.from_row_range(
                        start=start,
                        end=min(start + self.examples_per_shard, self.num_examples),
                        global_ordinal=len(shards),
                    )
                )
            return shards
        for ordinal in range(self.num_shards):
            start = ordinal * self.num_examples // self.num_shards
            end = (ordinal + 1) * self.num_examples // self.num_shards
            if start != end:
                shards.append(
                    Shard.from_row_range(
                        start=start,
                        end=end,
                        global_ordinal=len(shards),
                    )
                )
        return shards

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        """Read a planned row range from the TFDS split."""
        descriptor = shard.descriptor
        if not isinstance(descriptor, RowRangeDescriptor):
            raise TypeError("TfdsReader requires row-range shards")
        split = f"{self.split}[{descriptor.start}:{descriptor.end}]"
        dataset_kwargs = {
            "split": split,
            "shuffle_files": self.shuffle_files,
            "read_config": self.read_config,
            "decoders": self.decoders,
            "as_supervised": self.as_supervised,
        }
        dataset = self.builder.as_dataset(
            **dataset_kwargs,
            batch_size=None,
        )
        if not any(
            isinstance(spec, self.tf.data.DatasetSpec)
            for spec in self.tf.nest.flatten(dataset.element_spec)
        ):
            dataset = dataset.padded_batch(
                self.batch_size,
                self.tf.compat.v1.data.get_output_shapes(dataset),
            )
        for batch in dataset.prefetch(1):
            if self.as_supervised:
                batch = {"input": batch[0], "target": batch[1]}
            if self.video_paths:
                row: dict[str, Any] = {}
                for name, value in batch.items():
                    if isinstance(value, self.tf.data.Dataset):
                        paths = self.excluded_video_paths.get(name, ())
                        if paths:
                            value = value.map(
                                lambda step, paths=paths: _drop_tf_paths(step, paths)
                            )
                        row[name] = [
                            tensorflow_value_to_python(step)
                            for step in value.as_numpy_iterator()
                        ]
                    else:
                        row[name] = tensorflow_value_to_python(value)

                row["videos"] = {
                    name: VideoFrameSequence(
                        lambda dataset=batch[dataset_key], frame_path=frame_path: (
                            tensorflow_value_to_python(frame)
                            for frame in dataset.map(
                                lambda step, frame_path=frame_path: _get_tf_path(
                                    step, frame_path
                                )
                            ).as_numpy_iterator()
                        ),
                        fps=self.fps,
                        frame_count=len(row[dataset_key]),
                    )
                    for name, dataset_key, frame_path in self.video_paths
                }
                yield DictRow(row)
                continue
            table = tensorflow_batch_to_table(batch)
            if table.num_rows > 0:
                yield Tabular(table)


def _drop_tf_paths(value: Any, paths: Sequence[tuple[str, ...]]) -> Any:
    if not paths or not isinstance(value, Mapping):
        return value
    out = {}
    for name, child in value.items():
        child_paths = tuple(path[1:] for path in paths if path and path[0] == name)
        if () in child_paths:
            continue
        out[name] = _drop_tf_paths(child, child_paths)
    return out


def _get_tf_path(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = value
    for part in path:
        current = current[part]
    return current


__all__ = ["TfdsReader"]
