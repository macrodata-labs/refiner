from __future__ import annotations

import importlib
from collections.abc import Iterator, Mapping, Sequence
from functools import reduce
from operator import getitem
from pathlib import Path
import tempfile
from typing import Any

from refiner.io import DataFolder
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
        input: str,
        *,
        config: str | None = None,
        split: str = "train",
        data_dir: str | None = None,
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
            input: TFDS dataset name or prepared TFDS directory.
            config: Optional TFDS builder config.
            split: Plain split name from `builder.info.splits`, such as
                `"train"` or `"validation"`.
            data_dir: Optional local TFDS data directory.
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
        self.input = input
        self.config = config
        self.data_dir = data_dir
        self.download = download
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
        self._video_paths = tuple(
            (name, parts[0], parts[1:]) for name, parts in split_video_paths.items()
        )
        excluded_video_paths: dict[str, list[tuple[str, ...]]] = {}
        for _, dataset_key, frame_path in self._video_paths:
            excluded_video_paths.setdefault(dataset_key, []).append(frame_path)
        self._excluded_video_paths = {
            key: tuple(paths) for key, paths in excluded_video_paths.items()
        }
        self.fps = float(fps)
        self._builder: Any | None = None
        self._prepared_dir: str | None = None
        self._remote_folder: DataFolder | None = None
        self._remote_input = "://" in input
        self._downloaded_remote_files: set[str] = set()
        self.dataset_name: str | None = None
        self.num_examples: int | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_builder"] = None
        state["_prepared_dir"] = None
        state["_remote_folder"] = None
        state["_downloaded_remote_files"] = set()
        state["num_examples"] = None
        if self._remote_input:
            state["dataset_name"] = None
        return state

    def _ensure_builder(self) -> Any:
        if self._builder is not None:
            return self._builder

        input = self._ensure_prepared_dir()
        input_path = Path(input)
        is_prepared_dir = (
            input_path.is_dir()
            and (input_path / "dataset_info.json").exists()
            and (input_path / "features.json").exists()
        )
        if is_prepared_dir and any(
            value is not None for value in (self.config, self.data_dir)
        ):
            raise ValueError("config and data_dir cannot be used with a TFDS directory")
        if is_prepared_dir and self.download:
            raise ValueError("download cannot be used with a prepared TFDS directory")
        if is_prepared_dir:
            builder = self.tfds.builder_from_directory(input)
        else:
            builder = self.tfds.builder(
                input,
                config=self.config,
                data_dir=self.data_dir,
            )
        if self.download:
            builder.download_and_prepare()
        if self.split not in builder.info.splits:
            raise ValueError(
                "read_tfds currently shards plain split names only; pass a split "
                f"from builder.info.splits, got {self.split!r}"
            )
        self._builder = builder
        self.dataset_name = builder.info.name
        self.num_examples = int(builder.info.splits[self.split].num_examples)
        return builder

    def _ensure_prepared_dir(self) -> str:
        if not self._remote_input:
            return self.input
        if self._prepared_dir is None:
            local_dir = Path(tempfile.mkdtemp(prefix="refiner-tfds-"))
            folder = self._ensure_remote_folder()
            for name in ("dataset_info.json", "features.json"):
                folder.file(name).copy(str(local_dir / name))
            self._prepared_dir = str(local_dir)
        return self._prepared_dir

    def _ensure_remote_folder(self) -> DataFolder:
        if self._remote_folder is None:
            self._remote_folder = DataFolder.resolve(self.input)
        return self._remote_folder

    def _materialize_remote_shards(self, split: str) -> None:
        if not self._remote_input:
            return
        builder = self._ensure_builder()
        tfds_splits = importlib.import_module("tensorflow_datasets.core.splits")
        instructions = tfds_splits._make_file_instructions(  # noqa: SLF001
            list(builder.info.splits.values()),
            split,
        )
        for instruction in instructions:
            filename = Path(instruction.filename).name
            if filename in self._downloaded_remote_files:
                continue
            assert self._prepared_dir is not None
            local_file = Path(self._prepared_dir) / filename
            if not local_file.exists():
                self._ensure_remote_folder().file(filename).copy(str(local_file))
            self._downloaded_remote_files.add(filename)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "input": self.input,
            "dataset": self.dataset_name,
            "config": self.config,
            "split": self.split,
            "batch_size": self.batch_size,
            "examples_per_shard": self.examples_per_shard,
            "num_shards": self.num_shards,
            "videos": {
                name: "/".join((dataset_key, *frame_path))
                for name, dataset_key, frame_path in self._video_paths
            },
            "fps": self.fps,
        }

    def list_shards(self) -> list[Shard]:
        """Plan deterministic row ranges for the configured split."""
        self._ensure_builder()
        assert self.num_examples is not None
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
        builder = self._ensure_builder()
        self._materialize_remote_shards(split)
        dataset = builder.as_dataset(
            **dataset_kwargs,
            batch_size=None,
        )
        has_nested_datasets = any(
            isinstance(spec, self.tf.data.DatasetSpec)
            for spec in self.tf.nest.flatten(dataset.element_spec)
        )
        if not has_nested_datasets:
            dataset = dataset.ragged_batch(self.batch_size)
        for batch in dataset.prefetch(1):
            if self.as_supervised:
                batch = {"input": batch[0], "target": batch[1]}
            if has_nested_datasets:
                row: dict[str, Any] = {}
                for name, value in batch.items():
                    if isinstance(value, self.tf.data.Dataset):
                        paths = self._excluded_video_paths.get(name, ())
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

                if self._video_paths:
                    row["videos"] = {
                        name: VideoFrameSequence(
                            lambda dataset=batch[dataset_key], frame_path=frame_path: (
                                tensorflow_value_to_python(frame)
                                for frame in dataset.map(
                                    lambda step, frame_path=frame_path: reduce(
                                        getitem, frame_path, step
                                    )
                                ).as_numpy_iterator()
                            ),
                            fps=self.fps,
                            frame_count=len(row[dataset_key]),
                        )
                        for name, dataset_key, frame_path in self._video_paths
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


__all__ = ["TfdsReader"]
