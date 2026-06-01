from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from refiner.pipeline import read_tfds
from refiner.pipeline.data.shard import RowRangeDescriptor
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.sources.readers import TfdsReader
from refiner.robotics import RoboticsRow
from refiner.video import VideoFrameSequence

tfds = pytest.importorskip("tensorflow_datasets")
pytest.importorskip("tensorflow")


class TinyDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict(
                {
                    "id": tfds.features.Scalar(dtype=np.int64),
                    "text": tfds.features.Text(),
                    "nested": {
                        "value": tfds.features.Scalar(dtype=np.int64),
                    },
                    "values": tfds.features.Sequence(
                        tfds.features.Scalar(dtype=np.int64)
                    ),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        return {"train": self._generate_examples()}

    def _generate_examples(self):
        for idx in range(5):
            yield (
                str(idx),
                {
                    "id": idx,
                    "text": f"row-{idx}",
                    "nested": {"value": idx + 10},
                    "values": list(range(idx + 1)),
                },
            )


class TinyRldsDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "action": tfds.features.Tensor(
                                shape=(2,),
                                dtype=np.float32,
                            ),
                            "observation": {
                                "image": tfds.features.Tensor(
                                    shape=(2, 2, 3),
                                    dtype=np.uint8,
                                ),
                                "state": tfds.features.Tensor(
                                    shape=(2,),
                                    dtype=np.float32,
                                ),
                            },
                            "is_first": tfds.features.Scalar(dtype=np.bool_),
                        }
                    ),
                    "episode_metadata": {
                        "file_path": tfds.features.Text(),
                    },
                    "episode_vector": tfds.features.Tensor(
                        shape=(3,),
                        dtype=np.float32,
                    ),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        return {"train": self._generate_examples()}

    def _generate_examples(self):
        for idx in range(2):
            yield (
                str(idx),
                {
                    "steps": [
                        {
                            "action": [float(idx), 0.0],
                            "observation": {
                                "image": np.full((2, 2, 3), idx, dtype=np.uint8),
                                "state": [float(idx), 0.0],
                            },
                            "is_first": True,
                        },
                        {
                            "action": [float(idx), 1.0],
                            "observation": {
                                "image": np.full((2, 2, 3), idx + 1, dtype=np.uint8),
                                "state": [float(idx), 1.0],
                            },
                            "is_first": False,
                        },
                    ],
                    "episode_metadata": {"file_path": f"episode-{idx}"},
                    "episode_vector": [float(idx), float(idx + 1), float(idx + 2)],
                },
            )


def _rows_from_reader(reader: TfdsReader) -> list[dict]:
    rows = []
    for shard in reader.list_shards():
        for unit in reader.read_shard(shard):
            assert isinstance(unit, Tabular)
            rows.extend(row.to_dict() for row in unit)
    return rows


def test_tfds_reader_shards_by_example_range(tmp_path: Path, monkeypatch) -> None:
    builder = TinyDataset(data_dir=str(tmp_path))
    builder.download_and_prepare()
    monkeypatch.setattr(tfds, "builder", lambda *args, **kwargs: builder)

    reader = TfdsReader("tiny_dataset", examples_per_shard=2, batch_size=2)

    shards = reader.list_shards()
    rows = _rows_from_reader(reader)
    ranges = []
    for shard in shards:
        assert isinstance(shard.descriptor, RowRangeDescriptor)
        ranges.append((shard.descriptor.start, shard.descriptor.end))
    assert ranges == [
        (0, 2),
        (2, 4),
        (4, 5),
    ]
    assert sorted(row["id"] for row in rows) == [0, 1, 2, 3, 4]
    assert {row["id"]: row["nested"]["value"] for row in rows} == {
        0: 10,
        1: 11,
        2: 12,
        3: 13,
        4: 14,
    }


def test_read_tfds_pipeline_entrypoint(tmp_path: Path, monkeypatch) -> None:
    builder = TinyDataset(data_dir=str(tmp_path))
    builder.download_and_prepare()
    monkeypatch.setattr(tfds, "builder", lambda *args, **kwargs: builder)

    pipeline = read_tfds("tiny_dataset", num_shards=2, batch_size=3)

    assert sorted(row["id"] for row in pipeline.take(5)) == [0, 1, 2, 3, 4]


def test_read_tfds_accepts_prepared_tfds_directory(tmp_path: Path) -> None:
    builder = TinyDataset(data_dir=str(tmp_path))
    builder.download_and_prepare()

    pipeline = read_tfds(builder.data_dir, num_shards=2, batch_size=3)

    assert sorted(row["id"] for row in pipeline.take(5)) == [0, 1, 2, 3, 4]


def test_read_tfds_streams_dataset_valued_features(tmp_path: Path) -> None:
    builder = TinyRldsDataset(data_dir=str(tmp_path))
    builder.download_and_prepare()

    rows = read_tfds(builder.data_dir, batch_size=2).take(2)

    assert [row["episode_metadata"]["file_path"] for row in rows] == [
        b"episode-0",
        b"episode-1",
    ]
    assert len(rows[0]["steps"]) == 2
    assert rows[0]["steps"][1]["action"] == pytest.approx([0.0, 1.0])
    assert rows[0]["episode_vector"] == pytest.approx([0.0, 1.0, 2.0])


def test_read_tfds_preserves_variable_length_sequences(tmp_path: Path) -> None:
    builder = TinyDataset(data_dir=str(tmp_path))
    builder.download_and_prepare()

    rows = read_tfds(builder.data_dir, batch_size=3).take(5)

    assert {row["id"]: row["values"] for row in rows} == {
        0: [0],
        1: [0, 1],
        2: [0, 1, 2],
        3: [0, 1, 2, 3],
        4: [0, 1, 2, 3, 4],
    }


def test_read_tfds_lifts_rlds_images_into_video_sequences(tmp_path: Path) -> None:
    builder = TinyRldsDataset(data_dir=str(tmp_path))
    builder.download_and_prepare()

    row = read_tfds(
        builder.data_dir,
        videos={"front": "steps/observation/image"},
        fps=5,
    ).take(1)[0]

    video = row["videos"]["front"]
    assert isinstance(video, VideoFrameSequence)
    assert video.frame_count == 2
    assert video.fps == 5
    assert "image" not in row["steps"][0]["observation"]
    assert row["steps"][1]["action"] == pytest.approx([0.0, 1.0])
    assert [int(frame[0, 0, 0]) for frame in video.iter_frame_arrays()] == [0, 1]


def test_read_tfds_video_sequences_work_with_robot_rows(tmp_path: Path) -> None:
    builder = TinyRldsDataset(data_dir=str(tmp_path))
    builder.download_and_prepare()

    robot_row = cast(
        RoboticsRow,
        (
            read_tfds(
                builder.data_dir,
                videos={"front": "steps/observation/image"},
                fps=5,
            )
            .to_robot_rows(
                nested_frames_key="steps",
                state_key="observation/state",
                video_keys={"observation.images.front": "videos/front"},
            )
            .take(1)[0]
        ),
    )

    video = robot_row.videos["observation.images.front"]
    assert isinstance(video, VideoFrameSequence)
    assert video.frame_count == 2
    assert robot_row.actions[1].as_py() == pytest.approx([0.0, 1.0])
    assert [int(frame[0, 0, 0]) for frame in video.iter_frame_arrays()] == [0, 1]


def test_tfds_reader_requires_plain_split_name(tmp_path: Path, monkeypatch) -> None:
    builder = TinyDataset(data_dir=str(tmp_path))
    builder.download_and_prepare()
    monkeypatch.setattr(tfds, "builder", lambda *args, **kwargs: builder)

    with pytest.raises(ValueError, match="plain split names"):
        TfdsReader("tiny_dataset", split="train[:2]")


def test_tfds_reader_rejects_non_positive_num_shards(
    tmp_path: Path, monkeypatch
) -> None:
    builder = TinyDataset(data_dir=str(tmp_path))
    builder.download_and_prepare()
    monkeypatch.setattr(tfds, "builder", lambda *args, **kwargs: builder)

    with pytest.raises(ValueError, match="num_shards"):
        TfdsReader("tiny_dataset", num_shards=0)
