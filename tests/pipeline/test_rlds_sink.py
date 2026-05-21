from __future__ import annotations

from pathlib import Path

import pytest

import refiner as mdr
from refiner.pipeline.sinks.rlds import RldsSink


class _List:
    def __init__(self, value):
        self.value = list(value)


class _Feature:
    def __init__(self, bytes_list=None, float_list=None, int64_list=None):
        self.bytes_list = bytes_list
        self.float_list = float_list
        self.int64_list = int64_list


class _Features:
    def __init__(self, feature):
        self.feature = feature


class _Example:
    def __init__(self, features):
        self.features = features

    def SerializeToString(self):
        return b"example"


class _FakeTrain:
    BytesList = _List
    FloatList = _List
    Int64List = _List
    Feature = _Feature
    Features = _Features
    Example = _Example


class _FakeTf:
    train = _FakeTrain


def _robotics_pipeline():
    return mdr.from_items(
        [
            {
                "episode_id": "episode-0",
                "task": "push block",
                "frames": [
                    {
                        "timestamp": 0.0,
                        "action": [0.0, 0.1],
                        "observation.state": [1.0, 1.1],
                    },
                    {
                        "timestamp": 0.1,
                        "action": [0.2, 0.3],
                        "observation.state": [1.2, 1.3],
                    },
                ],
            }
        ]
    ).to_robot_rows(
        episode_id_key="episode_id",
        task_key="task",
        nested_frames_key="frames",
        fps=10,
        robot_type="testbot",
    )


def test_rlds_sink_maps_robotics_row_to_episode_features(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _robotics_pipeline().take(1)[0]
    sink = RldsSink(str(tmp_path))
    monkeypatch.setattr(
        "refiner.pipeline.sinks.rlds.require_tensorflow", lambda _: _FakeTf
    )

    example = sink._example(row)
    features = example.features.feature

    assert features["episode_id"].bytes_list.value == [b"episode-0"]
    assert features["task"].bytes_list.value == [b"push block"]
    assert features["language_instruction"].bytes_list.value == [b"push block"]
    assert features["robot_type"].bytes_list.value == [b"testbot"]
    assert features["fps"].float_list.value == [10.0]
    assert features["length"].int64_list.value == [2]
    assert features["steps/is_first"].int64_list.value == [1, 0]
    assert features["steps/is_last"].int64_list.value == [0, 1]
    assert features["steps/action/shape"].int64_list.value == [2, 2]
    assert features["steps/action"].float_list.value == pytest.approx(
        [0.0, 0.1, 0.2, 0.3]
    )


def _parse_example(path: Path) -> dict:
    tf = pytest.importorskip("tensorflow")
    dataset = tf.data.TFRecordDataset([str(path)])
    features = {
        "episode_id": tf.io.FixedLenFeature([], tf.string),
        "task": tf.io.FixedLenFeature([], tf.string),
        "language_instruction": tf.io.FixedLenFeature([], tf.string),
        "robot_type": tf.io.FixedLenFeature([], tf.string),
        "fps": tf.io.FixedLenFeature([], tf.float32),
        "length": tf.io.FixedLenFeature([], tf.int64),
        "steps/is_first": tf.io.VarLenFeature(tf.int64),
        "steps/is_last": tf.io.VarLenFeature(tf.int64),
        "steps/is_terminal": tf.io.VarLenFeature(tf.int64),
        "steps/action": tf.io.VarLenFeature(tf.float32),
        "steps/action/shape": tf.io.VarLenFeature(tf.int64),
        "steps/observation/state": tf.io.VarLenFeature(tf.float32),
        "steps/observation/state/shape": tf.io.VarLenFeature(tf.int64),
        "steps/timestamp": tf.io.VarLenFeature(tf.float32),
        "steps/timestamp/shape": tf.io.VarLenFeature(tf.int64),
    }
    record = next(iter(dataset))
    parsed = tf.io.parse_single_example(record, features)
    return {
        key: tf.sparse.to_dense(value).numpy().tolist()
        if isinstance(value, tf.SparseTensor)
        else value.numpy()
        for key, value in parsed.items()
    }


def test_write_rlds_writes_robotics_episode_tfrecord(tmp_path: Path) -> None:
    pytest.importorskip("tensorflow")
    out = tmp_path / "rlds"

    _robotics_pipeline().write_rlds(str(out)).launch_local(
        name="write-rlds", num_workers=1, rundir=str(tmp_path / "run")
    )

    [path] = list(out.glob("*.tfrecord"))
    row = _parse_example(path)

    assert row["episode_id"] == b"episode-0"
    assert row["task"] == b"push block"
    assert row["language_instruction"] == b"push block"
    assert row["robot_type"] == b"testbot"
    assert row["fps"] == pytest.approx(10.0)
    assert row["length"] == 2
    assert row["steps/is_first"] == [1, 0]
    assert row["steps/is_last"] == [0, 1]
    assert row["steps/is_terminal"] == [0, 1]
    assert row["steps/action/shape"] == [2, 2]
    assert row["steps/action"] == pytest.approx([0.0, 0.1, 0.2, 0.3])
    assert row["steps/observation/state/shape"] == [2, 2]
    assert row["steps/observation/state"] == pytest.approx([1.0, 1.1, 1.2, 1.3])
    assert row["steps/timestamp/shape"] == [2]
    assert row["steps/timestamp"] == pytest.approx([0.0, 0.1])
