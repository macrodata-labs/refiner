from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from fsspec import AbstractFileSystem
import numpy as np
import pyarrow as pa

from refiner.io.fileset import DataFileSetLike
from refiner.pipeline.data.row import DictRow
from refiner.pipeline.data.tabular import Tabular
from refiner.pipeline.data.shard import FilePartsDescriptor
from refiner.pipeline.sources.readers.base import BaseReader, Shard, SourceUnit
from refiner.pipeline.sources.readers.utils import (
    DEFAULT_TARGET_SHARD_BYTES,
    PathSelection,
    path_selection_map,
)
from refiner.utils import check_required_dependencies
from refiner.video import VideoFrameArray


@dataclass(frozen=True, slots=True)
class _McapEvent:
    timestamp_ns: int
    value: Any


@dataclass(frozen=True, slots=True)
class _EpisodeWindow:
    start_ns: int | None = None
    end_ns: int | None = None


class McapReader(BaseReader):
    """Read MCAP logs as robotics-ready episode rows."""

    name = "read_mcap"

    def __init__(
        self,
        inputs: DataFileSetLike,
        *,
        fs: AbstractFileSystem | None = None,
        storage_options: Mapping[str, Any] | None = None,
        recursive: bool = False,
        target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
        num_shards: int | None = None,
        topics: Sequence[str] | None = None,
        file_path_column: str | None = "file_path",
        frames_column: str = "frames",
        videos_column: str = "videos",
        fields: PathSelection | None = None,
        videos: PathSelection | None = None,
        primary: str | None = None,
        fps: float | None = None,
        fps_column: str | None = "fps",
        include_skew: bool = True,
        episode_splitting: str | Mapping[str, Any] = "single",
    ):
        time_gap_s, marker_topic = _parse_episode_splitting(episode_splitting)
        super().__init__(
            inputs,
            fs=fs,
            storage_options=storage_options,
            recursive=recursive,
            extensions=(".mcap",),
            target_shard_bytes=target_shard_bytes,
            num_shards=num_shards,
            file_path_column=file_path_column,
            split_by_bytes=False,
        )
        self.topics = tuple(topics) if topics is not None else None
        self.frames_column = frames_column
        self.videos_column = videos_column
        self.fields = path_selection_map(
            fields, format_name="MCAP", derive_names_from_paths=False
        )
        self.videos = path_selection_map(
            videos, format_name="MCAP", derive_names_from_paths=False
        )
        self.primary = primary
        self.fps = fps
        self.fps_column = fps_column
        self.include_skew = include_skew
        self.episode_splitting = episode_splitting
        self._time_gap_s = time_gap_s
        self._marker_topic = marker_topic

    def describe(self) -> dict[str, Any]:
        description = super().describe()
        description.update(
            {
                "topics": list(self.topics) if self.topics is not None else None,
                "frames_column": self.frames_column,
                "videos_column": self.videos_column,
                "fields": self.fields,
                "videos": self.videos,
                "primary": self.primary,
                "fps": self.fps,
                "fps_column": self.fps_column,
                "include_skew": self.include_skew,
                "episode_splitting": self.episode_splitting,
            }
        )
        return description

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        descriptor = shard.descriptor
        assert isinstance(descriptor, FilePartsDescriptor)
        check_required_dependencies("read_mcap", ["mcap"], dist="mcap")
        from mcap.reader import make_reader

        decoder_factories = _decoder_factories()
        for part in descriptor.parts:
            source = self.fileset.resolve_file(part.source_index, part.path)
            topic_events: dict[str, list[_McapEvent]] = defaultdict(list)
            with source.open(mode="rb") as stream:
                reader = make_reader(stream)
                summary = reader.get_summary()
                summary_topics = (
                    {channel.topic for channel in summary.channels.values()}
                    if summary is not None
                    else set()
                )
                read_topics = self.topics
                if read_topics is None and summary_topics:
                    selected = [*self.fields.values(), *self.videos.values()]
                    if (
                        self.primary is not None
                        and self.primary not in self.fields
                        and self.primary not in self.videos
                    ):
                        selected.append(self.primary)
                    if selected:
                        if self._marker_topic is not None:
                            selected.append(self._marker_topic)
                        read_topics = tuple(
                            sorted(
                                {
                                    _resolve_source(source, summary_topics)[0]
                                    for source in selected
                                }
                            )
                        )
                for schema, channel, message in reader.iter_messages(
                    topics=read_topics
                ):
                    decoded = _decode_message(
                        schema,
                        channel.message_encoding,
                        message.data,
                        decoder_factories,
                    )
                    topic_events[channel.topic].append(
                        _McapEvent(timestamp_ns=message.log_time, value=decoded)
                    )
            if self._time_gap_s is not None:
                windows = _time_gap_windows(topic_events, self._time_gap_s)
            elif self._marker_topic is not None:
                windows = _marker_windows(topic_events.get(self._marker_topic, ()))
            else:
                windows = [_EpisodeWindow()]
            for episode_index, window in enumerate(windows):
                window_events = _slice_events(topic_events, window)
                fields = self.fields or _default_fields(
                    window_events,
                    self.videos,
                    excluded_topic=self._marker_topic,
                )
                topics = set(window_events)
                resolved_fields = {
                    output: _resolve_source(source, topics)
                    for output, source in fields.items()
                }
                resolved_videos = {
                    output: _resolve_source(source, topics)
                    for output, source in self.videos.items()
                }
                primary = None
                if self.primary is not None:
                    primary = resolved_fields.get(self.primary) or resolved_videos.get(
                        self.primary
                    )
                    if primary is None:
                        primary = _resolve_source(self.primary, topics)
                frame_table = (
                    _sparse_frame_table(resolved_fields, window_events)
                    if primary is None
                    else _aligned_frame_table(
                        resolved_fields,
                        window_events,
                        primary=primary,
                        include_skew=self.include_skew,
                    )
                )
                primary_events = (
                    window_events.get(primary[0], ()) if primary is not None else None
                )
                inferred_fps = self.fps or _infer_fps(primary_events)
                videos = _video_map(
                    resolved_videos,
                    window_events,
                    primary_events=primary_events,
                    fps=int(round(inferred_fps or 30)),
                )
                row: dict[str, Any] = {
                    self.frames_column: Tabular(frame_table),
                    "episode_index": episode_index,
                    "message_count": sum(
                        len(events) for events in window_events.values()
                    ),
                    "topics": sorted(window_events),
                }
                if videos:
                    row[self.videos_column] = videos
                if self.fps_column is not None and inferred_fps is not None:
                    row[self.fps_column] = float(inferred_fps)
                yield DictRow(self._with_file_path(row, source))


def _parse_episode_splitting(
    splitting: str | Mapping[str, Any],
) -> tuple[float | None, str | None]:
    if splitting == "single":
        return None, None
    if isinstance(splitting, str) or not isinstance(splitting, Mapping):
        raise ValueError(
            "episode_splitting must be 'single', {'time_gap_s': seconds}, "
            "or {'marker_topic': topic}"
        )
    keys = set(splitting)
    if keys == {"time_gap_s"}:
        time_gap_s = float(splitting["time_gap_s"])
        if time_gap_s <= 0:
            raise ValueError("episode_splitting time_gap_s must be > 0")
        return time_gap_s, None
    if keys != {"marker_topic"}:
        raise ValueError(
            "episode_splitting must be 'single', {'time_gap_s': seconds}, "
            "or {'marker_topic': topic}"
        )
    if not isinstance(splitting["marker_topic"], str):
        raise TypeError("episode_splitting marker_topic must be a string")
    return None, splitting["marker_topic"]


def _time_gap_windows(
    topic_events: Mapping[str, Sequence[_McapEvent]],
    time_gap_s: float,
) -> list[_EpisodeWindow]:
    timestamps = sorted(
        event.timestamp_ns for events in topic_events.values() for event in events
    )
    if not timestamps:
        return [_EpisodeWindow()]
    gap_ns = int(time_gap_s * 1e9)
    windows: list[_EpisodeWindow] = []
    start = timestamps[0]
    for left, right in zip(timestamps, timestamps[1:], strict=False):
        if right - left > gap_ns:
            windows.append(_EpisodeWindow(start_ns=start, end_ns=right))
            start = right
    windows.append(_EpisodeWindow(start_ns=start, end_ns=timestamps[-1] + 1))
    return windows


def _marker_windows(marker_events: Sequence[_McapEvent]) -> list[_EpisodeWindow]:
    if not marker_events:
        return [_EpisodeWindow()]
    starts = [event.timestamp_ns for event in marker_events]
    ends: list[int | None] = [*starts[1:], None]
    return [
        _EpisodeWindow(start_ns=start, end_ns=end)
        for start, end in zip(starts, ends, strict=False)
    ]


def _slice_events(
    topic_events: Mapping[str, Sequence[_McapEvent]],
    window: _EpisodeWindow,
) -> dict[str, list[_McapEvent]]:
    out: dict[str, list[_McapEvent]] = {}
    for topic, events in topic_events.items():
        selected = [
            event
            for event in events
            if (window.start_ns is None or event.timestamp_ns >= window.start_ns)
            and (window.end_ns is None or event.timestamp_ns < window.end_ns)
        ]
        if selected:
            out[topic] = selected
    return out


def _decoder_factories() -> list[Any]:
    factories: list[Any] = []
    try:
        from mcap_ros2.decoder import DecoderFactory as Ros2DecoderFactory

        factories.append(Ros2DecoderFactory())
    except ImportError:
        pass
    try:
        from mcap_protobuf.decoder import DecoderFactory as ProtobufDecoderFactory

        factories.append(ProtobufDecoderFactory())
    except ImportError:
        pass
    return factories


def _decode_message(
    schema: Any,
    message_encoding: str,
    data: bytes,
    decoder_factories: Sequence[Any],
) -> Any:
    encoding = message_encoding.lower()
    if encoding in {"json", "application/json"}:
        return json.loads(data.decode("utf-8"))
    for factory in decoder_factories:
        decoder = factory.decoder_for(message_encoding, schema)
        if decoder is not None:
            return _plain_value(decoder(data))
    return data


def _plain_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain_value(item) for item in value]
    if hasattr(value, "DESCRIPTOR"):
        try:
            from google.protobuf.json_format import MessageToDict

            return MessageToDict(value, preserving_proto_field_name=True)
        except Exception:
            return value
    if type(value).__module__.startswith("mcap_ros2."):
        return {
            key: _plain_value(getattr(value, key))
            for key in dir(value)
            if not key.startswith("_") and not callable(getattr(value, key))
        }
    if hasattr(value, "__dict__"):
        return {
            key: _plain_value(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return value


def _default_fields(
    topic_events: Mapping[str, Sequence[_McapEvent]],
    videos: Mapping[str, str],
    *,
    excluded_topic: str | None,
) -> dict[str, str]:
    video_sources = set(videos.values())
    fields: dict[str, str] = {}
    for topic, events in topic_events.items():
        if (
            topic == excluded_topic
            or not events
            or any(
                source == topic or source.startswith(f"{topic}.")
                for source in video_sources
            )
        ):
            continue
        names = _flatten_names(events[0].value)
        if names:
            fields.update({f"{topic}.{name}": f"{topic}.{name}" for name in names})
        else:
            fields[topic] = topic
    return fields


def _flatten_names(value: Any, *, prefix: str = "") -> list[str]:
    if not isinstance(value, Mapping):
        return []
    names: list[str] = []
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            nested = _flatten_names(item, prefix=name)
            names.extend(nested or [name])
        else:
            names.append(name)
    return names


def _resolve_source(source: str, topics: set[str]) -> tuple[str, str | None]:
    if source in topics:
        return source, None
    for topic in sorted(topics, key=len, reverse=True):
        prefix = f"{topic}."
        if source.startswith(prefix):
            return str(topic), source[len(prefix) :]
    if topics:
        raise KeyError(f"MCAP source not found: {source}")
    return source, None


def _sparse_frame_table(
    fields: Mapping[str, tuple[str, str | None]],
    topic_events: Mapping[str, Sequence[_McapEvent]],
) -> pa.Table:
    values_by_field = {
        name: {
            event.timestamp_ns: _source_value(event.value, source[1])
            for event in topic_events.get(source[0], ())
        }
        for name, source in fields.items()
    }
    timestamps = sorted(
        {
            event.timestamp_ns
            for source in fields.values()
            for event in topic_events.get(source[0], ())
        }
    )
    rows: list[dict[str, Any]] = []
    for index, timestamp_ns in enumerate(timestamps):
        row: dict[str, Any] = {
            "frame_index": index,
            "timestamp": timestamp_ns / 1e9,
        }
        for name, values in values_by_field.items():
            if timestamp_ns in values:
                row[name] = values[timestamp_ns]
        rows.append(row)
    return Tabular.from_rows([DictRow(row) for row in rows]).table


def _aligned_frame_table(
    fields: Mapping[str, tuple[str, str | None]],
    topic_events: Mapping[str, Sequence[_McapEvent]],
    *,
    primary: tuple[str, str | None],
    include_skew: bool,
) -> pa.Table:
    primary_events = list(topic_events.get(primary[0], ()))
    primary_timestamps = [event.timestamp_ns for event in primary_events]
    aligned_events = {
        name: _nearest_events(topic_events.get(source[0], ()), primary_timestamps)
        for name, source in fields.items()
    }
    rows: list[dict[str, Any]] = []
    for index, primary_event in enumerate(primary_events):
        timestamp_ns = primary_event.timestamp_ns
        row: dict[str, Any] = {
            "frame_index": index,
            "timestamp": timestamp_ns / 1e9,
        }
        for name, source in fields.items():
            event = aligned_events[name][index]
            if event is None:
                row[name] = None
                continue
            row[name] = _source_value(event.value, source[1])
            if include_skew:
                row[f"mcap.{name}.timestamp"] = event.timestamp_ns / 1e9
                row[f"mcap.{name}.skew_ms"] = (event.timestamp_ns - timestamp_ns) / 1e6
        rows.append(row)
    return Tabular.from_rows([DictRow(row) for row in rows]).table


def _source_value(value: Any, field_path: str | None) -> Any:
    if field_path is None:
        return value
    current = value
    for part in field_path.split("."):
        if isinstance(current, Mapping):
            current = current[part]
        else:
            current = getattr(current, part)
    return current


def _nearest_events(
    events: Sequence[_McapEvent],
    timestamps_ns: Sequence[int],
) -> list[_McapEvent | None]:
    if not events:
        return [None] * len(timestamps_ns)
    sorted_events = sorted(events, key=lambda event: event.timestamp_ns)
    out: list[_McapEvent | None] = []
    cursor = 0
    for timestamp_ns in timestamps_ns:
        while (
            cursor + 1 < len(sorted_events)
            and sorted_events[cursor + 1].timestamp_ns <= timestamp_ns
        ):
            cursor += 1
        best = sorted_events[cursor]
        if cursor + 1 < len(sorted_events):
            right = sorted_events[cursor + 1]
            if abs(right.timestamp_ns - timestamp_ns) < abs(
                best.timestamp_ns - timestamp_ns
            ):
                best = right
        out.append(best)
    return out


def _infer_fps(events: Sequence[_McapEvent] | None) -> float | None:
    if events is None or len(events) < 2:
        return None
    deltas = [
        (right.timestamp_ns - left.timestamp_ns) / 1e9
        for left, right in zip(events, events[1:], strict=False)
        if right.timestamp_ns > left.timestamp_ns
    ]
    if not deltas:
        return None
    return 1.0 / float(np.median(np.asarray(deltas, dtype=np.float64)))


def _video_map(
    videos: Mapping[str, tuple[str, str | None]],
    topic_events: Mapping[str, Sequence[_McapEvent]],
    *,
    primary_events: Sequence[_McapEvent] | None,
    fps: int,
) -> dict[str, VideoFrameArray]:
    out: dict[str, VideoFrameArray] = {}
    primary_timestamps = (
        [event.timestamp_ns for event in primary_events] if primary_events else None
    )
    for name, source in videos.items():
        events = topic_events.get(source[0], ())
        if primary_timestamps is not None:
            frames = []
            for event in _nearest_events(events, primary_timestamps):
                if event is not None:
                    frames.append(
                        _frame_from_value(_source_value(event.value, source[1]))
                    )
        else:
            frames = [
                _frame_from_value(_source_value(event.value, source[1]))
                for event in events
            ]
        if frames:
            out[name] = VideoFrameArray(np.stack(frames), fps=fps)
    return out


def _frame_from_value(value: Any) -> np.ndarray:
    if isinstance(value, str):
        import base64

        return _frame_from_value(base64.b64decode(value))
    if isinstance(value, bytes):
        try:
            from PIL import Image

            return np.asarray(Image.open(BytesIO(value)).convert("RGB"), dtype=np.uint8)
        except Exception:
            raise ValueError("MCAP video payload bytes are not a decodable image")
    if isinstance(value, Mapping):
        if {"height", "width", "data"}.issubset(value):
            return _ros_image_frame(value)
        if {"data", "format"}.issubset(value):
            return _frame_from_value(value["data"])
        for key in ("image", "frame", "data"):
            if key in value:
                return _frame_from_value(value[key])
    array = np.asarray(value)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3 or int(array.shape[2]) != 3:
        raise ValueError("MCAP video frames must decode to [height, width, 3]")
    return np.asarray(np.clip(array, 0, 255), dtype=np.uint8)


def _ros_image_frame(value: Mapping[str, Any]) -> np.ndarray:
    height = int(value["height"])
    width = int(value["width"])
    encoding = str(value.get("encoding", "rgb8")).lower()
    data = value["data"]
    if isinstance(data, str):
        import base64

        raw = base64.b64decode(data)
    elif isinstance(data, bytes):
        raw = data
    else:
        raw = bytes(data)
    channels = 1 if encoding in {"mono8", "8uc1"} else 3
    array = np.frombuffer(raw, dtype=np.uint8)[: height * width * channels]
    array = array.reshape((height, width, channels))
    if channels == 1:
        return np.repeat(array, 3, axis=2)
    if encoding in {"bgr8", "bgr"}:
        array = array[:, :, ::-1]
    return array


__all__ = ["McapReader"]
