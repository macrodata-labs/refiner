from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Literal

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

_MISSING = object()
SyncMethod = Literal["nearest", "interpolate", "hold"]


@dataclass(frozen=True, slots=True)
class _McapEvent:
    timestamp_ns: int
    value: Any


@dataclass(frozen=True, slots=True)
class _EpisodeWindow:
    start_ns: int | None = None
    end_ns: int | None = None


@dataclass(frozen=True, slots=True)
class _AlignedValue:
    timestamp_ns: int
    value: Any
    skew_ns: int


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
        file_path_column: str | None = "file_path",
        fields: PathSelection | None = None,
        videos: PathSelection | None = None,
        primary: str | None = None,
        fps: float | None = None,
        sync_method: SyncMethod = "nearest",
        include_skew: bool = True,
        episode_splitting: str | Mapping[str, Any] = "single",
    ):
        time_gap_s, marker_topic = _parse_episode_splitting(episode_splitting)
        if sync_method not in ("nearest", "interpolate", "hold"):
            raise ValueError("sync_method must be 'nearest', 'interpolate', or 'hold'")
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
        self.fields = path_selection_map(
            fields, format_name="MCAP", derive_names_from_paths=False
        )
        self._read_default_fields = fields is None
        self.videos = path_selection_map(
            videos, format_name="MCAP", derive_names_from_paths=False
        )
        self.primary = primary
        self.fps = fps
        self.sync_method = sync_method
        self.include_skew = include_skew
        self.episode_splitting = episode_splitting
        self._time_gap_s = time_gap_s
        self._marker_topic = marker_topic

    def describe(self) -> dict[str, Any]:
        description = super().describe()
        description.update(
            {
                "fields": self.fields,
                "videos": self.videos,
                "primary": self.primary,
                "fps": self.fps,
                "sync_method": self.sync_method,
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
                stream_is_seekable = stream.seekable()
                reader = make_reader(stream)
                summary = reader.get_summary() if stream_is_seekable else None
                summary_topics = (
                    {channel.topic for channel in summary.channels.values()}
                    if summary is not None
                    else set()
                )
                primary_source = (
                    self.primary
                    if self.primary is not None
                    and self.primary not in self.fields
                    and self.primary not in self.videos
                    else None
                )
                selected_sources: list[str] | None = None
                if not self._read_default_fields:
                    selected_sources = [*self.fields.values(), *self.videos.values()]
                    selected_sources.extend(
                        source
                        for source in (primary_source, self._marker_topic)
                        if source is not None
                    )
                read_topics: tuple[str, ...] | None = None
                if selected_sources is not None:
                    if not selected_sources:
                        read_topics = ()
                    elif summary_topics:
                        read_topics = tuple(
                            sorted(
                                {
                                    _resolve_source(source, summary_topics)[0]
                                    for source in selected_sources
                                }
                            )
                        )
                for schema, channel, message in reader.iter_messages(
                    topics=read_topics,
                    log_time_order=False,
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
                fields = (
                    _default_fields(
                        window_events,
                        self.videos,
                        excluded_topic=self._marker_topic,
                    )
                    if self._read_default_fields
                    else self.fields
                )
                available_topics = set(window_events)
                resolved_fields = {
                    output: _resolve_source(source, available_topics)
                    for output, source in fields.items()
                }
                resolved_videos = {
                    output: _resolve_source(source, available_topics)
                    for output, source in self.videos.items()
                }
                primary = None
                if self.primary is not None:
                    primary = resolved_fields.get(self.primary) or resolved_videos.get(
                        self.primary
                    )
                    if primary is None:
                        primary = _resolve_source(self.primary, available_topics)
                primary_events = (
                    sorted(
                        window_events.get(primary[0], ()),
                        key=lambda event: event.timestamp_ns,
                    )
                    if primary is not None
                    else None
                )
                frame_table = (
                    _sparse_frame_table(resolved_fields, window_events)
                    if primary is None
                    else _aligned_frame_table(
                        resolved_fields,
                        window_events,
                        primary_events=primary_events or (),
                        sync_method=self.sync_method,
                        include_skew=self.include_skew,
                    )
                )
                inferred_fps = self.fps or _infer_fps(primary_events)
                videos = _video_map(
                    resolved_videos,
                    window_events,
                    primary_events=primary_events,
                    sync_method=self.sync_method,
                    fps=int(round(inferred_fps or 30)),
                )
                row: dict[str, Any] = {
                    "frames": Tabular(frame_table),
                    "episode_index": episode_index,
                }
                if videos:
                    row["videos"] = videos
                if inferred_fps is not None:
                    row["fps"] = float(inferred_fps)
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
    starts = sorted(event.timestamp_ns for event in marker_events)
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
        names = sorted(
            {name for event in events for name in _flatten_names(event.value)}
        )
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
    values_by_field: dict[str, dict[int, list[Any]]] = {}
    for name, source in fields.items():
        values: dict[int, list[Any]] = defaultdict(list)
        for event in topic_events.get(source[0], ()):
            values[event.timestamp_ns].append(
                _source_value(event.value, source[1], default=None)
            )
        values_by_field[name] = values
    timestamps = sorted(
        {
            event.timestamp_ns
            for source in fields.values()
            for event in topic_events.get(source[0], ())
        }
    )
    rows: list[dict[str, Any]] = []
    for timestamp_ns in timestamps:
        repeats = max(
            (len(values.get(timestamp_ns, ())) for values in values_by_field.values()),
            default=1,
        )
        for duplicate_index in range(repeats):
            row: dict[str, Any] = {
                "frame_index": len(rows),
                "timestamp": timestamp_ns / 1e9,
            }
            for name, values in values_by_field.items():
                timestamp_values = values.get(timestamp_ns, ())
                if duplicate_index < len(timestamp_values):
                    row[name] = timestamp_values[duplicate_index]
            rows.append(row)
    return Tabular.from_rows([DictRow(row) for row in rows]).table


def _aligned_frame_table(
    fields: Mapping[str, tuple[str, str | None]],
    topic_events: Mapping[str, Sequence[_McapEvent]],
    *,
    primary_events: Sequence[_McapEvent],
    sync_method: SyncMethod,
    include_skew: bool,
) -> pa.Table:
    primary_timestamps = [event.timestamp_ns for event in primary_events]
    aligned_values = {
        name: _align_values(
            topic_events.get(source[0], ()),
            primary_timestamps,
            source[1],
            method=sync_method,
        )
        for name, source in fields.items()
    }
    rows: list[dict[str, Any]] = []
    for index, primary_event in enumerate(primary_events):
        timestamp_ns = primary_event.timestamp_ns
        row: dict[str, Any] = {
            "frame_index": index,
            "timestamp": timestamp_ns / 1e9,
        }
        for name in fields:
            aligned = aligned_values[name][index]
            if aligned is None:
                row[name] = None
                continue
            row[name] = aligned.value
            if include_skew:
                row[f"mcap.{name}.timestamp"] = aligned.timestamp_ns / 1e9
                row[f"mcap.{name}.skew_ms"] = aligned.skew_ns / 1e6
        rows.append(row)
    return Tabular.from_rows([DictRow(row) for row in rows]).table


def _source_value(
    value: Any, field_path: str | None, *, default: Any = _MISSING
) -> Any:
    if field_path is None:
        return value
    current = value
    try:
        for part in field_path.split("."):
            if isinstance(current, Mapping):
                current = current[part]
            else:
                current = getattr(current, part)
    except (KeyError, AttributeError):
        if default is not _MISSING:
            return default
        raise
    return current


def _align_values(
    events: Sequence[_McapEvent],
    timestamps_ns: Sequence[int],
    field_path: str | None,
    *,
    method: SyncMethod,
) -> list[_AlignedValue | None]:
    if not events:
        return [None] * len(timestamps_ns)
    sorted_events = sorted(events, key=lambda event: event.timestamp_ns)
    source_timestamps = [event.timestamp_ns for event in sorted_events]
    source_values = [
        _source_value(event.value, field_path, default=None) for event in sorted_events
    ]
    align = _nearest_value
    if method == "hold":
        align = _hold_value
    elif method == "interpolate":
        align = _interpolate_value
    return [
        align(timestamp_ns, source_timestamps, source_values)
        for timestamp_ns in timestamps_ns
    ]


def _nearest_value(
    timestamp_ns: int,
    source_timestamps: Sequence[int],
    source_values: Sequence[Any],
) -> _AlignedValue:
    index = bisect_left(source_timestamps, timestamp_ns)
    if index == 0:
        source_index = 0
    elif index == len(source_timestamps):
        source_index = len(source_timestamps) - 1
    else:
        left = source_timestamps[index - 1]
        right = source_timestamps[index]
        source_index = (
            index - 1 if timestamp_ns - left <= right - timestamp_ns else index
        )
    source_timestamp = int(source_timestamps[source_index])
    return _AlignedValue(
        timestamp_ns=source_timestamp,
        value=source_values[source_index],
        skew_ns=source_timestamp - timestamp_ns,
    )


def _hold_value(
    timestamp_ns: int,
    source_timestamps: Sequence[int],
    source_values: Sequence[Any],
) -> _AlignedValue:
    index = bisect_right(source_timestamps, timestamp_ns) - 1
    source_index = max(0, index)
    source_timestamp = int(source_timestamps[source_index])
    return _AlignedValue(
        timestamp_ns=source_timestamp,
        value=source_values[source_index],
        skew_ns=source_timestamp - timestamp_ns,
    )


def _interpolate_value(
    timestamp_ns: int,
    source_timestamps: Sequence[int],
    source_values: Sequence[Any],
) -> _AlignedValue:
    index = bisect_left(source_timestamps, timestamp_ns)
    if index == 0 or index == len(source_timestamps):
        return _nearest_value(timestamp_ns, source_timestamps, source_values)

    left_value = source_values[index - 1]
    right_value = source_values[index]
    left_array = np.asarray(left_value)
    right_array = np.asarray(right_value)
    if (
        left_array.dtype.kind not in "biufc"
        or right_array.dtype.kind not in "biufc"
        or left_array.shape != right_array.shape
    ):
        return _nearest_value(timestamp_ns, source_timestamps, source_values)

    left_timestamp = int(source_timestamps[index - 1])
    right_timestamp = int(source_timestamps[index])
    span = right_timestamp - left_timestamp
    if span <= 0:
        return _nearest_value(timestamp_ns, source_timestamps, source_values)

    alpha = (timestamp_ns - left_timestamp) / span
    value = (1.0 - alpha) * left_array + alpha * right_array
    if value.shape == ():
        value = value.item()
    else:
        value = value.tolist()

    nearest_timestamp = (
        left_timestamp
        if timestamp_ns - left_timestamp <= right_timestamp - timestamp_ns
        else right_timestamp
    )
    return _AlignedValue(
        timestamp_ns=nearest_timestamp,
        value=value,
        skew_ns=nearest_timestamp - timestamp_ns,
    )


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
    sync_method: SyncMethod,
    fps: int,
) -> dict[str, VideoFrameArray]:
    out: dict[str, VideoFrameArray] = {}
    primary_timestamps = (
        [event.timestamp_ns for event in primary_events] if primary_events else None
    )
    for name, source in videos.items():
        events = topic_events.get(source[0], ())
        if primary_timestamps is not None:
            video_sync_method: SyncMethod = (
                "nearest" if sync_method == "interpolate" else sync_method
            )
            frames = [
                _frame_from_value(aligned.value)
                for aligned in _align_values(
                    events, primary_timestamps, source[1], method=video_sync_method
                )
                if aligned is not None
            ]
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
    if encoding in {"mono8", "8uc1"}:
        channels = 1
    elif encoding in {"rgba8", "bgra8", "8uc4"}:
        channels = 4
    else:
        channels = 3
    step = int(value.get("step") or width * channels)
    expected_bytes = height * step
    array = np.frombuffer(raw, dtype=np.uint8)
    if array.size < expected_bytes:
        raise ValueError("MCAP ROS image payload is smaller than height * step")
    array = array[:expected_bytes].reshape((height, step))
    array = array[:, : width * channels].reshape((height, width, channels))
    if channels == 1:
        return np.repeat(array, 3, axis=2)
    if channels == 4:
        array = array[:, :, [2, 1, 0]] if encoding == "bgra8" else array[:, :, :3]
        return array
    if encoding in {"bgr8", "bgr"}:
        array = array[:, :, ::-1]
    return array


__all__ = ["McapReader", "SyncMethod"]
