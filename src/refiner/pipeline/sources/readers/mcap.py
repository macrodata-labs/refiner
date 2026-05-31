from __future__ import annotations

import base64
import json
import warnings
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from itertools import chain
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
from refiner.video import VideoFrameArray, decode_raw_h264_frames

_MISSING = object()
_RESERVED_FRAME_COLUMNS = frozenset({"frame_index", "timestamp"})
_ROW_COLUMNS = frozenset({"records", "episode_index", "videos", "fps"})
_EPISODE_SPLITTING_ERROR = (
    "episode_splitting must be 'single', {'time_gap_s': seconds}, "
    "or {'marker_topic': topic}"
)
SyncMethod = Literal["nearest", "interpolate", "hold"]
_McapEvent = tuple[int, Any]
_EpisodeWindow = tuple[int | None, int | None]


@dataclass(frozen=True, slots=True)
class _AlignedValue:
    timestamp_ns: int
    value: Any
    skew_ns: int


class McapReader(BaseReader):
    """Read MCAP logs as episode rows with decoded records and optional videos."""

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
        episode_splitting: str | Mapping[str, Any] = "single",
        stream_episodes: bool = False,
        fields: PathSelection | None = None,
        videos: PathSelection | None = None,
        sync_primary: str | None = None,
        sync_method: SyncMethod = "nearest",
        include_skew: bool = True,
        fps: float | None = None,
    ):
        """Create an MCAP reader.

        Each emitted row represents one episode and contains a `records`
        `Tabular` table, plus optional `videos`, `fps`, and source file path
        columns. MCAP files are planned atomically; `episode_splitting` controls
        how each file is split into episode rows after it is read.

        Args:
            inputs: MCAP file, glob, directory, or sequence of inputs.
            fs: Optional fsspec filesystem for string inputs.
            storage_options: Optional fsspec storage options.
            recursive: Whether directory inputs should be expanded recursively.
            target_shard_bytes: Target shard size used when planning files.
            num_shards: Requested number of planned shards. MCAP files are
                atomic, so readers may emit fewer shards when there are fewer
                files.
            file_path_column: Output column for the source file path, or `None`
                to omit it.
            episode_splitting: `"single"`, `{"time_gap_s": seconds}`, or
                `{"marker_topic": topic}`.
            stream_episodes: When splitting episodes, buffer one episode at a
                time for seekable indexed MCAPs.
            fields: Record-table selections as output-name to MCAP source
                mapping, a single source string, a source sequence, or `None`
                to derive default fields from decoded non-video messages.
            videos: Video selections as video-name to MCAP source mapping, a
                single source string, a source sequence, or `None`.
            sync_primary: Optional selected field/video name, topic, or dotted
                MCAP source that defines aligned record timestamps.
            sync_method: Alignment method for non-primary fields and videos:
                `"nearest"`, `"hold"`, or `"interpolate"`.
            include_skew: Whether to add alignment timestamp/skew columns for
                non-primary aligned fields.
            fps: Explicit video/frame rate. If omitted, aligned reads infer it
                from `sync_primary` timestamps when possible.
        """
        time_gap_s, marker_topic = _parse_episode_splitting(episode_splitting)
        if sync_method not in ("nearest", "interpolate", "hold"):
            raise ValueError("sync_method must be 'nearest', 'interpolate', or 'hold'")
        if file_path_column in _ROW_COLUMNS:
            raise ValueError(
                f"file_path_column cannot use reserved MCAP row column {file_path_column!r}"
            )
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
        field_names = set(self.fields)
        reserved_fields = sorted(field_names & _RESERVED_FRAME_COLUMNS)
        if reserved_fields:
            raise ValueError(
                "MCAP fields cannot use reserved frame columns: "
                + ", ".join(reserved_fields)
            )
        if include_skew and sync_primary is not None:
            skew_field_names = field_names - {sync_primary}
            generated_fields = {
                generated
                for field in skew_field_names
                for generated in (f"mcap.{field}.timestamp", f"mcap.{field}.skew_ms")
            }
            collisions = sorted(field_names & generated_fields)
            if collisions:
                raise ValueError(
                    "MCAP fields cannot use generated skew/timestamp columns: "
                    + ", ".join(collisions)
                )
        self._read_default_fields = fields is None
        self.videos = path_selection_map(
            videos, format_name="MCAP", derive_names_from_paths=False
        )
        self.sync_primary = sync_primary
        self.fps = fps
        self.sync_method = sync_method
        self.include_skew = include_skew
        self.stream_episodes = stream_episodes
        self.episode_splitting = episode_splitting
        self._time_gap_s = time_gap_s
        self._marker_topic = marker_topic

    def describe(self) -> dict[str, Any]:
        description = super().describe()
        description.update(
            {
                "fields": self.fields,
                "videos": self.videos,
                "sync_primary": self.sync_primary,
                "fps": self.fps,
                "sync_method": self.sync_method,
                "include_skew": self.include_skew,
                "stream_episodes": self.stream_episodes,
                "episode_splitting": self.episode_splitting,
            }
        )
        return description

    def _episode_row(
        self,
        topic_events: Mapping[str, Sequence[_McapEvent]],
        *,
        file_topics: set[str],
        file_fields: Mapping[str, tuple[str, str | None]] | None,
        file_videos: Mapping[str, tuple[str, str | None]],
        episode_index: int,
        source: Any,
    ) -> DictRow:
        if self._read_default_fields:
            fields = _default_fields(
                topic_events,
                self.videos,
                excluded_topic=self._marker_topic,
            )
            source_topics = set(topic_events)
            resolved_fields = {
                output: _resolve_source(source, source_topics)
                for output, source in fields.items()
            }
        else:
            source_topics = file_topics
            resolved_fields = file_fields or {}
        sync_primary = None
        if self.sync_primary is not None:
            sync_primary = resolved_fields.get(self.sync_primary) or file_videos.get(
                self.sync_primary
            )
            if sync_primary is None:
                sync_primary = _resolve_source(self.sync_primary, file_topics)
        field_names = set(resolved_fields)
        reserved_fields = sorted(field_names & _RESERVED_FRAME_COLUMNS)
        if reserved_fields:
            raise ValueError(
                "MCAP fields cannot use reserved frame columns: "
                + ", ".join(reserved_fields)
            )
        if self.include_skew and sync_primary is not None:
            skew_field_names = {
                name
                for name, source in resolved_fields.items()
                if source[0] != sync_primary[0]
            }
            generated_fields = {
                generated
                for field in skew_field_names
                for generated in (
                    f"mcap.{field}.timestamp",
                    f"mcap.{field}.skew_ms",
                )
            }
            collisions = sorted(field_names & generated_fields)
            if collisions:
                raise ValueError(
                    "MCAP fields cannot use generated skew/timestamp columns: "
                    + ", ".join(collisions)
                )
        sync_primary_events = (
            sorted(
                topic_events.get(sync_primary[0], ()),
                key=lambda event: event[0],
            )
            if sync_primary is not None
            else None
        )
        frame_table = (
            _sparse_frame_table(resolved_fields, topic_events)
            if sync_primary is None
            else _aligned_frame_table(
                resolved_fields,
                topic_events,
                sync_primary_events=sync_primary_events or (),
                sync_primary=sync_primary,
                sync_method=self.sync_method,
                include_skew=self.include_skew,
            )
        )
        inferred_fps = self.fps or _infer_fps(sync_primary_events)
        video_fps = inferred_fps or 30
        if self.videos and video_fps != int(video_fps):
            raise ValueError("MCAP videos require integer fps")
        videos = _video_map(
            file_videos,
            topic_events,
            sync_primary_events=sync_primary_events,
            sync_primary=sync_primary,
            sync_method=self.sync_method,
            fps=int(video_fps),
        )
        row: dict[str, Any] = {
            "records": Tabular(frame_table),
            "episode_index": episode_index,
        }
        if videos:
            row["videos"] = videos
        if inferred_fps is not None:
            row["fps"] = float(inferred_fps)
        return DictRow(self._with_file_path(row, source))

    def read_shard(self, shard: Shard) -> Iterator[SourceUnit]:
        descriptor = shard.descriptor
        assert isinstance(descriptor, FilePartsDescriptor)
        check_required_dependencies("read_mcap", ["mcap"], dist="mcap")
        from mcap.reader import make_reader

        decoder_factories = _decoder_factories()
        for part in descriptor.parts:
            source = self.fileset.resolve_file(part.source_index, part.path)
            topic_events: dict[str, list[_McapEvent]] = defaultdict(list)
            # Streaming only helps when a file can be split into smaller episodes.
            wants_stream = self.stream_episodes and (
                self._time_gap_s is not None or self._marker_topic is not None
            )
            with source.open(mode="rb") as stream:
                stream_is_seekable = stream.seekable()
                reader = make_reader(stream)
                # If sync_primary is already a selected output name, its topic is
                # already covered by fields/videos.
                sync_primary_source = (
                    self.sync_primary
                    if self.sync_primary is not None
                    and self.sync_primary not in self.fields
                    and self.sync_primary not in self.videos
                    else None
                )
                selected_sources: list[str] | None = None
                if not self._read_default_fields:
                    selected_sources = [*self.fields.values(), *self.videos.values()]
                    selected_sources.extend(
                        source
                        for source in (sync_primary_source, self._marker_topic)
                        if source is not None
                    )
                # The summary lets us turn dotted sources into exact MCAP topics
                # before scanning messages.
                summary = (
                    reader.get_summary()
                    if stream_is_seekable
                    and (selected_sources is not None or wants_stream)
                    else None
                )
                summary_topics = (
                    {channel.topic for channel in summary.channels.values()}
                    if summary is not None
                    else set()
                )
                read_topics: tuple[str, ...] | None = None
                # None means read all topics; () means read no topics.
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
                can_stream = (
                    wants_stream
                    and stream_is_seekable
                    and summary is not None
                    and bool(summary.chunk_indexes)
                )
                if wants_stream and not can_stream:
                    # Non-seekable or unindexed inputs fall back because MCAP
                    # log-time ordering would require buffering the stream.
                    warnings.warn(
                        "read_mcap stream_episodes fell back to buffered reading",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                if can_stream:
                    file_topics = summary_topics
                    file_fields = (
                        None
                        if self._read_default_fields
                        else {
                            output: _resolve_source(source, file_topics)
                            for output, source in self.fields.items()
                        }
                    )
                    file_videos = {
                        output: _resolve_source(source, file_topics)
                        for output, source in self.videos.items()
                    }
                    episode_index = 0
                    # Group same-timestamp messages before deciding boundaries
                    # so marker messages include peer events at the same time.
                    gap_ns = (
                        int(self._time_gap_s * 1e9)
                        if self._time_gap_s is not None
                        else None
                    )
                    last_timestamp_ns: int | None = None
                    pending_ts: int | None = None
                    pending: list[tuple[str, _McapEvent]] = []
                    saw_marker = False
                    saw_message = False
                    for item in chain(
                        reader.iter_messages(
                            topics=read_topics,
                            log_time_order=True,
                        ),
                        (None,),
                    ):
                        timestamp_ns = item[2].log_time if item is not None else None
                        if pending and timestamp_ns != pending_ts:
                            assert pending_ts is not None
                            if gap_ns is None:
                                boundary = any(
                                    topic == self._marker_topic for topic, _ in pending
                                )
                                should_yield = saw_marker
                                saw_marker = saw_marker or boundary
                            else:
                                boundary = (
                                    last_timestamp_ns is not None
                                    and pending_ts - last_timestamp_ns > gap_ns
                                )
                                should_yield = True
                                last_timestamp_ns = pending_ts
                            if boundary:
                                # A boundary starts a new episode at this
                                # timestamp group, so flush the previous group
                                # before appending the pending messages.
                                if should_yield:
                                    yield self._episode_row(
                                        topic_events,
                                        file_topics=file_topics,
                                        file_fields=file_fields,
                                        file_videos=file_videos,
                                        episode_index=episode_index,
                                        source=source,
                                    )
                                    episode_index += 1
                                topic_events = defaultdict(list)
                            for topic, event in pending:
                                topic_events[topic].append(event)
                            pending = []
                        if item is None:
                            yield self._episode_row(
                                topic_events if saw_message else {},
                                file_topics=file_topics,
                                file_fields=file_fields,
                                file_videos=file_videos,
                                episode_index=episode_index,
                                source=source,
                            )
                            break
                        schema, channel, message = item
                        decoded = _decode_message(
                            schema,
                            channel.message_encoding,
                            message.data,
                            decoder_factories,
                        )
                        pending.append((channel.topic, (message.log_time, decoded)))
                        pending_ts = message.log_time
                        saw_message = True
                    continue
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
                    topic_events[channel.topic].append((message.log_time, decoded))
            # Buffered path: split after reading the whole file.
            if self._time_gap_s is not None:
                windows = _time_gap_windows(topic_events, self._time_gap_s)
            elif self._marker_topic is not None:
                windows = _marker_windows(topic_events.get(self._marker_topic, ()))
            else:
                windows = [(None, None)]
            file_topics = set(topic_events)
            file_fields = (
                None
                if self._read_default_fields
                else {
                    output: _resolve_source(source, file_topics)
                    for output, source in self.fields.items()
                }
            )
            file_videos = {
                output: _resolve_source(source, file_topics)
                for output, source in self.videos.items()
            }
            for episode_index, window in enumerate(windows):
                window_events = _slice_events(topic_events, window)
                yield self._episode_row(
                    window_events,
                    file_topics=file_topics,
                    file_fields=file_fields,
                    file_videos=file_videos,
                    episode_index=episode_index,
                    source=source,
                )


def _parse_episode_splitting(
    splitting: str | Mapping[str, Any],
) -> tuple[float | None, str | None]:
    if splitting == "single":
        return None, None
    if (
        isinstance(splitting, str)
        or not isinstance(splitting, Mapping)
        or len(splitting) != 1
        or next(iter(splitting)) not in {"time_gap_s", "marker_topic"}
    ):
        raise ValueError(_EPISODE_SPLITTING_ERROR)
    key, value = next(iter(splitting.items()))
    if key == "time_gap_s":
        time_gap_s = float(value)
        if time_gap_s <= 0:
            raise ValueError("episode_splitting time_gap_s must be > 0")
        return time_gap_s, None
    if not isinstance(value, str):
        raise TypeError("episode_splitting marker_topic must be a string")
    return None, value


def _time_gap_windows(
    topic_events: Mapping[str, Sequence[_McapEvent]],
    time_gap_s: float,
) -> list[_EpisodeWindow]:
    timestamps = sorted(
        event[0] for events in topic_events.values() for event in events
    )
    if not timestamps:
        return [(None, None)]
    gap_ns = int(time_gap_s * 1e9)
    windows: list[_EpisodeWindow] = []
    start = timestamps[0]
    for left, right in zip(timestamps, timestamps[1:], strict=False):
        if right - left > gap_ns:
            windows.append((start, right))
            start = right
    windows.append((start, timestamps[-1] + 1))
    return windows


def _marker_windows(marker_events: Sequence[_McapEvent]) -> list[_EpisodeWindow]:
    if not marker_events:
        return [(None, None)]
    starts = sorted(event[0] for event in marker_events)
    ends: list[int | None] = [*starts[1:], None]
    return [(start, end) for start, end in zip(starts, ends, strict=False)]


def _slice_events(
    topic_events: Mapping[str, Sequence[_McapEvent]],
    window: _EpisodeWindow,
) -> dict[str, list[_McapEvent]]:
    out: dict[str, list[_McapEvent]] = {}
    for topic, events in topic_events.items():
        selected = [
            event
            for event in events
            if (window[0] is None or event[0] >= window[0])
            and (window[1] is None or event[0] < window[1])
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
        names = sorted({name for event in events for name in _flatten_names(event[1])})
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
            values[event[0]].append(_source_value(event[1], source[1], default=None))
        values_by_field[name] = values
    timestamps = sorted(
        {
            event[0]
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
                else:
                    row[name] = None
            rows.append(row)
    return Tabular.from_rows([DictRow(row) for row in rows]).table


def _aligned_frame_table(
    fields: Mapping[str, tuple[str, str | None]],
    topic_events: Mapping[str, Sequence[_McapEvent]],
    *,
    sync_primary_events: Sequence[_McapEvent],
    sync_primary: tuple[str, str | None],
    sync_method: SyncMethod,
    include_skew: bool,
) -> pa.Table:
    sync_primary_timestamps = [event[0] for event in sync_primary_events]
    aligned_values = {
        name: _align_values(
            topic_events.get(source[0], ()),
            sync_primary_timestamps,
            source[1],
            method=sync_method,
        )
        for name, source in fields.items()
        if source[0] != sync_primary[0]
    }
    rows: list[dict[str, Any]] = []
    for index, sync_primary_event in enumerate(sync_primary_events):
        timestamp_ns = sync_primary_event[0]
        row: dict[str, Any] = {
            "frame_index": index,
            "timestamp": timestamp_ns / 1e9,
        }
        for name in fields:
            source = fields[name]
            if source[0] == sync_primary[0]:
                row[name] = _source_value(
                    sync_primary_event[1], source[1], default=None
                )
                continue
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
    sorted_events = sorted(events, key=lambda event: event[0])
    source_timestamps = [event[0] for event in sorted_events]
    source_values = [
        _source_value(event[1], field_path, default=None) for event in sorted_events
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
) -> _AlignedValue | None:
    index = bisect_right(source_timestamps, timestamp_ns) - 1
    if index < 0:
        return None
    source_timestamp = int(source_timestamps[index])
    return _AlignedValue(
        timestamp_ns=source_timestamp,
        value=source_values[index],
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
        left_array.dtype.kind not in "iufc"
        or right_array.dtype.kind not in "iufc"
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
        (right[0] - left[0]) / 1e9
        for left, right in zip(events, events[1:], strict=False)
        if right[0] > left[0]
    ]
    if not deltas:
        return None
    return 1.0 / float(np.median(np.asarray(deltas, dtype=np.float64)))


def _video_map(
    videos: Mapping[str, tuple[str, str | None]],
    topic_events: Mapping[str, Sequence[_McapEvent]],
    *,
    sync_primary_events: Sequence[_McapEvent] | None,
    sync_primary: tuple[str, str | None] | None,
    sync_method: SyncMethod,
    fps: int,
) -> dict[str, VideoFrameArray]:
    out: dict[str, VideoFrameArray] = {}
    sync_primary_timestamps = (
        [event[0] for event in sync_primary_events]
        if sync_primary_events is not None
        else None
    )
    for name, source in videos.items():
        events = topic_events.get(source[0], ())
        if sync_primary is not None and source[0] == sync_primary[0]:
            source_events = sync_primary_events or ()
            h264_events = _h264_frame_events(source_events, source[1])
            frames = (
                [event[1] for event in h264_events]
                if h264_events is not None
                else [
                    _frame_from_value(_source_value(event[1], source[1]))
                    for event in source_events
                ]
            )
        elif sync_primary_timestamps is not None:
            video_sync_method: SyncMethod = (
                "nearest" if sync_method == "interpolate" else sync_method
            )
            h264_events = _h264_frame_events(events, source[1])
            frames = []
            for aligned in _align_values(
                h264_events if h264_events is not None else events,
                sync_primary_timestamps,
                None if h264_events is not None else source[1],
                method=video_sync_method,
            ):
                if aligned is None:
                    raise ValueError(
                        f"MCAP video {name!r} has no aligned frame for a sync_primary row"
                    )
                frames.append(
                    aligned.value
                    if h264_events is not None
                    else _frame_from_value(aligned.value)
                )
        else:
            h264_events = _h264_frame_events(events, source[1])
            frames = (
                [event[1] for event in h264_events]
                if h264_events is not None
                else [
                    _frame_from_value(_source_value(event[1], source[1]))
                    for event in sorted(events, key=lambda event: event[0])
                ]
            )
        if frames:
            out[name] = VideoFrameArray(np.stack(frames), fps=fps)
    return out


def _h264_frame_events(
    events: Sequence[_McapEvent],
    field_path: str | None,
) -> list[_McapEvent] | None:
    if not events:
        return []
    chunks: list[bytes] = []
    timestamps: list[int] = []
    for timestamp_ns, value in sorted(events, key=lambda event: event[0]):
        packet = _source_value(value, field_path, default=None)
        if not isinstance(packet, Mapping) or str(
            packet.get("format", "")
        ).lower() not in {"h264", "h.264", "video/h264"}:
            return None
        data = packet.get("data")
        if isinstance(data, str):
            payload = base64.b64decode(data)
        elif isinstance(data, bytes):
            payload = data
        elif data is not None:
            payload = bytes(data)
        else:
            return None
        timestamps.append(timestamp_ns)
        chunks.append(payload)

    frames = decode_raw_h264_frames(chunks)
    if len(frames) != len(timestamps):
        raise ValueError("MCAP H.264 video frame count does not match message count")
    return list(zip(timestamps, frames, strict=True))


def _frame_from_value(value: Any) -> np.ndarray:
    if isinstance(value, str):
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
            data = value["data"]
            if not isinstance(data, str | bytes):
                data = bytes(data)
            return _frame_from_value(data)
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
