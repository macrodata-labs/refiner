from __future__ import annotations

import asyncio
from concurrent.futures import Future
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any

import numpy as np

from refiner.execution.asyncio.runtime import submit
from refiner.io import DataFolder
from refiner.media import Video
from refiner.pipeline.sinks.lerobot._lerobot_video import (
    _PendingVideoRun,
    _PendingVideoSegment,
    VideoTrackWriter,
    _append_video_segment,
    _resolve_video_fps,
    run_can_extend,
    video_from_timestamp_s,
    video_to_timestamp_s,
    video_uri,
)
from refiner.pipeline.sinks.lerobot._lerobot_video_remux import (
    RemuxWriter,
    _VideoProbeCache,
    _VideoSourceProbe,
    append_remux_run,
    probe_run_for_remux,
    probes_are_remux_compatible,
)

if TYPE_CHECKING:
    from refiner.pipeline.sinks.lerobot._lerobot_writer import (
        LeRobotStatsConfig,
        LeRobotVideoConfig,
    )

DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index}/file-{file_index:03d}.mp4"


def _format_video_path(*, chunk_key: str, video_key: str, file_idx: int) -> str:
    return DEFAULT_VIDEO_PATH.format(
        video_key=video_key,
        chunk=chunk_key,
        chunk_key=chunk_key,
        chunk_index=chunk_key,
        file=file_idx,
        file_idx=file_idx,
        file_index=file_idx,
    )


def _video_feature(
    *,
    fps: int,
    height: int,
    width: int,
    codec: str,
    pix_fmt: str,
) -> dict[str, Any]:
    return {
        "dtype": "video",
        "shape": [3, int(height), int(width)],
        "names": ["channels", "height", "width"],
        "info": {
            "video.fps": int(fps),
            "video.height": int(height),
            "video.width": int(width),
            "video.channels": 3,
            "video.codec": codec,
            "video.pix_fmt": pix_fmt,
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }


@dataclass(slots=True)
class _CompletedVideoSegment:
    episode_index: int
    video_key: str
    video_meta: dict[str, Any]
    video_stats: dict[str, np.ndarray]


@dataclass(slots=True)
class _CompletedVideoRun:
    video_key: str
    feature: dict[str, Any] | None
    segments: list[_CompletedVideoSegment]


@dataclass(slots=True)
class _PreparedVideoRun:
    run: _PendingVideoRun
    probe: _VideoSourceProbe | None


@dataclass(slots=True)
class _SequencedVideoRun:
    seq: int
    run: _PendingVideoRun


@dataclass(slots=True)
class LeRobotVideoWriter:
    folder: DataFolder
    chunk_key: str
    video_key: str
    video_config: "LeRobotVideoConfig"
    stats_config: "LeRobotStatsConfig"
    default_fps: int | None
    video_bytes_limit: int
    prepare_max_in_flight: int = 10

    _coordinator_future: Future[None] = field(init=False)
    _input_runs: Queue[_SequencedVideoRun | None] = field(
        default_factory=Queue,
        init=False,
    )
    _completed_runs: Queue[_CompletedVideoRun] = field(
        default_factory=Queue,
        init=False,
    )
    _probe_cache: _VideoProbeCache = field(default_factory=_VideoProbeCache, init=False)
    _pending_run: _PendingVideoRun | None = field(default=None, init=False)
    _writer: RemuxWriter | VideoTrackWriter | None = field(
        default=None,
        init=False,
    )
    _next_file_index: int = field(default=0, init=False)
    _next_run_seq: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._coordinator_future = submit(self._run_coordinator())

    def submit(
        self,
        *,
        episode_index: int,
        video: Video,
        source_stats: dict[str, np.ndarray] | None,
    ) -> list[_CompletedVideoRun]:
        self._raise_if_failed()

        current_run = self._pending_run
        if current_run is not None and not run_can_extend(current_run, video):
            self._schedule_pending_run()

        if self._pending_run is None:
            self._pending_run = _PendingVideoRun(video_key=self.video_key)

        self._pending_run.segments.append(
            _PendingVideoSegment(
                episode_index=episode_index,
                video=video,
                source_stats=source_stats,
            )
        )
        return self._drain_completed_runs()

    def flush(self) -> list[_CompletedVideoRun]:
        self._raise_if_failed()
        self._schedule_pending_run()
        self._input_runs.put(None)
        self._coordinator_future.result()
        self._close_writer()
        return self._drain_completed_runs()

    def drain_completed(
        self, *, force_schedule_pending_run: bool = False
    ) -> list[_CompletedVideoRun]:
        self._raise_if_failed()
        if force_schedule_pending_run:
            self._schedule_pending_run()
        return self._drain_completed_runs()

    def _schedule_pending_run(self) -> bool:
        current_run = self._pending_run
        if current_run is None or not current_run.segments:
            return False
        self._input_runs.put(
            _SequencedVideoRun(seq=self._next_run_seq, run=current_run)
        )
        self._next_run_seq += 1
        self._pending_run = None
        return True

    def _raise_if_failed(self) -> None:
        if not self._coordinator_future.done():
            return
        if (exc := self._coordinator_future.exception()) is not None:
            raise exc

    def _drain_completed_runs(self) -> list[_CompletedVideoRun]:
        self._raise_if_failed()
        completed: list[_CompletedVideoRun] = []
        while True:
            try:
                completed.append(self._completed_runs.get_nowait())
            except Empty:
                return completed

    async def _run_coordinator(self) -> None:
        prepared_queue: asyncio.Queue[tuple[int, _PreparedVideoRun] | None] = (
            asyncio.Queue()
        )
        await asyncio.gather(
            self._produce_prepared_runs(prepared_queue),
            self._commit_prepared_runs(prepared_queue),
        )

    async def _produce_prepared_runs(
        self,
        prepared_queue: asyncio.Queue[tuple[int, _PreparedVideoRun] | None],
    ) -> None:
        active_prepares: set[asyncio.Task[None]] = set()

        while True:
            sequenced = await asyncio.to_thread(self._input_runs.get)
            if sequenced is None:
                break

            while len(active_prepares) >= self.prepare_max_in_flight:
                done, pending = await asyncio.wait(
                    active_prepares,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                active_prepares = set(pending)
                for task in done:
                    task.result()

            active_prepares.add(
                asyncio.create_task(self._prepare_and_queue(sequenced, prepared_queue))
            )

        if active_prepares:
            done, _ = await asyncio.wait(active_prepares)
            for task in done:
                task.result()
        await prepared_queue.put(None)

    async def _prepare_and_queue(
        self,
        sequenced: _SequencedVideoRun,
        prepared_queue: asyncio.Queue[tuple[int, _PreparedVideoRun] | None],
    ) -> None:
        await prepared_queue.put(
            (sequenced.seq, await self._prepare_closed_run(sequenced.run))
        )

    async def _commit_prepared_runs(
        self,
        prepared_queue: asyncio.Queue[tuple[int, _PreparedVideoRun] | None],
    ) -> None:
        prepared_by_seq: dict[int, _PreparedVideoRun] = {}
        next_commit_seq = 0
        producer_done = False

        while not producer_done or prepared_by_seq:
            prepared_item = await prepared_queue.get()
            if prepared_item is None:
                producer_done = True
            else:
                prepared_by_seq[prepared_item[0]] = prepared_item[1]

            # Ensures order
            while next_commit_seq in prepared_by_seq:
                prepared = prepared_by_seq.pop(next_commit_seq)
                for completed in await self._commit_prepared_run(prepared):
                    self._completed_runs.put(completed)
                next_commit_seq += 1

    async def _prepare_closed_run(self, run: _PendingVideoRun) -> _PreparedVideoRun:
        probe = await asyncio.to_thread(
            probe_run_for_remux,
            run=run,
            probe_cache=self._probe_cache,
            default_fps=self.default_fps,
        )
        return _PreparedVideoRun(run=run, probe=probe)

    async def _commit_prepared_run(
        self, prepared: _PreparedVideoRun
    ) -> list[_CompletedVideoRun]:
        if prepared.probe is not None:
            return [self._remux_run(prepared.run, prepared.probe)]
        return [await self._transcode_run(prepared.run)]

    def _remux_run(
        self,
        run: _PendingVideoRun,
        probe: _VideoSourceProbe,
    ) -> _CompletedVideoRun:
        if isinstance(self._writer, VideoTrackWriter):
            self._rotate_writer()

        writer = self._writer if isinstance(self._writer, RemuxWriter) else None
        if writer is None or (
            not probes_are_remux_compatible(writer.probe, probe)
            or writer.size_bytes >= self.video_bytes_limit
        ):
            if writer is not None:
                self._rotate_writer()
            writer = RemuxWriter.open(
                folder=self.folder,
                video_key=self.video_key,
                output_rel=self._current_output_rel,
                file_idx=self._next_file_index,
                probe=probe,
            )
            self._writer = writer

        base_duration_s = float(writer.duration_s)
        append_remux_run(writer=writer, run=run, probe=probe)
        return _completed_remux_run(
            video_key=self.video_key,
            chunk_key=self.chunk_key,
            run=run,
            probe=probe,
            file_idx=writer.file_idx,
            base_duration_s=base_duration_s,
            probe_cache=self._probe_cache,
            video_config=self.video_config,
        )

    async def _transcode_run(self, run: _PendingVideoRun) -> _CompletedVideoRun:
        completed_segments: list[_CompletedVideoSegment] = []
        feature: dict[str, Any] | None = None

        for segment in run.segments:
            if isinstance(self._writer, RemuxWriter):
                self._rotate_writer()

            writer = await self._ensure_transcode_writer(segment.video)
            if writer.size_bytes >= self.video_bytes_limit:
                self._rotate_writer()
                writer = await self._ensure_transcode_writer(segment.video)

            from_timestamp = float(writer.duration_s)
            clip_duration_s, clip_stats = await _append_video_segment(
                writer=writer,
                video=segment.video,
                clip_from=video_from_timestamp_s(segment.video),
                clip_to=video_to_timestamp_s(segment.video),
                video_config=self.video_config,
                stats_config=self.stats_config,
            )
            writer.duration_s = from_timestamp + float(clip_duration_s)
            if writer.output_file is not None:
                writer.output_file.flush()
            if writer.stream is None:
                raise RuntimeError("Video writer stream was not initialized")

            completed_segments.append(
                _CompletedVideoSegment(
                    episode_index=segment.episode_index,
                    video_key=self.video_key,
                    video_meta={
                        f"videos/{self.video_key}/chunk_index": self.chunk_key,
                        f"videos/{self.video_key}/file_index": writer.file_idx,
                        f"videos/{self.video_key}/from_timestamp": from_timestamp,
                        f"videos/{self.video_key}/to_timestamp": writer.duration_s,
                    },
                    video_stats=clip_stats,
                )
            )
            feature = _video_feature(
                fps=writer.fps,
                height=int(writer.stream.height),
                width=int(writer.stream.width),
                codec=self.video_config.codec,
                pix_fmt=self.video_config.pix_fmt,
            )
            self._probe_cache.cleanup_video(segment.video)
            if writer.size_bytes >= self.video_bytes_limit:
                self._rotate_writer()

        return _CompletedVideoRun(
            video_key=self.video_key,
            feature=feature,
            segments=completed_segments,
        )

    async def _ensure_transcode_writer(self, video: Video) -> VideoTrackWriter:
        writer = self._writer
        if isinstance(writer, VideoTrackWriter):
            return writer
        if writer is not None:
            raise RuntimeError("Expected transcoder writer state")

        writer = VideoTrackWriter.open(
            folder=self.folder,
            output_rel=self._current_output_rel,
            chunk_key=self.chunk_key,
            video_key=self.video_key,
            file_idx=self._next_file_index,
            fps=await _resolve_video_fps(
                video=video,
                default_fps=self.default_fps,
                video_key=self.video_key,
            ),
            config=self.video_config,
        )
        self._writer = writer
        return writer

    def _close_writer(self) -> None:
        if self._writer is None:
            return
        self._writer.close()
        self._writer = None

    def _rotate_writer(self) -> None:
        writer = self._writer
        if writer is None:
            return
        writer.close()
        self._writer = None
        self._next_file_index += 1

    @property
    def _current_output_rel(self) -> str:
        return _format_video_path(
            chunk_key=self.chunk_key,
            video_key=self.video_key,
            file_idx=self._next_file_index,
        )


def _completed_remux_run(
    *,
    video_key: str,
    chunk_key: str,
    run: _PendingVideoRun,
    probe: _VideoSourceProbe,
    file_idx: int,
    base_duration_s: float,
    probe_cache: _VideoProbeCache,
    video_config: "LeRobotVideoConfig",
) -> _CompletedVideoRun:
    run_start_s = video_from_timestamp_s(run.segments[0].video)
    segments: list[_CompletedVideoSegment] = []
    for segment in run.segments:
        segment_to = video_to_timestamp_s(segment.video)
        if segment_to is None:
            raise ValueError(
                f"Video segment for {video_uri(segment.video)!r} is missing an end time"
            )
        segments.append(
            _CompletedVideoSegment(
                episode_index=segment.episode_index,
                video_key=video_key,
                video_meta={
                    f"videos/{video_key}/chunk_index": chunk_key,
                    f"videos/{video_key}/file_index": file_idx,
                    f"videos/{video_key}/from_timestamp": (
                        base_duration_s
                        + video_from_timestamp_s(segment.video)
                        - run_start_s
                    ),
                    f"videos/{video_key}/to_timestamp": (
                        base_duration_s + segment_to - run_start_s
                    ),
                },
                video_stats=dict(segment.source_stats or {}),
            )
        )
        probe_cache.cleanup_video(segment.video)

    return _CompletedVideoRun(
        video_key=video_key,
        feature=_video_feature(
            fps=int(probe.fps),
            height=int(probe.height),
            width=int(probe.width),
            codec=probe.codec or video_config.codec,
            pix_fmt=probe.pix_fmt or video_config.pix_fmt,
        ),
        segments=segments,
    )


__all__ = [
    "DEFAULT_VIDEO_PATH",
    "LeRobotVideoWriter",
    "_CompletedVideoRun",
    "_CompletedVideoSegment",
    "_PreparedVideoRun",
]
