from __future__ import annotations

import io

from refiner.video.remux import (
    RemuxWriter,
    prepared_source_is_remuxable,
    prepare_video_source,
)
from refiner.video.transcode import TranscodeWriter, VideoTranscodeConfig
from refiner.video.types import VideoFile


class _NonClosingBytesIO(io.BytesIO):
    def close(self) -> None:
        pass


async def export_clip_bytes(
    video: VideoFile,
    *,
    stream_key: str = "clip",
    force_transcode: bool = False,
    transcode_config: VideoTranscodeConfig | None = None,
) -> bytes:
    config = transcode_config or VideoTranscodeConfig()
    prepared = await prepare_video_source(cache_key=stream_key, video=video)
    output_file = _NonClosingBytesIO()
    try:
        if not force_transcode and prepared_source_is_remuxable(prepared):
            probe = prepared.probe
            if probe is None:
                raise RuntimeError("Remux path selected without a source probe")
            writer = RemuxWriter.open_file(
                output_file=output_file,
                probe=probe,
                movflags=None,
            )
            writer.append_prepared_video(prepared)
        else:
            fps = (
                int(prepared.probe.fps)
                if prepared.probe is not None and prepared.probe.fps is not None
                else None
            )
            if fps is None:
                raise ValueError("Prepared transcode item is missing FPS")
            writer = TranscodeWriter.open_file(
                output_file=output_file,
                config=config,
                fps=fps,
                movflags=None,
            )
            writer.append_prepared_video(
                prepared_source=prepared,
            )
        writer.close()
        return output_file.getvalue()
    finally:
        prepared.close()


__all__ = ["export_clip_bytes"]
