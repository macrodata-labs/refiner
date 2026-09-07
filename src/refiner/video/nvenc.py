from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Literal, TypeVar

from refiner.io import DataFile
from refiner.io.datafile import DataFileLike
from refiner.video.types import VideoFile

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class NVENCConfig:
    """H.264 settings for L4 dataset preparation; requires NVIDIA FFmpeg."""

    preset: Literal["p1", "p2", "p3", "p4", "p5", "p6", "p7"] = "p1"
    cq: int = 29
    gop: int = 17
    max_width: int = 1920
    max_height: int = 1080
    decode: Literal["auto", "cuda", "cpu"] = "auto"
    gpu: int = 0
    cpu_threads: int = 2
    audio: Literal["drop", "aac"] = "drop"
    timeout_s: float = 3600
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"

    def __post_init__(self) -> None:
        if self.preset not in {f"p{i}" for i in range(1, 8)}:
            raise ValueError("preset must be p1 through p7")
        for name, minimum in (
            ("cq", 0),
            ("gop", 1),
            ("max_width", 2),
            ("max_height", 2),
            ("gpu", 0),
            ("cpu_threads", 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.cq > 51:
            raise ValueError("cq must be <= 51")
        if self.decode not in {"auto", "cuda", "cpu"}:
            raise ValueError("decode must be auto, cuda, or cpu")
        if self.audio not in {"drop", "aac"}:
            raise ValueError("audio must be drop or aac")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be finite and > 0")


async def _run(command: list[str], timeout: float) -> bytes:
    # Files avoid unbounded pipe buffers and retain the useful end of FFmpeg errors.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            process = await asyncio.create_subprocess_exec(
                *command, stdin=asyncio.subprocess.DEVNULL, stdout=stdout, stderr=stderr
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"{command[0]} is missing; install FFmpeg/ffprobe with NVENC and scale_cuda"
            ) from exc
        try:
            await asyncio.wait_for(process.wait(), timeout)
        except asyncio.TimeoutError as exc:
            raise asyncio.TimeoutError(
                f"{command[0]} exceeded {timeout} seconds"
            ) from exc
        finally:
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
        if process.returncode:
            stderr.seek(max(0, stderr.tell() - 16384))
            detail = stderr.read().decode(errors="replace")
            raise RuntimeError(
                f"{command[0]} exited with {process.returncode}: {detail}"
            )
        stdout.seek(0)
        return stdout.read()


async def _io(fn: Callable[..., _T], *args: Any) -> _T:
    # A cancelled copy must finish before its temporary directory can be removed.
    task = asyncio.create_task(asyncio.to_thread(fn, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


async def _probe(path: Path, config: NVENCConfig) -> dict[str, Any]:
    payload = await _run(
        [
            config.ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        config.timeout_s,
    )
    streams = json.loads(payload).get("streams", [])
    if (
        not streams
        or int(streams[0].get("width", 0)) < 2
        or int(streams[0].get("height", 0)) < 2
    ):
        raise ValueError("source must contain a video stream at least 2x2 pixels")
    return streams[0]


def _dimensions(stream: dict[str, Any], config: NVENCConfig) -> tuple[int, int]:
    width, height = int(stream["width"]), int(stream["height"])
    ratio = min(1.0, config.max_width / width, config.max_height / height)
    return max(2, int(width * ratio) // 2 * 2), max(2, int(height * ratio) // 2 * 2)


def _command(
    source: Path,
    output: Path,
    stream: dict[str, Any],
    config: NVENCConfig,
    *,
    fps: float | None = None,
) -> list[str]:
    if stream.get("color_transfer") in {"smpte2084", "arib-std-b67"}:
        raise ValueError("HDR inputs require tone mapping before SDR H.264 transcoding")
    width, height = _dimensions(stream, config)
    # Conservative auto path: unusual chroma, full range and legacy codecs use
    # software decode/conversion, but encoding is always NVENC. Never retry on CPU.
    cuda = fps is None and (
        config.decode == "cuda"
        or (
            config.decode == "auto"
            and stream.get("codec_name") in {"h264", "hevc", "vp9", "av1"}
            and stream.get("pix_fmt") in {"yuv420p", "nv12"}
            and stream.get("color_range") != "pc"
        )
    )
    if cuda and stream.get("color_range") == "pc":
        raise ValueError("full-range video requires decode='cpu' for range conversion")
    command = [
        config.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-noautorotate",
        "-threads",
        str(config.cpu_threads),
        "-filter_threads",
        "1",
    ]
    if cuda:
        command += [
            "-hwaccel",
            "cuda",
            "-hwaccel_device",
            str(config.gpu),
            "-hwaccel_output_format",
            "cuda",
            "-extra_hw_frames",
            "8",
        ]
    if fps is not None:
        command += ["-framerate", str(fps), "-start_number", "0"]
    command += ["-i", str(source), "-map", "0:v:0"]
    if config.audio == "aac" and fps is None:
        command += ["-map", "0:a:0?", "-c:a", "aac", "-b:a", "128k"]
    else:
        command += ["-an"]
    if cuda:
        # passthrough=0 gives the encoder its own frames, releasing decoder surfaces.
        filters = f"scale_cuda={width}:{height}:format=yuv420p:passthrough=0"
    else:
        filters = f"scale={width}:{height}:out_range=tv,format=yuv420p"
    command += [
        "-vf",
        filters,
        "-c:v",
        "h264_nvenc",
        "-gpu",
        str(config.gpu),
        "-preset",
        config.preset,
        "-tune",
        "hq",
        "-rc",
        "vbr",
        "-cq",
        str(config.cq),
        "-b:v",
        "0",
        "-multipass",
        "disabled",
        "-rc-lookahead",
        "0",
        "-bf",
        "0",
        "-g",
        str(config.gop),
        "-keyint_min",
        str(config.gop),
        "-no-scenecut",
        "1",
        "-strict_gop",
        "1",
        "-color_range",
        "tv",
        "-bsf:v",
        "h264_metadata=video_full_range_flag=0",
        "-fps_mode",
        "passthrough",
        "-enc_time_base",
        "demux",
        "-map_metadata",
        "-1",
        "-metadata:s:v:0",
        "rotate=0",
        "-movflags",
        "+faststart",
        str(output),
    ]
    return command


async def _stage(source: DataFile, directory: Path, name: str) -> Path:
    if source.is_local:
        return Path(source.path).resolve()
    local = directory / name
    await _io(source.copy, local)
    return local


async def _encode(
    source: Path,
    destination: DataFile,
    directory: Path,
    config: NVENCConfig,
    *,
    fps: float | None = None,
    probe_source: Path | None = None,
    expected_frames: int | None = None,
) -> VideoFile:
    before = await _probe(probe_source or source, config)
    output = directory / "encoded.mp4"
    await _run(_command(source, output, before, config, fps=fps), config.timeout_s)
    after = await _probe(output, config)
    if (after.get("codec_name"), after.get("pix_fmt")) != ("h264", "yuv420p"):
        raise RuntimeError("NVENC output must be H.264 yuv420p")
    if after.get("color_range") != "tv":
        raise RuntimeError("NVENC output must use limited color range")
    if (int(after["width"]), int(after["height"])) != _dimensions(before, config):
        raise RuntimeError("NVENC output dimensions do not match the requested bounds")
    expected = (
        expected_frames if expected_frames is not None else before.get("nb_frames")
    )
    actual = after.get("nb_frames")
    if actual is None or actual == "N/A" or int(actual) <= 0:
        raise RuntimeError("NVENC produced no verifiable video frames")
    if expected not in (None, "N/A") and int(expected) != int(actual):
        raise RuntimeError(f"NVENC frame count changed: {expected} -> {actual}")
    if destination.is_local:
        # Same-filesystem link publishes atomically and refuses existing files.
        await _io(os.link, output, destination.path)
    else:
        await _io(DataFile.resolve(output).copy, destination)
    return VideoFile(destination)


def _stage_images(sources: list[DataFile], directory: Path, suffix: str) -> None:
    # One I/O task per sequence avoids a thread-pool roundtrip per image.
    for index, item in enumerate(sources):
        link = directory / f"{index:08d}{suffix}"
        if item.is_local:
            os.symlink(Path(item.path).resolve(), link)
        else:
            item.copy(link)


async def _output_directory(destination: DataFile) -> str | None:
    if await _io(destination.exists):
        raise FileExistsError(str(destination))
    if not destination.is_local:
        return None
    parent = Path(destination.path).resolve().parent
    await _io(parent.mkdir, 0o777, True, True)
    return str(parent)


async def transcode_video(
    source: DataFileLike | VideoFile,
    destination: DataFileLike,
    *,
    config: NVENCConfig | None = None,
) -> VideoFile:
    """Transcode a whole video to a new MP4 using NVENC, asynchronously.

    Remote inputs are staged on disk. Existing outputs and clipped views are
    rejected. No Python frame materialization or silent software encode fallback.
    """
    if isinstance(source, VideoFile):
        if source.from_timestamp_s is not None or source.to_timestamp_s is not None:
            raise ValueError("transcode_video accepts whole videos, not clipped views")
        source = source.data_file
    target = DataFile.resolve(destination)
    parent = await _output_directory(target)
    with tempfile.TemporaryDirectory(prefix="refiner-nvenc-", dir=parent) as temp:
        directory = Path(temp)
        local = await _stage(DataFile.resolve(source), directory, "input")
        return await _encode(local, target, directory, config or NVENCConfig())


async def encode_image_sequence(
    images: Sequence[DataFileLike],
    destination: DataFileLike,
    *,
    fps: float,
    config: NVENCConfig | None = None,
) -> VideoFile:
    """Encode an explicitly ordered, uniformly sized JPEG or PNG sequence on GPU."""
    if not images or isinstance(images, (str, bytes)):
        raise ValueError("images must be a nonempty ordered sequence of files")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and > 0")
    sources = [DataFile.resolve(item) for item in images]
    suffixes = [
        Path(item.path).suffix.lower().replace(".jpeg", ".jpg") for item in sources
    ]
    if len(set(suffixes)) != 1 or suffixes[0] not in {".jpg", ".png"}:
        raise ValueError("images must be all JPEG or all PNG")
    target = DataFile.resolve(destination)
    parent = await _output_directory(target)
    with tempfile.TemporaryDirectory(prefix="refiner-nvenc-", dir=parent) as temp:
        directory = Path(temp)
        await _io(_stage_images, sources, directory, suffixes[0])
        return await _encode(
            directory / f"%08d{suffixes[0]}",
            target,
            directory,
            config or NVENCConfig(),
            fps=fps,
            probe_source=directory / f"00000000{suffixes[0]}",
            expected_frames=len(sources),
        )
