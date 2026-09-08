from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import cloudpickle
import pytest

import refiner as mdr
from refiner.video import blocks, nvenc


def _stream(**patch):
    return {
        "codec_name": "h264",
        "pix_fmt": "yuv420p",
        "color_range": "tv",
        "width": 640,
        "height": 480,
        "nb_frames": "34",
        **patch,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cq": -1},
        {"cq": 52},
        {"gop": 0},
        {"gpu": -1},
        {"max_width": 1},
        {"max_height": 0},
        {"cpu_threads": 0},
        {"gop": 1.5},
        {"preset": "fast"},
        {"decode": "invalid"},
        {"audio": "copy"},
        {"timeout_s": float("nan")},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        nvenc.NVENCConfig(**kwargs)


def test_gpu_path_has_no_host_download_or_frame_rate_conversion():
    command = nvenc._command(
        Path("input.mp4"), Path("output.mp4"), _stream(), nvenc.NVENCConfig(gpu=1)
    )
    assert command[command.index("-hwaccel_device") + 1] == "1"
    assert command[command.index("-gpu") + 1] == "1"
    assert "scale_cuda=640:480:format=yuv420p:passthrough=0" in command
    assert command[command.index("-c:v") + 1] == "h264_nvenc"
    assert command[command.index("-preset") + 1] == "p1"
    assert command[command.index("-fps_mode") + 1] == "passthrough"
    assert command[command.index("-enc_time_base") + 1] == "demux"
    assert "hwdownload" not in " ".join(command)
    assert "-r" not in command
    assert command[command.index("-g") + 1] == "17"
    assert command[command.index("-bf") + 1] == "0"


@pytest.mark.parametrize(
    "stream",
    [
        _stream(codec_name="mpeg4"),
        _stream(codec_name="mjpeg", pix_fmt="yuvj420p"),
        _stream(color_range="pc"),
        _stream(pix_fmt="yuv444p"),
    ],
)
def test_auto_selects_cpu_conversion_but_always_gpu_encoding(stream):
    command = nvenc._command(Path("in"), Path("out"), stream, nvenc.NVENCConfig())
    assert "-hwaccel" not in command
    assert "scale=640:480:out_range=tv,format=yuv420p" in command
    assert command[command.index("-c:v") + 1] == "h264_nvenc"


def test_hdr_and_forced_cuda_full_range_are_rejected():
    for stream, config in [
        (_stream(color_transfer="smpte2084"), nvenc.NVENCConfig()),
        (_stream(color_range="pc"), nvenc.NVENCConfig(decode="cuda")),
    ]:
        with pytest.raises(ValueError):
            nvenc._command(Path("in"), Path("out"), stream, config)


@pytest.mark.parametrize(
    "width,height,expected",
    [
        (3840, 2160, (1920, 1080)),
        (1080, 1920, (606, 1080)),
        (641, 481, (640, 480)),
        (320, 240, (320, 240)),
    ],
)
def test_dimensions_bound_even_without_upscaling(width, height, expected):
    assert (
        nvenc._dimensions(_stream(width=width, height=height), nvenc.NVENCConfig())
        == expected
    )


def _fake_encoder(monkeypatch, *, output_stream=None, failure=False):
    commands = []

    async def run(command, timeout):
        commands.append(command)
        if command[0] == "ffprobe":
            stream = output_stream if command[-1].endswith("encoded.mp4") else _stream()
            return json.dumps({"streams": [stream or _stream()]}).encode()
        if failure:
            raise RuntimeError("NVENC unavailable")
        Path(command[-1]).write_bytes(b"validated video")
        return b""

    monkeypatch.setattr(nvenc, "_run", run)
    return commands


def test_transcode_reads_local_input_and_publishes_after_validation(
    tmp_path, monkeypatch
):
    commands = _fake_encoder(monkeypatch)
    source = tmp_path / "input.mp4"
    source.write_bytes(b"local input")
    target = tmp_path / "output.mp4"
    video = asyncio.run(nvenc.transcode_video(source, target))
    assert video.data_file.path == str(target)
    assert target.read_bytes() == b"validated video"
    assert commands[0][-1] == str(source)
    assert set(tmp_path.iterdir()) == {source, target}


@pytest.mark.parametrize(
    "remote",
    [
        "s3://bucket/video.mp4",
        "gs://bucket/video.mp4",
        "https://example.com/video.mp4",
        "memory://video.mp4",
        "simplecache::s3://bucket/video.mp4",
    ],
)
def test_remote_inputs_and_outputs_fail_before_filesystem_or_encoder_access(
    tmp_path, monkeypatch, remote
):
    def no_resolution(self):
        raise AssertionError("must not resolve a remote filesystem")

    async def no_encode(*args, **kwargs):
        raise AssertionError("must not start FFmpeg")

    monkeypatch.setattr(mdr.io.DataFile, "_resolve", no_resolution)
    monkeypatch.setattr(nvenc, "_run", no_encode)
    with pytest.raises(ValueError, match="local files only"):
        asyncio.run(nvenc.transcode_video(remote, tmp_path / "out.mp4"))
    # Remote destinations can be tested directly without resolving a local source.
    with pytest.raises(ValueError, match="local files only"):
        nvenc._local_path(remote)
    with pytest.raises(ValueError, match="local files only"):
        asyncio.run(nvenc.encode_image_sequence([remote], tmp_path / "out.mp4", fps=30))
    with pytest.raises(ValueError, match="local files only"):
        mdr.video.transcode_videos(output_folder=remote, video_key="video")
    with pytest.raises(ValueError, match="local files only"):
        mdr.video.encode_image_sequences(
            output_folder=remote, images_key="images", fps=30
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("source_kind", ["path", "data_file", "video_file"])
def test_local_wrappers_and_file_urls_are_supported(tmp_path, monkeypatch, source_kind):
    _fake_encoder(monkeypatch)
    source = tmp_path / "input.mp4"
    source.write_bytes(b"input")
    value = source
    if source_kind == "data_file":
        value = mdr.io.DataFile.resolve(source)
    elif source_kind == "video_file":
        value = mdr.video.VideoFile(mdr.io.DataFile.resolve(source))
    target = tmp_path / "output.mp4"
    asyncio.run(nvenc.transcode_video(value, target.as_uri()))
    assert target.exists()


def test_remote_destinations_rejected_before_output_creation(tmp_path, monkeypatch):
    commands = _fake_encoder(monkeypatch)
    with pytest.raises(ValueError, match="local files only"):
        asyncio.run(
            nvenc.transcode_video(tmp_path / "input.mp4", "s3://bucket/out.mp4")
        )
    with pytest.raises(ValueError, match="local files only"):
        asyncio.run(
            nvenc.encode_image_sequence(
                [tmp_path / "input.jpg"], "gs://bucket/out.mp4", fps=30
            )
        )
    assert not commands
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "output_stream",
    [
        _stream(nb_frames="33"),
        _stream(nb_frames="0"),
        _stream(pix_fmt="yuvj420p"),
        _stream(width=320),
        _stream(codec_name="mpeg4"),
        _stream(color_range="pc"),
    ],
)
def test_invalid_output_is_not_published(tmp_path, monkeypatch, output_stream):
    _fake_encoder(monkeypatch, output_stream=output_stream)
    target = tmp_path / "output.mp4"
    with pytest.raises(RuntimeError):
        asyncio.run(nvenc.transcode_video(tmp_path / "input.mp4", target))
    assert not target.exists()
    assert not list(tmp_path.iterdir())


def test_failure_cleans_temporary_files_and_never_retries_cpu(tmp_path, monkeypatch):
    commands = _fake_encoder(monkeypatch, failure=True)
    with pytest.raises(RuntimeError, match="NVENC unavailable"):
        asyncio.run(
            nvenc.transcode_video(tmp_path / "input.mp4", tmp_path / "output.mp4")
        )
    assert len([c for c in commands if c[0] == "ffmpeg"]) == 1
    assert not list(tmp_path.iterdir())


def test_existing_output_and_clipped_video_are_rejected(tmp_path):
    target = tmp_path / "output.mp4"
    target.write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        asyncio.run(nvenc.transcode_video("input.mp4", target))
    assert target.read_bytes() == b"keep me"
    clipped = mdr.video.VideoFile(
        mdr.io.DataFile.resolve("input.mp4"), to_timestamp_s=1
    )
    with pytest.raises(ValueError, match="clipped"):
        asyncio.run(nvenc.transcode_video(clipped, target))


def test_images_keep_explicit_order_and_verify_count(tmp_path, monkeypatch):
    sources = [tmp_path / "z.jpg", tmp_path / "a.jpg"]
    for index, path in enumerate(sources):
        path.write_bytes(bytes([index]))
    original_run = _fake_encoder(monkeypatch, output_stream=_stream(nb_frames="2"))
    fake_run = nvenc._run

    async def inspect(command, timeout):
        if command[0] == "ffmpeg":
            pattern = Path(command[command.index("-i") + 1])
            assert (pattern.parent / "00000000.jpg").read_bytes() == b"\x00"
            assert (pattern.parent / "00000001.jpg").read_bytes() == b"\x01"
            assert "-hwaccel" not in command
            assert command[command.index("-framerate") + 1] == "30"
        return await fake_run(command, timeout)

    monkeypatch.setattr(nvenc, "_run", inspect)
    asyncio.run(nvenc.encode_image_sequence(sources, tmp_path / "out.mp4", fps=30))
    assert len(original_run) == 3
    assert not list(tmp_path.glob("refiner-nvenc-*"))


def test_blocks_are_serializable_and_use_pipeline_concurrency(tmp_path, monkeypatch):
    active = peak = 0

    async def transcode(source, destination, *, config):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return mdr.video.VideoFile(mdr.io.DataFile.resolve(destination))

    monkeypatch.setattr(blocks, "transcode_video", transcode)
    block = mdr.video.transcode_videos(video_key="video", output_folder=tmp_path)
    assert callable(cloudpickle.loads(cloudpickle.dumps(block)))
    rows = (
        mdr.from_items([{"video": f"{i}.mp4", "id": i} for i in range(6)])
        .map_async(block, max_in_flight=3)
        .take(6)
    )
    assert peak == 3
    assert [row["id"] for row in rows] == list(range(6))
    assert len({row["transcoded_video"] for row in rows}) == 6


def test_subprocess_reports_stderr():
    with pytest.raises(RuntimeError, match="encoder failed"):
        asyncio.run(
            nvenc._run(
                [
                    sys.executable,
                    "-c",
                    "import sys; sys.stderr.write('encoder failed'); sys.exit(2)",
                ],
                5,
            )
        )


def test_blocks_can_be_imported_first_and_have_json_pipeline_plans(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import refiner as m; m.video.transcode_videos(video_key='v', output_folder='/tmp/out')",
        ],
        check=True,
    )
    from refiner.pipeline.planning import compile_pipeline_plan

    for block in [
        mdr.video.transcode_videos(video_key="video", output_folder=tmp_path),
        mdr.video.encode_image_sequences(
            images_key="images", output_folder=tmp_path, fps=30
        ),
    ]:
        pipeline = mdr.from_items([]).map_async(block, max_in_flight=4)
        encoded = json.dumps(compile_pipeline_plan(pipeline))
        assert "output_folder" in encoded
        assert callable(cloudpickle.loads(cloudpickle.dumps(block)))


@pytest.mark.parametrize("cancel", [False, True])
def test_subprocess_timeout_and_cancellation_reap_child(monkeypatch, cancel):
    real_spawn = asyncio.create_subprocess_exec
    children = []

    async def spawn(*args, **kwargs):
        process = await real_spawn(*args, **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)

    async def execute():
        task = asyncio.create_task(
            nvenc._run(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                0.1 if not cancel else 60,
            )
        )
        if cancel:
            while not children:
                await asyncio.sleep(0.001)
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else asyncio.TimeoutError):
            await task

    asyncio.run(execute())
    assert len(children) == 1 and children[0].returncode is not None


@pytest.mark.skipif(
    os.environ.get("REFINER_TEST_NVENC") != "1",
    reason="set REFINER_TEST_NVENC=1 on an NVIDIA GPU worker",
)
@pytest.mark.parametrize("decode", ["cuda", "cpu"])
def test_real_nvenc_frame_timing_keyframes_and_faststart(tmp_path, decode):
    assert shutil.which("ffmpeg") and shutil.which("ffprobe")
    source, output = tmp_path / "in.mp4", tmp_path / "out.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=30",
            "-frames:v",
            "68",
            "-vf",
            "setpts='if(lt(N,34),N,N+4)/(30*TB)'",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            str(source),
        ],
        check=True,
    )
    asyncio.run(
        nvenc.transcode_video(source, output, config=nvenc.NVENCConfig(decode=decode))
    )

    def frames(path):
        return json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_frames",
                    "-show_entries",
                    "frame=best_effort_timestamp_time,key_frame",
                    "-of",
                    "json",
                    str(path),
                ]
            )
        )["frames"]

    before, after = frames(source), frames(output)
    assert len(before) == len(after) == 68
    assert [float(f["best_effort_timestamp_time"]) for f in after] == pytest.approx(
        [float(f["best_effort_timestamp_time"]) for f in before], abs=1e-6
    )
    assert [i for i, frame in enumerate(after) if frame["key_frame"]] == [0, 17, 34, 51]
    payload = output.read_bytes()
    assert payload.index(b"moov") < payload.index(b"mdat")


@pytest.mark.skipif(
    os.environ.get("REFINER_TEST_NVENC") != "1", reason="requires NVIDIA GPU"
)
def test_real_image_nvenc_preserves_order_and_limited_range(tmp_path):
    from PIL import Image
    import numpy as np
    import av

    images = []
    for index, value in enumerate([32, 128, 224]):
        path = tmp_path / f"{2 - index}.jpg"
        Image.fromarray(np.full((240, 320, 3), value, dtype=np.uint8)).save(path)
        images.append(path)
    output = tmp_path / "images.mp4"
    asyncio.run(nvenc.encode_image_sequence(images, output, fps=30))
    with av.open(str(output)) as container:
        frames = list(container.decode(video=0))
    assert len(frames) == 3
    assert [
        float(f.to_ndarray(format="rgb24").mean()) for f in frames
    ] == pytest.approx([32, 128, 224], abs=5)
    assert frames[0].format.name == "yuv420p"
