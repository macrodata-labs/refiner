---
title: "GPU video transcoding"
description: "Prepare H.264 training videos and ordered image sequences with NVIDIA NVENC"
---

Use `mdr.video.transcode_videos` to transcode whole video files, or
`mdr.video.encode_image_sequences` to turn ordered JPEG or PNG sequences into
videos. Both are async map blocks that always encode with NVIDIA NVENC.

## Transcode videos

```python
import refiner as mdr

pipeline = mdr.read_videos("/data/source/*.mp4", file_path_column="video").map_async(
    mdr.video.transcode_videos(
        video_key="video",
        output_folder="/data/encoded",
        output_key="training_video",
        config=mdr.video.NVENCConfig(),
    ),
    max_in_flight=4,
    preserve_order=False,
    dtypes={"training_video": mdr.datatype.video_path()},
)
```

Input values can be paths, `mdr.io.DataFile` objects, or whole
`mdr.video.VideoFile` objects. Each output row keeps its original fields and adds
the new MP4 path. Use a durable shared `output_folder` for cloud pipelines,
for example an S3 bucket accessible to the worker. Clipped views are rejected;
materialize the desired clip before using this block.

Request one L4 per worker on the launch call and begin with three or four in-flight rows.
Each row uses one FFmpeg process. `max_in_flight` bounds decoding, encoding, and
file staging together; it is per worker, not a global limit. Start with 8 CPUs
and 8 GiB RAM per worker and provision scratch disk for the concurrently staged
inputs and outputs. GPU allocation belongs on the launcher, not the block.

## Encode ordered images

```python
import refiner as mdr

pipeline = mdr.from_items([
    {"episode_id": "episode-1", "images": ["/data/0001.jpg", "/data/0002.jpg"]},
]).map_async(
    mdr.video.encode_image_sequences(
        images_key="images",
        output_folder="/data/encoded",
        fps=30,
    ),
    max_in_flight=4,
    dtypes={"encoded_video": mdr.datatype.video_path()},
)
```

Supply a nonempty list of uniformly sized files, all JPEG or all PNG. List order
is frame order; the block does not sort filenames. Image decode and conversion
to limited-range YUV happen on the CPU; encoding happens on the GPU. No Python
RGB frame arrays are created. The output frame count must match the list length.

## Output and tuning

| Setting | Default | Behavior |
| --- | --- | --- |
| Codec | H.264 NVENC | 8-bit `yuv420p`, limited range, MP4 faststart |
| `preset` | `"p1"` | Fastest NVENC preset; `"p6"` spends more work on compression |
| `cq` | `29` | VBR constant-quality target; lower values request higher quality |
| `gop` | `17` | Fixed keyframe interval in frames, no scene cuts or B-frames |
| Bounds | 1920 × 1080 | Keep aspect ratio approximately, round down to even dimensions, never upscale |
| `decode` | `"auto"` | CUDA for common 8-bit 4:2:0 H.264/HEVC/VP9/AV1; CPU conversion otherwise |
| `audio` | `"drop"` | Choose `"aac"` to retain the first audio stream as 128 kbps AAC |
| `gpu` | `0` | CUDA device ordinal as visible inside the worker |
| `cpu_threads` | `2` | Decoder thread budget per FFmpeg process |
| `timeout_s` | `3600` | Timeout for each FFmpeg/ffprobe subprocess |

Video frame timing passes through without an FPS conversion. Source timestamps
are normalized by FFmpeg's normal input handling; do not use output timestamps
as an absolute source clock. Image sequences receive a regular timeline at `fps`.
Automatic rotation is disabled so pixels remain in source annotation coordinates.
Resize associated keypoints, boxes, and intrinsics to match output dimensions;
these blocks do not modify annotation columns. HDR inputs require tone mapping
before this SDR block.

On the CUDA path, decode surfaces remain on the GPU through `scale_cuda` and
NVENC. Full-range inputs use CPU range conversion in `auto` mode. `decode="cpu"`
still uses NVENC for encoding. `decode="cuda"` forces hardware decoding and
reports unsupported input/build errors; it does not silently retry on the CPU.

The defaults prioritize throughput. Choosing `preset="p6"` restores the preset
used in earlier dataset converters, but does not promise bit-identical output:
this block disables lookahead, multipass, and B-frames and fixes the GOP.
CQ 29 is an encoder quality target, not an equivalence to x264 CRF 29.

Measure on your own videos before scaling worker count:

```bash
python examples/benchmark_video_nvenc.py /data/representative.mp4 --copies 8
```

The benchmark compares `p1` and `p6` with one through four concurrent lanes.
It reports aggregate frames per second including probing and local publication,
after a warmup. It excludes remote transfer and does not measure perceptual
quality. Short clips, image decoding, storage, and scene complexity can change
which configuration wins.

An L4 validation run on September 8, 2026 used 8 CPUs, 8 GiB RAM, NVIDIA driver
580.95.05, and FFmpeg 8.1. It encoded eight copies of a 600-frame, 1920×1080
`testsrc2` H.264 video for each configuration:

| Concurrent lanes | `p1` aggregate fps | `p6` aggregate fps |
| --- | --- | --- |
| 1 | 308.4 | 205.8 |
| 2 | 550.7 | 364.4 |
| 3 | 663.1 | 395.3 |
| 4 | 660.6 | 394.7 |

Three and four lanes performed similarly on the final implementation. These are
single-run synthetic local-file measurements, not a throughput guarantee for
datasets or remote storage. Earlier runs varied, so benchmark both settings on
your worker image and representative input.

## Worker setup and validation

Install a recent FFmpeg build (8.x is used for validation) and `ffprobe` on the
worker's `PATH`, or set `NVENCConfig(ffmpeg=..., ffprobe=...)`. The build must
include `h264_nvenc` and, for CUDA decoding/scaling, `scale_cuda`. Install a
compatible NVIDIA driver and expose the GPU's video capabilities to the
container. Installing Refiner's Python `video` extra alone does not install
these executables or the NVIDIA driver.

Run the hardware tests on the actual worker image:

```bash
REFINER_TEST_NVENC=1 pytest tests/test_video_nvenc.py -q
```

The CUDA and CPU-decode tests check frame count, a variable-rate timeline,
keyframe positions, and faststart MP4. The image test also checks frame order and
color conversion. All 40 tests passed on the L4 validation worker. Ordinary unit
tests run without a GPU.

Each encode checks codec, pixel format, dimensions, and positive output frame
count before publishing. It also compares frame counts when the source container
reports one; image sequence counts are always checked. This fast path does not
fully decode every source/output for a timestamp audit. Use the hardware tests
and dataset-specific validation when strict annotation alignment is required.

## Direct calls

```python
import refiner as mdr

# Inside an async function:
video = await mdr.video.transcode_video("source.mp4", "new.mp4")
video = await mdr.video.encode_image_sequence(
    ["0001.jpg", "0002.jpg"], "images.mp4", fps=30,
)
```

Direct calls return `mdr.video.VideoFile`. Existing destinations are rejected.
Local files publish atomically. For remote storage, use a unique destination
per invocation: object-store publication is not a cross-worker lock or transaction.
The map blocks generate unique names automatically. Retries can leave orphaned
objects from earlier attempts; these helpers do not implement dataset checkpoints
or garbage collection.

Very small images or extreme resize bounds can fall below the GPU encoder's
minimum supported dimensions. The block reports FFmpeg's error rather than
upscaling or switching to a software encoder.

## Internal Notes

Remote files are staged through `DataFile` so configured storage credentials
remain with fsspec. Local videos are read directly; local image sequences use
numbered links. Completed MP4s are validated on disk before upload. Timeout and
cancellation kill and reap FFmpeg; outstanding file copies finish before scratch
cleanup. The async block uses the existing pipeline window, without another pool
of GPU workers.
The H.264 bitstream filter explicitly signals limited range even when NVENC
omits that tag. Callable descriptions live in a lightweight module so video
blocks can be imported before the pipeline planner.

Spark's partition tasks and Beam/Dataflow's element processing could host an
external encoder, but would require their own scheduling integration. Hugging
Face Datasets offers process-based mapping. Daft and Ray Data provide GPU task
or actor scheduling; adding either scheduler here would duplicate Refiner's
worker and bounded async execution. Refiner keeps those controls and delegates
codec execution to FFmpeg. This also avoids transferring decoded frame arrays
through Python merely to feed an encoder.

The device-resident path and concurrent-session design follow
[NVIDIA's FFmpeg guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/ffmpeg-with-nvidia-gpu/index.html).
