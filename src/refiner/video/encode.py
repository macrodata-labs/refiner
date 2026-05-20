from __future__ import annotations

import io

import numpy as np

from typing import IO

from refiner.utils import check_required_dependencies
from refiner.video.arrays import rgb24_frame_array


class NonClosingBytesIO(io.BytesIO):
    def close(self) -> None:
        pass


def encode_frame_array_to_mp4(frames: np.ndarray, *, fps: int) -> bytes:
    check_required_dependencies("video frame array encoding", ["av"], dist="video")
    import av

    output = NonClosingBytesIO()
    container = av.open(output, mode="w", format="mp4")
    try:
        stream = container.add_stream("mpeg4", rate=int(fps))
        stream.width = int(frames.shape[2])
        stream.height = int(frames.shape[1])
        stream.pix_fmt = "yuv420p"

        for frame_array in frames:
            frame = av.VideoFrame.from_ndarray(
                rgb24_frame_array(frame_array),
                format="rgb24",
            )
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode(None):
            container.mux(packet)
    finally:
        container.close()
    return output.getvalue()


def open_frame_array_as_mp4(frames: np.ndarray, *, fps: int) -> IO[bytes]:
    return io.BytesIO(encode_frame_array_to_mp4(frames, fps=fps))


def open_video_bytes(data: bytes) -> IO[bytes]:
    return io.BytesIO(data)


__all__ = [
    "NonClosingBytesIO",
    "encode_frame_array_to_mp4",
    "open_frame_array_as_mp4",
    "open_video_bytes",
]
