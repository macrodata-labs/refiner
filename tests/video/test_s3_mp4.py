from __future__ import annotations

import io

import numpy as np
import pytest

from refiner.video.s3_mp4 import S3MultipartSeekableWriter, supports_s3_multipart
from refiner.video.transcode import TranscodeWriter, VideoTranscodeConfig

_PART_SIZE = 5 * 1024 * 1024


class _FakeS3FileSystem:
    protocol = "s3"

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.parts: dict[int, bytes] = {}
        self.objects: dict[tuple[str, str], bytes] = {}

    @staticmethod
    def split_path(path: str) -> tuple[str, str, None]:
        bucket, key = path.split("/", 1)
        return bucket, key, None

    def call_s3(self, method: str, *args, **kwargs):
        assert args == ({},)
        self.calls.append((method, kwargs))
        if method == "create_multipart_upload":
            return {"UploadId": "upload-1"}
        if method == "upload_part":
            self.parts[kwargs["PartNumber"]] = bytes(kwargs["Body"])
            return {"ETag": f"etag-{kwargs['PartNumber']}"}
        if method == "complete_multipart_upload":
            parts = kwargs["MultipartUpload"]["Parts"]
            self.objects[(kwargs["Bucket"], kwargs["Key"])] = b"".join(
                self.parts[part["PartNumber"]] for part in parts
            )
            return {}
        if method == "abort_multipart_upload":
            self.parts.clear()
            return {}
        if method == "put_object":
            self.objects[(kwargs["Bucket"], kwargs["Key"])] = bytes(kwargs["Body"])
            return {}
        raise AssertionError(method)


def test_s3_multipart_writer_rewrites_only_prefix_and_streams_later_parts() -> None:
    fs = _FakeS3FileSystem()
    writer = S3MultipartSeekableWriter(fs, "bucket/video.mp4")
    payload = bytearray(b"\0" * (_PART_SIZE * 2 + 23))
    writer.write(payload)
    writer.seek(36)
    writer.write(b"mdat")
    writer.seek(0, io.SEEK_END)
    writer.close()

    assert fs.objects[("bucket", "video.mp4")][36:40] == b"mdat"
    assert len(fs.objects[("bucket", "video.mp4")]) == len(payload)
    assert [method for method, _ in fs.calls] == [
        "create_multipart_upload",
        "upload_part",
        "upload_part",
        "upload_part",
        "upload_part",
        "complete_multipart_upload",
    ]
    assert [call[1]["PartNumber"] for call in fs.calls if call[0] == "upload_part"] == [
        1,
        2,
        1,
        3,
    ]
    uploaded_parts = [call[1] for call in fs.calls if call[0] == "upload_part"]
    assert uploaded_parts[0]["Body"][36:40] == b"\0\0\0\0"
    assert uploaded_parts[2]["Body"][36:40] == b"mdat"
    assert [
        part["PartNumber"] for part in fs.calls[-1][1]["MultipartUpload"]["Parts"]
    ] == [
        1,
        2,
        3,
    ]


def test_s3_multipart_writer_aborts_after_upload_failure() -> None:
    class FailingS3(_FakeS3FileSystem):
        def call_s3(self, method: str, *args, **kwargs):
            if method == "complete_multipart_upload":
                raise RuntimeError("completion failed")
            return super().call_s3(method, *args, **kwargs)

    fs = FailingS3()
    writer = S3MultipartSeekableWriter(fs, "bucket/video.mp4")
    writer.write(b"\0" * (_PART_SIZE + 1))

    with pytest.raises(RuntimeError, match="completion failed"):
        writer.close()

    assert fs.calls[-1][0] == "abort_multipart_upload"


def test_conventional_mp4_uses_moov_not_moof() -> None:
    fs = _FakeS3FileSystem()
    output = S3MultipartSeekableWriter(fs, "bucket/video.mp4")
    writer = TranscodeWriter.open_file(
        output_file=output,
        config=VideoTranscodeConfig(codec="h264"),
        fps=5,
        movflags=None,
    )
    writer.append_frame_arrays(
        [np.full((16, 16, 3), value, dtype=np.uint8) for value in range(10)]
    )
    writer.close()

    data = fs.objects[("bucket", "video.mp4")]
    assert b"moov" in data
    assert b"moof" not in data

    import av

    with av.open(io.BytesIO(data), mode="r") as container:
        stream = container.streams.video[0]
        assert stream.frames == 10
        assert float(stream.duration * stream.time_base) == pytest.approx(2.0)


def test_s3_capability_requires_s3fs_multipart_surface() -> None:
    assert supports_s3_multipart(_FakeS3FileSystem())
    assert not supports_s3_multipart(object())
