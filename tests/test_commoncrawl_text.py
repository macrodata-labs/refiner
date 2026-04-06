from __future__ import annotations

import gzip
import io
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
from warcio.statusandheaders import StatusAndHeaders
from warcio.warcwriter import WARCWriter

import refiner as mdr
from refiner.pipeline.data.row import Row
from refiner.pipeline.data.shard import FilePartsDescriptor
from refiner.worker.metrics.context import set_active_user_metrics_emitter


class _RecordingEmitter:
    def __init__(self) -> None:
        self.counters: list[dict[str, object]] = []

    def emit_user_counter(self, **kwargs) -> None:
        self.counters.append(kwargs)

    def emit_user_gauge(self, **kwargs) -> None:
        del kwargs

    def register_user_gauge(self, **kwargs) -> None:
        del kwargs

    def emit_user_histogram(self, **kwargs) -> None:
        del kwargs

    def force_flush_user_metrics(self) -> None:
        return None

    def force_flush_resource_metrics(self) -> None:
        return None

    def force_flush_logs(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


def _write_gzip_lines(path: Path, lines: list[str]) -> None:
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        handle.write(payload)


def _write_warc_gz(path: Path) -> list[tuple[int, int]]:
    return _write_warc_gz_records(
        path,
        [
            ("https://example.com/a", "<html>a</html>"),
            ("https://example.com/b", "<html>b</html>"),
        ],
    )


def _write_warc_gz_records(
    path: Path,
    records: list[
        tuple[str, str]
        | tuple[str, bytes, list[tuple[str, str]], dict[str, str] | None]
    ],
) -> list[tuple[int, int]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets: list[tuple[int, int]] = []
    with path.open("wb") as raw:
        writer = WARCWriter(raw, gzip=True)
        for spec in records:
            start = raw.tell()
            if len(spec) == 2 and isinstance(spec[1], str):
                url, body = cast(tuple[str, str], spec)
                payload = io.BytesIO(body.encode("utf-8"))
                http_headers = [("Content-Type", "text/html; charset=utf-8")]
                warc_headers_dict = None
            else:
                url, body, http_headers, warc_headers_dict = cast(
                    tuple[str, bytes, list[tuple[str, str]], dict[str, str] | None],
                    spec,
                )
                payload = io.BytesIO(body)
            record = writer.create_warc_record(
                url,
                "response",
                payload=payload,
                http_headers=StatusAndHeaders(
                    "200 OK",
                    http_headers,
                    protocol="HTTP/1.0",
                ),
                warc_headers_dict=warc_headers_dict,
            )
            writer.write_record(record)
            offsets.append((start, raw.tell() - start))
    return offsets


def _write_wet_gz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        writer = WARCWriter(raw, gzip=True)
        record = writer.create_warc_record(
            "https://example.com/wet",
            "conversion",
            payload=io.BytesIO(b"plain text body"),
            warc_headers_dict={"WARC-Identified-Payload-Type": "text/plain"},
        )
        writer.write_record(record)


def test_read_commoncrawl_warc_uses_file_backed_reader(tmp_path: Path) -> None:
    dump = "CC-MAIN-TEST"
    warc_rel = f"crawl-data/{dump}/segments/00000/warc/test.warc.gz"
    warc_path = tmp_path / warc_rel
    _write_warc_gz(warc_path)

    pipeline = mdr.text.read_commoncrawl(
        dump,
        base_url=tmp_path.as_uri(),
        use_https=True,
    )
    rows = [row.to_dict() for row in pipeline.take(10)]

    assert [row["content_bytes"].decode("utf-8") for row in rows] == [
        "<html>a</html>",
        "<html>b</html>",
    ]
    assert [row["WARC-Target-URI"] for row in rows] == [
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert all(row["warc_path"] == str(warc_path) for row in rows)


def test_read_commoncrawl_num_files_limits_file_backed_inputs(tmp_path: Path) -> None:
    dump = "CC-MAIN-TEST"
    first_rel = f"crawl-data/{dump}/segments/00000/warc/first.warc.gz"
    second_rel = f"crawl-data/{dump}/segments/00000/warc/second.warc.gz"
    first_path = tmp_path / first_rel
    second_path = tmp_path / second_rel
    _write_warc_gz(first_path)
    _write_warc_gz(second_path)

    pipeline = mdr.text.read_commoncrawl(
        dump,
        num_files=1,
        base_url=tmp_path.as_uri(),
        use_https=True,
    )
    rows = [row.to_dict() for row in pipeline.take(10)]

    assert len(rows) == 2
    assert {row["warc_path"] for row in rows} == {str(first_path)}


def test_read_commoncrawl_warc_supports_parallel_fetch(tmp_path: Path) -> None:
    dump = "CC-MAIN-TEST"
    warc_rel = f"crawl-data/{dump}/segments/00000/warc/test.warc.gz"
    warc_path = tmp_path / warc_rel
    offsets = _write_warc_gz_records(
        warc_path,
        [
            ("https://example.com/a", "<html>a</html>"),
            ("https://example.com/b", "<html>b</html>"),
            ("https://example.com/c", "<html>c</html>"),
        ],
    )

    index_rel = (
        f"cc-index/table/cc-main/warc/crawl={dump}/subset=warc/part-00000.parquet"
    )
    index_path = tmp_path / index_rel
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "url": [
                    "https://example.com/a",
                    "https://example.com/b",
                    "https://example.com/c",
                ],
                "warc_filename": [warc_rel, warc_rel, warc_rel],
                "warc_record_offset": [offset for offset, _ in offsets],
                "warc_record_length": [length for _, length in offsets],
                "warc_segment": ["00000", "00000", "00000"],
            }
        ),
        index_path,
    )
    _write_gzip_lines(
        tmp_path / "crawl-data" / dump / "cc-index-table.paths.gz",
        [index_rel],
    )

    rows = [
        row.to_dict()
        for row in mdr.text.read_commoncrawl_from_index(
            dump,
            max_inflight=2,
            base_url=tmp_path.as_uri(),
            use_https=True,
        ).take(10)
    ]

    assert [row["WARC-Target-URI"] for row in rows] == [
        "https://example.com/a",
        "https://example.com/b",
        "https://example.com/c",
    ]
    assert all(row["warc_path"] == str(warc_path) for row in rows)


def test_read_commoncrawl_can_return_binary_pdf_bytes(tmp_path: Path) -> None:
    dump = "CC-MAIN-TEST"
    warc_rel = f"crawl-data/{dump}/segments/00000/warc/test.warc.gz"
    warc_path = tmp_path / warc_rel
    pdf_bytes = b"%PDF-1.4\\n1 0 obj\\n<<>>\\nendobj\\n"
    offsets = _write_warc_gz_records(
        warc_path,
        [
            (
                "https://example.com/a.pdf",
                pdf_bytes,
                [("Content-Type", "application/pdf")],
                {"WARC-Identified-Payload-Type": "application/pdf"},
            ),
        ],
    )

    index_rel = (
        f"cc-index/table/cc-main/warc/crawl={dump}/subset=warc/part-00000.parquet"
    )
    index_path = tmp_path / index_rel
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "url": ["https://example.com/a.pdf"],
                "warc_filename": [warc_rel],
                "warc_record_offset": [offsets[0][0]],
                "warc_record_length": [offsets[0][1]],
                "warc_segment": ["00000"],
                "content_mime_type": ["application/pdf"],
                "content_mime_detected": ["application/pdf"],
            }
        ),
        index_path,
    )
    _write_gzip_lines(
        tmp_path / "crawl-data" / dump / "cc-index-table.paths.gz",
        [index_rel],
    )

    rows = [
        row.to_dict()
        for row in mdr.text.read_commoncrawl_from_index(
            dump,
            filter=mdr.text.commoncrawl.filter_pdf,
            output_fields=[
                "WARC-Target-URI",
                "WARC-Identified-Payload-Type",
                "content_bytes",
            ],
            base_url=tmp_path.as_uri(),
            use_https=True,
        ).take(10)
    ]

    assert len(rows) == 1
    assert rows[0]["WARC-Target-URI"] == "https://example.com/a.pdf"
    assert rows[0]["WARC-Identified-Payload-Type"] == "application/pdf"
    assert rows[0]["content_bytes"] == pdf_bytes
    assert rows[0]["warc_path"] == str(warc_path)


def test_read_commoncrawl_can_return_all_original_fields(tmp_path: Path) -> None:
    dump = "CC-MAIN-TEST"
    warc_rel = f"crawl-data/{dump}/segments/00000/warc/test.warc.gz"
    warc_path = tmp_path / warc_rel
    offsets = _write_warc_gz(tmp_path / warc_rel)

    index_rel = (
        f"cc-index/table/cc-main/warc/crawl={dump}/subset=warc/part-00000.parquet"
    )
    index_path = tmp_path / index_rel
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "url": ["https://example.com/a"],
                "warc_filename": [warc_rel],
                "warc_record_offset": [offsets[0][0]],
                "warc_record_length": [offsets[0][1]],
                "warc_segment": ["00000"],
            }
        ),
        index_path,
    )
    _write_gzip_lines(
        tmp_path / "crawl-data" / dump / "cc-index-table.paths.gz",
        [index_rel],
    )

    row = (
        mdr.text.read_commoncrawl_from_index(
            dump,
            output_fields="all",
            base_url=tmp_path.as_uri(),
            use_https=True,
        )
        .take(1)[0]
        .to_dict()
    )

    assert row["WARC-Type"] == "response"
    assert row["WARC-Target-URI"] == "https://example.com/a"
    assert row["Content-Type"] == [
        "application/http; msgtype=response",
        "text/html; charset=utf-8",
    ]
    assert row["content_bytes"]
    assert row["warc_path"] == str(warc_path)


def test_read_commoncrawl_all_preserves_duplicate_http_headers(tmp_path: Path) -> None:
    dump = "CC-MAIN-TEST"
    warc_rel = f"crawl-data/{dump}/segments/00000/warc/test.warc.gz"
    warc_path = tmp_path / warc_rel
    _write_warc_gz_records(
        warc_path,
        [
            (
                "https://example.com/a",
                b"<html>a</html>",
                [
                    ("Content-Type", "text/html; charset=utf-8"),
                    ("Link", "</a>; rel=prev"),
                    ("Link", "</b>; rel=next"),
                ],
                None,
            ),
        ],
    )

    row = (
        mdr.text.read_commoncrawl(
            dump,
            output_fields="all",
            base_url=tmp_path.as_uri(),
            use_https=True,
        )
        .take(1)[0]
        .to_dict()
    )

    assert row["Link"] == ["</a>; rel=prev", "</b>; rel=next"]
    assert row["warc_path"] == str(warc_path)


def test_read_commoncrawl_warc_ignores_non_warc_index_subsets(tmp_path: Path) -> None:
    dump = "CC-MAIN-TEST"
    warc_rel = f"crawl-data/{dump}/segments/00000/warc/test.warc.gz"
    offsets = _write_warc_gz(tmp_path / warc_rel)

    ignored_rel = (
        f"cc-index/table/cc-main/warc/crawl={dump}/subset=crawldiagnostics/"
        "part-00000-ignored.parquet"
    )
    ignored_path = tmp_path / ignored_rel
    ignored_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "url": ["https://example.com/ignored"],
                "warc_filename": [warc_rel],
                "warc_record_offset": [offsets[0][0]],
                "warc_record_length": [offsets[0][1]],
                "warc_segment": ["00000"],
            }
        ),
        ignored_path,
    )

    index_rel = (
        f"cc-index/table/cc-main/warc/crawl={dump}/subset=warc/part-00000.parquet"
    )
    index_path = tmp_path / index_rel
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "url": ["https://example.com/a", "https://example.com/b"],
                "warc_filename": [warc_rel, warc_rel],
                "warc_record_offset": [offset for offset, _ in offsets],
                "warc_record_length": [length for _, length in offsets],
                "warc_segment": ["00000", "00000"],
            }
        ),
        index_path,
    )

    _write_gzip_lines(
        tmp_path / "crawl-data" / dump / "cc-index-table.paths.gz",
        [ignored_rel, index_rel],
    )

    pipeline = mdr.text.read_commoncrawl_from_index(
        dump,
        base_url=tmp_path.as_uri(),
        use_https=True,
    )
    rows = [row.to_dict() for row in pipeline.take(10)]

    assert [row["WARC-Target-URI"] for row in rows] == [
        "https://example.com/a",
        "https://example.com/b",
    ]


def test_read_commoncrawl_wet_uses_globbed_files(tmp_path: Path) -> None:
    dump = "CC-MAIN-TEST"
    wet_rel = f"crawl-data/{dump}/segments/00000/wet/test.warc.wet.gz"
    wet_path = tmp_path / wet_rel
    _write_wet_gz(wet_path)

    pipeline = mdr.text.read_commoncrawl(
        [dump],
        format="wet",
        base_url=tmp_path.as_uri(),
        use_https=True,
    )
    rows = [row.to_dict() for row in pipeline.take(10)]

    assert [row["content_bytes"].decode("utf-8") for row in rows] == ["plain text body"]
    assert rows[0]["WARC-Target-URI"] == "https://example.com/wet"
    assert rows[0]["wet_path"] == str(wet_path)


def test_read_commoncrawl_segments_restricts_warc_files(tmp_path: Path) -> None:
    dump = "CC-MAIN-TEST"
    keep_rel = f"crawl-data/{dump}/segments/00000/warc/keep.warc.gz"
    skip_rel = f"crawl-data/{dump}/segments/00001/warc/skip.warc.gz"
    _write_warc_gz(tmp_path / keep_rel)
    _write_warc_gz(tmp_path / skip_rel)

    pipeline = mdr.text.read_commoncrawl(
        dump,
        segments=["00000"],
        base_url=tmp_path.as_uri(),
        use_https=True,
    )
    rows = [row.to_dict() for row in pipeline.take(10)]

    assert len(rows) == 2
    assert {row["warc_path"] for row in rows} == {str(tmp_path / keep_rel)}


def test_read_commoncrawl_filter_restricts_index_rows(tmp_path: Path) -> None:
    dump = "CC-MAIN-TEST"
    warc_rel = f"crawl-data/{dump}/segments/00000/warc/test.warc.gz"
    warc_path = tmp_path / warc_rel
    offsets = _write_warc_gz(warc_path)

    index_rel = (
        f"cc-index/table/cc-main/warc/crawl={dump}/subset=warc/part-00000.parquet"
    )
    index_path = tmp_path / index_rel
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "url": ["https://example.com/a", "https://example.com/b"],
                "warc_filename": [warc_rel, warc_rel],
                "warc_record_offset": [offset for offset, _ in offsets],
                "warc_record_length": [length for _, length in offsets],
                "warc_segment": ["00000", "00000"],
            }
        ),
        index_path,
    )
    _write_gzip_lines(
        tmp_path / "crawl-data" / dump / "cc-index-table.paths.gz",
        [index_rel],
    )

    pipeline = mdr.text.read_commoncrawl_from_index(
        dump,
        filter=mdr.col("url").str.endswith("/b"),
        base_url=tmp_path.as_uri(),
        use_https=True,
    )
    rows = [row.to_dict() for row in pipeline.take(10)]

    assert len(rows) == 1
    assert rows[0]["WARC-Target-URI"] == "https://example.com/b"
    assert rows[0]["warc_path"] == str(warc_path)


def test_read_commoncrawl_filter_fn_restricts_index_rows(tmp_path: Path) -> None:
    dump = "CC-MAIN-TEST"
    warc_rel = f"crawl-data/{dump}/segments/00000/warc/test.warc.gz"
    warc_path = tmp_path / warc_rel
    offsets = _write_warc_gz(warc_path)

    index_rel = (
        f"cc-index/table/cc-main/warc/crawl={dump}/subset=warc/part-00000.parquet"
    )
    index_path = tmp_path / index_rel
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "url": ["https://example.com/a", "https://example.com/b"],
                "warc_filename": [warc_rel, warc_rel],
                "warc_record_offset": [offset for offset, _ in offsets],
                "warc_record_length": [length for _, length in offsets],
                "warc_segment": ["00000", "00000"],
            }
        ),
        index_path,
    )
    _write_gzip_lines(
        tmp_path / "crawl-data" / dump / "cc-index-table.paths.gz",
        [index_rel],
    )

    pipeline = mdr.text.read_commoncrawl_from_index(
        dump,
        filter_fn=lambda row: str(row["url"]).endswith("/a"),
        base_url=tmp_path.as_uri(),
        use_https=True,
    )
    rows = [row.to_dict() for row in pipeline.take(10)]

    assert len(rows) == 1
    assert rows[0]["WARC-Target-URI"] == "https://example.com/a"
    assert rows[0]["warc_path"] == str(warc_path)


def test_read_commoncrawl_filter_fn_can_use_non_core_index_columns(
    tmp_path: Path,
) -> None:
    dump = "CC-MAIN-TEST"
    warc_rel = f"crawl-data/{dump}/segments/00000/warc/test.warc.gz"
    warc_path = tmp_path / warc_rel
    offsets = _write_warc_gz(warc_path)

    index_rel = (
        f"cc-index/table/cc-main/warc/crawl={dump}/subset=warc/part-00000.parquet"
    )
    index_path = tmp_path / index_rel
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "url": ["https://example.com/a", "https://example.com/b"],
                "warc_filename": [warc_rel, warc_rel],
                "warc_record_offset": [offset for offset, _ in offsets],
                "warc_record_length": [length for _, length in offsets],
                "warc_segment": ["00000", "00000"],
                "content_mime_type": ["text/html", "application/pdf"],
            }
        ),
        index_path,
    )
    _write_gzip_lines(
        tmp_path / "crawl-data" / dump / "cc-index-table.paths.gz",
        [index_rel],
    )

    rows = [
        row.to_dict()
        for row in mdr.text.read_commoncrawl_from_index(
            dump,
            filter_fn=lambda row: row["content_mime_type"] == "application/pdf",
            base_url=tmp_path.as_uri(),
            use_https=True,
        ).take(10)
    ]

    assert len(rows) == 1
    assert rows[0]["WARC-Target-URI"] == "https://example.com/b"
    assert rows[0]["warc_path"] == str(warc_path)


def test_read_commoncrawl_from_index_rejects_non_positive_max_inflight(
    tmp_path: Path,
) -> None:
    dump = "CC-MAIN-TEST"
    try:
        mdr.text.read_commoncrawl_from_index(
            dump,
            base_url=tmp_path.as_uri(),
            use_https=True,
            max_inflight=0,
        )
    except ValueError as exc:
        assert str(exc) == "max_inflight must be positive"
    else:
        raise AssertionError("expected ValueError for non-positive max_inflight")


def test_read_commoncrawl_filter_fn_logs_dropped_rows(tmp_path: Path) -> None:
    dump = "CC-MAIN-TEST"
    warc_rel = f"crawl-data/{dump}/segments/00000/warc/test.warc.gz"
    offsets = _write_warc_gz(tmp_path / warc_rel)

    index_rel = (
        f"cc-index/table/cc-main/warc/crawl={dump}/subset=warc/part-00000.parquet"
    )
    index_path = tmp_path / index_rel
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "url": ["https://example.com/a", "https://example.com/b"],
                "warc_filename": [warc_rel, warc_rel],
                "warc_record_offset": [offset for offset, _ in offsets],
                "warc_record_length": [length for _, length in offsets],
                "warc_segment": ["00000", "00000"],
            }
        ),
        index_path,
    )
    _write_gzip_lines(
        tmp_path / "crawl-data" / dump / "cc-index-table.paths.gz",
        [index_rel],
    )

    emitter = _RecordingEmitter()
    with set_active_user_metrics_emitter(emitter):
        rows = [
            row.to_dict()
            for row in mdr.text.read_commoncrawl_from_index(
                dump,
                filter_fn=lambda row: str(row["url"]).endswith("/a"),
                base_url=tmp_path.as_uri(),
                use_https=True,
            ).take(10)
        ]

    assert len(rows) == 1
    dropped = [
        counter
        for counter in emitter.counters
        if counter["label"] == "filter_fn_rows_filtered"
    ]
    assert len(dropped) == 1
    assert dropped[0]["value"] == 1.0
    total = [
        counter
        for counter in emitter.counters
        if counter["label"] == "total_rows_filtered"
    ]
    assert len(total) == 1
    assert total[0]["value"] == 1.0


def test_filter_commoncrawl_html_filters_index_mime_types(tmp_path: Path) -> None:
    dump = "CC-MAIN-TEST"
    warc_rel = f"crawl-data/{dump}/segments/00000/warc/test.warc.gz"
    warc_path = tmp_path / warc_rel
    offsets = _write_warc_gz(warc_path)

    index_rel = (
        f"cc-index/table/cc-main/warc/crawl={dump}/subset=warc/part-00000.parquet"
    )
    index_path = tmp_path / index_rel
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "url": ["https://example.com/a", "https://example.com/b"],
                "warc_filename": [warc_rel, warc_rel],
                "warc_record_offset": [offset for offset, _ in offsets],
                "warc_record_length": [length for _, length in offsets],
                "warc_segment": ["00000", "00000"],
                "content_mime_type": ["text/html", "application/pdf"],
                "content_mime_detected": pa.array([None, None], type=pa.string()),
            }
        ),
        index_path,
    )
    _write_gzip_lines(
        tmp_path / "crawl-data" / dump / "cc-index-table.paths.gz",
        [index_rel],
    )

    pipeline = mdr.text.read_commoncrawl_from_index(
        dump,
        filter=mdr.text.commoncrawl.filter_html,
        base_url=tmp_path.as_uri(),
        use_https=True,
    )
    rows = [row.to_dict() for row in pipeline.take(10)]

    assert len(rows) == 1
    assert rows[0]["WARC-Target-URI"] == "https://example.com/a"
    assert rows[0]["warc_path"] == str(warc_path)


def test_filter_commoncrawl_pdf_matches_vendor_mime(tmp_path: Path) -> None:
    dump = "CC-MAIN-TEST"
    warc_rel = f"crawl-data/{dump}/segments/00000/warc/test.warc.gz"
    warc_path = tmp_path / warc_rel
    offsets = _write_warc_gz_records(
        warc_path,
        [("https://example.com/a", "<html>a</html>")],
    )

    index_rel = (
        f"cc-index/table/cc-main/warc/crawl={dump}/subset=warc/part-00000.parquet"
    )
    index_path = tmp_path / index_rel
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "url": ["https://example.com/a"],
                "warc_filename": [warc_rel],
                "warc_record_offset": [offsets[0][0]],
                "warc_record_length": [offsets[0][1]],
                "warc_segment": ["00000"],
                "content_mime_type": ["application/vnd.pdf"],
                "content_mime_detected": pa.array([None], type=pa.string()),
            }
        ),
        index_path,
    )
    _write_gzip_lines(
        tmp_path / "crawl-data" / dump / "cc-index-table.paths.gz",
        [index_rel],
    )

    rows = [
        row.to_dict()
        for row in mdr.text.read_commoncrawl_from_index(
            dump,
            filter=mdr.text.commoncrawl.filter_pdf,
            base_url=tmp_path.as_uri(),
            use_https=True,
        ).take(10)
    ]

    assert len(rows) == 1
    assert rows[0]["WARC-Target-URI"] == "https://example.com/a"
    assert rows[0]["warc_path"] == str(warc_path)


def test_filter_commoncrawl_pdf_filters_only_mime_confirmed_pdfs(
    tmp_path: Path,
) -> None:
    dump = "CC-MAIN-TEST"
    warc_rel = f"crawl-data/{dump}/segments/00000/warc/test.warc.gz"
    warc_path = tmp_path / warc_rel
    offsets = _write_warc_gz_records(
        warc_path,
        [
            ("https://example.com/a.pdf", "<html>a</html>"),
            ("https://example.com/b", "<html>b</html>"),
            ("https://example.com/c", "<html>c</html>"),
        ],
    )

    index_rel = (
        f"cc-index/table/cc-main/warc/crawl={dump}/subset=warc/part-00000.parquet"
    )
    index_path = tmp_path / index_rel
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "url": [
                    "https://example.com/a.pdf",
                    "https://example.com/b",
                    "https://example.com/c",
                ],
                "warc_filename": [warc_rel, warc_rel, warc_rel],
                "warc_record_offset": [offset for offset, _ in offsets],
                "warc_record_length": [length for _, length in offsets],
                "warc_segment": ["00000", "00000", "00000"],
                "content_mime_type": pa.array(
                    [None, "application/pdf", "text/html"], type=pa.string()
                ),
                "content_mime_detected": pa.array([None, None, None], type=pa.string()),
                "url_path": ["/a.pdf", "/b", "/c"],
            }
        ),
        index_path,
    )
    _write_gzip_lines(
        tmp_path / "crawl-data" / dump / "cc-index-table.paths.gz",
        [index_rel],
    )

    pipeline = mdr.text.read_commoncrawl_from_index(
        dump,
        filter=mdr.text.commoncrawl.filter_pdf,
        base_url=tmp_path.as_uri(),
        use_https=True,
    )
    rows = [row.to_dict() for row in pipeline.take(10)]

    assert len(rows) == 1
    assert rows[0]["WARC-Target-URI"] == "https://example.com/b"
    assert rows[0]["warc_path"] == str(warc_path)


def test_filter_commoncrawl_domain_suffixes_filters_registry_suffix(
    tmp_path: Path,
) -> None:
    dump = "CC-MAIN-TEST"
    warc_rel = f"crawl-data/{dump}/segments/00000/warc/test.warc.gz"
    warc_path = tmp_path / warc_rel
    offsets = _write_warc_gz_records(
        warc_path,
        [
            ("https://example.pt/a", "<html>a</html>"),
            ("https://example.com/b", "<html>b</html>"),
        ],
    )

    index_rel = (
        f"cc-index/table/cc-main/warc/crawl={dump}/subset=warc/part-00000.parquet"
    )
    index_path = tmp_path / index_rel
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "url": ["https://example.pt/a", "https://example.com/b"],
                "warc_filename": [warc_rel, warc_rel],
                "warc_record_offset": [offset for offset, _ in offsets],
                "warc_record_length": [length for _, length in offsets],
                "warc_segment": ["00000", "00000"],
                "url_host_registry_suffix": ["pt", "com"],
            }
        ),
        index_path,
    )
    _write_gzip_lines(
        tmp_path / "crawl-data" / dump / "cc-index-table.paths.gz",
        [index_rel],
    )

    pipeline = mdr.text.read_commoncrawl_from_index(
        dump,
        filter=mdr.text.commoncrawl.filter_domain_suffixes("pt", "com.pt"),
        base_url=tmp_path.as_uri(),
        use_https=True,
    )
    rows = [row.to_dict() for row in pipeline.take(10)]

    assert len(rows) == 1
    assert rows[0]["WARC-Target-URI"] == "https://example.pt/a"
    assert rows[0]["warc_path"] == str(warc_path)


def test_filter_commoncrawl_truncation_helpers(tmp_path: Path) -> None:
    dump = "CC-MAIN-TEST"
    warc_rel = f"crawl-data/{dump}/segments/00000/warc/test.warc.gz"
    warc_path = tmp_path / warc_rel
    offsets = _write_warc_gz_records(
        warc_path,
        [
            ("https://example.com/a", "<html>a</html>"),
            ("https://example.com/b", "<html>b</html>"),
        ],
    )

    index_rel = (
        f"cc-index/table/cc-main/warc/crawl={dump}/subset=warc/part-00000.parquet"
    )
    index_path = tmp_path / index_rel
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "url": ["https://example.com/a", "https://example.com/b"],
                "warc_filename": [warc_rel, warc_rel],
                "warc_record_offset": [offset for offset, _ in offsets],
                "warc_record_length": [length for _, length in offsets],
                "warc_segment": ["00000", "00000"],
                "content_truncated": pa.array([None, "length"], type=pa.string()),
            }
        ),
        index_path,
    )
    _write_gzip_lines(
        tmp_path / "crawl-data" / dump / "cc-index-table.paths.gz",
        [index_rel],
    )

    kept = [
        row.to_dict()
        for row in mdr.text.read_commoncrawl_from_index(
            dump,
            filter=mdr.text.commoncrawl.filter_not_truncated,
            base_url=tmp_path.as_uri(),
            use_https=True,
        ).take(10)
    ]
    truncated = [
        row.to_dict()
        for row in mdr.text.read_commoncrawl_from_index(
            dump,
            filter=mdr.text.commoncrawl.filter_truncated,
            base_url=tmp_path.as_uri(),
            use_https=True,
        ).take(10)
    ]

    assert [row["WARC-Target-URI"] for row in kept] == ["https://example.com/a"]
    assert [row["WARC-Target-URI"] for row in truncated] == ["https://example.com/b"]
    assert all(row["warc_path"] == str(warc_path) for row in kept + truncated)


def test_read_commoncrawl_warc_shards_on_files(tmp_path: Path) -> None:
    dump = "CC-MAIN-TEST"
    first_rel = f"crawl-data/{dump}/segments/00000/warc/first.warc.gz"
    second_rel = f"crawl-data/{dump}/segments/00000/warc/second.warc.gz"
    _write_warc_gz(tmp_path / first_rel)
    _write_warc_gz(tmp_path / second_rel)

    pipeline = mdr.text.read_commoncrawl(
        dump,
        num_shards=2,
        base_url=tmp_path.as_uri(),
        use_https=True,
    )
    shards = pipeline.source.list_shards()

    assert len(shards) == 2
    shard_paths = {
        part.path
        for shard in shards
        for part in cast(FilePartsDescriptor, shard.descriptor).parts
    }
    assert shard_paths == {str(tmp_path / first_rel), str(tmp_path / second_rel)}

    rows = [
        cast(Row, row).to_dict()
        for shard in shards
        for row in pipeline.source.read_shard(shard)
    ]
    assert {tuple(sorted(row["warc_path"] for row in rows))} == {
        (
            str(tmp_path / first_rel),
            str(tmp_path / first_rel),
            str(tmp_path / second_rel),
            str(tmp_path / second_rel),
        ),
    }


def test_read_commoncrawl_logs_pushdown_pruning_metrics(tmp_path: Path) -> None:
    dump = "CC-MAIN-TEST"
    warc_rel = f"crawl-data/{dump}/segments/00000/warc/test.warc.gz"
    offsets = _write_warc_gz_records(
        tmp_path / warc_rel,
        [
            ("https://example.com/a", "<html>a</html>"),
            ("https://example.com/b", "<html>b</html>"),
        ],
    )

    index_rel = (
        f"cc-index/table/cc-main/warc/crawl={dump}/subset=warc/part-00000.parquet"
    )
    index_path = tmp_path / index_rel
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "url": ["https://example.com/a", "https://example.com/b"],
                "warc_filename": [warc_rel, warc_rel],
                "warc_record_offset": [offset for offset, _ in offsets],
                "warc_record_length": [length for _, length in offsets],
                "warc_segment": ["00000", "00000"],
                "url_host_registry_suffix": ["com", "pt"],
            }
        ),
        index_path,
        row_group_size=1,
    )
    _write_gzip_lines(
        tmp_path / "crawl-data" / dump / "cc-index-table.paths.gz",
        [index_rel],
    )

    emitter = _RecordingEmitter()
    with set_active_user_metrics_emitter(emitter):
        rows = [
            row.to_dict()
            for row in mdr.text.read_commoncrawl_from_index(
                dump,
                filter=mdr.text.commoncrawl.filter_domain_suffixes("pt"),
                base_url=tmp_path.as_uri(),
                use_https=True,
            ).take(10)
        ]

    assert [row["WARC-Target-URI"] for row in rows] == ["https://example.com/b"]
    counters_by_label = {
        counter["label"]: counter["value"] for counter in emitter.counters
    }
    assert counters_by_label["pushdown_row_groups_filtered"] == 1.0
    assert counters_by_label["total_rows_filtered"] == 1.0


def test_read_commoncrawl_logs_in_memory_filter_metrics(tmp_path: Path) -> None:
    dump = "CC-MAIN-TEST"
    warc_rel = f"crawl-data/{dump}/segments/00000/warc/test.warc.gz"
    offsets = _write_warc_gz(tmp_path / warc_rel)

    index_rel = (
        f"cc-index/table/cc-main/warc/crawl={dump}/subset=warc/part-00000.parquet"
    )
    index_path = tmp_path / index_rel
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "url": ["https://example.com/a", "https://example.com/b"],
                "warc_filename": [warc_rel, warc_rel],
                "warc_record_offset": [offset for offset, _ in offsets],
                "warc_record_length": [length for _, length in offsets],
                "warc_segment": ["00000", "00000"],
            }
        ),
        index_path,
    )
    _write_gzip_lines(
        tmp_path / "crawl-data" / dump / "cc-index-table.paths.gz",
        [index_rel],
    )

    emitter = _RecordingEmitter()
    with set_active_user_metrics_emitter(emitter):
        rows = [
            row.to_dict()
            for row in mdr.text.read_commoncrawl_from_index(
                dump,
                filter=mdr.col("url").str.endswith("/b"),
                base_url=tmp_path.as_uri(),
                use_https=True,
            ).take(10)
        ]

    assert [row["WARC-Target-URI"] for row in rows] == ["https://example.com/b"]
    counters_by_label = {
        counter["label"]: counter["value"] for counter in emitter.counters
    }
    assert counters_by_label["total_rows_filtered"] == 1.0


def test_read_commoncrawl_segments_restricts_wet_files(tmp_path: Path) -> None:
    dump = "CC-MAIN-TEST"
    keep_rel = f"crawl-data/{dump}/segments/00000/wet/keep.warc.wet.gz"
    skip_rel = f"crawl-data/{dump}/segments/00001/wet/skip.warc.wet.gz"
    _write_wet_gz(tmp_path / keep_rel)
    _write_wet_gz(tmp_path / skip_rel)

    pipeline = mdr.text.read_commoncrawl(
        dump,
        format="wet",
        segments="00000",
        base_url=tmp_path.as_uri(),
        use_https=True,
    )
    rows = [row.to_dict() for row in pipeline.take(10)]

    assert len(rows) == 1
    assert rows[0]["wet_path"].endswith("keep.warc.wet.gz")
