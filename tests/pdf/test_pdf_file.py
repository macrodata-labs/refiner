from __future__ import annotations

import asyncio

import refiner as mdr
from refiner.io import DataFile


def _minimal_pdf(page_count: int = 2) -> bytes:
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        (
            b"<< /Type /Pages /Kids ["
            + b" ".join(f"{3 + i * 2} 0 R".encode() for i in range(page_count))
            + b"] /Count "
            + str(page_count).encode()
            + b" >>"
        ),
    ]
    for index in range(page_count):
        page_object = 3 + index * 2
        content_object = page_object + 1
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 80] "
            + f"/Contents {content_object} 0 R >>".encode()
        )
        objects.append(b"<< /Length 0 >>\nstream\n\nendstream")

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{index} 0 obj\n".encode())
        out.extend(obj)
        out.extend(b"\nendobj\n")
    xref_offset = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    out.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        out.extend(f"{offset:010d} 00000 n \n".encode())
    out.extend(
        b"trailer\n"
        + f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
        + b"startxref\n"
        + str(xref_offset).encode()
        + b"\n%%EOF\n"
    )
    return bytes(out)


async def _collect_pages(pdf: mdr.pdf.PdfFile, *, scale: float):
    return [page async for page in pdf.iter_rendered_pages(scale=scale)]


def test_iter_rendered_pages_yields_scaled_images(tmp_path) -> None:
    path = tmp_path / "sample.pdf"
    path.write_bytes(_minimal_pdf(page_count=2))
    pdf = mdr.pdf.PdfFile(DataFile.resolve(path))

    pages = asyncio.run(_collect_pages(pdf, scale=2.0))

    assert [page.index for page in pages] == [0, 1]
    assert [page.image.size for page in pages] == [(200, 160), (200, 160)]
    assert [page.image.mode for page in pages] == ["RGB", "RGB"]


def test_pdf_file_uri_uses_underlying_data_file(tmp_path) -> None:
    path = tmp_path / "sample.pdf"
    path.write_bytes(_minimal_pdf(page_count=1))

    pdf = mdr.pdf.PdfFile(DataFile.resolve(path))

    assert pdf.uri.endswith("sample.pdf")


def test_iter_rendered_pages_supports_byte_backed_pdf() -> None:
    pdf = mdr.pdf.PdfFile(_minimal_pdf(page_count=1), name="inline.pdf")

    pages = asyncio.run(_collect_pages(pdf, scale=1.0))

    assert pdf.uri == "inline.pdf"
    assert [page.index for page in pages] == [0]
    assert pages[0].image.size == (100, 80)
