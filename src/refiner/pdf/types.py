from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import IO, TYPE_CHECKING

from refiner.execution.asyncio.runtime import io_executor
from refiner.io import DataFile
from refiner.io.datafile import DataFileLike
from refiner.utils import check_required_dependencies

if TYPE_CHECKING:
    from PIL.Image import Image


_pdfium_lock = asyncio.Lock()


@dataclass(frozen=True, slots=True)
class RenderedPdfPage:
    index: int
    image: Image


@dataclass(frozen=True, slots=True)
class PdfFile:
    data: DataFileLike | bytes
    name: str | None = None

    @property
    def uri(self) -> str:
        if isinstance(self.data, bytes):
            return self.name or "<bytes>"
        return str(DataFile.resolve(self.data))

    def open(self, mode: str = "rb") -> IO[bytes]:
        if isinstance(self.data, bytes):
            raise TypeError("byte-backed PdfFile does not support open()")
        return DataFile.resolve(self.data).open(mode=mode)

    async def iter_rendered_pages(
        self,
        *,
        scale: float = 2.0,
    ) -> AsyncIterator[RenderedPdfPage]:
        check_required_dependencies(
            "PDF rendering",
            [("PIL", "pillow"), "pypdfium2"],
            dist="pdf",
        )

        import pypdfium2 as pdfium

        pdf_bytes = await _read_pdf_bytes(self)
        async with _pdfium_lock:
            doc = pdfium.PdfDocument(pdf_bytes)
            try:
                page_count = len(doc)
                for index in range(page_count):
                    page = doc[index]
                    try:
                        image = page.render(scale=scale).to_pil()
                        image.load()
                    finally:
                        page.close()
                    yield RenderedPdfPage(index=index, image=image)
            finally:
                doc.close()


async def _read_pdf_bytes(pdf: PdfFile) -> bytes:
    if isinstance(pdf.data, bytes):
        return pdf.data
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(io_executor(), _read_pdf_bytes_sync, pdf)


def _read_pdf_bytes_sync(pdf: PdfFile) -> bytes:
    with pdf.open("rb") as handle:
        data = handle.read()
    if not isinstance(data, bytes):
        raise TypeError(
            f"Expected bytes when reading PDF {pdf.uri!r}, got {type(data)!r}"
        )
    return data


__all__ = ["PdfFile", "RenderedPdfPage"]
