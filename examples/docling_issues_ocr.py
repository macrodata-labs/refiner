from __future__ import annotations

import base64
import io
import json
from collections.abc import Mapping
from typing import Any

import refiner as mdr

INPUT_PATH = "hf://datasets/hynky/docling-issues/data/train-00000-of-00001.parquet"
OUTPUT_PATH = "output/docling-issues-ocr"
PDF_SCALE = 200.0 / 72.0
DOTS_OCR_PROMPT = "Extract the text content from this image."
DOTS_MAX_COMPLETION_TOKENS = 16384

DOTS_PROVIDER = mdr.inference.VLLMProvider(model="rednote-hilab/dots.mocr")

BLANK_PAGE_PROVIDER = mdr.inference.VLLMProvider(model="Qwen/Qwen3.5-4B")


def _image_data_url(image) -> str:
    if image.mode not in {"RGB", "L"}:
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _image_message(prompt: str, image_url: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _dots_image_message(prompt: str, image_url: str) -> list[dict[str, Any]]:
    # dots.mocr's official vLLM example expects the image sentinel tokens inside the
    # text block, even when the request also carries an image_url content item.
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {
                    "type": "text",
                    "text": f"<|img|><|imgpad|><|endofimg|>{prompt}",
                },
            ],
        }
    ]


def _repetition_score(text: str) -> float:
    words = text.lower().split()
    if len(words) < 24:
        return 0.0
    windows = [" ".join(words[index : index + 6]) for index in range(len(words) - 5)]
    if not windows:
        return 0.0
    unique = len(set(windows))
    return 1.0 - unique / len(windows)


def _has_repetition(page_texts: list[str]) -> bool:
    for text in page_texts:
        stripped = text.strip()
        if not stripped:
            continue
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if lines and len(set(lines)) / len(lines) < 0.35:
            return True
        if _repetition_score(stripped) > 0.4:
            return True
    return False


async def transcribe_pdf(row, generate):
    pdf = mdr.pdf.PdfFile(row["media_bytes"], name=f"{row['id']}.pdf")
    page_texts: list[str] = []
    suspect_blank_pages: list[int] = []
    page_images: list[str] = []

    try:
        async for page in pdf.iter_rendered_pages(scale=PDF_SCALE):
            image_url = _image_data_url(page.image)
            response = await generate(
                {
                    "messages": _dots_image_message(
                        DOTS_OCR_PROMPT,
                        image_url,
                    ),
                    "temperature": 0.1,
                    "top_p": 1.0,
                    "max_completion_tokens": DOTS_MAX_COMPLETION_TOKENS,
                }
            )
            text = response.text.strip()
            page_texts.append(text)
            page_images.append(image_url)
            if not text or text.startswith("The "):
                suspect_blank_pages.append(page.index)
    except Exception as exc:
        return row.update(
            {
                "invalid_pdf": True,
                "pdf_error": str(exc),
                "page_texts": [],
                "transcription": "",
                "page_count": 0,
                "has_repetition": False,
                "suspect_blank_pages": [],
                "_page_images": [],
            }
        )

    return row.update(
        {
            "invalid_pdf": False,
            "pdf_error": None,
            "page_texts": page_texts,
            "transcription": "\n\n".join(page_texts),
            "page_count": len(page_texts),
            "has_repetition": _has_repetition(page_texts),
            "suspect_blank_pages": suspect_blank_pages,
            "_page_images": page_images,
        }
    )


def keep_valid_pdf(row) -> bool:
    return not bool(row["invalid_pdf"])


def keep_non_repetitive(row) -> bool:
    return not bool(row["has_repetition"])


async def check_blank_pages(row, generate):
    suspect_pages = list(row["suspect_blank_pages"])
    if not suspect_pages:
        return row.update(
            {
                "blank_pages": [],
                "has_blank_pages": False,
            }
        )

    blank_pages: list[int] = []
    page_images = list(row["_page_images"])
    for page_index in suspect_pages:
        response = await generate(
            {
                "messages": _image_message(
                    (
                        "Decide whether this PDF page is blank or effectively blank. "
                        'Return JSON only: {"blank": true|false, "reason": "..."}'
                    ),
                    page_images[page_index],
                ),
                "temperature": 0.0,
                "max_tokens": 256,
                "response_format": {"type": "json_object"},
            }
        )
        try:
            payload = json.loads(response.text)
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, Mapping) and bool(payload.get("blank")):
            blank_pages.append(page_index)

    return row.update(
        {
            "blank_pages": blank_pages,
            "has_blank_pages": bool(blank_pages),
        }
    ).drop("_page_images")


def keep_non_blank(row) -> bool:
    return not bool(row["has_blank_pages"])


if __name__ == "__main__":
    (
        mdr.read_parquet(
            INPUT_PATH,
            columns_to_read=["id", "media_bytes", "problem", "url"],
        )
        .map_async(
            mdr.inference.generate(
                fn=transcribe_pdf,
                provider=DOTS_PROVIDER,
                max_concurrent_requests=64,
            ),
            max_in_flight=16,
            preserve_order=False,
        )
        .filter(keep_valid_pdf)
        .filter(keep_non_repetitive)
        .map_async(
            mdr.inference.generate(
                fn=check_blank_pages,
                provider=BLANK_PAGE_PROVIDER,
                max_concurrent_requests=64,
            ),
            max_in_flight=16,
            preserve_order=False,
        )
        .filter(keep_non_blank)
        .write_parquet(OUTPUT_PATH)
        .launch_cloud(
            name="docling-issues-ocr",
            num_workers=1,
        )
    )
