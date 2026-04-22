from __future__ import annotations

import argparse
import base64
import importlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import refiner as mdr
from refiner.pipeline.data.row import Row

DEFAULT_OUTPUT_PATH = "output/dots-pdf-ocr.jsonl"
DOTS_MODEL = "rednote-hilab/dots.mocr"
QWEN_BLANK_MODEL = "Qwen/Qwen3.5-4B"
DOTS_IMAGE_MARKER = "<|img|><|imgpad|><|endofimg|>"
DOTS_PROVIDER = mdr.inference.VLLMProvider(model=DOTS_MODEL)
QWEN_PROVIDER = mdr.inference.VLLMProvider(model=QWEN_BLANK_MODEL)


def render_pdf_pages(row: Row) -> Iterable[dict[str, Any]]:
    try:
        fitz = importlib.import_module("fitz")
    except ImportError as exc:
        raise RuntimeError(
            "PDF rendering requires PyMuPDF. Install it in the worker image before "
            "running this example."
        ) from exc

    pdf_path = Path(str(row["pdf_path"]))
    zoom = float(row.get("zoom", 2.0))
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            png_bytes = pixmap.tobytes("png")
            yield {
                "pdf_path": str(pdf_path),
                "page_index": page_index,
                "image_data_url": _image_data_url(png_bytes),
            }


async def transcribe_page(row, generate):
    response = await generate(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": row["image_data_url"]},
                        },
                        {
                            "type": "text",
                            "text": (
                                f"{DOTS_IMAGE_MARKER}Extract the text content from this page. "
                                "Return only the page text."
                            ),
                        },
                    ],
                }
            ]
        }
    )
    text = response.text.strip()
    return {
        **row,
        "text": text,
        "ocr_finish_reason": response.finish_reason,
    }


def add_repetition_flags(row: Row) -> dict[str, Any]:
    text = str(row.get("text", ""))
    return {
        **row,
        "starts_with_the": text.lstrip().startswith("The "),
        "has_repetition": has_repetition(text),
        "text_chars": len(text),
    }


def keep_after_repetition_check(row: Row) -> bool:
    return bool(row.get("text")) and not bool(row.get("has_repetition"))


async def check_blank_page_candidate(row, generate):
    if not bool(row.get("starts_with_the")):
        return {**row, "blank_page_check": "skipped", "is_blank_page": False}

    response = await generate(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": row["image_data_url"]},
                        },
                        {
                            "type": "text",
                            "text": (
                                "Decide whether this PDF page is blank or only contains "
                                "non-content marks. Reply as JSON with exactly "
                                '{"is_blank_page": true|false}.'
                            ),
                        },
                    ],
                }
            ],
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
    )
    is_blank = parse_blank_page_response(response.text)
    return {
        **row,
        "blank_page_check": response.text.strip(),
        "is_blank_page": is_blank,
    }


def keep_after_blank_check(row: Row) -> bool:
    return not bool(row.get("is_blank_page"))


def to_output_row(row: Row) -> dict[str, Any]:
    return {
        "pdf_path": row["pdf_path"],
        "page_index": row["page_index"],
        "text": row["text"],
        "text_chars": row["text_chars"],
        "starts_with_the": row["starts_with_the"],
        "has_repetition": row["has_repetition"],
        "blank_page_check": row["blank_page_check"],
    }


def has_repetition(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 12]
    line_counts: dict[str, int] = {}
    for line in lines:
        line_counts[line] = line_counts.get(line, 0) + 1
        if line_counts[line] >= 3:
            return True

    normalized = " ".join(text.split())
    if len(normalized) < 120:
        return False

    words = normalized.split()
    for window_size in (4, 6, 8):
        if _has_repeated_word_window(words, window_size=window_size, min_repeats=4):
            return True
    return False


def parse_blank_page_response(raw: str) -> bool:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        lowered = cleaned.lower()
        return '"is_blank_page": true' in lowered or "is_blank_page: true" in lowered
    if not isinstance(parsed, dict):
        return False
    return bool(parsed.get("is_blank_page"))


def build_pipeline(
    pdf_paths: list[str],
    *,
    output_path: str = DEFAULT_OUTPUT_PATH,
    zoom: float = 2.0,
):
    rows = [{"pdf_path": pdf_path, "zoom": zoom} for pdf_path in pdf_paths]
    return (
        mdr.from_items(rows)
        .flat_map(render_pdf_pages)
        .map_async(
            mdr.inference.generate(
                fn=transcribe_page,
                provider=DOTS_PROVIDER,
                default_generation_params={"temperature": 0.0, "max_tokens": 4096},
                max_concurrent_requests=64,
            ),
            max_in_flight=64,
            preserve_order=False,
        )
        .map(add_repetition_flags)
        .filter(keep_after_repetition_check)
        .map_async(
            mdr.inference.generate(
                fn=check_blank_page_candidate,
                provider=QWEN_PROVIDER,
                default_generation_params={"temperature": 0.0, "max_tokens": 128},
                max_concurrent_requests=32,
            ),
            max_in_flight=32,
            preserve_order=False,
        )
        .filter(keep_after_blank_check)
        .map(to_output_row)
        .write_jsonl(output_path)
    )


def _has_repeated_word_window(
    words: list[str], *, window_size: int, min_repeats: int
) -> bool:
    if len(words) < window_size * min_repeats:
        return False
    counts: dict[tuple[str, ...], int] = {}
    for index in range(0, len(words) - window_size + 1):
        window = tuple(words[index : index + window_size])
        counts[window] = counts.get(window, 0) + 1
        if counts[window] >= min_repeats:
            return True
    return False


def _image_data_url(png_bytes: bytes) -> str:
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR PDFs with Dots OCR on Refiner.")
    parser.add_argument("pdf", nargs="+", help="PDF file paths to transcribe.")
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_PATH, help="Output JSONL path."
    )
    parser.add_argument(
        "--zoom", type=float, default=2.0, help="PDF render zoom factor."
    )
    parser.add_argument("--workers", type=int, default=1, help="Cloud worker count.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_pipeline(args.pdf, output_path=args.output, zoom=args.zoom).launch_cloud(
        name="dots-pdf-ocr",
        num_workers=args.workers,
    )
