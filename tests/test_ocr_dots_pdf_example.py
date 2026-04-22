from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_example_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "ocr_dots_pdf.py"
    spec = importlib.util.spec_from_file_location("ocr_dots_pdf", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repetition_check_flags_repeated_lines() -> None:
    example = _load_example_module()
    text = "\n".join(["Repeated OCR header"] * 3)

    assert example.has_repetition(text)


def test_repetition_check_allows_short_normal_text() -> None:
    example = _load_example_module()

    assert not example.has_repetition("A concise page of normal text.")


def test_blank_page_response_parser_accepts_json() -> None:
    example = _load_example_module()

    assert example.parse_blank_page_response('{"is_blank_page": true}')
    assert not example.parse_blank_page_response('{"is_blank_page": false}')


def test_pipeline_uses_dots_and_qwen_runtime_services() -> None:
    example = _load_example_module()
    from refiner.services.discovery import collect_pipeline_services

    pipeline = example.build_pipeline(["sample.pdf"], output_path="output/test.jsonl")
    services = collect_pipeline_services(pipeline)

    models = {service.config["model_name_or_path"] for service in services}
    assert models == {"rednote-hilab/dots.mocr", "Qwen/Qwen3.5-4B"}
