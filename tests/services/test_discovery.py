from __future__ import annotations

import refiner as mdr
from refiner.services.discovery import collect_pipeline_services


async def _noop_inference(row, generate):
    del generate
    return row


def test_collect_pipeline_services_supports_nested_service_config() -> None:
    pipeline = mdr.from_items([{"text": "hello"}]).map_async(
        mdr.inference.generate(
            fn=_noop_inference,
            provider=mdr.inference.VLLMProvider(
                model="Qwen/Qwen2.5-VL-7B-Instruct",
                extra_kwargs={"limit-mm-per-prompt": "video=1"},
            ),
        )
    )

    services = collect_pipeline_services(pipeline)

    assert len(services) == 1
    assert services[0].config == {
        "model_name_or_path": "Qwen/Qwen2.5-VL-7B-Instruct",
        "extra_kwargs": {"limit-mm-per-prompt": "video=1"},
    }
