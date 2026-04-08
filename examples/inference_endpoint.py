from __future__ import annotations

import refiner as mdr

INPUT_PATH = "input.jsonl"
OUTPUT_PATH = "output/endpoint-inference"
ENDPOINT = mdr.inference.OpenAIEndpointProvider(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)


async def summarize(row, generate):
    response = await generate(
        {
            "messages": [
                {"role": "system", "content": "Summarize the input briefly."},
                {"role": "user", "content": row["text"]},
            ]
        }
    )
    return {"summary": response.text}


if __name__ == "__main__":
    (
        mdr.from_items([{"text": "Hello, world!"}])
        .map_async(
            mdr.inference.generate(
                fn=summarize,
                provider=ENDPOINT,
                default_generation_params={
                    "temperature": 0.1,
                    "max_tokens": 256,
                    "model": "gpt-4o-mini",
                },
                max_concurrent_requests=64,
            ),
            max_in_flight=64,
        )
        .write_jsonl(OUTPUT_PATH)
        .launch_cloud(
            name="endpoint-inference", num_workers=1, gpus_per_worker=1, gpu_type="h100"
        )
    )
