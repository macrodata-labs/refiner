from __future__ import annotations

import refiner as mdr

INPUT_PATH = "input.jsonl"
OUTPUT_PATH = "output/endpoint-inference"
ENDPOINT = mdr.inference.OpenAIEndpointProvider(
    base_url="https://api.openai.com",
    api_key="YOUR_API_KEY",
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
        mdr.read_jsonl(INPUT_PATH)
        .map_async(
            mdr.inference.generate(
                fn=summarize,
                provider=ENDPOINT,
                default_generation_params={"temperature": 0.1, "max_tokens": 256},
                max_concurrent_requests=64,
            ),
            max_in_flight=64,
        )
        .write_parquet(OUTPUT_PATH)
        .launch_local(name="endpoint-inference", num_workers=1)
    )
