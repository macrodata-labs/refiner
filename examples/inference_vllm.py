from __future__ import annotations

import refiner as mdr

INPUT_PATH = "input.jsonl"
OUTPUT_PATH = "output/vllm-inference.jsonl"
PROVIDER = mdr.inference.VLLMProvider(
    model_name_or_path="Qwen/Qwen3.5-9B",
    model_max_context=8192,
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
                provider=PROVIDER,
                default_generation_params={"temperature": 0.1, "max_tokens": 256},
            ),
            max_in_flight=64,
        )
        .write_parquet(OUTPUT_PATH)
        .launch_cloud(
            name="vllm-inference",
            num_workers=1,
        )
    )
