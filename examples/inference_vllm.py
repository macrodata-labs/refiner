from __future__ import annotations

import refiner as mdr

OUTPUT_PATH = "output/vllm-inference.jsonl"
PROVIDER = mdr.inference.VLLMProvider(
    model="Qwen/Qwen3.5-9B",
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
        mdr.from_items([{"text": "Hello, world!"}] * 10000)
        .map_async(
            mdr.inference.generate(
                fn=summarize,
                provider=PROVIDER,
                default_generation_params={"temperature": 0.1, "max_tokens": 256},
            ),
            max_in_flight=256,
            preserve_order=False,
        )
        .write_parquet(OUTPUT_PATH)
        .launch_cloud(
            name="vllm-inference",
            num_workers=1,
        )
    )
