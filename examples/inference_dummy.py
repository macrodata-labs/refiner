from __future__ import annotations

import refiner as mdr

OUTPUT_PATH = "output/dummy-inference.jsonl"
PROVIDER = mdr.inference.DummyRequestProvider(
    model="dummy-local",
    response_text="dummy response",
)


async def summarize(row, generate):
    response = await generate(
        {
            "messages": [
                {"role": "system", "content": "Return the canned answer."},
                {"role": "user", "content": row["text"]},
            ]
        }
    )
    return {"summary": response.text}


if __name__ == "__main__":
    (
        mdr.from_items([{"text": "Hello, world!"}] * 10)
        .map_async(
            mdr.inference.generate(
                fn=summarize,
                provider=PROVIDER,
            ),
            max_in_flight=16,
        )
        .write_jsonl(OUTPUT_PATH)
        .launch_local(
            name="dummy-inference",
            num_workers=1,
        )
    )
