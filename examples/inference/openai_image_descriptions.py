from __future__ import annotations

import refiner as mdr

MAX_IMAGES = 1_000

PROVIDER = mdr.inference.OpenAIResponsesProvider(
    model="gpt-4.1",
)


def first_1k_images():
    seen = 0

    def keep(row):
        nonlocal seen
        if seen >= MAX_IMAGES:
            return False
        seen += 1
        return True

    return keep


async def describe_image(row, generate_text):
    image = row["image"]
    image_data = image.get("bytes") if isinstance(image, dict) else image
    response = await generate_text(
        messages=[
            {
                "role": "system",
                "content": "Describe the food in the image briefly.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What food is shown in this image?",
                    },
                    {
                        "type": "image",
                        "image": image_data,
                    },
                ],
            },
        ],
    )
    return {"label": row["label"], "description": response.text}


if __name__ == "__main__":
    (
        mdr.read_hf_dataset(
            "ethz/food101",
            split="train",
            columns_to_read=["image", "label"],
            num_shards=1,
        )
        .filter(first_1k_images())
        .map_async(
            mdr.inference.generate_text(
                fn=describe_image,
                provider=PROVIDER,
                default_generation_params={
                    "temperature": 0.1,
                    "max_tokens": 128,
                },
            ),
            max_in_flight=256,
            preserve_order=False,
        )
        .write_parquet(
            "hf://buckets/macrodata/test_bucket/food101-openai-descriptions-gpt-4.1-1k.parquet"
        )
        .launch_cloud(
            name="food101-openai-descriptions",
            num_workers=1,
            secrets=mdr.Secrets.dict({"HF_TOKEN": None, "OPENAI_API_KEY": None}),
        )
    )
