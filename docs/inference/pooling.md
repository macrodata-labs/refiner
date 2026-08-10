---
title: "Pooling"
description: "Use pooling-style inference for reward and progress scoring"
---

# Pooling

Pooling inference is used for model APIs that return token logits, embeddings,
or classification-like outputs instead of chat text.

`reward_score` uses pooling inference internally:

```python
pipeline = pipeline.map_async(
    mdr.robotics.reward_score(max_frames=8),
    max_in_flight=32,
)
```

For direct use, wrap a row function with `generate_pooling`:

```python
async def score(row, generate_pooling):
    response = await generate_pooling(
        {
            "task": "token_classify",
            "messages": [{"role": "user", "content": "score this"}],
        }
    )
    return row.update({"raw_score": response})
```

Most users should start with [Reward Scoring](../episode-operations/reward-scoring.md)
instead of calling pooling directly.

