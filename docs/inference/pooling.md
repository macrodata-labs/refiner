---
title: "Pooling"
description: "Use pooling-style inference for reward and progress scoring"
---

# Pooling

Pooling inference is used for model APIs that return token logits, embeddings,
or classification-like outputs instead of chat text.

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
