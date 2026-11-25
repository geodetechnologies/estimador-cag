# OpenAI embedding adapter

`OpenAIEmbedder` turns chunks into in-memory vectors using `text-embedding-3-small` at its default 1,536 dimensions. No persistence or router is included here.

## Use

```python
from app.embedding_pipeline.embedder import OpenAIEmbedder, estimated_cost_usd

embedder = OpenAIEmbedder()  # or OpenAIEmbedder(client=configured_client)
embedded_chunks = embedder.embed_many(chunks)
input_cost = estimated_cost_usd(sum(chunk.token_count for chunk in chunks))
vector = embedder.embed_one("text to compare")
```

The default client is constructed lazily with SDK environment authentication and SDK retries disabled. Callers may inject a configured OpenAI client; they should disable its SDK retries as well to retain the adapter's retry bound. Configuration and credentials are the caller's responsibility. Tests may also inject `sleep`.

## Behavior

- Requests contain up to 100 texts. Empty input makes no request. Provider indices restore input order; missing, duplicate, malformed, non-finite, or incorrectly sized vectors raise `ValueError`.
- Rate limits get three retries after 1, 2, and 4 seconds; other errors propagate. API input limits are left to the provider; text is never truncated or split here.
- Each successful batch logs only count, estimated input tokens, elapsed time including retries, and model. Text, vectors, credentials, provider objects and errors are not logged by this adapter.
- Cost uses the exercise tariff of $0.02 per million input tokens, not actual provider usage or invoice. Prices may change.
