# Ingest budget embeddings

`POST /embeddings/ingest` accepts `{"budgets": [...]}` using the budget schema shown in Swagger at `/docs`. It returns ordered embedded component chunks and `stats` with `total_budgets`, `total_chunks`, `total_tokens`, and `estimated_cost_usd`. The estimate uses the exercise tariff, not provider billing usage.

For nonempty chunks, configure the existing `OPENAI_API_KEY` setting. The request runs synchronously in FastAPI's threadpool; the OpenAI SDK's automatic retries are disabled so the embedder owns rate-limit retries. Empty budgets or budgets without components return zero chunks and cost without accessing credentials or constructing a client.

Invalid input returns 422. Configuration, client construction, and embedding failures return a generic 500; operational logs contain only the stage and exception type, not provider messages or input content. `/health` remains independent of provider configuration. This endpoint does not persist vectors or perform retrieval.
