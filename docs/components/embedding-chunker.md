# Structural budget chunking

`JSONStructuralChunker` turns each budget component into one traceable chunk, preserving input order. It does not call an embedding provider or modify its inputs.

```python
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.schemas import IngestRequest

request = IngestRequest.model_validate({"budgets": budgets})  # budgets is a list of JSON objects
chunks = JSONStructuralChunker().chunk(request.budgets)
```

Each chunk contains the parent project summary, sector, year and main technology followed by the component's name, description, technology stack, complexity and estimated hours. Its ID is `budget_id::component_id`; metadata contains these IDs plus sector, main technology, year, complexity and estimated hours. Client names are not included in chunk text or metadata.

Token counts use `tiktoken.encoding_for_model("text-embedding-3-small")` on the full text. There is no overlap, truncation or splitting, even for unusually long descriptions; consumers should check model limits before embedding. Sector is an open string, not a closed list. The provided `data/budgets_sample.json` is the exercise sample (15 budgets, 52 components); it produces 52 chunks. Embedding and HTTP ingestion are separate future steps.
