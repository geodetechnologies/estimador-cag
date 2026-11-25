"""Batch embeddings with bounded retries and content-free operational logs."""

import math
import time
from collections.abc import Callable

import structlog
import tiktoken
from openai import OpenAI, RateLimitError

from .schemas import Chunk, EmbeddedChunk

MODEL = "text-embedding-3-small"
DIMENSIONS = 1536
BATCH_SIZE = 100
# Exercise assignment tariff; provider prices may change.
PRICE_USD_PER_MILLION_INPUT_TOKENS = 0.02
logger = structlog.get_logger(__name__)


def estimated_cost_usd(total_tokens: int) -> float:
    """Estimate input cost from chunk token counts, not provider billing usage."""
    return total_tokens * PRICE_USD_PER_MILLION_INPUT_TOKENS / 1_000_000


def _ordered_vectors(response, count: int) -> list[list[float]]:
    """Reject missing, duplicate, malformed, or non-finite provider output."""
    data = getattr(response, "data", None)
    if not isinstance(data, list) or len(data) != count:
        raise ValueError("Invalid embedding response")
    ordered = [None] * count
    for item in data:
        index = getattr(item, "index", None)
        embedding = getattr(item, "embedding", None)
        if (type(index) is not int or not 0 <= index < count
                or ordered[index] is not None or not isinstance(embedding, list)
                or len(embedding) != DIMENSIONS
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       for value in embedding)):
            raise ValueError("Invalid embedding response")
        ordered[index] = [float(value) for value in embedding]
    return ordered


class OpenAIEmbedder:
    def __init__(self, client=None, sleep: Callable[[float], None] | None = None):
        self._client = client
        self._sleep = sleep if sleep is not None else time.sleep

    def _batch(self, texts: list[str], token_count: int) -> list[list[float]]:
        if self._client is None:
            # SDK environment authentication; no settings or credential file read here.
            self._client = OpenAI(max_retries=0)
        start = time.perf_counter()
        for attempt in range(4):
            try:
                response = self._client.embeddings.create(model=MODEL, input=texts)
                break
            except RateLimitError:
                if attempt == 3:
                    raise
                self._sleep(2 ** attempt)
        vectors = _ordered_vectors(response, len(texts))
        logger.info(
            "embedding_batch_processed",
            batch_count=len(texts),
            estimated_input_tokens=token_count,
            elapsed_seconds=time.perf_counter() - start,
            model=MODEL,
        )
        return vectors

    def embed_one(self, text: str) -> list[float]:
        encoding = tiktoken.encoding_for_model(MODEL)
        token_count = len(encoding.encode(text, disallowed_special=()))
        return self._batch([text], token_count)[0]

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        result = []
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start:start + BATCH_SIZE]
            vectors = self._batch([chunk.text for chunk in batch], sum(chunk.token_count for chunk in batch))
            result.extend(EmbeddedChunk(**chunk.model_dump(), embedding=vector)
                          for chunk, vector in zip(batch, vectors, strict=True))
        return result
