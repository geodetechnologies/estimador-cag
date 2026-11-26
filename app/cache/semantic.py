import hashlib
import json
import math
from typing import Any, Protocol, Sequence

from pydantic import BaseModel, ConfigDict

from app.cache.base import AsyncCacheBackend
from app.schemas import EstimationRequest, EstimationResult


class AsyncEmbedder(Protocol):
    async def embed(self, text: str) -> Sequence[float]: ...


class _Entry(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    key: str
    vector: list[Any]
    result: EstimationResult


def semantic_cache_key(
    request: EstimationRequest, *, bucket: str, prompt_version: str, model: str,
    embedding_model: str,
) -> str:
    """Identify an isolated semantic bucket without exposing its inputs."""
    for value in (bucket, prompt_version, model, embedding_model):
        if not isinstance(value, str) or not value.strip():
            raise ValueError('identity fields must be nonempty strings')
    identity = {
        'bucket': bucket, 'prompt_version': prompt_version, 'model': model,
        'embedding_model': embedding_model,
        'request': request.model_dump(mode='json', exclude={'description'}),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return 'estimation:semantic:v1:' + hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def _unit_vector(vector: Sequence[float]) -> list[float]:
    if not isinstance(vector, (list, tuple)) or not 1 <= len(vector) <= 4096:
        raise ValueError('invalid vector')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in vector):
        raise ValueError('invalid vector')
    values = [float(x) for x in vector]
    if not all(math.isfinite(x) for x in values):
        raise ValueError('invalid vector')
    norm = math.hypot(*values)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError('invalid vector')
    return [x / norm for x in values]


class SemanticCache:
    def __init__(
        self, backend: AsyncCacheBackend | None = None, embedder: AsyncEmbedder | None = None,
        *, enabled: bool = False, ttl_seconds: int = 86400, threshold: float = 0.95,
    ) -> None:
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int) or ttl_seconds <= 0:
            raise ValueError('ttl_seconds must be a positive integer')
        if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold) or not 0 <= threshold <= 1):
            raise ValueError('threshold must be finite and between 0 and 1')
        self._backend, self._embedder = backend, embedder
        self._enabled, self._ttl_seconds, self._threshold = enabled, ttl_seconds, threshold

    async def get(
        self, request: EstimationRequest, *, bucket: str, prompt_version: str, model: str,
        embedding_model: str,
    ) -> EstimationResult | None:
        if not self._enabled or self._backend is None or self._embedder is None:
            return None
        try:
            request = EstimationRequest.model_validate(request.model_dump(mode='json'))
            key = semantic_cache_key(request, bucket=bucket, prompt_version=prompt_version,
                                     model=model, embedding_model=embedding_model)
            payload = await self._backend.get(key)
            if not isinstance(payload, (str, bytes)) or len(payload) > 1_000_000:
                return None
            entry = _Entry.model_validate_json(payload)
            if entry.key != key:
                return None
            stored = _unit_vector(entry.vector)
            query = _unit_vector(await self._embedder.embed(request.description))
            if len(stored) != len(query):
                return None
            score = math.fsum(a * b for a, b in zip(stored, query))
            return entry.result if score >= self._threshold else None
        except Exception:
            return None

    async def set(
        self, request: EstimationRequest, result: EstimationResult, *, bucket: str,
        prompt_version: str, model: str, embedding_model: str,
    ) -> None:
        if not self._enabled or self._backend is None or self._embedder is None:
            return
        try:
            if not isinstance(result, EstimationResult):
                return
            result = EstimationResult.model_validate(result.model_dump(mode='json'))
            request = EstimationRequest.model_validate(request.model_dump(mode='json'))
            key = semantic_cache_key(request, bucket=bucket, prompt_version=prompt_version,
                                     model=model, embedding_model=embedding_model)
            vector = _unit_vector(await self._embedder.embed(request.description))
            payload = _Entry(key=key, vector=vector, result=result).model_dump_json()
            await self._backend.setex(key, self._ttl_seconds, payload)
        except Exception:
            return
