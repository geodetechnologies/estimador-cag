import hashlib
import json

from pydantic import BaseModel, ConfigDict

from app.cache.base import AsyncCacheBackend
from app.schemas import EstimationRequest, EstimationResult


class _CacheEntry(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    key: str
    result: EstimationResult


def exact_cache_key(request: EstimationRequest, *, prompt_version: str, model: str) -> str:
    """Hash a canonical typed estimation identity without exposing request text."""
    identity = {
        'request': request.model_dump(mode='json'),
        'prompt_version': prompt_version,
        'model': model,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    digest = hashlib.sha256(encoded.encode('utf-8')).hexdigest()
    return f'estimation:exact:v1:{digest}'


class ExactCache:
    def __init__(
        self, backend: AsyncCacheBackend | None = None, *, enabled: bool = False,
        ttl_seconds: int = 86400,
    ) -> None:
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int) or ttl_seconds <= 0:
            raise ValueError('ttl_seconds must be a positive integer')
        self._backend = backend
        self._enabled = enabled
        self._ttl_seconds = ttl_seconds

    async def get(
        self, request: EstimationRequest, *, prompt_version: str, model: str,
    ) -> EstimationResult | None:
        if not self._enabled or self._backend is None:
            return None
        try:
            key = exact_cache_key(request, prompt_version=prompt_version, model=model)
            payload = await self._backend.get(key)
            if payload is None:
                return None
            entry = _CacheEntry.model_validate_json(payload)
            return entry.result if entry.key == key else None
        except Exception:
            return None

    async def set(
        self, request: EstimationRequest, result: EstimationResult, *,
        prompt_version: str, model: str,
    ) -> None:
        if not self._enabled or self._backend is None:
            return
        try:
            if not isinstance(result, EstimationResult):
                return
            # Revalidate the data, including instances created or mutated without validation.
            validated = EstimationResult.model_validate(result.model_dump(mode='json'))
            key = exact_cache_key(request, prompt_version=prompt_version, model=model)
            payload = _CacheEntry(key=key, result=validated).model_dump_json()
            await self._backend.setex(key, self._ttl_seconds, payload)
        except Exception:
            return
