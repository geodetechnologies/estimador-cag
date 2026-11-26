from typing import Any, Protocol


class AsyncCacheBackend(Protocol):
    """Minimal asynchronous Redis-compatible backend, supplied by the caller."""

    async def get(self, key: str) -> str | bytes | None: ...

    async def setex(self, key: str, ttl_seconds: int, value: str) -> Any: ...
