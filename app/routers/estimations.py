"""Validate estimation requests and expose safe service errors over HTTP."""

import json
from collections.abc import AsyncIterator
from typing import Annotated

import anyio
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from app.config import Settings, get_settings
from app.prompts.loader import render_estimation_prompt
from app.schemas import EstimationRequest, EstimationResponse
from app.services import llm_service

router = APIRouter(prefix='/api/v1', tags=['estimations'])


@router.post('/estimate', response_model=EstimationResponse)
async def estimate(
    payload: EstimationRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> EstimationResponse:
    system, user = render_estimation_prompt(payload)
    try:
        result = await llm_service.structured_estimate(user, settings, system=system)
    except llm_service.EstimationError as error:
        raise HTTPException(status_code=error.status_code, detail=error.detail) from None
    return EstimationResponse(result=result, prompt_version='v1')


_SAFE_ERROR = 'Estimation provider unavailable. Try again later.'
_EMPTY_ERROR = 'Estimation provider returned an empty result.'


def _event(name: str, data: dict[str, str]) -> str:
    return f'event: {name}\ndata: {json.dumps(data, ensure_ascii=True)}\n\n'


class _EstimationStreamResponse(StreamingResponse):
    """Close a prefetched provider stream even if the client disconnects immediately."""

    def __init__(self, content: AsyncIterator[str], provider: AsyncIterator[str]):
        super().__init__(content, media_type='text/event-stream',
                         headers={'Cache-Control': 'no-cache'})
        self.provider = provider

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await self.provider.aclose()


class _DisconnectedResponse(Response):
    """Do not send headers or a body after an observed disconnect."""

    async def __call__(self, scope, receive, send):
        pass


@router.post('/estimate/stream', response_class=StreamingResponse,
             responses={200: {'content': {'text/event-stream': {}}}})
async def estimate_stream(
    payload: EstimationRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    request: Request,
) -> Response:
    """Stream JSON SSE events; failures before the first delta retain HTTP status."""
    system, user = render_estimation_prompt(payload)
    provider = llm_service.stream_estimate([{'role': 'user', 'content': user}],
                                           settings, system=system)
    first = None
    failure = None
    disconnected = False

    try:
        async with anyio.create_task_group() as group:
            async def prefetch():
                nonlocal first, failure
                try:
                    first = await anext(provider)
                except Exception as error:
                    failure = error
                finally:
                    group.cancel_scope.cancel()

            async def watch_disconnect():
                nonlocal disconnected
                while True:
                    if (await request.receive())['type'] == 'http.disconnect':
                        disconnected = True
                        group.cancel_scope.cancel()
                        return

            group.start_soon(prefetch)
            group.start_soon(watch_disconnect)
    except BaseException:
        with anyio.CancelScope(shield=True):
            await provider.aclose()
        raise

    if disconnected:
        with anyio.CancelScope(shield=True):
            await provider.aclose()
        return _DisconnectedResponse()
    if isinstance(failure, llm_service.EstimationError):
        raise HTTPException(status_code=failure.status_code, detail=failure.detail) from None
    if isinstance(failure, StopAsyncIteration):
        raise HTTPException(status_code=502, detail=_EMPTY_ERROR) from None
    if failure is not None:
        raise HTTPException(status_code=502, detail=_SAFE_ERROR) from None

    async def events() -> AsyncIterator[str]:
        yield _event('metadata', {'prompt_version': 'v1'})
        yield _event('delta', {'text': first})
        try:
            async for text in provider:
                yield _event('delta', {'text': text})
        except llm_service.EstimationError as error:
            yield _event('error', {'detail': error.detail})
        except Exception:
            yield _event('error', {'detail': _SAFE_ERROR})
        else:
            yield _event('done', {})

    return _EstimationStreamResponse(events(), provider)
