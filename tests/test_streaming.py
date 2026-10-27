import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import APIConnectionError
import httpx

from app.config import Settings
from app.services import llm_service
from app.prompts import SYSTEM_PROMPT


def settings(key='test-key'):
    return Settings(_env_file=None, openai_api_key=key)


def fake_stream(monkeypatch, parts, error=None, usage=None):
    async def chunks():
        for part in parts:
            yield SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content=part))])
        if error:
            raise error
        if usage is not None:
            yield SimpleNamespace(choices=[], usage=usage, model='gpt-4o-mini-provider')

    stream = AsyncMock()
    stream.__aiter__.side_effect = chunks
    stream.__aenter__.return_value = stream
    create = AsyncMock(return_value=stream)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    context = AsyncMock()
    context.__aenter__.return_value = client
    factory = MagicMock(return_value=context)
    monkeypatch.setattr(llm_service, 'AsyncOpenAI', factory)
    return factory, create, context, stream


def test_stream_yields_deltas_and_sends_shared_prompt_with_history(monkeypatch):
    factory, create, context, stream = fake_stream(monkeypatch, [None, 'Estimate ', '42 hours'])
    history = [{'role': 'user', 'content': 'A portal'},
               {'role': 'assistant', 'content': 'Which integrations?'},
               {'role': 'user', 'content': 'HubSpot'}]

    async def consume():
        output = llm_service.stream_estimate(history, settings())
        assert await anext(output) == 'Estimate '
        assert await anext(output) == '42 hours'
        with pytest.raises(StopAsyncIteration):
            await anext(output)

    asyncio.run(consume())
    assert create.call_args.kwargs == {
        'model': 'gpt-4o-mini', 'stream': True,
        'messages': [{'role': 'system', 'content': SYSTEM_PROMPT}, *history],
    }
    assert len(history) == 3
    assert factory.call_args.kwargs['timeout'] == 30.0
    assert factory.call_args.kwargs['max_retries'] == 0
    stream.__aexit__.assert_awaited_once()
    context.__aexit__.assert_awaited_once()


@pytest.mark.parametrize('parts', [[], [None, '', '  ']])
def test_stream_rejects_empty_output(monkeypatch, parts):
    fake_stream(monkeypatch, parts)

    async def consume():
        return [part async for part in llm_service.stream_estimate([], settings())]

    with pytest.raises(llm_service.EstimationError, match='empty result'):
        asyncio.run(consume())


def test_stream_safely_reports_midstream_failure(monkeypatch):
    _, _, context, stream = fake_stream(monkeypatch, ['Partial'], APIConnectionError(
        message='secret diagnostics', request=httpx.Request('POST', 'https://example.invalid')))

    async def consume():
        output = llm_service.stream_estimate([], settings())
        assert await anext(output) == 'Partial'
        with pytest.raises(llm_service.EstimationError) as caught:
            await anext(output)
        assert str(caught.value) == 'Estimation provider unavailable. Try again later.'

    asyncio.run(consume())
    stream.__aexit__.assert_awaited_once()
    context.__aexit__.assert_awaited_once()


@pytest.mark.parametrize('key', [None, '', '   '])
def test_stream_missing_key_never_calls_provider(monkeypatch, key):
    factory, _, _, _ = fake_stream(monkeypatch, [])

    async def consume():
        return [part async for part in llm_service.stream_estimate([], settings(key))]

    with pytest.raises(llm_service.EstimationError, match='OPENAI_API_KEY'):
        asyncio.run(consume())
    factory.assert_not_called()


def test_stream_collects_provider_usage_without_yielding_metadata(monkeypatch):
    _, create, _, _ = fake_stream(monkeypatch, ['Estimate ', '42 hours'],
        usage=SimpleNamespace(prompt_tokens=123, completion_tokens=17))
    clock = iter([10.0, 12.5])
    monkeypatch.setattr(llm_service, 'perf_counter', lambda: next(clock), raising=False)
    metrics = llm_service.CallMetrics()

    async def consume():
        return [part async for part in llm_service.stream_estimate(
            [], settings(), metrics=metrics)]

    assert asyncio.run(consume()) == ['Estimate ', '42 hours']
    assert create.call_args.kwargs['stream_options'] == {'include_usage': True}
    assert metrics.model == 'gpt-4o-mini-provider'
    assert metrics.input_tokens == 123
    assert metrics.output_tokens == 17
    assert metrics.elapsed_seconds == 2.5


def test_stream_missing_usage_is_not_reported_as_zero(monkeypatch):
    fake_stream(monkeypatch, ['Estimate'])
    metrics = llm_service.CallMetrics()

    async def consume():
        return [part async for part in llm_service.stream_estimate(
            [], settings(), metrics=metrics)]

    assert asyncio.run(consume()) == ['Estimate']
    assert metrics.model == 'gpt-4o-mini'
    assert metrics.input_tokens is None
    assert metrics.output_tokens is None
    assert metrics.elapsed_seconds >= 0


def test_failed_stream_does_not_publish_success_metrics(monkeypatch):
    fake_stream(monkeypatch, ['Partial'], APIConnectionError(
        message='secret diagnostics', request=httpx.Request('POST', 'https://example.invalid')))
    metrics = llm_service.CallMetrics()

    async def consume():
        return [part async for part in llm_service.stream_estimate(
            [], settings(), metrics=metrics)]

    with pytest.raises(llm_service.EstimationError):
        asyncio.run(consume())
    assert metrics.model is None
    assert metrics.input_tokens is None
    assert metrics.output_tokens is None
    assert metrics.elapsed_seconds is None
