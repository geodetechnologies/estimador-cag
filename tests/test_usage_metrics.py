import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.config import Settings, get_settings
from app.main import app
from app.services import llm_service
from test_sessions import EMPTY_METADATA, RESULT


def install_provider(monkeypatch, responses):
    create = AsyncMock(side_effect=responses)
    context = AsyncMock()
    context.__aenter__.return_value = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(llm_service, 'AsyncOpenAI', MagicMock(return_value=context))
    return create


def provider_response(content, model='gpt-4o-mini-2024-07-18', usage=(1000, 500)):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        model=model, usage=None if usage is None else SimpleNamespace(
            prompt_tokens=usage[0], completion_tokens=usage[1]))


def test_opt_in_nonstream_metrics_and_unchanged_request(monkeypatch):
    create = install_provider(monkeypatch, [provider_response(' answer '), provider_response('again')])
    settings = Settings(_env_file=None, openai_api_key='test-key')
    metrics = llm_service.CallMetrics()
    assert asyncio.run(llm_service.estimate('private text', settings, system='private system', metrics=metrics)) == 'answer'
    assert (metrics.model, metrics.input_tokens, metrics.output_tokens) == ('gpt-4o-mini-2024-07-18', 1000, 500)
    assert metrics.elapsed_seconds is not None and metrics.elapsed_seconds >= 0
    assert asyncio.run(llm_service.estimate('private text', settings, system='private system')) == 'again'
    assert create.call_args_list[0] == create.call_args_list[1]
    assert create.call_args.kwargs == {'model': 'gpt-4o-mini', 'messages': [
        {'role': 'system', 'content': 'private system'}, {'role': 'user', 'content': 'private text'}]}


@pytest.mark.parametrize('model,usage', [(None, None), ('unpriced-model', (7, 4)),
                                        ('gpt-4o-mini', None)])
def test_unknowns_never_invented(monkeypatch, model, usage):
    install_provider(monkeypatch, [provider_response('ok', model, usage)])
    metrics = llm_service.CallMetrics()
    asyncio.run(llm_service.estimate('text', Settings(_env_file=None, openai_api_key='x'), metrics=metrics))
    assert metrics.model == model
    assert metrics.input_tokens == (usage[0] if usage else None)
    assert metrics.output_tokens == (usage[1] if usage else None)
    assert metrics.cost_usd is None


def test_failure_does_not_publish_metrics(monkeypatch):
    install_provider(monkeypatch, [provider_response(' '), provider_response('not json')])
    settings = Settings(_env_file=None, openai_api_key='x')
    for call in (lambda m: llm_service.estimate('text', settings, metrics=m),
                 lambda m: llm_service.structured_estimate('text', settings, system='system', metrics=m)):
        metrics = llm_service.CallMetrics()
        with pytest.raises(llm_service.EstimationError):
            asyncio.run(call(metrics))
        assert metrics == llm_service.CallMetrics()


def test_conversation_observes_both_calls_without_sensitive_data(monkeypatch, caplog):
    create = install_provider(monkeypatch, [provider_response(json.dumps(RESULT)),
        provider_response(json.dumps(EMPTY_METADATA), 'not-priced', None)])
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, openai_api_key='x')
    caplog.set_level(logging.INFO, logger='app.services.estimation')
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            session = (await client.post('/sessions')).json()['session_id']
            result = await client.post(f'/sessions/{session}/estimate', data={'transcript': 'private transcript'})
            assert result.status_code == 200
            assert result.json() == {'result': RESULT, 'project_metadata': EMPTY_METADATA,
                                     'prompt_version': 'v1', 'cached': False}
    try:
        asyncio.run(run())
    finally:
        app.dependency_overrides.clear()
    records = [r for r in caplog.records if r.name == 'app.services.estimation']
    assert len(records) == 2
    first, second = [r.getMessage() for r in records]
    assert 'estimation' in first and 'gpt-4o-mini-2024-07-18' in first
    assert '1000' in first and '500' in first and '0.00045' in first
    assert 'metadata' in second and 'not-priced' in second and 'None' in second
    assert all('elapsed' in r.getMessage() for r in records)
    for secret in ('private transcript', 'A detailed project estimate', 'project metadata extractor',
                   'previous_metadata', 'test-key'):
        assert secret not in caplog.text
    assert create.await_count == 2
    assert all(set(c.kwargs) == {'model', 'response_format', 'messages'} for c in create.call_args_list)


def test_logging_failure_does_not_change_success_or_commit(monkeypatch):
    create = install_provider(monkeypatch, [provider_response(json.dumps(RESULT)),
        provider_response(json.dumps(EMPTY_METADATA)), provider_response(json.dumps(RESULT)),
        provider_response(json.dumps(EMPTY_METADATA))])
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, openai_api_key='x')
    from app.services import estimation
    monkeypatch.setattr(estimation.logger, 'info', MagicMock(side_effect=RuntimeError('private log error')))
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            session = (await client.post('/sessions')).json()['session_id']
            first = await client.post(f'/sessions/{session}/estimate', data={'transcript': 'first'})
            assert first.status_code == 200
            second = await client.post(f'/sessions/{session}/estimate', data={'transcript': 'second'})
            assert second.status_code == 200
            assert [m['role'] for m in create.call_args_list[2].kwargs['messages']] == [
                'system', 'user', 'assistant', 'user']
    try:
        asyncio.run(run())
    finally:
        app.dependency_overrides.clear()


def test_failed_turn_emits_no_success_and_preserves_session(monkeypatch, caplog):
    create = install_provider(monkeypatch, [provider_response(json.dumps(RESULT)),
        provider_response('private invalid metadata'), provider_response(json.dumps(RESULT)),
        provider_response(json.dumps(EMPTY_METADATA))])
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, openai_api_key='x')
    caplog.set_level(logging.INFO, logger='app.services.estimation')
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            session = (await client.post('/sessions')).json()['session_id']
            rejected = await client.post(f'/sessions/{session}/estimate', data={'transcript': 'private failure'})
            assert rejected.status_code == 502
            assert rejected.json() == {'detail': 'Estimation provider returned an invalid result.'}
            assert not [r for r in caplog.records if r.name == 'app.services.estimation']
            accepted = await client.post(f'/sessions/{session}/estimate', data={'transcript': 'fresh'})
            assert accepted.status_code == 200
            assert [m['role'] for m in create.call_args_list[2].kwargs['messages']] == ['system', 'user']
    try:
        asyncio.run(run())
    finally:
        app.dependency_overrides.clear()
    assert 'private failure' not in caplog.text and 'private invalid metadata' not in caplog.text
