import asyncio
import json

import anyio
import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app
from app.schemas import EstimationRequest
from app.services import llm_service


PAYLOAD = {'description': 'Build a customer portal for scheduling appointments.',
           'project_type': 'web_saas', 'detail_level': 'summary', 'output_format': 'narrative'}


@pytest.fixture
def client():
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, openai_api_key='test-key')
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def events(text):
    return [(parts[0].removeprefix('event: '), json.loads(parts[1].removeprefix('data: ')))
            for block in text.strip().split('\n\n') if (parts := block.splitlines())]


def test_stream_emits_metadata_deltas_and_completion(client, monkeypatch):
    calls = []

    async def stream(messages, settings, *, system):
        calls.append((messages, system))
        yield 'Hello\n'
        yield 'world'

    monkeypatch.setattr(llm_service, 'stream_estimate', stream)
    response = client.post('/api/v1/estimate/stream', json=PAYLOAD)
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('text/event-stream')
    assert events(response.text) == [('metadata', {'prompt_version': 'v1'}),
                                     ('delta', {'text': 'Hello\n'}),
                                     ('delta', {'text': 'world'}), ('done', {})]
    assert calls[0][0][0]['role'] == 'user'
    assert PAYLOAD['description'] in calls[0][0][0]['content']
    assert 'español' in calls[0][1]


def test_stream_rejects_invalid_request(client):
    assert client.post('/api/v1/estimate/stream', json=PAYLOAD | {'description': 'short'}).status_code == 422


def test_stream_reports_initial_provider_error_as_http(client, monkeypatch):
    async def stream(messages, settings, *, system):
        raise llm_service.EstimationError(503, 'Provider unavailable.')
        yield
    monkeypatch.setattr(llm_service, 'stream_estimate', stream)
    response = client.post('/api/v1/estimate/stream', json=PAYLOAD)
    assert response.status_code == 503
    assert response.json() == {'detail': 'Provider unavailable.'}


def test_stream_empty_provider_result_is_http_error(client, monkeypatch):
    async def stream(messages, settings, *, system):
        raise llm_service.EstimationError(502, 'Estimation provider returned an empty result.')
        yield
    monkeypatch.setattr(llm_service, 'stream_estimate', stream)
    response = client.post('/api/v1/estimate/stream', json=PAYLOAD)
    assert response.status_code == 502
    assert response.json() == {'detail': 'Estimation provider returned an empty result.'}


def test_stream_missing_key_uses_existing_service_error(client):
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, openai_api_key=None)
    response = client.post('/api/v1/estimate/stream', json=PAYLOAD)
    assert response.status_code == 503
    assert response.json() == {'detail': 'Set OPENAI_API_KEY on the server to enable estimation.'}


def test_stream_unexpected_initial_failure_is_safe(client, monkeypatch):
    async def stream(messages, settings, *, system):
        raise RuntimeError('secret diagnostics')
        yield
    monkeypatch.setattr(llm_service, 'stream_estimate', stream)
    response = client.post('/api/v1/estimate/stream', json=PAYLOAD)
    assert response.status_code == 502
    assert response.json() == {'detail': 'Estimation provider unavailable. Try again later.'}


def test_stream_unexpected_midstream_failure_is_safe(client, monkeypatch):
    async def stream(messages, settings, *, system):
        yield 'Partial'
        raise RuntimeError('secret diagnostics')
    monkeypatch.setattr(llm_service, 'stream_estimate', stream)
    response = client.post('/api/v1/estimate/stream', json=PAYLOAD)
    assert response.status_code == 200
    assert events(response.text) == [('metadata', {'prompt_version': 'v1'}),
                                     ('delta', {'text': 'Partial'}),
                                     ('error', {'detail': 'Estimation provider unavailable. Try again later.'})]


def test_stream_closes_prefetched_provider_on_immediate_disconnect(monkeypatch):
    closed = []

    async def stream(messages, settings, *, system):
        try:
            yield 'Partial'
            yield 'rest'
        finally:
            closed.append(True)

    monkeypatch.setattr(llm_service, 'stream_estimate', stream)

    async def exercise():
        from app.routers.estimations import estimate_stream

        async def wait_for_disconnect():
            await anyio.sleep_forever()

        response = await estimate_stream(EstimationRequest(**PAYLOAD),
                                         Settings(_env_file=None, openai_api_key='test-key'),
                                         Request({'type': 'http'}, wait_for_disconnect))

        async def send(message):
            raise OSError('client disconnected')

        async def receive():
            return {'type': 'http.disconnect'}

        from starlette.requests import ClientDisconnect
        with pytest.raises(ClientDisconnect):
            await response({'type': 'http', 'asgi': {'spec_version': '2.4'}}, receive, send)

    asyncio.run(exercise())
    assert closed == [True]


def test_stream_closes_provider_after_partial_output_disconnect(monkeypatch):
    closed = []
    sent = []

    async def stream(messages, settings, *, system):
        try:
            yield 'Partial'
            yield 'rest'
        finally:
            closed.append(True)

    monkeypatch.setattr(llm_service, 'stream_estimate', stream)

    async def exercise():
        from app.routers.estimations import estimate_stream
        from starlette.requests import ClientDisconnect

        async def wait_for_disconnect():
            await anyio.sleep_forever()

        response = await estimate_stream(EstimationRequest(**PAYLOAD),
                                         Settings(_env_file=None, openai_api_key='test-key'),
                                         Request({'type': 'http'}, wait_for_disconnect))

        async def send(message):
            sent.append(message)
            if message.get('body', b'').startswith(b'event: delta'):
                raise OSError('client disconnected')

        async def receive():
            return {'type': 'http.disconnect'}

        with pytest.raises(ClientDisconnect):
            await response({'type': 'http', 'asgi': {'spec_version': '2.4'}}, receive, send)

    asyncio.run(exercise())
    assert closed == [True]
    assert any(message.get('body', b'').startswith(b'event: delta') for message in sent)
    assert not any(b'event: done' in message.get('body', b'') for message in sent)


def test_disconnect_during_prefetch_cancels_and_closes_provider(monkeypatch):
    closed = []
    sent = []

    async def exercise():
        started = anyio.Event()
        received_body = False

        async def stream(messages, settings, *, system):
            try:
                started.set()
                await anyio.sleep_forever()
                yield 'never'
            finally:
                with anyio.CancelScope(shield=True):
                    await anyio.sleep(0)
                    closed.append(True)

        monkeypatch.setattr(llm_service, 'stream_estimate', stream)
        app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, openai_api_key='test-key')

        async def receive():
            nonlocal received_body
            if not received_body:
                received_body = True
                return {'type': 'http.request', 'body': json.dumps(PAYLOAD).encode(), 'more_body': False}
            await started.wait()
            return {'type': 'http.disconnect'}

        async def send(message):
            sent.append(message)

        scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.4'},
                 'http_version': '1.1', 'method': 'POST', 'scheme': 'http',
                 'path': '/api/v1/estimate/stream', 'raw_path': b'/api/v1/estimate/stream',
                 'query_string': b'', 'headers': [(b'content-type', b'application/json')],
                 'client': ('test', 1), 'server': ('test', 80)}
        try:
            with anyio.fail_after(0.25):
                await app(scope, receive, send)
        finally:
            app.dependency_overrides.clear()

    anyio.run(exercise)
    assert closed == [True]
    assert sent == []


def test_prefetch_disconnect_watcher_stops_before_streaming_receive(monkeypatch):
    sent = []
    closed = []

    async def exercise():
        watcher_started = anyio.Event()
        delta_sent = anyio.Event()
        receive_count = 0
        active = 0
        max_active = 0
        watcher_cancelled = False

        async def stream(messages, settings, *, system):
            try:
                await watcher_started.wait()
                yield 'first'
                await anyio.sleep_forever()
            finally:
                closed.append(True)

        monkeypatch.setattr(llm_service, 'stream_estimate', stream)
        app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, openai_api_key='test-key')

        async def receive():
            nonlocal receive_count, active, max_active, watcher_cancelled
            receive_count += 1
            if receive_count == 1:
                return {'type': 'http.request', 'body': json.dumps(PAYLOAD).encode(), 'more_body': False}
            active += 1
            max_active = max(max_active, active)
            try:
                if receive_count == 2:
                    watcher_started.set()
                    try:
                        await anyio.sleep_forever()
                    finally:
                        watcher_cancelled = True
                assert watcher_cancelled
                await delta_sent.wait()
                return {'type': 'http.disconnect'}
            finally:
                active -= 1

        async def send(message):
            sent.append(message)
            if message.get('body', b'').startswith(b'event: delta'):
                delta_sent.set()

        scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.3'},
                 'http_version': '1.1', 'method': 'POST', 'scheme': 'http',
                 'path': '/api/v1/estimate/stream', 'raw_path': b'/api/v1/estimate/stream',
                 'query_string': b'', 'headers': [(b'content-type', b'application/json')],
                 'client': ('test', 1), 'server': ('test', 80)}
        try:
            with anyio.fail_after(1):
                await app(scope, receive, send)
        finally:
            app.dependency_overrides.clear()
        assert watcher_cancelled
        assert max_active == 1
        assert receive_count == 3

    anyio.run(exercise)
    assert closed == [True]
    assert sent[0]['status'] == 200
    assert events(b''.join(message.get('body', b'') for message in sent).decode()) == [
        ('metadata', {'prompt_version': 'v1'}), ('delta', {'text': 'first'})]


def test_stream_reports_error_after_partial_output(client, monkeypatch):
    async def stream(messages, settings, *, system):
        yield 'Partial'
        raise llm_service.EstimationError(502, 'Provider unavailable.')
    monkeypatch.setattr(llm_service, 'stream_estimate', stream)
    response = client.post('/api/v1/estimate/stream', json=PAYLOAD)
    assert response.status_code == 200
    assert events(response.text) == [('metadata', {'prompt_version': 'v1'}),
                                     ('delta', {'text': 'Partial'}),
                                     ('error', {'detail': 'Provider unavailable.'})]
