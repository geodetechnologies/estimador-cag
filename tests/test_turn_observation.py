import asyncio
import io
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from docx import Document

from app.config import Settings, get_settings
from app.main import app
from app.services import llm_service
from test_sessions import EMPTY_METADATA, RESULT
from test_usage_metrics import provider_response


FIELDS = {'turn_index', 'session_id', 'enriched_transcript_chars', 'attachments_total_chars',
          'messages_in_window', 'anchors_count', 'summary_chars', 'tokens_in', 'tokens_out',
          'cost_usd', 'latency_ms', 'cache_hit_kind', 'last_resolved_tier'}


@pytest.fixture
def observed(monkeypatch, caplog):
    calls = []
    failures = {'metadata': False}

    async def respond(**kwargs):
        calls.append(kwargs)
        system = kwargs['messages'][0]['content']
        if 'project metadata extractor' in system:
            content = 'invalid private metadata' if failures['metadata'] else json.dumps(EMPTY_METADATA)
        elif 'cumulative record' in system:
            content = 'private cumulative summary'
        else:
            content = json.dumps(RESULT)
        return provider_response(content, usage=(10, 4))

    create = AsyncMock(side_effect=respond)
    context = AsyncMock()
    context.__aenter__.return_value = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(llm_service, 'AsyncOpenAI', MagicMock(return_value=context))
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, openai_api_key='private-key', max_turns=1)
    caplog.set_level(logging.INFO)
    yield calls, failures, create, caplog
    app.dependency_overrides.clear()


def events(caplog):
    records = [r.getMessage() for r in caplog.records if r.getMessage().startswith('turn_observed ')]
    result = [json.loads(record.removeprefix('turn_observed ')) for record in records]
    assert all(set(event) == FIELDS for event in result)
    assert all(isinstance(event['latency_ms'], (int, float)) and event['latency_ms'] >= 0 for event in result)
    return result


def document_bytes():
    document = Document()
    document.add_paragraph('private attachment contents')
    output = io.BytesIO()
    document.save(output)
    return output.getvalue()


def test_observe_success_and_attachment_counts_without_content(observed):
    calls, _, _, caplog = observed
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            session = (await client.post('/sessions')).json()['session_id']
            response = await client.post(f'/sessions/{session}/estimate', data={'transcript': 'private transcript'},
                files={'attachments': ('private-name.docx', document_bytes(), 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')})
            assert response.status_code == 200
            assert response.json() == {'result': RESULT, 'project_metadata': EMPTY_METADATA,
                                       'prompt_version': 'v1', 'cached': False}
            assert (await client.get(f'/sessions/{session}')).json() == {
                'session_id': session, 'history_window_message_count': 2, 'max_turns': 1}
            [event] = events(caplog)
            assert event == {'turn_index': 1, 'session_id': session,
                'enriched_transcript_chars': len(calls[0]['messages'][-1]['content']),
                'attachments_total_chars': len('private attachment contents'),
                'messages_in_window': 2, 'anchors_count': 0, 'summary_chars': 0,
                'tokens_in': 20, 'tokens_out': 8, 'cost_usd': pytest.approx(0.0000078),
                'latency_ms': event['latency_ms'], 'cache_hit_kind': 'none', 'last_resolved_tier': 'default'}
    asyncio.run(run())
    for secret in ('private transcript', 'private attachment contents', 'private-name.docx',
                   'private-key', 'A detailed project estimate'):
        assert secret not in '\n'.join(r.getMessage() for r in caplog.records if r.getMessage().startswith('turn_observed '))


def test_failures_and_validation_each_emit_unknowns_without_committing(observed):
    calls, failure, _, caplog = observed
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            session = (await client.post('/sessions')).json()['session_id']
            url = f'/sessions/{session}/estimate'
            missing = await client.post('/sessions/private-unknown-id/estimate', data={'transcript': 'private text'})
            assert missing.status_code == 404 and missing.json() == {'detail': 'Session not found.'}
            invalid = await client.post(url, data={})
            assert invalid.status_code == 422
            blank = await client.post(url, data={'transcript': ' '})
            assert blank.status_code == 422 and blank.json() == {'detail': 'Transcript must not be blank.'}
            attachment = await client.post(url, data={'transcript': 'private text'},
                files={'attachments': ('private.txt', b'private attachment', 'text/plain')})
            assert attachment.status_code == 422 and attachment.json() == {'detail': 'Unsupported attachment; use PDF or DOCX.'}
            failure['metadata'] = True
            rejected = await client.post(url, data={'transcript': 'private failure'})
            assert rejected.status_code == 502 and rejected.json() == {'detail': 'Estimation provider returned an invalid result.'}
            assert (await client.get(f'/sessions/{session}')).json() == {
                'session_id': session, 'history_window_message_count': 0, 'max_turns': 1}
            failure['metadata'] = False
            assert (await client.post(url, data={'transcript': 'fresh'})).status_code == 200
            assert [m['role'] for m in calls[-2]['messages']] == ['system', 'user']
            observed_events = events(caplog)
            assert len(observed_events) == 6
            assert observed_events[0]['session_id'] is None
            for event in observed_events[:5]:
                assert event['turn_index'] is None
                assert event['tokens_in'] is event['tokens_out'] is event['cost_usd'] is None
            assert observed_events[1]['session_id'] == session
            assert observed_events[1]['enriched_transcript_chars'] is None
            assert observed_events[3]['attachments_total_chars'] is None
            assert observed_events[4]['enriched_transcript_chars'] == len('private failure')
            assert observed_events[4]['attachments_total_chars'] == 0
            assert observed_events[5]['turn_index'] == 1
    asyncio.run(run())
    event_log = '\n'.join(r.getMessage() for r in caplog.records if r.getMessage().startswith('turn_observed '))
    for secret in ('private-unknown-id', 'private text', 'private.txt', 'private attachment',
                   'private failure', 'invalid private metadata', 'private-key'):
        assert secret not in event_log


def test_concurrent_turns_commit_number_and_snapshot_in_lock_order(observed):
    calls, _, create, caplog = observed
    original = create.side_effect
    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        async def delayed(**kwargs):
            if not calls:
                started.set()
                await release.wait()
            return await original(**kwargs)
        create.side_effect = delayed
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            session = (await client.post('/sessions')).json()['session_id']
            url = f'/sessions/{session}/estimate'
            first = asyncio.create_task(client.post(url, data={'transcript': 'first'}))
            await asyncio.wait_for(started.wait(), 1)
            second = asyncio.create_task(client.post(url, data={'transcript': 'second'}))
            await asyncio.sleep(0.02)
            assert not second.done()
            release.set()
            one, two = await asyncio.gather(first, second)
            assert one.status_code == two.status_code == 200
            first_event, second_event = events(caplog)
            assert [first_event['turn_index'], second_event['turn_index']] == [1, 2]
            assert first_event['session_id'] == second_event['session_id'] == session
            assert first_event['summary_chars'] == 0
            assert second_event['summary_chars'] == len('private cumulative summary')
            assert second_event['tokens_in'] == 30 and second_event['tokens_out'] == 12
            assert second_event['messages_in_window'] == 2
    asyncio.run(run())


def test_success_with_unknown_provider_usage_is_unknown(observed):
    _, _, create, caplog = observed
    async def no_usage(**kwargs):
        system = kwargs['messages'][0]['content']
        content = json.dumps(EMPTY_METADATA if 'project metadata extractor' in system else RESULT)
        return provider_response(content, usage=None)
    create.side_effect = no_usage
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            session = (await client.post('/sessions')).json()['session_id']
            assert (await client.post(f'/sessions/{session}/estimate', data={'transcript': 'hello'})).status_code == 200
            [event] = events(caplog)
            assert event['turn_index'] == 1
            assert event['tokens_in'] is event['tokens_out'] is event['cost_usd'] is None
    asyncio.run(run())


def test_provider_model_cannot_inject_secret_into_usage_logs(observed):
    _, _, create, caplog = observed
    async def untrusted_model(**kwargs):
        system = kwargs['messages'][0]['content']
        content = json.dumps(EMPTY_METADATA if 'project metadata extractor' in system else RESULT)
        return provider_response(content, model='private-key', usage=(10, 4))
    create.side_effect = untrusted_model
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            session = (await client.post('/sessions')).json()['session_id']
            assert (await client.post(f'/sessions/{session}/estimate', data={'transcript': 'hello'})).status_code == 200
            [event] = events(caplog)
            assert event['tokens_in'] == 20 and event['tokens_out'] == 8
            assert event['cost_usd'] is None
    asyncio.run(run())
    assert 'private-key' not in caplog.text


def test_concurrent_failure_does_not_consume_a_turn_index(observed):
    calls, failure, create, caplog = observed
    original = create.side_effect
    failure['metadata'] = True
    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        async def delayed(**kwargs):
            if not calls:
                started.set()
                await release.wait()
            response = await original(**kwargs)
            if 'project metadata extractor' in kwargs['messages'][0]['content']:
                failure['metadata'] = False
            return response
        create.side_effect = delayed
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            session = (await client.post('/sessions')).json()['session_id']
            url = f'/sessions/{session}/estimate'
            first = asyncio.create_task(client.post(url, data={'transcript': 'failed'}))
            await asyncio.wait_for(started.wait(), 1)
            second = asyncio.create_task(client.post(url, data={'transcript': 'committed'}))
            await asyncio.sleep(0.02)
            assert not second.done()
            release.set()
            rejected, accepted = await asyncio.gather(first, second)
            assert rejected.status_code == 502 and accepted.status_code == 200
            assert [event['turn_index'] for event in events(caplog)] == [None, 1]
            assert [message['role'] for message in calls[2]['messages']] == ['system', 'user']
            assert (await client.get(f'/sessions/{session}')).json()['history_window_message_count'] == 2
    asyncio.run(run())


def test_observation_logging_failure_does_not_change_response(observed, monkeypatch):
    from app.routers import sessions
    monkeypatch.setattr(sessions.logger, 'info', MagicMock(side_effect=RuntimeError('private log error')))
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            session = (await client.post('/sessions')).json()['session_id']
            result = await client.post(f'/sessions/{session}/estimate', data={'transcript': 'hello'})
            assert result.status_code == 200 and result.json()['result'] == RESULT
            assert (await client.get(f'/sessions/{session}')).json()['history_window_message_count'] == 2
    asyncio.run(run())
