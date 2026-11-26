"""Behavioral tests for the unexposed synthetic stress ASGI boundary."""
import asyncio
import json
from functools import partial
from types import SimpleNamespace

import httpx
import pytest

from evals.stress.service import StressRun

TOKEN = b'0123456789abcdef0123456789abcdef'
AUTH = {'authorization': 'Bearer ' + TOKEN.decode()}
ERROR = {'error': 'invalid request'}


def exercise(coro):
    return asyncio.run(coro)


def make_app(calls):
    from evals.stress.http_service import StressHTTP

    async def estimate(session, text):
        calls.append(text)
        session.history.add_turn(text, 'PRIVATE ANSWER')
        session.summary = 'Harbor PRIVATE SNAPSHOT'
        return 'PRIVATE PROVIDER RESULT'

    return StressHTTP(StressRun('growing', 3, estimate), TOKEN, '127.0.0.1:8000')


def request(app, *, method='POST', path='/step', headers=None, content=b'{"turn":1}', client=('127.0.0.1', 1234)):
    async def send():
        transport = httpx.ASGITransport(app=app, client=client)
        async with httpx.AsyncClient(transport=transport, base_url='http://127.0.0.1:8000') as http:
            return await http.request(method, path, headers=AUTH if headers is None else headers, content=content)
    return exercise(send())


def test_ordered_steps_return_only_derived_results_and_no_private_content(caplog):
    calls = []
    app = make_app(calls)
    first = request(app)
    second = request(app, content=b'{"turn":2}')
    assert first.status_code == second.status_code == 200
    assert first.json() == {'status': 'ok', 'turn': 1, 'memory': [], 'cost_usd': None}
    assert second.json() == {'status': 'ok', 'turn': 2, 'memory': [{'fact_turn': 1, 'score': 1.0}], 'cost_usd': None}
    assert len(calls) == 2
    visible = first.text + second.text + caplog.text + repr(app)
    for secret in ('PRIVATE', 'Harbor', calls[0], TOKEN.decode()):
        assert secret not in visible


@pytest.mark.parametrize(('changes'), [
    {'headers': {}},
    {'headers': {'authorization': 'Bearer WRONG PRIVATE'}},
    {'headers': [('authorization', AUTH['authorization']), ('authorization', AUTH['authorization'])]},
    {'headers': {**AUTH, 'host': 'evil.example'}},
    {'headers': {**AUTH, 'origin': 'http://evil.example'}},
    {'headers': {**AUTH, 'origin': 'http://127.0.0.1:8001'}},
    {'headers': {**AUTH, 'x-forwarded-for': '127.0.0.1'}},
    {'client': ('192.0.2.1', 1234)},
    {'client': None},
    {'path': '/step?PRIVATE=1'},
    {'path': '/PRIVATE'},
    {'method': 'GET'},
    {'content': b'{"turn":1,"turn":2}'},
    {'content': b'{"turn":1,"transcript":"PRIVATE"}'},
    {'content': b'{"turn":true}'},
    {'content': b'{"turn":1.0}'},
    {'content': b'{"turn":0}'},
    {'content': b'{"turn":22}'},
    {'content': b'[]'},
    {'content': b'{bad PRIVATE'},
    {'content': b'\xff'},
    {'content': b' ' * 1025},
])
def test_rejection_is_fixed_nonreflecting_and_has_no_effect(changes, caplog):
    calls = []
    app = make_app(calls)
    response = request(app, **changes)
    assert response.status_code == 400
    assert response.json() == ERROR
    assert 'PRIVATE' not in response.text + caplog.text
    assert TOKEN.decode() not in response.text + caplog.text
    assert calls == []
    assert request(app).json()['status'] == 'ok'
    assert len(calls) == 1


def test_matching_origin_is_allowed_and_wrong_turn_invalidates():
    calls = []
    app = make_app(calls)
    headers = {**AUTH, 'origin': 'http://127.0.0.1:8000'}
    first = request(app, headers=headers, content=b'{"turn":2}')
    second = request(app, headers=headers)
    assert first.status_code == second.status_code == 409
    assert first.json() == second.json() == {'error': 'run invalid'}
    assert calls == []


def test_injected_token_is_not_taken_from_request_and_configuration_is_checked():
    from evals.stress.http_service import StressHTTP
    async def estimate(session, text):
        pass
    run = StressRun('growing', 1, estimate)
    for token, host in [(b'short', '127.0.0.1:8000'), (TOKEN, '0.0.0.0:8000'),
                        (TOKEN, 'localhost:8000'), (TOKEN, '127.0.0.1:0')]:
        with pytest.raises(ValueError):
            StressHTTP(run, token, host)


def test_failure_to_send_a_result_poisons_the_run():
    async def check():
        calls = []
        app = make_app(calls)
        scope = {'type': 'http', 'method': 'POST', 'path': '/step', 'query_string': b'',
                 'client': ('127.0.0.1', 1234), 'headers': [(b'host', b'127.0.0.1:8000'),
                 (b'authorization', b'Bearer ' + TOKEN)]}
        sent = False
        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {'type': 'http.request', 'body': b'{"turn":1}', 'more_body': False}
            await asyncio.Event().wait()
        async def send(message):
            raise OSError('PRIVATE SEND FAILURE')
        await app(scope, receive, send)
        assert len(calls) == 1
        transport = httpx.ASGITransport(app=app, client=('127.0.0.1', 1234))
        async with httpx.AsyncClient(transport=transport, base_url='http://127.0.0.1:8000') as client:
            result = await client.post('/step', headers=AUTH, json={'turn': 2})
        assert result.status_code == 409 and result.json() == {'error': 'run invalid'}
        assert len(calls) == 1
    exercise(check())


def test_unbounded_empty_body_chunks_are_rejected_without_advancing():
    async def check():
        calls = []
        app = make_app(calls)
        scope = {'type': 'http', 'method': 'POST', 'path': '/step', 'query_string': b'',
                 'client': ('127.0.0.1', 1234), 'headers': [(b'host', b'127.0.0.1:8000'),
                 (b'authorization', b'Bearer ' + TOKEN)]}
        count = 0
        output = []
        async def receive():
            nonlocal count
            count += 1
            if count <= 1025:
                return {'type': 'http.request', 'body': b'', 'more_body': True}
            if count == 1026:
                return {'type': 'http.request', 'body': b'{"turn":1}', 'more_body': False}
            await asyncio.Event().wait()
        async def send(message):
            output.append(message)
        await app(scope, receive, send)
        assert output[0]['status'] == 400
        assert json.loads(output[1]['body']) == ERROR
        assert calls == []
    exercise(check())


def test_trusted_core_invalidation_preserves_earlier_success_and_prevents_later_step():
    async def check():
        calls = []
        async def estimate(session, text):
            calls.append(text)
            session.history.add_turn(text, 'synthetic')
        run = StressRun('growing', 3, estimate)
        assert (await run.step(1))['status'] == 'ok'
        run.invalidate()
        assert await run.step(2) == {'status': 'invalid'}
        assert len(calls) == 1
    exercise(check())


def test_proven_invalid_estimate_receipt_is_authenticated_and_does_not_leak(monkeypatch, caplog):
    from pydantic import SecretStr
    from app.services import estimation, llm_service
    from evals.stress.run import FakeProvider
    from evals.stress.http_service import StressHTTP
    from evals.stress.scenarios import generate_scenario

    calls = []
    caplog.set_level('DEBUG')

    class Provider(FakeProvider):
        async def create(self, **kwargs):
            metadata = 'project metadata extractor' in kwargs['messages'][0]['content']
            if not metadata:
                calls.append(kwargs['messages'])
                if len(calls) == 2:
                    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{PRIVATE'))])
            return await super().create(**kwargs)

    monkeypatch.setattr(llm_service, 'AsyncOpenAI', Provider)
    settings = SimpleNamespace(openai_api_key=SecretStr('PRIVATE KEY'), openai_model='simulated')
    app = StressHTTP(StressRun('growing', 3, partial(estimation.estimate_conversational, settings=settings),
                               attachment_size=0), TOKEN, '127.0.0.1:8000')
    assert request(app, headers={}).status_code == 400
    first = request(app)
    rejected = request(app, content=b'{"turn":2}')
    third = request(app, content=b'{"turn":3}')
    assert [first.status_code, rejected.status_code, third.status_code] == [200, 200, 200]
    receipt = rejected.json()
    assert receipt == {'status': 'invalid_estimate', 'turn': 2, 'reason': 'estimate_validation',
                       'attachment_bytes': 0, 'enriched_transcript_chars': len(generate_scenario('growing', 3).turns[1].message),
                       'attachments_total_chars': 0}
    assert third.json()['observation']['turn_index'] == 2
    assert third.json()['memory'] == [{'fact_turn': 1, 'score': 1.0}]
    assert len(calls) == 3
    assert all(secret not in first.text + rejected.text + third.text + caplog.text
               for secret in ('PRIVATE', TOKEN.decode(), 'Harbor'))


def test_malformed_receipt_is_never_reflected_and_invalidates(monkeypatch):
    from evals.stress.http_service import StressHTTP
    calls = []
    async def estimate(session, text):
        calls.append(text)
    stress = StressRun('growing', 3, estimate)
    async def forged(turn):
        return {'status': 'invalid_estimate', 'turn': turn, 'reason': 'estimate_validation',
                'attachment_bytes': 0, 'enriched_transcript_chars': 1,
                'attachments_total_chars': 0, 'provider': 'PRIVATE'}
    monkeypatch.setattr(stress, 'step', forged)
    app = StressHTTP(stress, TOKEN, '127.0.0.1:8000')
    response = request(app)
    assert response.status_code == 409 and response.json() == {'error': 'run invalid'}
    assert 'PRIVATE' not in response.text and calls == []


def test_disconnect_during_invalid_estimate_poisons_run(monkeypatch):
    from pydantic import SecretStr
    from app.services import estimation, llm_service
    from evals.stress.run import FakeProvider
    from evals.stress.http_service import StressHTTP

    async def check():
        started = asyncio.Event()
        calls = []
        class Provider(FakeProvider):
            async def create(self, **kwargs):
                calls.append(1)
                started.set()
                await asyncio.Event().wait()
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{PRIVATE'))])
        monkeypatch.setattr(llm_service, 'AsyncOpenAI', Provider)
        settings = SimpleNamespace(openai_api_key=SecretStr('PRIVATE KEY'), openai_model='simulated')
        stress = StressRun('growing', 3, partial(estimation.estimate_conversational, settings=settings),
                           attachment_size=0)
        app = StressHTTP(stress, TOKEN, '127.0.0.1:8000')
        scope = {'type': 'http', 'method': 'POST', 'path': '/step', 'query_string': b'',
                 'client': ('127.0.0.1', 1234), 'headers': [(b'host', b'127.0.0.1:8000'),
                 (b'authorization', b'Bearer ' + TOKEN)]}
        first = True
        output = []
        async def receive():
            nonlocal first
            if first:
                first = False
                return {'type': 'http.request', 'body': b'{"turn":1}'}
            await started.wait()
            return {'type': 'http.disconnect'}
        async def send(message):
            output.append(message)
        await asyncio.wait_for(app(scope, receive, send), 1)
        assert output == [] and len(calls) == 1
        assert await stress.step(2) == {'status': 'invalid'}
    exercise(check())


def test_disconnected_waiter_invalidates_active_step_before_it_can_report_success():
    async def check():
        from evals.stress.http_service import StressHTTP
        started, release = asyncio.Event(), asyncio.Event()
        calls, outputs = [], [[], []]

        async def estimate(session, text):
            calls.append(text)
            started.set()
            await release.wait()
            session.history.add_turn(text, 'synthetic')

        app = StressHTTP(StressRun('growing', 3, estimate), TOKEN, '127.0.0.1:8000')
        scope = {'type': 'http', 'method': 'POST', 'path': '/step', 'query_string': b'',
                 'client': ('127.0.0.1', 1234), 'headers': [(b'host', b'127.0.0.1:8000'),
                 (b'authorization', b'Bearer ' + TOKEN)]}

        def receiver(turn, disconnect):
            first = True
            async def receive():
                nonlocal first
                if first:
                    first = False
                    return {'type': 'http.request', 'body': json.dumps({'turn': turn}).encode()}
                if disconnect:
                    return {'type': 'http.disconnect'}
                await asyncio.Event().wait()
            return receive

        async def invoke(index, disconnect):
            async def send(message):
                outputs[index].append(message)
            await app(scope, receiver(index + 1, disconnect), send)

        first = asyncio.create_task(invoke(0, False))
        await asyncio.wait_for(started.wait(), 1)
        second = asyncio.create_task(invoke(1, True))
        await asyncio.sleep(0.01)
        release.set()
        await asyncio.wait_for(asyncio.gather(first, second), 1)
        assert len(calls) == 1
        assert outputs[0][0]['status'] == 409
        assert json.loads(outputs[0][1]['body']) == {'error': 'run invalid'}
    exercise(check())


@pytest.mark.parametrize('disconnect', [False, True])
def test_cancel_or_disconnect_during_step_poisons_run(disconnect):
    async def check():
        from evals.stress.http_service import StressHTTP
        started = asyncio.Event()
        calls = []
        async def estimate(session, text):
            calls.append(text)
            started.set()
            await asyncio.Event().wait()
        app = StressHTTP(StressRun('growing', 3, estimate), TOKEN, '127.0.0.1:8000')
        scope = {'type': 'http', 'method': 'POST', 'path': '/step', 'query_string': b'',
                 'client': ('127.0.0.1', 1234), 'headers': [(b'host', b'127.0.0.1:8000'),
                 (b'authorization', b'Bearer ' + TOKEN)]}
        received = False
        async def receive():
            nonlocal received
            if not received:
                received = True
                return {'type': 'http.request', 'body': b'{"turn":1}', 'more_body': False}
            await started.wait()
            if disconnect:
                return {'type': 'http.disconnect'}
            await asyncio.Event().wait()
        async def send(message):
            pass
        task = asyncio.create_task(app(scope, receive, send))
        await asyncio.wait_for(started.wait(), 1)
        if not disconnect:
            task.cancel('PRIVATE CANCELLATION')
        try:
            await asyncio.wait_for(task, 1)
        except asyncio.CancelledError as error:
            assert 'PRIVATE' not in str(error)
        transport = httpx.ASGITransport(app=app, client=('127.0.0.1', 1234))
        async with httpx.AsyncClient(transport=transport, base_url='http://127.0.0.1:8000') as client:
            result = await client.post('/step', headers=AUTH, json={'turn': 2})
        assert result.status_code == 409 and result.json() == {'error': 'run invalid'}
        assert len(calls) == 1
    exercise(check())
