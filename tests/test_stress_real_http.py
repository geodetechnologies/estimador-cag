"""Real HTTP mode uses fake SDK, fake output and an in-memory local transport only."""
import asyncio
import csv
import io
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, AuthenticationError, OpenAIError

from app.config import Settings
from app.services import llm_service
from evals.stress import http_client, run, service


class Output(io.StringIO):
    def close(self):
        pass


@pytest.fixture
def harness(monkeypatch):
    files, calls, requests, servers = {}, [], [], []

    def open_output(path, mode, **kwargs):
        assert mode == 'x'
        if str(path) in files:
            raise FileExistsError()
        files[str(path)] = Output(newline='')
        return files[str(path)]

    class Provider(run.FakeProvider):
        def __init__(self, **kwargs):
            calls.append(('client', kwargs))
            super().__init__(**kwargs)

        async def create(self, **kwargs):
            calls.append(('create', kwargs))
            response = await super().create(**kwargs)
            response.model = 'gpt-4o-mini-2024-07-18'
            response.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
            return response

    class Socket:
        family = http_client.socket.AF_INET
        def __init__(self, *args):
            self.closed = False
        def set_inheritable(self, value): pass
        def bind(self, address): self.address = (address[0], 45123)
        def getsockname(self): return self.address
        def listen(self, value): pass
        def setblocking(self, value): pass
        def close(self): self.closed = True

    class Server:
        def __init__(self, config):
            self.config, self.started, self.should_exit = config, False, False
            servers.append(self)
        async def serve(self, sockets):
            self.started = True
            while not self.should_exit:
                await asyncio.sleep(0.001)

    def make_client(app, host):
        class Transport(httpx.ASGITransport):
            async def handle_async_request(self, request):
                requests.append(request)
                return await super().handle_async_request(request)
        return httpx.AsyncClient(transport=Transport(app=app, client=('127.0.0.1', 1234)),
                                 base_url='http://' + host)

    monkeypatch.setattr(os, 'environ', {'OPENAI_API_KEY': 'fake-secret'})
    monkeypatch.setattr(Path, 'open', open_output)
    monkeypatch.setattr(llm_service, 'AsyncOpenAI', Provider)
    monkeypatch.setattr(http_client, '_socket_factory', Socket)
    monkeypatch.setattr(http_client.uvicorn, 'Server', Server)
    monkeypatch.setattr(http_client, '_make_client', make_client)
    return SimpleNamespace(files=files, calls=calls, requests=requests, servers=servers, provider=Provider)


def rows(harness):
    return list(csv.DictReader(io.StringIO(harness.files['report.csv'].getvalue())))


def test_real_http_usage_private_data_and_latency(harness, monkeypatch, capsys, caplog):
    times = iter([1.0, 1.125, 2.0, 2.25, 3.0, 3.375])
    settings_calls = []
    init = Settings.__init__
    def settings_init(self, *args, **kwargs):
        settings_calls.append(kwargs.copy())
        return init(self, *args, **kwargs)
    monkeypatch.setattr(Settings, '__init__', settings_init)
    monkeypatch.setattr(service, 'perf_counter', lambda: next(times))
    with caplog.at_level(logging.DEBUG):
        code = run.main(['--real', '--http', 'http://localhost:0', '--lengths', '3',
                         '--attachment-sizes', '5', '--output', 'report.csv'])
    assert (code, *capsys.readouterr()) == (0, 'Wrote 3 real turns.\n', '')
    assert settings_calls == [{'_env_file': None}]
    result = rows(harness)
    assert [row['mode'] for row in result] == ['real-http'] * 3
    assert [row['turn_index'] for row in result] == ['1', '2', '3']
    assert [float(row['latency_ms']) for row in result] == [125, 250, 375]
    assert all(row['tokens_in'] == '20' and row['tokens_out'] == '10' for row in result)
    assert all(float(row['cost_usd']) == pytest.approx(0.000009) for row in result)
    assert all(row['session_id'] == '' and row['attachment_bytes'] == '5120' for row in result)
    assert len(harness.requests) == 4
    assert all(req.url.host == '127.0.0.1' and req.headers['authorization'].startswith('Bearer ') for req in harness.requests)
    assert len([call for call in harness.calls if call[0] == 'create']) == 6
    assert all(call[1]['max_retries'] == 0 and call[1]['timeout'] == 30.0 for call in harness.calls if call[0] == 'client')
    visible = harness.files['report.csv'].getvalue() + caplog.text
    for private in ('fake-secret', 'Harbor', 'SYNTHETIC STRESS FIXTURE', 'Synthetic project estimate.',
                    harness.requests[0].headers['authorization']):
        assert private not in visible


def test_partial_unknown_usage_is_not_zero(harness, monkeypatch, capsys):
    original = harness.provider.create
    async def partial(self, **kwargs):
        response = await original(self, **kwargs)
        if len([c for c in harness.calls if c[0] == 'create']) == 2:
            response.usage = None
        return response
    monkeypatch.setattr(harness.provider, 'create', partial)
    code = run.main(['--real', '--http', 'http://127.0.0.1:0', '--lengths', '1', '--output', 'report.csv'])
    assert (code, *capsys.readouterr()) == (0, 'Wrote 1 real turns.\n', '')
    row = rows(harness)[0]
    assert row['tokens_in'] == row['tokens_out'] == row['cost_usd'] == ''
    assert float(row['latency_ms']) >= 0


@pytest.mark.parametrize('url', ['https://localhost:1', 'http://example.com:80', 'http://localhost:65536'])
def test_invalid_http_and_existing_output_cannot_call_provider(harness, capsys, url):
    code = run.main(['--real', '--http', url, '--output', 'report.csv'])
    assert (code, *capsys.readouterr()) == (2, '', 'Error: invalid configuration.\n')
    assert not harness.calls and not harness.files and not harness.requests
    harness.files['report.csv'] = io.StringIO('existing')
    code = run.main(['--real', '--http', 'http://localhost:0', '--output', 'report.csv'])
    assert (code, *capsys.readouterr()) == (1, '', 'Error: output already exists.\n')
    assert harness.files['report.csv'].getvalue() == 'existing'
    assert not harness.calls and not harness.requests


@pytest.mark.parametrize(('options', 'mode', 'message'), [
    ([], 'simulated', 'simulated'), (['--real'], 'real', 'real'),
])
def test_in_process_modes_remain_available(harness, capsys, options, mode, message):
    code = run.main([*options, '--lengths', '1', '--output', 'report.csv'])
    assert (code, *capsys.readouterr()) == (0, f'Wrote 1 {message} turns.\n', '')
    row = rows(harness)[0]
    assert row['mode'] == mode and row['turn_index'] == '1'
    if mode == 'simulated':
        assert not harness.calls and row['tokens_in'] == row['latency_ms'] == ''
    else:
        assert row['tokens_in'] == '20' and float(row['latency_ms']) >= 0


def test_simulated_http_keeps_unknown_measurements(harness, capsys):
    code = run.main(['--http', 'http://localhost:0', '--lengths', '1', '--output', 'report.csv'])
    assert (code, *capsys.readouterr()) == (0, 'Wrote 1 simulated turns.\n', '')
    row = rows(harness)[0]
    assert row['mode'] == 'simulated-http'
    assert all(row[key] == '' for key in ('tokens_in', 'tokens_out', 'cost_usd', 'latency_ms'))
    assert not harness.calls


def test_failed_provider_invalidates_without_retries(harness, monkeypatch, capsys, caplog):
    async def fail(self, **kwargs):
        harness.calls.append(('failure', None))
        logging.error('fake-secret PRIVATE')
        raise RuntimeError('fake-secret PRIVATE')
    monkeypatch.setattr(harness.provider, 'create', fail)
    with caplog.at_level(logging.DEBUG):
        code = run.main(['--real', '--http', 'http://localhost:0', '--lengths', '3', '--output', 'report.csv'])
    assert (code, *capsys.readouterr()) == (1, '', 'Error: real execution or output failed.\n')
    assert len([c for c in harness.calls if c[0] == 'failure']) == 1
    assert len(harness.requests) == 1
    assert rows(harness) == []
    assert 'PRIVATE' not in caplog.text and 'fake-secret' not in caplog.text


def test_http_provider_category_is_sanitized_to_generic(harness, monkeypatch, capsys, caplog):
    async def fail(self, **kwargs):
        harness.calls.append(('failure', None))
        raise OpenAIError('PRIVATE fake-secret provider-content')
    monkeypatch.setattr(harness.provider, 'create', fail)
    with caplog.at_level(logging.DEBUG):
        code = run.main(['--real', '--http', 'http://localhost:0', '--lengths', '1', '--output', 'report.csv'])
    out, err = capsys.readouterr()
    assert (code, out, err) == (1, '', 'Error: real execution or output failed.\n')
    assert len([c for c in harness.calls if c[0] == 'failure']) == 1
    assert len(harness.requests) == 1 and rows(harness) == []
    assert all(secret not in out + err + caplog.text for secret in ('PRIVATE', 'fake-secret', 'provider-content'))


@pytest.mark.parametrize('kind', ['authentication', 'connection'])
def test_http_known_sdk_errors_remain_generic(harness, monkeypatch, capsys, caplog, kind):
    async def fail(self, **kwargs):
        harness.calls.append(('failure', None))
        request = httpx.Request('POST', 'https://example.invalid/PRIVATE?key=fake-secret')
        if kind == 'authentication':
            raise AuthenticationError('PRIVATE fake-secret provider-content',
                                      response=httpx.Response(401, request=request),
                                      body={'PRIVATE': 'fake-secret'})
        raise APIConnectionError(message='PRIVATE fake-secret provider-content', request=request)

    monkeypatch.setattr(harness.provider, 'create', fail)
    with caplog.at_level(logging.DEBUG):
        code = run.main(['--real', '--http', 'http://localhost:0', '--lengths', '1', '--output', 'report.csv'])
    out, err = capsys.readouterr()
    assert (code, out, err) == (1, '', 'Error: real execution or output failed.\n')
    assert len([c for c in harness.calls if c[0] == 'failure']) == 1
    assert len(harness.requests) == 1 and rows(harness) == []
    assert all(secret not in out + err + caplog.text for secret in
               ('PRIVATE', 'fake-secret', 'provider-content', 'example.invalid'))


@pytest.mark.parametrize('kind', ['estimate', 'metadata', 'missing', 'estimate_constraint', 'estimate_sum', 'estimate_confidence', 'estimate_multiple'])
def test_http_invalid_stages_remain_generic_and_private(harness, monkeypatch, capsys, caplog, kind):
    original = harness.provider.create
    async def malformed(self, **kwargs):
        metadata = 'project metadata extractor' in kwargs['messages'][0]['content']
        if (kind in ('estimate', 'estimate_constraint', 'estimate_sum', 'estimate_confidence', 'estimate_multiple') and not metadata) or (kind in ('metadata', 'missing') and metadata):
            harness.calls.append(('failure', None))
            if kind in ('estimate_constraint', 'estimate_sum', 'estimate_confidence', 'estimate_multiple'):
                payload = {'summary': 'Synthetic project estimate.',
                    'confidence_pct': -1 if kind == 'estimate_constraint' else (29 if kind == 'estimate_confidence' else 80),
                    'phases': [{'name': 'Build', 'duration_weeks': 2, 'cost_eur': 500,
                                'summary': 'Build the synthetic project.'}],
                    'total_duration_weeks': 2, 'total_cost_eur': 501 if kind == 'estimate_sum' else 500,
                    'PRIVATE': 'fake-secret provider-content'}
                if kind == 'estimate_multiple':
                    del payload['summary']
                    del payload['confidence_pct']
                content = json.dumps(payload)
            else:
                content = ('{"project_name":"PRIVATE fake-secret"}' if kind == 'missing'
                           else 'PRIVATE fake-secret provider-content')
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
        return await original(self, **kwargs)
    monkeypatch.setattr(harness.provider, 'create', malformed)
    with caplog.at_level(logging.DEBUG):
        code = run.main(['--real', '--http', 'http://localhost:0', '--lengths', '1', '--output', 'report.csv'])
    out, err = capsys.readouterr()
    rejected = kind not in ('metadata', 'missing')
    assert (code, out, err) == ((1, 'Wrote 1 real turns (1 rejected).\n',
        'Error: real provider returned an invalid structured result (stage: estimate).\n') if rejected else
        (1, '', 'Error: real execution or output failed.\n'))
    assert len([c for c in harness.calls if c[0] == 'failure']) == 1
    assert len([c for c in harness.calls if c[0] == 'client']) == (2 if not rejected else 1)
    assert len(harness.requests) == (2 if rejected else 1)
    assert [row['status'] for row in rows(harness)] == (['invalid_estimate'] if rejected else [])
    assert all(secret not in out + err + caplog.text for secret in ('PRIVATE', 'fake-secret', 'provider-content'))


def test_cancelled_waiter_poisons_active_run(harness):
    async def check():
        started, release = asyncio.Event(), asyncio.Event()
        calls = []
        async def estimate(session, text, *, observation):
            calls.append(1)
            started.set()
            await release.wait()
        async with http_client.LocalStress('growing', 3, attachment_size=0, estimate=estimate) as service:
            first = asyncio.create_task(service.step(1))
            await asyncio.wait_for(started.wait(), 1)
            waiter = asyncio.create_task(service.step(2))
            await asyncio.sleep(0)
            waiter.cancel('PRIVATE')
            with pytest.raises(asyncio.CancelledError):
                await waiter
            release.set()
            with pytest.raises(RuntimeError, match='^Stress run invalid\\.$'):
                await first
            with pytest.raises(RuntimeError, match='^Stress run invalid\\.$'):
                await service.step(1)
            assert calls == [1] and len(harness.requests) == 1
    asyncio.run(check())


def test_real_timeout_cancels_provider_and_never_retries(harness, monkeypatch, capsys):
    original_timeout = asyncio.timeout
    async def stall(self, **kwargs):
        harness.calls.append(('stalled', None))
        await asyncio.Event().wait()
    monkeypatch.setattr(harness.provider, 'create', stall)
    monkeypatch.setattr(http_client.asyncio, 'timeout',
                        lambda seconds: original_timeout(0.02 if seconds == 120 else seconds))
    code = run.main(['--real', '--http', 'http://localhost:0', '--lengths', '1', '--output', 'report.csv'])
    assert (code, *capsys.readouterr()) == (1, '', 'Error: real execution or output failed.\n')
    assert len([c for c in harness.calls if c[0] == 'stalled']) == 1
    assert len(harness.requests) == 1 and rows(harness) == []


def test_three_call_eviction_aggregates_usage(harness, capsys):
    code = run.main(['--real', '--http', 'http://localhost:0', '--lengths', '10', '--output', 'report.csv'])
    assert (code, *capsys.readouterr()) == (0, 'Wrote 10 real turns.\n', '')
    result = rows(harness)
    assert result[6]['tokens_in'] == '30' and result[6]['tokens_out'] == '15'
    assert float(result[6]['cost_usd']) == pytest.approx(0.0000135)
    assert len([c for c in harness.calls if c[0] == 'create']) == 24


def test_unpriced_model_keeps_tokens_but_not_cost(harness, monkeypatch, capsys):
    original = harness.provider.create
    async def unpriced(self, **kwargs):
        response = await original(self, **kwargs)
        response.model = 'unknown-model'
        return response
    monkeypatch.setattr(harness.provider, 'create', unpriced)
    code = run.main(['--real', '--http', 'http://localhost:0', '--lengths', '1', '--output', 'report.csv'])
    assert (code, *capsys.readouterr()) == (0, 'Wrote 1 real turns.\n', '')
    row = rows(harness)[0]
    assert (row['tokens_in'], row['tokens_out'], row['cost_usd']) == ('20', '10', '')


def test_malformed_measurement_invalidates_without_retry(harness, monkeypatch):
    original = http_client._make_client
    def corrupt(app, host):
        client = original(app, host)
        send = client._transport.handle_async_request
        async def replace(request):
            response = await send(request)
            data = response.json()
            data['observation']['cost_usd'] = float('nan')
            return httpx.Response(200, content=json.dumps(data).encode(), request=request)
        client._transport.handle_async_request = replace
        return client
    monkeypatch.setattr(http_client, '_make_client', corrupt)
    async def check():
        async def estimate(session, text, *, observation):
            await run.estimation.estimate_conversational(session, text,
                SimpleNamespace(openai_api_key=run.SecretStr('fake-secret'), openai_model='fake'),
                observation=observation)
            observation['latency_ms'] = 1.0
        async with http_client.LocalStress('growing', 1, attachment_size=0, estimate=estimate, real=True) as service:
            with pytest.raises(RuntimeError, match='^Stress run invalid\\.$'):
                await service.step(1)
            with pytest.raises(RuntimeError, match='^Stress run invalid\\.$'):
                await service.step(1)
            assert len(harness.requests) == 1
            assert len([c for c in harness.calls if c[0] == 'create']) == 2
    asyncio.run(check())


def test_http_input_validation_does_not_advance_real_run(harness):
    async def check():
        calls = []
        async def estimate(session, text, *, observation):
            calls.append(1)
            await run.estimation.estimate_conversational(session, text,
                SimpleNamespace(openai_api_key=run.SecretStr('fake-secret'), openai_model='fake'),
                observation=observation)
            observation['latency_ms'] = 1.0
        async with http_client.LocalStress('growing', 1, attachment_size=0, estimate=estimate, real=True) as service:
            app = harness.servers[0].config.app
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=('127.0.0.1', 1)),
                                         base_url='http://127.0.0.1:45123') as client:
                for headers, body in [({}, {'turn': 1}),
                                      ({'authorization': 'Bearer ' + service._token.decode()},
                                       {'turn': 1, 'transcript': 'PRIVATE'})]:
                    response = await client.post('/step', headers=headers, json=body)
                    assert response.status_code == 400 and response.json() == {'error': 'invalid request'}
                    assert 'PRIVATE' not in response.text
                assert calls == []
            assert (await service.step(1))['status'] == 'ok'
            assert calls == [1]
    asyncio.run(check())
