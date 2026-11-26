"""Local supervisor contract: mocked sockets and in-memory ASGI only."""
import asyncio
import csv
from functools import partial
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def harness(monkeypatch):
    from evals.stress import http_client as module

    sockets, servers, requests = [], [], []

    class Socket:
        family = module.socket.AF_INET
        type = module.socket.SOCK_STREAM

        def __init__(self, *args):
            self.address = None
            self.closed = False
            sockets.append(self)

        def bind(self, address):
            self.address = (address[0], 45123)

        def listen(self, backlog):
            self.backlog = backlog

        def getsockname(self):
            return self.address

        def setblocking(self, value):
            pass

        def set_inheritable(self, value):
            pass

        def close(self):
            self.closed = True

    class Server:
        def __init__(self, config):
            self.config = config
            self.started = False
            self.should_exit = False
            self.force_exit = False
            self.stopped = False
            servers.append(self)

        async def serve(self, sockets):
            self.started = True
            try:
                while not self.should_exit:
                    await asyncio.sleep(0.001)
            finally:
                self.stopped = True

    def make_client(app, host):
        class Transport(httpx.ASGITransport):
            async def handle_async_request(self, request):
                requests.append(request)
                return await super().handle_async_request(request)
        return httpx.AsyncClient(transport=Transport(app=app, client=('127.0.0.1', 1234)),
                                 base_url='http://' + host)

    monkeypatch.setattr(module, '_socket_factory', Socket)
    monkeypatch.setattr(module.uvicorn, 'Server', Server)
    monkeypatch.setattr(module, '_make_client', make_client)
    return module, sockets, servers, requests


def test_local_synthetic_run_uses_verified_loopback_and_fresh_private_tokens(harness, caplog):
    module, sockets, servers, requests = harness

    async def check():
        tokens = []
        for _ in range(2):
            async with module.LocalStress('growing', 1) as client:
                tokens.append(servers[-1].config.app._token)
                assert sockets[-1].address == ('127.0.0.1', 45123)
                assert await client.step(1) == {'status': 'ok', 'turn': 1, 'memory': [], 'cost_usd': None}
                assert await client.step(2) == {'status': 'complete'}
                assert tokens[-1].decode() not in repr(client)
        assert tokens[0] != tokens[1] and len(tokens[0]) >= 43
        assert all(s.closed for s in sockets)
        assert all(s.config.access_log is False and s.config.log_config is None for s in servers)
        assert all(r.url.host == '127.0.0.1' and not r.url.query for r in requests)
        assert all(t.decode() not in caplog.text for t in tokens)
        assert all(r.headers['authorization'] == 'Bearer ' + tokens[i // 2].decode()
                   for i, r in enumerate(requests))
    run(check())


@pytest.fixture
def continuation(harness, monkeypatch):
    from app.services import llm_service
    from evals.stress import run as runner

    class Output(io.StringIO):
        def __init__(self):
            super().__init__(newline='')
            self.flushes = 0
            self.fail_flush = None
            self.short_write = None
            self.writes = 0
        def flush(self):
            self.flushes += 1
            if self.flushes == self.fail_flush:
                raise OSError('PRIVATE')
            return super().flush()
        def write(self, text):
            self.writes += 1
            if self.writes == self.short_write:
                super().write(text[:-1])
                return len(text) - 1
            return super().write(text)
        def close(self):
            pass

    state = SimpleNamespace(output=Output(), calls=[], fail_at=2)
    class Provider(runner.FakeProvider):
        def __init__(self, **kwargs):
            assert kwargs['max_retries'] == 0
            super().__init__(**kwargs)
        async def create(self, **kwargs):
            metadata = 'project metadata extractor' in kwargs['messages'][0]['content']
            if not metadata:
                state.calls.append((kwargs['messages'], state.output.flushes))
                if len(state.calls) == state.fail_at:
                    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{PRIVATE'))])
            return await super().create(**kwargs)
    monkeypatch.setattr(os, 'environ', {'OPENAI_API_KEY': 'CREDENTIAL_SENTINEL'})
    monkeypatch.setattr(Path, 'open', lambda *args, **kwargs: state.output)
    monkeypatch.setattr(llm_service, 'AsyncOpenAI', Provider)
    return state


@pytest.mark.parametrize(('failed', 'indices', 'scores'), [
    (1, ['', '1', '2'], ['', '', '0.0']),
    (2, ['1', '', '2'], ['', '', '1.0']),
    (3, ['1', '2', ''], ['', '1.0', '']),
])
def test_http_cli_continues_proven_rejection_with_sparse_scores(continuation, harness, capsys, caplog,
                                                                 failed, indices, scores):
    from evals.stress import run as runner
    continuation.fail_at = failed
    code = runner.main(['--real', '--http', 'http://localhost:0', '--lengths', '3', '--output', 'report.csv'])
    out, err = capsys.readouterr()
    assert (code, out, err) == (1, 'Wrote 3 real turns (1 rejected).\n',
                                 'Error: real provider returned an invalid structured result (stage: estimate).\n')
    rows = list(csv.DictReader(io.StringIO(continuation.output.getvalue())))
    assert [row['scenario_turn'] for row in rows] == ['1', '2', '3']
    assert [row['turn_index'] for row in rows] == indices
    assert [row['memory_score'] for row in rows] == scores
    assert [row['status'] for row in rows] == ['invalid_estimate' if n == failed else 'committed' for n in (1, 2, 3)]
    assert [row['reason'] for row in rows] == ['estimate_validation' if n == failed else '' for n in (1, 2, 3)]
    assert all(rows[failed - 1][key] == '' for key in ('tokens_in', 'tokens_out', 'cost_usd', 'latency_ms', 'turn_index'))
    assert [flushes for _, flushes in continuation.calls] == [1, 2, 3]
    assert len(harness[3]) == 4
    if failed == 2:
        assert 'Add customer accounts.' not in json.dumps(continuation.calls[2][0])
        assert 'Harbor' in json.dumps(continuation.calls[2][0])
    assert all(secret not in out + err + caplog.text + continuation.output.getvalue()
               for secret in ('PRIVATE', 'CREDENTIAL_SENTINEL', harness[3][0].headers['authorization']))


@pytest.mark.parametrize(('kind', 'number', 'calls'), [('flush', 1, 0), ('flush', 2, 1), ('flush', 3, 2),
                                                       ('short_write', 1, 0), ('short_write', 2, 1),
                                                       ('short_write', 3, 2)])
def test_http_cli_write_failure_stops_before_next_request(continuation, harness, capsys, kind, number, calls):
    from evals.stress import run as runner
    setattr(continuation.output, 'fail_flush' if kind == 'flush' else 'short_write', number)
    code = runner.main(['--real', '--http', 'http://localhost:0', '--lengths', '3', '--output', 'report.csv'])
    assert (code, *capsys.readouterr()) == (1, '', 'Error: real execution or output failed.\n')
    assert len(continuation.calls) == calls and len(harness[3]) == calls


@pytest.mark.parametrize('corruption', ['extra', 'reason', 'wrong_turn', 'bool_count', 'negative',
                                         'oversize', 'wrong_bytes', 'wrong_counts', 'unknown_status',
                                         'duplicate', 'forged'])
def test_bad_receipt_invalidates_client_without_retry(continuation, harness, monkeypatch, corruption):
    from app.services import estimation
    from evals.stress import run as runner
    module, sockets, servers, requests = harness
    if corruption == 'forged':
        continuation.fail_at = 99
    original = module._make_client
    def corrupt(app, host):
        client = original(app, host)
        send = client._transport.handle_async_request
        async def replace(request):
            response = await send(request)
            await response.aread()
            data = response.json()
            if json.loads(request.content)['turn'] != 2:
                return response
            if corruption == 'forged':
                data = {'status': 'invalid_estimate', 'turn': 2, 'reason': 'estimate_validation',
                        'attachment_bytes': 0, 'enriched_transcript_chars': 42, 'attachments_total_chars': 0}
            elif corruption == 'extra': data['private'] = 'PRIVATE'
            elif corruption == 'reason': data['reason'] = 'PRIVATE'
            elif corruption == 'wrong_turn': data['turn'] = 3
            elif corruption == 'bool_count': data['enriched_transcript_chars'] = True
            elif corruption == 'negative': data['attachments_total_chars'] = -1
            elif corruption == 'oversize': data['enriched_transcript_chars'] = 1000001
            elif corruption == 'wrong_bytes': data['attachment_bytes'] = 5120
            elif corruption == 'wrong_counts': data['enriched_transcript_chars'] += 1
            elif corruption == 'unknown_status': data['status'] = 'PRIVATE'
            body = json.dumps(data).encode()
            if corruption == 'duplicate':
                body = body[:-1] + b',"reason":"estimate_validation"}'
            return httpx.Response(200, content=body, request=request)
        client._transport.handle_async_request = replace
        return client
    monkeypatch.setattr(module, '_make_client', corrupt)
    settings = SimpleNamespace(openai_api_key=runner.SecretStr('CREDENTIAL_SENTINEL'), openai_model='simulated')
    async def check():
        async with module.LocalStress('growing', 3, attachment_size=0,
                estimate=partial(estimation.estimate_conversational, settings=settings), real=True) as client:
            assert (await client.step(1))['status'] == 'ok'
            with pytest.raises(RuntimeError, match='^Stress run invalid\\.$'):
                await client.step(2)
            with pytest.raises(RuntimeError, match='^Stress run invalid\\.$'):
                await client.step(3)
            assert len(requests) == 2 and len(continuation.calls) == 2
    run(check())


def test_http_cli_lost_receipt_preserves_previous_row_and_never_retries(continuation, harness, monkeypatch, capsys):
    from evals.stress import run as runner
    module, _, _, requests = harness
    original = module._make_client
    def lose(app, host):
        client = original(app, host)
        send = client._transport.handle_async_request
        async def receive(request):
            response = await send(request)
            if json.loads(request.content)['turn'] == 2:
                raise httpx.ReadError('PRIVATE', request=request)
            return response
        client._transport.handle_async_request = receive
        return client
    monkeypatch.setattr(module, '_make_client', lose)
    code = runner.main(['--real', '--http', 'http://localhost:0', '--lengths', '3', '--output', 'report.csv'])
    assert (code, *capsys.readouterr()) == (1, '', 'Error: real execution or output failed.\n')
    rows = list(csv.DictReader(io.StringIO(continuation.output.getvalue())))
    assert [(row['scenario_turn'], row['status'], row['turn_index']) for row in rows] == [('1', 'committed', '1')]
    assert len(requests) == len(continuation.calls) == 2


def test_http_existing_output_never_calls_provider(harness, monkeypatch, capsys):
    from evals.stress import run as runner
    def exists(*args, **kwargs):
        raise FileExistsError()
    monkeypatch.setattr(Path, 'open', exists)
    code = runner.main(['--real', '--http', 'http://localhost:0', '--output', 'report.csv'])
    assert (code, *capsys.readouterr()) == (1, '', 'Error: output already exists.\n')
    assert not harness[1] and not harness[3]


def test_requested_port_is_bound_and_other_ports_are_rejected(harness):
    module, sockets, servers, requests = harness
    async def check():
        async with module.LocalStress('growing', 1, port=45123) as client:
            assert sockets[-1].address == ('127.0.0.1', 45123)
            assert (await client.step(1))['status'] == 'ok'
        with pytest.raises(RuntimeError, match='^Local stress service unavailable\\.$'):
            async with module.LocalStress('growing', 1, port=8000):
                pytest.fail('misbound port must not start')
        assert sockets[-1].closed and len(servers) == 1
    run(check())


def test_real_loopback_rejects_unauthenticated_input_and_shuts_down(monkeypatch):
    import socket
    from evals.stress.http_client import LocalStress

    connect = socket.socket.connect
    def only_loopback(sock, address):
        assert isinstance(address, tuple) and address[0] == '127.0.0.1'
        return connect(sock, address)
    monkeypatch.setattr(socket.socket, 'connect', only_loopback)

    async def check():
        service = LocalStress('growing', 1)
        async with service:
            host, port = service._socket.getsockname()
            assert host == '127.0.0.1'
            async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{port}', trust_env=False) as client:
                response = await client.post('/step', json={'turn': 1, 'transcript': 'PRIVATE'})
                assert response.status_code == 400
                assert response.json() == {'error': 'invalid request'}
                assert 'PRIVATE' not in response.text
                response = await client.post('/step', json={'turn': 1, 'session_id': 'PRIVATE'},
                    headers={'authorization': 'Bearer ' + service._token.decode('ascii')})
                assert response.status_code == 400
                assert response.json() == {'error': 'invalid request'}
                assert 'PRIVATE' not in response.text
            assert await service.step(1) == {'status': 'ok', 'turn': 1, 'memory': [], 'cost_usd': None}
            assert await service.step(2) == {'status': 'complete'}
        assert service._socket.fileno() == -1 and service._task.done()
        assert service._token is None
    run(check())


def test_misbound_socket_is_rejected_without_starting_server(harness, monkeypatch):
    module, sockets, servers, requests = harness
    original = module._socket_factory
    class Misbound(original):
        def getsockname(self):
            return ('0.0.0.0', 45123)
    monkeypatch.setattr(module, '_socket_factory', Misbound)
    async def check():
        with pytest.raises(RuntimeError, match='^Local stress service unavailable\\.$'):
            async with module.LocalStress('growing', 1):
                pytest.fail('must not start')
        assert len(sockets) == 1 and sockets[0].closed
        assert not servers and not requests
    run(check())


def test_ambiguous_client_receive_invalidates_without_retry(harness, monkeypatch):
    module, sockets, servers, requests = harness
    original = module._make_client
    def broken(app, host):
        client = original(app, host)
        transport = client._transport
        original_send = transport.handle_async_request
        async def lose_response(request):
            await original_send(request)
            raise httpx.ReadError('PRIVATE FAILURE', request=request)
        transport.handle_async_request = lose_response
        return client
    monkeypatch.setattr(module, '_make_client', broken)
    async def check():
        async with module.LocalStress('growing', 3) as client:
            with pytest.raises(RuntimeError, match='^Stress run invalid\\.$') as error:
                await client.step(1)
            assert 'PRIVATE' not in str(error.value)
            with pytest.raises(RuntimeError, match='^Stress run invalid\\.$'):
                await client.step(1)
            assert len(requests) == 1
            assert (await servers[0].config.app._run.step(2)) == {'status': 'invalid'}
    run(check())


def test_cancelled_waiter_poisons_inflight_client_and_never_retries(harness, monkeypatch):
    module, sockets, servers, requests = harness
    original = module._make_client
    started, release = asyncio.Event(), asyncio.Event()
    def stalled(app, host):
        client = original(app, host)
        transport = client._transport
        original_send = transport.handle_async_request
        async def delay(request):
            started.set()
            await release.wait()
            return await original_send(request)
        transport.handle_async_request = delay
        return client
    monkeypatch.setattr(module, '_make_client', stalled)
    async def check():
        async with module.LocalStress('growing', 3) as client:
            first = asyncio.create_task(client.step(1))
            await asyncio.wait_for(started.wait(), 1)
            waiter = asyncio.create_task(client.step(2))
            await asyncio.sleep(0)
            waiter.cancel('PRIVATE')
            with pytest.raises(asyncio.CancelledError) as error:
                await waiter
            assert 'PRIVATE' not in str(error.value)
            release.set()
            with pytest.raises(RuntimeError, match='^Stress run invalid\\.$'):
                await first
            assert len(requests) <= 1
            with pytest.raises(RuntimeError, match='^Stress run invalid\\.$'):
                await client.step(1)
    run(check())


def test_client_close_failure_still_shuts_down_listener(harness, monkeypatch):
    module, sockets, servers, requests = harness
    original = module._make_client
    failure = RuntimeError('client close failed')

    def failing_client(app, host):
        client = original(app, host)
        async def fail_close():
            raise failure
        client.aclose = fail_close
        return client

    monkeypatch.setattr(module, '_make_client', failing_client)

    async def check():
        service = module.LocalStress('growing', 1)
        with pytest.raises(RuntimeError) as error:
            async with service:
                pass
        assert error.value is failure
        assert len(servers) == 1 and servers[0].should_exit and servers[0].stopped
        assert len(sockets) == 1 and sockets[0].closed
        assert not requests
        with pytest.raises(RuntimeError, match='^Stress run invalid\\.$'):
            await service.step(1)
    run(check())


def test_cancelled_client_close_still_shuts_down_listener(harness, monkeypatch):
    module, sockets, servers, requests = harness
    original = module._make_client
    started = asyncio.Event()

    def stalled_client(app, host):
        client = original(app, host)
        async def stall_close():
            started.set()
            await asyncio.Event().wait()
        client.aclose = stall_close
        return client

    monkeypatch.setattr(module, '_make_client', stalled_client)

    async def check():
        service = module.LocalStress('growing', 1)
        async def use_service():
            async with service:
                pass
        task = asyncio.create_task(use_service())
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(servers) == 1 and servers[0].should_exit and servers[0].stopped
        assert len(sockets) == 1 and sockets[0].closed
        assert not requests
        with pytest.raises(RuntimeError, match='^Stress run invalid\\.$'):
            await service.step(1)
    run(check())


def test_cancelled_server_shutdown_propagates_and_closes_listener(harness, monkeypatch):
    module, sockets, servers, requests = harness
    original = module.uvicorn.Server
    shutdown_started = asyncio.Event()

    class StalledShutdown(original):
        async def serve(self, sockets):
            self.started = True
            try:
                while not self.should_exit:
                    await asyncio.sleep(0.001)
                shutdown_started.set()
                await asyncio.Event().wait()
            finally:
                self.stopped = True

    monkeypatch.setattr(module.uvicorn, 'Server', StalledShutdown)

    async def check():
        service = module.LocalStress('growing', 1)

        async def use_service():
            async with service:
                pass

        task = asyncio.create_task(use_service())
        await asyncio.wait_for(shutdown_started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(servers) == 1 and servers[0].should_exit and servers[0].stopped
        assert len(sockets) == 1 and sockets[0].closed
        assert not requests
        with pytest.raises(RuntimeError, match='^Stress run invalid\\.$'):
            await service.step(1)

    run(check())
