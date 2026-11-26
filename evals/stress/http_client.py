"""Trusted, local-only supervisor for one offline synthetic stress run."""
import asyncio
import json
import secrets
import socket
from io import BytesIO

from fastapi import UploadFile
import httpx
import uvicorn

from app.services.attachments import enrich_transcript
from evals.stress.http_service import StressHTTP
from evals.stress.service import StressRun


_socket_factory = socket.socket


async def _synthetic_estimate(session, text):
    async with session.lock:
        session.history.add_turn(text, 'Synthetic response.')


def _make_client(app, host):
    # No proxy, redirects, environment configuration, HTTP/2 or transport retries.
    transport = httpx.AsyncHTTPTransport(retries=0, trust_env=False,
                                          limits=httpx.Limits(max_connections=1))
    return httpx.AsyncClient(base_url='http://' + host, transport=transport,
                             timeout=5, trust_env=False, follow_redirects=False)


class LocalStress:
    """Single-use async context manager; all requests and credentials stay local."""

    def __init__(self, profile, length, *, attachment_size=None, estimate=None, port=0, real=False):
        if (type(port) is not int or not 0 <= port <= 65535 or
                (attachment_size is None) != (estimate is None)):
            raise ValueError('Invalid service configuration.')
        self._run = StressRun(profile, length, estimate or _synthetic_estimate,
                              attachment_size=attachment_size, real=real)
        self._real = real
        self._observed = attachment_size is not None
        self._attachment_size = attachment_size
        self._port = port
        self._length = length
        self._next = 1
        self._facts = []
        self._lock = asyncio.Lock()
        self._invalid = False
        self._used = False
        self._socket = self._server = self._task = self._client = None
        self._token = None

    def _invalidate(self):
        self._invalid = True
        self._run.invalidate()

    async def __aenter__(self):
        if self._used:
            raise RuntimeError('Local stress service unavailable.')
        self._used = True
        try:
            self._token = secrets.token_urlsafe(32).encode('ascii')
            self._socket = _socket_factory(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.set_inheritable(False)
            self._socket.bind(('127.0.0.1', self._port))
            address = self._socket.getsockname()
            if (self._socket.family != socket.AF_INET or address[0] != '127.0.0.1' or
                    type(address[1]) is not int or not 1 <= address[1] <= 65535 or
                    (self._port and address[1] != self._port)):
                raise ValueError('Invalid bind.')
            self._socket.listen(16)
            self._socket.setblocking(False)
            host = '127.0.0.1:' + str(address[1])
            app = StressHTTP(self._run, self._token, host)
            config = uvicorn.Config(app, host='127.0.0.1', port=address[1], workers=1,
                                    http='h11', ws='none', lifespan='off', proxy_headers=False,
                                    log_config=None, access_log=False, limit_concurrency=16,
                                    backlog=16, timeout_keep_alive=1, timeout_graceful_shutdown=2,
                                    h11_max_incomplete_event_size=8192)
            self._server = uvicorn.Server(config)
            # Embedded operation must not install process signal handlers.
            from contextlib import nullcontext
            self._server.capture_signals = nullcontext
            self._task = asyncio.create_task(self._server.serve(sockets=[self._socket]))
            async with asyncio.timeout(5):
                while not self._server.started:
                    if self._task.done():
                        raise RuntimeError('Server stopped.')
                    await asyncio.sleep(0.001)
            if self._task.done():
                raise RuntimeError('Server stopped.')
            self._client = _make_client(app, host)
            if self._real:
                self._client.timeout = httpx.Timeout(120.0)
            return self
        except asyncio.CancelledError:
            self._invalidate()
            await self._close()
            raise asyncio.CancelledError() from None
        except BaseException:
            self._invalidate()
            await self._close()
            raise RuntimeError('Local stress service unavailable.') from None

    async def _close(self):
        try:
            try:
                if self._client is not None:
                    await self._client.aclose()
            finally:
                if self._server is not None:
                    self._server.should_exit = True
                if self._task is not None:
                    try:
                        await asyncio.wait_for(self._task, 5)
                    except asyncio.CancelledError:
                        self._task.cancel()
                        await asyncio.gather(self._task, return_exceptions=True)
                        if asyncio.current_task().cancelling():
                            raise
                    except Exception:
                        self._task.cancel()
                        await asyncio.gather(self._task, return_exceptions=True)
        finally:
            try:
                if self._socket is not None:
                    self._socket.close()
            finally:
                self._token = None

    async def __aexit__(self, exc_type, exc, tb):
        self._invalidate()
        await self._close()

    async def step(self, turn):
        try:
            async with self._lock:
                if (self._invalid or self._client is None or self._task.done() or
                        type(turn) is not int or turn != self._next or turn > self._length + 1):
                    raise ValueError('Invalid step.')
                async with asyncio.timeout(120 if self._real else 10):
                    async with self._client.stream('POST', '/step',
                            headers={'authorization': 'Bearer ' + self._token.decode('ascii')},
                            json={'turn': turn}) as response:
                        if response.status_code != 200:
                            raise ValueError('Invalid response.')
                        body = bytearray()
                        async for chunk in response.aiter_bytes():
                            body.extend(chunk)
                            if len(body) > 4096:
                                raise ValueError('Invalid response.')
                def unique(pairs):
                    result = {}
                    for key, item in pairs:
                        if key in result:
                            raise ValueError('Duplicate key.')
                        result[key] = item
                    return result
                def invalid_constant(_):
                    raise ValueError('Invalid constant.')
                value = json.loads(body, object_pairs_hook=unique, parse_constant=invalid_constant)
                if self._invalid or not self._valid_result(value, turn):
                    raise ValueError('Invalid response.')
                if turn <= self._length:
                    if value['status'] == 'invalid_estimate':
                        if not await self._valid_receipt(value, turn) or self._invalid:
                            raise ValueError('Invalid receipt.')
                    else:
                        self._facts.append(turn)
                self._next += 1
                return value
        except asyncio.CancelledError:
            self._invalidate()
            raise asyncio.CancelledError() from None
        except BaseException:
            self._invalidate()
            raise RuntimeError('Stress run invalid.') from None

    async def _valid_receipt(self, value, turn):
        # The local service is trusted, but a matching JSON shape alone cannot
        # prove a rejection: confirm its direct-estimator state and known counts.
        if (not self._observed or not self._run._direct or self._run._invalid or
                self._run._next != turn + 1 or self._run._facts != self._facts or
                self._run._session.committed_turns != len(self._facts)):
            return False
        text = self._run._scenario.turns[turn - 1].message
        if turn == 1:
            text += ' We commit to ship. docker kafka.'
        observed = {}
        uploads = ([UploadFile(filename='synthetic.pdf', file=BytesIO(self._run._pdf))]
                   if self._run._pdf else [])
        await enrich_transcript(text, uploads, observation=observed)
        return (value['enriched_transcript_chars'] == observed['enriched_transcript_chars'] and
                value['attachments_total_chars'] == observed['attachments_total_chars'])

    def _valid_result(self, value, turn):
        if turn == self._length + 1:
            return type(value) is dict and value == {'status': 'complete'}
        if type(value) is dict and value.get('status') == 'invalid_estimate':
            counts = ('attachment_bytes', 'enriched_transcript_chars', 'attachments_total_chars')
            return (self._observed and set(value) == {'status', 'turn', 'reason', *counts} and
                    type(value['turn']) is int and value['turn'] == turn and
                    value['reason'] == 'estimate_validation' and
                    all(type(value[key]) is int and 0 <= value[key] <= 1_000_000 for key in counts) and
                    value['attachment_bytes'] == self._attachment_size * 1024 and
                    value['attachments_total_chars'] <= value['enriched_transcript_chars'])
        keys = {'status', 'turn', 'memory', 'cost_usd'}
        if self._observed:
            keys.update(('observation', 'attachment_bytes'))
        if (type(value) is not dict or set(value) != keys or
                value['status'] != 'ok' or type(value['turn']) is not int or
                value['turn'] != turn or value['cost_usd'] is not None or
                type(value['memory']) is not list or len(value['memory']) != len(self._facts)):
            return False
        if self._observed:
            try:
                if (type(value['attachment_bytes']) is not int or
                        value['attachment_bytes'] != self._attachment_size * 1024 or
                        StressRun._observation(value['observation'], len(self._facts) + 1, real=self._real) != value['observation']):
                    return False
            except (TypeError, ValueError, AttributeError):
                return False
        for index, item in zip(self._facts, value['memory']):
            if (type(item) is not dict or set(item) != {'fact_turn', 'score'} or
                    type(item['fact_turn']) is not int or item['fact_turn'] != index or
                    type(item['score']) is not float or item['score'] not in (0.0, 1.0)):
                return False
        return True
