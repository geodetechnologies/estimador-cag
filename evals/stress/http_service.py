"""Unexposed ASGI boundary for one trusted synthetic stress run.

This module does not create a listener, generate credentials, or configure logging.
"""
import asyncio
import hmac
import ipaddress
import json
import re

from evals.stress.service import StressRun


class StressHTTP:
    def __init__(self, run: StressRun, token: bytes, host: str):
        # Entropy and freshness are the trusted supervisor's responsibility.
        if (type(run) is not StressRun or type(token) is not bytes or
                not re.fullmatch(rb'[A-Za-z0-9_-]{32,128}', token) or
                type(host) is not str or not re.fullmatch(r'127\.0\.0\.1:[1-9][0-9]{0,4}', host) or
                int(host.split(':')[1]) > 65535):
            raise ValueError('Invalid service configuration.')
        self._run = run
        self._token = token
        self._host = host.encode('ascii')
        self._lock = asyncio.Lock()

    def _valid_scope(self, scope):
        try:
            peer = scope['client']
            if (scope['type'] != 'http' or not ipaddress.ip_address(peer[0]).is_loopback or
                    scope['method'] != 'POST' or scope['path'] != '/step' or
                    scope.get('raw_path', b'/step') != b'/step' or scope.get('root_path', '') or
                    scope.get('query_string', b'') or sum(len(k) + len(v) for k, v in scope['headers']) > 8192):
                return False
            headers = scope['headers']
            def values(name):
                return [v for k, v in headers if k.lower() == name]
            auth, host, origin = values(b'authorization'), values(b'host'), values(b'origin')
            return (len(auth) == 1 and hmac.compare_digest(auth[0], b'Bearer ' + self._token) and
                    host == [self._host] and origin in ([], [b'http://' + self._host]) and
                    not any(k.lower() in (b'forwarded', b'x-forwarded-for', b'x-forwarded-host',
                                          b'x-forwarded-proto') for k, _ in headers))
        except (KeyError, TypeError, ValueError, IndexError):
            return False

    @staticmethod
    async def _body(receive):
        body = bytearray()
        for _ in range(1024):
            message = await receive()
            if message['type'] != 'http.request':
                raise ValueError('Invalid request.')
            chunk = message.get('body', b'')
            if len(body) + len(chunk) > 1024:
                raise ValueError('Invalid request.')
            body.extend(chunk)
            if not message.get('more_body', False):
                break
        else:
            raise ValueError('Invalid request.')
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError('Invalid request.')
                result[key] = value
            return result
        value = json.loads(body.decode('utf-8'), object_pairs_hook=unique)
        if (type(value) is not dict or set(value) != {'turn'} or
                type(value['turn']) is not int or not 1 <= value['turn'] <= 21):
            raise ValueError('Invalid request.')
        return value['turn']

    @staticmethod
    async def _reply(send, status, value):
        body = json.dumps(value, separators=(',', ':')).encode('ascii')
        await send({'type': 'http.response.start', 'status': status,
                    'headers': [(b'content-type', b'application/json'),
                                (b'cache-control', b'no-store')]})
        await send({'type': 'http.response.body', 'body': body})

    async def __call__(self, scope, receive, send):
        if scope.get('type') != 'http':
            return
        try:
            if not self._valid_scope(scope):
                await self._reply(send, 400, {'error': 'invalid request'})
                return
            try:
                turn = await asyncio.wait_for(self._body(receive), timeout=5)
            except (ValueError, UnicodeError, KeyError, TypeError, RecursionError, TimeoutError):
                await self._reply(send, 400, {'error': 'invalid request'})
                return
        except asyncio.CancelledError:
            raise asyncio.CancelledError() from None
        except Exception:
            return

        sent_complete = False

        async def execute():
            nonlocal sent_complete
            async with self._lock:
                result = await self._run.step(turn)
                status = result.get('status') if type(result) is dict else None
                if status == 'invalid_estimate':
                    counts = ('attachment_bytes', 'enriched_transcript_chars', 'attachments_total_chars')
                    valid = (set(result) == {'status', 'turn', 'reason', *counts} and
                             type(result['turn']) is int and result['turn'] == turn and
                             result['reason'] == 'estimate_validation' and
                             all(type(result[key]) is int and 0 <= result[key] <= 1_000_000 for key in counts) and
                             result['attachment_bytes'] in (0, 5120, 20480, 51200, 102400) and
                             result['attachments_total_chars'] <= result['enriched_transcript_chars'])
                    if not valid:
                        self._run.invalidate()
                        status = None
                if status not in ('ok', 'complete', 'invalid_estimate'):
                    self._run.invalidate()
                    await self._reply(send, 409, {'error': 'run invalid'})
                else:
                    await self._reply(send, 200, result)
                sent_complete = True

        step = asyncio.create_task(execute())
        watcher = asyncio.create_task(receive())
        try:
            done, _ = await asyncio.wait((step, watcher), return_when=asyncio.FIRST_COMPLETED)
            if watcher in done and not sent_complete:
                self._run.invalidate()
            else:
                step.result()
        except asyncio.CancelledError:
            self._run.invalidate()
            raise asyncio.CancelledError() from None
        except BaseException:
            self._run.invalidate()
        finally:
            for task in (step, watcher):
                if not task.done():
                    task.cancel()
            await asyncio.gather(step, watcher, return_exceptions=True)
