"""Offline public CLI checks for explicitly selected real-provider execution."""
import csv
import io
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError, AuthenticationError, OpenAIError, RateLimitError

from app.config import Settings
from app.services import llm_service
from evals.stress import run


class Output(io.StringIO):
    def close(self):
        # Retain in-memory contents for assertions after the CLI closes the handle.
        pass


@pytest.fixture
def harness(monkeypatch):
    files = {}
    opens = []
    calls = []
    settings_calls = []
    original_init = Settings.__init__

    def settings_init(self, *args, **kwargs):
        settings_calls.append(kwargs.copy())
        return original_init(self, *args, **kwargs)

    def open_output(path, mode, **kwargs):
        opens.append((str(path), mode))
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

    monkeypatch.setattr(os, 'environ', {'OPENAI_API_KEY': 'secret-key'})
    monkeypatch.setattr(Settings, '__init__', settings_init)
    monkeypatch.setattr(Path, 'open', open_output)
    monkeypatch.setattr(llm_service, 'AsyncOpenAI', Provider)
    return SimpleNamespace(files=files, opens=opens, calls=calls,
                           settings_calls=settings_calls, provider=Provider)


def output_rows(harness):
    return list(csv.DictReader(io.StringIO(harness.files['report.csv'].getvalue())))


def test_real_turns_use_private_sessions_attachments_usage_and_timing(harness, monkeypatch, capsys, caplog):
    times = iter([1.0, 1.125, 2.0, 2.250, 3.0, 3.375,
                  4.0, 4.5, 5.0, 5.625, 6.0, 6.75])
    monkeypatch.setattr(run, 'perf_counter', lambda: next(times), raising=False)
    with caplog.at_level(logging.DEBUG):
        exit_code = run.main(['--real', '--lengths', '3', '--attachment-sizes', '5',
                              '--repeats', '2', '--output', 'report.csv'])
    assert (exit_code, *capsys.readouterr()) == (0, 'Wrote 6 real turns.\n', '')
    assert harness.settings_calls == [{'_env_file': None}]
    assert harness.opens == [('report.csv', 'x')]
    rows = output_rows(harness)
    assert [row['turn_index'] for row in rows] == ['1', '2', '3'] * 2
    assert [float(row['latency_ms']) for row in rows] == [125, 250, 375, 500, 625, 750]
    assert [row['memory_score'] for row in rows] == ['', '1.0', '0.5'] * 2
    assert all(row['mode'] == 'real' and row['session_id'] == '' for row in rows)
    assert all(row['tokens_in'] == '20' and row['tokens_out'] == '10' for row in rows)
    assert all(float(row['cost_usd']) == pytest.approx(0.000009) for row in rows)
    assert all(int(row['attachments_total_chars']) > 0 and row['attachment_bytes'] == '5120' for row in rows)
    assert len([call for call in harness.calls if call[0] == 'create']) == 12
    assert all(call[1]['api_key'] == 'secret-key' and call[1]['max_retries'] == 0 and
               call[1]['timeout'] == 30.0 for call in harness.calls if call[0] == 'client')
    output = harness.files['report.csv'].getvalue() + caplog.text
    for private in ('secret-key', 'Harbor', 'SYNTHETIC STRESS FIXTURE', 'Synthetic project estimate.'):
        assert private not in output


def test_unknown_usage_and_unpriced_model_remain_blank(harness, monkeypatch, capsys):
    original = harness.provider.create

    async def missing(self, **kwargs):
        result = await original(self, **kwargs)
        result.model = 'unknown-model'
        result.usage = None
        return result

    monkeypatch.setattr(harness.provider, 'create', missing)
    exit_code = run.main(['--real', '--lengths', '1', '--output', 'report.csv'])
    assert (exit_code, *capsys.readouterr()) == (0, 'Wrote 1 real turns.\n', '')
    row = output_rows(harness)[0]
    assert row['tokens_in'] == row['tokens_out'] == row['cost_usd'] == ''
    assert float(row['latency_ms']) >= 0


@pytest.mark.parametrize('args,existing,expected', [
    (['--real'], True, (1, '', 'Error: output already exists.\n')),
    (['--real', '--lengths', '2'], False, (2, '', 'Error: invalid configuration.\n')),
    (['--real', '--repeats', '0'], False, (2, '', 'Error: invalid configuration.\n')),
])
def test_rejected_runs_never_load_settings_or_call_provider(harness, capsys, args, existing, expected):
    if existing:
        harness.files['report.csv'] = io.StringIO('user artifact')
    result = run.main([*args, '--output', 'report.csv'])
    assert (result, *capsys.readouterr()) == expected
    assert not harness.settings_calls and not harness.calls
    if existing:
        assert harness.files['report.csv'].getvalue() == 'user artifact'
    else:
        assert not harness.files and not harness.opens


def test_provider_failure_has_fixed_error_and_no_sensitive_logs(harness, monkeypatch, capsys, caplog):
    async def failure(self, **kwargs):
        logging.getLogger('provider').error('secret-key transcript raw provider error')
        raise RuntimeError('secret-key transcript raw provider error')

    monkeypatch.setattr(harness.provider, 'create', failure)
    with caplog.at_level(logging.DEBUG):
        result = run.main(['--real', '--lengths', '1', '--output', 'report.csv'])
    assert (result, *capsys.readouterr()) == (1, '', 'Error: real execution or output failed.\n')
    assert 'secret-key' not in caplog.text and 'transcript' not in caplog.text
    assert 'raw provider error' not in caplog.text
    assert 'secret-key' not in harness.files['report.csv'].getvalue()
    assert len([call for call in harness.calls if call[0] == 'client']) == 1


@pytest.mark.parametrize(('failure', 'message'), [
    ('timeout', 'Error: real provider timed out.\n'),
    ('unavailable', 'Error: real provider unavailable.\n'),
    ('busy', 'Error: real provider busy or rate limited.\n'),
    ('invalid', 'Error: real provider returned an invalid structured result (stage: estimate; category: invalid JSON).\n'),
    ('empty', 'Error: real provider returned an empty result.\n'),
    ('unknown', 'Error: real execution or output failed.\n'),
])
def test_real_provider_diagnostics_are_fixed_and_private(harness, monkeypatch, capsys, caplog,
                                                         failure, message):
    async def fail(self, **kwargs):
        harness.calls.append(('failure', None))
        logging.error('PRIVATE secret-key transcript provider-content')
        if failure == 'timeout':
            raise APITimeoutError(request=httpx.Request('POST', 'https://example.invalid/PRIVATE'))
        if failure == 'unavailable':
            raise OpenAIError('PRIVATE secret-key transcript provider-content')
        if failure == 'busy':
            response = httpx.Response(429, request=httpx.Request('POST', 'https://example.invalid/PRIVATE'))
            raise RateLimitError('PRIVATE secret-key', response=response, body=None)
        if failure == 'unknown':
            raise llm_service.EstimationError(502, 'PRIVATE secret-key transcript provider-content')
        content = 'PRIVATE secret-key transcript provider-content' if failure == 'invalid' else ''
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    monkeypatch.setattr(harness.provider, 'create', fail)
    with caplog.at_level(logging.DEBUG):
        code = run.main(['--real', '--lengths', '1', '--output', 'report.csv'])
    out, err = capsys.readouterr()
    assert (code, out, err) == (1, 'Wrote 1 real turns (1 rejected).\n' if failure == 'invalid' else '', message)
    assert [row['status'] for row in output_rows(harness)] == (['invalid_estimate'] if failure == 'invalid' else [])
    assert len([c for c in harness.calls if c[0] == 'failure']) == 1
    assert all(private not in out + err + caplog.text for private in
               ('PRIVATE', 'secret-key', 'transcript', 'provider-content'))
    assert all(c[1]['max_retries'] == 0 for c in harness.calls if c[0] == 'client')


@pytest.mark.parametrize(('kind', 'message'), [
    ('authentication', 'Error: real provider authentication rejected.\n'),
    ('connection', 'Error: real provider connection failed.\n'),
])
def test_known_sdk_unavailable_context_is_safe(harness, monkeypatch, capsys, caplog, kind, message):
    async def fail(self, **kwargs):
        harness.calls.append(('failure', None))
        request = httpx.Request('POST', 'https://example.invalid/PRIVATE?key=secret-key')
        if kind == 'authentication':
            response = httpx.Response(401, request=request)
            raise AuthenticationError('PRIVATE secret-key provider-content', response=response,
                                      body={'PRIVATE': 'secret-key'})
        raise APIConnectionError(message='PRIVATE secret-key provider-content', request=request)

    monkeypatch.setattr(harness.provider, 'create', fail)
    with caplog.at_level(logging.DEBUG):
        code = run.main(['--real', '--lengths', '1', '--output', 'report.csv'])
    out, err = capsys.readouterr()
    assert (code, out, err) == (1, '', message)
    assert output_rows(harness) == []
    assert len([c for c in harness.calls if c[0] == 'failure']) == 1
    assert all(c[1]['max_retries'] == 0 for c in harness.calls if c[0] == 'client')
    assert all(secret not in out + err + caplog.text for secret in
               ('PRIVATE', 'secret-key', 'provider-content', 'example.invalid'))


@pytest.mark.parametrize('kind', ['bare', 'unknown', 'nested', 'subclass', 'explicit_cause'])
def test_unavailable_without_trusted_direct_context_stays_generic(harness, monkeypatch, capsys, caplog, kind):
    class MaliciousConnection(APIConnectionError):
        def __str__(self):
            raise AssertionError('Must not stringify malicious error')

    async def fail(*args, **kwargs):
        request = httpx.Request('POST', 'https://example.invalid/PRIVATE')
        trusted = APIConnectionError(message='PRIVATE secret-key', request=request)
        if kind == 'bare':
            raise llm_service.EstimationError(502, 'Estimation provider unavailable. Try again later.')
        if kind == 'explicit_cause':
            raise llm_service.EstimationError(502, 'Estimation provider unavailable. Try again later.') from trusted
        try:
            if kind == 'unknown':
                raise OpenAIError('PRIVATE secret-key')
            if kind == 'subclass':
                raise MaliciousConnection(message='PRIVATE secret-key', request=request)
            try:
                raise trusted
            except APIConnectionError:
                raise RuntimeError('PRIVATE secret-key')
        except Exception:
            raise llm_service.EstimationError(502, 'Estimation provider unavailable. Try again later.') from None

    monkeypatch.setattr(run.estimation, 'estimate_conversational', fail)
    with caplog.at_level(logging.DEBUG):
        code = run.main(['--real', '--lengths', '1', '--output', 'report.csv'])
    out, err = capsys.readouterr()
    assert (code, out, err) == (1, '', 'Error: real provider unavailable.\n')
    assert output_rows(harness) == [] and not harness.calls
    assert all(secret not in out + err + caplog.text for secret in ('PRIVATE', 'secret-key', 'example.invalid'))


@pytest.mark.parametrize(('wrapper', 'expected'), [
    ('busy', 'Error: real provider busy or rate limited.\n'),
    ('subclass', 'Error: real provider unavailable.\n'),
])
def test_sdk_context_does_not_change_other_contracts_or_malicious_wrapper(
        harness, monkeypatch, capsys, wrapper, expected):
    class MaliciousWrapper(llm_service.EstimationError):
        def __getattribute__(self, name):
            if name in ('__context__', '__cause__', '__suppress_context__'):
                raise AssertionError('Must not inspect malicious wrapper')
            return super().__getattribute__(name)

    calls = []
    async def fail(*args, **kwargs):
        calls.append(1)
        try:
            raise APIConnectionError(message='PRIVATE secret-key',
                                     request=httpx.Request('POST', 'https://example.invalid/PRIVATE'))
        except APIConnectionError:
            if wrapper == 'busy':
                raise llm_service.EstimationError(503, 'Estimation provider is busy. Try again later.') from None
            raise MaliciousWrapper(502, 'Estimation provider unavailable. Try again later.') from None

    monkeypatch.setattr(run.estimation, 'estimate_conversational', fail)
    code = run.main(['--real', '--lengths', '1', '--output', 'report.csv'])
    assert (code, *capsys.readouterr()) == (1, '', expected)
    assert calls == [1] and not harness.calls and output_rows(harness) == []


@pytest.mark.parametrize(('status', 'detail'), [
    (503, 'Estimation provider unavailable. Try again later.'),
    (502, 'Estimation provider is busy. Try again later.'),
    (503, 'Estimation provider returned an empty result.'),
    (503, 'Estimation provider returned an invalid result.'),
    (502, 'Estimation provider unavailable. Try again later. PRIVATE secret-key transcript'),
    (502, 'Estimation provider returned an empty result'),
    ('502', 'Estimation provider returned an invalid result.'),
])
def test_unrecognized_provider_contract_stays_generic(harness, monkeypatch, capsys, caplog,
                                                       status, detail):
    calls = []

    async def fail(*args, **kwargs):
        calls.append(1)
        logging.error('PRIVATE secret-key transcript provider-content')
        raise llm_service.EstimationError(status, detail)

    monkeypatch.setattr(run.estimation, 'estimate_conversational', fail)
    with caplog.at_level(logging.DEBUG):
        code = run.main(['--real', '--lengths', '1', '--output', 'report.csv'])
    out, err = capsys.readouterr()
    assert (code, out, err) == (1, '', 'Error: real execution or output failed.\n')
    assert calls == [1] and not harness.calls
    assert output_rows(harness) == []
    assert all(private not in out + err + caplog.text for private in
               ('PRIVATE', 'secret-key', 'transcript', 'provider-content'))


def test_real_missing_key_diagnostic(harness, monkeypatch, capsys):
    monkeypatch.setattr(os, 'environ', {})
    code = run.main(['--real', '--lengths', '1', '--output', 'report.csv'])
    assert (code, *capsys.readouterr()) == (1, '', 'Error: real provider API key is missing.\n')
    assert output_rows(harness) == [] and not harness.calls


def test_real_invalid_settings_diagnostic_is_private(harness, monkeypatch, capsys, caplog):
    monkeypatch.setattr(os, 'environ', {'OPENAI_API_KEY': 'secret-key', 'LOG_LEVEL': 'PRIVATE'})
    with caplog.at_level(logging.DEBUG):
        code = run.main(['--real', '--output', 'report.csv'])
    out, err = capsys.readouterr()
    assert (code, out, err) == (1, '', 'Error: invalid real-mode configuration.\n')
    assert output_rows(harness) == [] and not harness.calls
    assert 'PRIVATE' not in out + err + caplog.text and 'secret-key' not in out + err + caplog.text


def test_real_output_os_error_is_fixed_and_private(harness, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise OSError('PRIVATE secret-key transcript')
    monkeypatch.setattr(Path, 'open', fail)
    code = run.main(['--real', '--output', 'report.csv'])
    assert (code, *capsys.readouterr()) == (1, '', 'Error: local output operation failed.\n')
    assert not harness.calls and not harness.settings_calls and not harness.files


def test_default_simulation_does_not_load_settings(harness, capsys):
    result = run.main(['--lengths', '1', '--output', 'report.csv'])
    assert (result, *capsys.readouterr()) == (0, 'Wrote 1 simulated turns.\n', '')
    assert not harness.settings_calls and not harness.calls
    assert output_rows(harness)[0]['mode'] == 'simulated'


def test_simulated_http_remains_available_without_settings(harness, monkeypatch, capsys):
    class Service:
        def __init__(self, profile, length, *, attachment_size, estimate, port):
            assert (profile, length, attachment_size, port) == ('growing', 1, 0, 0)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def step(self, turn):
            return ({'status': 'ok', 'observation': {'turn_index': 1}, 'memory': [], 'attachment_bytes': 0}
                    if turn == 1 else {'status': 'complete'})

    monkeypatch.setattr(run, 'LocalStress', Service)
    result = run.main(['--http', 'http://localhost:0', '--lengths', '1', '--output', 'report.csv'])
    assert (result, *capsys.readouterr()) == (0, 'Wrote 1 simulated turns.\n', '')
    assert output_rows(harness)[0]['mode'] == 'simulated-http'
    assert tuple(output_rows(harness)[0]) == run.FIELDS
    assert not harness.settings_calls and not harness.calls


def test_help_explains_opt_in_and_http(harness, capsys):
    with pytest.raises(SystemExit) as exit_info:
        run.main(['--help'])
    assert exit_info.value.code == 0
    out, err = capsys.readouterr()
    assert err == '' and '--real' in out and 'simulated' in out and 'HTTP' in out
    normalized = ' '.join(out.split())
    assert 'charges' in normalized and 'empty or partial CSV' in normalized and 'no automatic retries' in normalized
    assert 'authentication rejected' in normalized and 'connection failed' in normalized
    assert 'unavailable, busy or rate limited, empty result and invalid structured result' in normalized
    assert 'estimate or metadata validation' in normalized
    assert 'fixed field and validation category' in normalized
    assert 'known business rule' in normalized
    assert 'HTTP failures may remain generic' in normalized
    assert not harness.settings_calls and not harness.calls and not harness.opens
