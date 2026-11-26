"""Public CLI continuation checks using only fake providers and in-memory CSVs."""
import asyncio
import csv
import io
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services import llm_service
from evals.stress import run


class Output(io.StringIO):
    def __init__(self, fail_flush=None):
        super().__init__(newline='')
        self.flushes = 0
        self.fail_flush = fail_flush

    def flush(self):
        self.flushes += 1
        if self.flushes == self.fail_flush:
            raise OSError('PRIVATE write error')
        return super().flush()

    def close(self):
        pass


@pytest.fixture
def fake(monkeypatch):
    state = SimpleNamespace(output=Output(), calls=[], estimate_messages=[], failures=set())

    class Provider(run.FakeProvider):
        def __init__(self, **kwargs):
            assert kwargs['max_retries'] == 0
            super().__init__(**kwargs)

        async def create(self, **kwargs):
            metadata = 'project metadata extractor' in kwargs['messages'][0]['content']
            if not metadata:
                state.estimate_messages.append(kwargs['messages'])
                turn = len(state.estimate_messages)
            else:
                turn = len(state.estimate_messages)
            state.calls.append(('metadata' if metadata else 'estimate', turn, state.output.flushes))
            if not metadata and turn in state.failures:
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                    content='{PRIVATE CREDENTIAL_SENTINEL'))])
            return await super().create(**kwargs)

    monkeypatch.setattr(os, 'environ', {'OPENAI_API_KEY': 'CREDENTIAL_SENTINEL'})
    monkeypatch.setattr(Path, 'open', lambda *args, **kwargs: state.output)
    monkeypatch.setattr(llm_service, 'AsyncOpenAI', Provider)
    return state


def rows(fake):
    return list(csv.DictReader(io.StringIO(fake.output.getvalue())))


@pytest.mark.parametrize(('failure', 'indices', 'scores'), [
    (1, ['', '1', '2'], ['', '', '0.0']),
    (2, ['1', '', '2'], ['', '', '1.0']),
    (3, ['1', '2', ''], ['', '1.0', '']),
])
def test_rejected_turn_continues_only_after_flushed_receipt(fake, capsys, caplog, failure, indices, scores):
    fake.failures = {failure}
    with caplog.at_level(logging.DEBUG):
        code = run.main(['--real', '--lengths', '3', '--output', 'report.csv'])
    out, err = capsys.readouterr()
    assert (code, out, err) == (1, 'Wrote 3 real turns (1 rejected).\n',
        'Error: real provider returned an invalid structured result (stage: estimate; category: invalid JSON).\n')
    result = rows(fake)
    assert [r['scenario_turn'] for r in result] == ['1', '2', '3']
    assert [r['turn_index'] for r in result] == indices
    assert [r['memory_score'] for r in result] == scores
    assert [r['status'] for r in result] == ['invalid_estimate' if i == failure else 'committed' for i in (1, 2, 3)]
    assert [r['reason'] for r in result] == ['estimate_validation' if i == failure else '' for i in (1, 2, 3)]
    rejected = result[failure - 1]
    assert all(rejected[name] == '' for name in ('turn_index', 'tokens_in', 'tokens_out', 'cost_usd',
               'latency_ms', 'messages_in_window', 'anchors_count', 'summary_chars', 'cache_hit_kind',
               'last_resolved_tier', 'memory_score'))
    assert [c[2] for c in fake.calls if c[0] == 'estimate'] == [1, 2, 3]
    assert len(fake.calls) == 5 and fake.output.flushes >= 4
    if failure < 3:
        later = json.dumps(fake.estimate_messages[-1])
        assert ('Add customer accounts.' in later) is (failure != 2 and failure == 1)
        if failure == 2:
            assert 'Add customer accounts.' not in later and 'Harbor' in later
        if failure == 1:
            assert 'Build a React web portal' not in later
    assert all(value not in out + err + caplog.text + fake.output.getvalue()
               for value in ('PRIVATE', 'CREDENTIAL_SENTINEL', 'Traceback'))


@pytest.mark.parametrize('failed_flush,expected_calls', [(1, []), (2, [('estimate', 1), ('metadata', 1)]),
                                                         (3, [('estimate', 1), ('metadata', 1), ('estimate', 2)])])
def test_write_failure_stops_before_next_provider(fake, capsys, failed_flush, expected_calls):
    fake.failures = {2}
    fake.output.fail_flush = failed_flush
    code = run.main(['--real', '--lengths', '3', '--output', 'report.csv'])
    assert (code, *capsys.readouterr()) == (1, '', 'Error: real execution or output failed.\n')
    assert [(a, b) for a, b, _ in fake.calls] == expected_calls


@pytest.mark.parametrize('failure', ['transport', 'metadata', 'spoof'])
def test_other_failures_stop_without_fabricating_receipt(fake, monkeypatch, capsys, failure):
    original = llm_service.AsyncOpenAI.create

    async def fail(self, **kwargs):
        metadata = 'project metadata extractor' in kwargs['messages'][0]['content']
        if len(fake.estimate_messages) == 1 and not metadata:
            if failure == 'transport':
                fake.calls.append(('transport', 2, fake.output.flushes))
                raise RuntimeError('PRIVATE')
            if failure == 'spoof':
                fake.calls.append(('spoof', 2, fake.output.flushes))
                raise llm_service.EstimationError(502, 'Estimation provider returned an invalid result.')
        if failure == 'metadata' and metadata and len(fake.estimate_messages) == 2:
            fake.calls.append(('metadata', 2, fake.output.flushes))
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{PRIVATE'))])
        return await original(self, **kwargs)

    monkeypatch.setattr(llm_service.AsyncOpenAI, 'create', fail)
    code = run.main(['--real', '--lengths', '3', '--output', 'report.csv'])
    out, err = capsys.readouterr()
    assert code == 1 and out == '' and 'PRIVATE' not in err
    assert len(rows(fake)) == 1 and rows(fake)[0]['turn_index'] == '1'
    assert len(fake.estimate_messages) <= 2


@pytest.mark.parametrize('short_write,expected_calls', [(1, []), (2, [('estimate', 1), ('metadata', 1)])])
def test_short_write_is_uncertain_and_stops(fake, capsys, short_write, expected_calls):
    original = fake.output.write
    count = 0

    def write(value):
        nonlocal count
        count += 1
        if count == short_write:
            original(value[:-1])
            return len(value) - 1
        return original(value)

    fake.output.write = write
    code = run.main(['--real', '--lengths', '3', '--output', 'report.csv'])
    assert (code, *capsys.readouterr()) == (1, '', 'Error: real execution or output failed.\n')
    assert [(a, b) for a, b, _ in fake.calls] == expected_calls


def test_cancellation_does_not_fabricate_receipt_or_continue(fake, monkeypatch, capsys):
    original = llm_service.AsyncOpenAI.create

    async def cancel(self, **kwargs):
        metadata = 'project metadata extractor' in kwargs['messages'][0]['content']
        if not metadata and len(fake.estimate_messages) == 1:
            raise asyncio.CancelledError()
        return await original(self, **kwargs)

    monkeypatch.setattr(llm_service.AsyncOpenAI, 'create', cancel)
    with pytest.raises(asyncio.CancelledError):
        run.main(['--real', '--lengths', '3', '--output', 'report.csv'])
    assert tuple(capsys.readouterr()) == ('', '')
    assert len(rows(fake)) == 1 and rows(fake)[0]['turn_index'] == '1'
    assert len(fake.estimate_messages) == 1
