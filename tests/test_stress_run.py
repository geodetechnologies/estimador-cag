"""Public CLI contract for the offline synthetic stress runner."""
import csv
import io
import subprocess
import sys
from tempfile import TemporaryDirectory
from pathlib import Path

import pytest


FIELDS = ('turn_index', 'session_id', 'enriched_transcript_chars', 'attachments_total_chars',
          'messages_in_window', 'anchors_count', 'summary_chars', 'tokens_in', 'tokens_out',
          'cost_usd', 'latency_ms', 'cache_hit_kind', 'last_resolved_tier')


def cli(*args):
    return subprocess.run([sys.executable, '-m', 'evals.stress.run', *map(str, args)],
                          capture_output=True, text=True, check=False)


def test_simulated_rows_are_content_free_and_account_for_every_turn():
    with TemporaryDirectory() as directory:
        target = Path(directory) / 'report.csv'
        result = cli('--scenarios', 'growing,pivot', '--lengths', '1,3',
                     '--attachment-sizes', '0,5', '--repeats', '2', '--output', target)
        assert (result.returncode, result.stdout, result.stderr) == (0, 'Wrote 32 simulated turns.\n', '')
        content = target.read_text(encoding='utf-8')
        rows = list(csv.DictReader(io.StringIO(content)))
        assert len(rows) == 32
        assert set(FIELDS) <= set(rows[0])
        assert all(row['mode'] == 'simulated' for row in rows)
        assert all(row['session_id'] == row['tokens_in'] == row['tokens_out'] ==
                   row['cost_usd'] == row['latency_ms'] == '' for row in rows)
        assert all(row['cache_hit_kind'] == 'none' for row in rows)
        assert all(row['last_resolved_tier'] == ('developer' if row['turn_index'] == '1' else
                   'pm' if row['attachment_size_kib'] == '5' else 'default') for row in rows)
        assert all(int(row['messages_in_window']) == 2 * int(row['turn_index']) for row in rows)
        assert all(row['anchors_count'] == row['summary_chars'] == '0' for row in rows)
        growing = [row for row in rows if row['scenario'] == 'growing' and row['turn_index'] == '2']
        assert all(row['memory_score'] == '1.0' for row in growing)
        for row in rows:
            assert int(row['enriched_transcript_chars']) > 0
            assert (int(row['attachments_total_chars']) > 0) == (row['attachment_size_kib'] == '5')
            assert int(row['attachment_bytes']) == int(row['attachment_size_kib']) * 1024
        for forbidden in ('Harbor', 'customer accounts', 'SYNTHETIC STRESS FIXTURE',
                          'Synthetic project estimate.', 'Synthetic folded summary.', 'We commit to ship.'):
            assert forbidden not in content + result.stdout + result.stderr
        assert sorted(p.name for p in Path(directory).iterdir()) == ['report.csv']


def test_all_attachment_sizes_and_twenty_turn_window():
    with TemporaryDirectory() as directory:
        target = Path(directory) / 'report.csv'
        result = cli('--scenarios', 'contradiction', '--lengths', '20',
                     '--attachment-sizes', '0,5,20,50,100', '--output', target)
        assert (result.returncode, result.stdout, result.stderr) == (0, 'Wrote 100 simulated turns.\n', '')
        rows = list(csv.DictReader(io.StringIO(target.read_text(encoding='utf-8'))))
        assert len(rows) == 100
        for size in (0, 5, 20, 50, 100):
            group = [row for row in rows if row['attachment_size_kib'] == str(size)]
            assert [int(row['turn_index']) for row in group] == list(range(1, 21))
            assert all(int(row['attachment_bytes']) == size * 1024 for row in group)
            assert all(int(row['messages_in_window']) == min(12, 2 * int(row['turn_index'])) for row in group)
            assert all((int(row['attachments_total_chars']) > 0) == bool(size) for row in group)
            assert all(row['anchors_count'] == ('0' if int(row['turn_index']) <= 6 else '1') for row in group)
            assert all((int(row['summary_chars']) > 0) == (int(row['turn_index']) >= 7) for row in group)


def test_cli_calls_actual_estimator_with_enriched_synthetic_input():
    # Instrument only the public CLI subprocess; the real implementation still runs.
    wrapper = '''
import runpy, socket, sys
from app.config import Settings
from app.services import estimation

def forbidden(*args, **kwargs):
    raise AssertionError('Network or settings access')
socket.socket.connect = forbidden
socket.socket.connect_ex = forbidden
socket.create_connection = forbidden
Settings.__init__ = forbidden
original = estimation.estimate_conversational
calls = []
async def checked(session, user, settings, *, observation=None):
    assert "SYNTHETIC STRESS FIXTURE" in user
    assert observation["attachments_total_chars"] > 0
    result = await original(session, user, settings, observation=observation)
    assert observation["turn_index"] == session.committed_turns
    assert observation["anchors_count"] == len(session.anchors)
    assert observation["summary_chars"] == len(session.summary)
    assert observation["last_resolved_tier"] == session.tier
    calls.append(1)
    return result
estimation.estimate_conversational = checked
try:
    runpy.run_module("evals.stress.run", run_name="__main__")
except SystemExit as error:
    assert error.code == 0 and len(calls) == 10
'''
    with TemporaryDirectory() as directory:
        target = Path(directory) / 'report.csv'
        result = subprocess.run([sys.executable, '-c', wrapper, '--scenarios', 'growing',
                                 '--lengths', '10', '--attachment-sizes', '5', '--output', str(target)],
                                capture_output=True, text=True, check=False)
        assert (result.returncode, result.stdout, result.stderr) == (0, 'Wrote 10 simulated turns.\n', '')
        rows = list(csv.DictReader(io.StringIO(target.read_text(encoding='utf-8'))))
        assert rows[0]['last_resolved_tier'] == 'developer'
        assert rows[1]['last_resolved_tier'] == 'pm'
        assert rows[6]['anchors_count'] == '1' and int(rows[6]['summary_chars']) > 0
        assert 'SYNTHETIC STRESS FIXTURE' not in target.read_text(encoding='utf-8')


def test_existing_output_and_invalid_inputs_leave_data_unchanged():
    with TemporaryDirectory() as directory:
        target = Path(directory) / 'report.csv'
        target.write_text('user artifact', encoding='utf-8')
        result = cli('--output', target)
        assert (result.returncode, result.stdout, result.stderr) == (1, '', 'Error: output already exists.\n')
        assert target.read_text(encoding='utf-8') == 'user artifact'
        missing = Path(directory) / 'missing.csv'
        result = cli('--scenarios', 'private-secret', '--output', missing)
        assert (result.returncode, result.stdout, result.stderr) == (2, '', 'Error: invalid configuration.\n')
        assert not missing.exists()
        assert target.read_text(encoding='utf-8') == 'user artifact'


@pytest.mark.parametrize('arguments', [('--attachment-sizes', '10'), ('--lengths', '2'),
                                        ('--repeats', '0'), ('--http', 'http://example.invalid:8000'),
                                        ('--http', 'http://localhost:65536'),
                                        ('--http', 'http://localhost:8000/?PRIVATE'),
                                        ('--http', 'http://0.0.0.0:8000')])
def test_unsupported_options_fail_closed(arguments):
    with TemporaryDirectory() as directory:
        target = Path(directory) / 'report.csv'
        result = cli(*arguments, '--output', target)
        assert (result.returncode, result.stdout, result.stderr) == (2, '', 'Error: invalid configuration.\n')
        assert not target.exists()


def test_http_simulation_exercises_real_estimator_and_enriched_pdfs_without_exposing_content():
    wrapper = '''
import csv, runpy, socket, sys
from app.config import Settings
from app.services import estimation, llm_service
original_connect = socket.socket.connect
def local_connect(self, address):
    assert isinstance(address, tuple) and address[0] == '127.0.0.1'
    return original_connect(self, address)
socket.socket.connect = local_connect
def forbidden(*args, **kwargs):
    raise AssertionError('settings access')
Settings.__init__ = forbidden
original = estimation.estimate_conversational
calls = []
async def checked(session, user, settings, *, observation=None, _validation_origin=None):
    assert llm_service.AsyncOpenAI.__name__ == 'FakeProvider'
    assert 'SYNTHETIC STRESS FIXTURE' in user
    assert observation['attachments_total_chars'] > 0
    result = await original(session, user, settings, observation=observation,
                            _validation_origin=_validation_origin)
    assert observation['turn_index'] == session.committed_turns
    assert observation['messages_in_window'] == len(session.history.to_messages_list('')) - 1
    assert observation['anchors_count'] == len(session.anchors)
    assert observation['summary_chars'] == len(session.summary)
    assert observation['last_resolved_tier'] == session.tier
    calls.append(dict(observation))
    return result
estimation.estimate_conversational = checked
try:
    runpy.run_module('evals.stress.run', run_name='__main__')
except SystemExit as error:
    assert error.code == 0 and len(calls) == 3
    with open(sys.argv[-1], encoding='utf-8', newline='') as output:
        rows = list(csv.DictReader(output))
    keys = ('turn_index', 'session_id', 'enriched_transcript_chars', 'attachments_total_chars',
            'messages_in_window', 'anchors_count', 'summary_chars', 'tokens_in', 'tokens_out',
            'cost_usd', 'latency_ms', 'cache_hit_kind', 'last_resolved_tier')
    assert all(row[key] == ('' if observed.get(key) is None else str(observed[key]))
               for row, observed in zip(rows, calls) for key in keys)
'''
    with TemporaryDirectory() as directory:
        target = Path(directory) / 'report.csv'
        result = subprocess.run([sys.executable, '-c', wrapper, '--http', 'http://localhost:0',
                                 '--lengths', '3', '--attachment-sizes', '5', '--output', str(target)],
                                capture_output=True, text=True, check=False)
        assert (result.returncode, result.stdout, result.stderr) == (0, 'Wrote 3 simulated turns.\n', '')
        content = target.read_text(encoding='utf-8')
        rows = list(csv.DictReader(io.StringIO(content)))
        assert len(rows) == 3
        assert all(row['mode'] == 'simulated-http' for row in rows)
        assert [row['turn_index'] for row in rows] == ['1', '2', '3']
        assert all(int(row['attachments_total_chars']) > 0 for row in rows)
        assert all(row['session_id'] == row['tokens_in'] == row['tokens_out'] ==
                   row['cost_usd'] == row['latency_ms'] == '' for row in rows)
        for private in ('Harbor', 'SYNTHETIC STRESS FIXTURE', 'Synthetic project estimate.', 'Bearer'):
            assert private not in content + result.stdout + result.stderr
        assert sorted(p.name for p in Path(directory).iterdir()) == ['report.csv']


def test_http_simulation_60_rows_and_existing_output_is_never_overwritten():
    with TemporaryDirectory() as directory:
        target = Path(directory) / 'report.csv'
        result = cli('--http', 'http://localhost:0', '--scenarios', 'growing,pivot', '--lengths', '10',
                     '--attachment-sizes', '0,5,20', '--output', target)
        assert (result.returncode, result.stdout, result.stderr) == (0, 'Wrote 60 simulated turns.\n', '')
        rows = list(csv.DictReader(io.StringIO(target.read_text(encoding='utf-8'))))
        assert len(rows) == 60 and all(row['mode'] == 'simulated-http' for row in rows)
        assert all((int(row['attachments_total_chars']) > 0) == (row['attachment_size_kib'] != '0')
                   for row in rows)
        original = target.read_text(encoding='utf-8')
        result = cli('--http', 'http://localhost:0', '--output', target)
        assert (result.returncode, result.stdout, result.stderr) == (1, '', 'Error: output already exists.\n')
        assert target.read_text(encoding='utf-8') == original


def test_http_explicit_loopback_port_supports_multiple_private_runs():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    with TemporaryDirectory() as directory:
        target = Path(directory) / 'report.csv'
        result = cli('--http', f'http://localhost:{port}', '--scenarios', 'growing,pivot',
                     '--lengths', '1', '--output', target)
        assert (result.returncode, result.stdout, result.stderr) == (0, 'Wrote 2 simulated turns.\n', '')
        rows = list(csv.DictReader(io.StringIO(target.read_text(encoding='utf-8'))))
        assert len(rows) == 2 and all(row['mode'] == 'simulated-http' for row in rows)


def test_help_describes_offline_and_supported_options():
    result = cli('--help')
    assert result.returncode == 0
    assert result.stderr == ''
    for text in ('simulated', '--scenarios', '--lengths', '--attachment-sizes', '--repeats', '--output', 'HTTP'):
        assert text in result.stdout
