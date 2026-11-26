import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.config import Settings, get_settings
from app.main import app
from app.services import llm_service


CASE = {'id': 'portal-case', 'transcript': 'Build a customer portal for appointments.',
        'project_type': 'web_saas', 'expected_in_summary': ['portal']}
RESULT = {'summary': 'A customer portal for appointments', 'confidence_pct': 80,
          'phases': [{'name': 'Build', 'duration_weeks': 2, 'cost_eur': 500,
                      'summary': 'Build the customer portal'}],
          'total_duration_weeks': 2, 'total_cost_eur': 500}
METADATA = {'project_name': None, 'assumed_team_size': None, 'mentioned_technologies': [],
            'agreed_scope': [], 'explicit_constraints': [], 'rejected_options': []}
SESSION = '00000000-0000-4000-8000-000000000001'


def dataset(tmp_path, cases):
    path = tmp_path / 'cases.json'
    path.write_text(json.dumps(cases), encoding='utf-8')
    return str(path)


def cli(args, capsys):
    from evals.run import main
    code = main(args)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def report(case_id, passed=True):
    return {'cases': [{'id': case_id, 'passed': passed, 'metrics': [
        {'name': 'schema_adherence', 'score': 1.0, 'passed': True, 'details': 'Schema checks passed.'},
        {'name': 'cost_bounds', 'score': 1.0, 'passed': True, 'details': 'Cost and duration checks passed.'},
        {'name': 'content_recall', 'score': 1.0 if passed else 0.0, 'passed': passed,
         'details': 'Content checks passed.' if passed else 'Expected content not found.'}]}]}


def test_http_runner_posts_each_transcript_to_fresh_session_and_scores(tmp_path, monkeypatch, capsys):
    requests = []
    def handle(request):
        requests.append(request)
        if request.url.path == '/sessions':
            return httpx.Response(200, json={'session_id': SESSION})
        assert request.url.path == f'/sessions/{SESSION}/estimate'
        assert request.headers['content-type'].startswith('application/x-www-form-urlencoded')
        assert request.content == b'transcript=Build+a+customer+portal+for+appointments.'
        return httpx.Response(200, json={'result': RESULT, 'prompt_version': 'v1',
                                        'cached': False, 'project_metadata': METADATA})
    monkeypatch.setattr(httpx, 'AsyncHTTPTransport', lambda **kwargs: httpx.MockTransport(handle))
    code, out, err = cli(['--transport', 'http', '--base-url', 'http://example.test',
                          '--dataset', dataset(tmp_path, [CASE, CASE | {'id': 'second-case'}])], capsys)
    assert (code, err) == (0, '')
    assert json.loads(out) == {'cases': report('portal-case')['cases'] + report('second-case')['cases']}
    assert [r.method for r in requests] == ['POST', 'POST', 'POST', 'POST']
    assert CASE['transcript'] not in out


def test_metric_failure_is_reported_without_response_content(tmp_path, monkeypatch, capsys):
    def handle(request):
        return httpx.Response(200, json=({'session_id': SESSION} if request.url.path == '/sessions'
            else {'result': RESULT, 'prompt_version': 'v1', 'cached': False, 'project_metadata': METADATA}))
    monkeypatch.setattr(httpx, 'AsyncHTTPTransport', lambda **kwargs: httpx.MockTransport(handle))
    code, out, err = cli(['--transport', 'http', '--base-url', 'http://example.test', '--dataset',
                          dataset(tmp_path, [CASE | {'expected_in_summary': ['missing']}])], capsys)
    assert (code, err) == (1, '')
    assert json.loads(out) == report('portal-case', False)
    assert RESULT['summary'] not in out


@pytest.mark.parametrize('failure', ['create', 'estimate', 'invalid-result'])
def test_remote_failures_do_not_echo_sensitive_details(tmp_path, monkeypatch, capsys, failure):
    paths = []
    def handle(request):
        paths.append(request.url.path)
        if request.url.path == '/sessions':
            return httpx.Response(503, text='private provider data') if failure == 'create' else httpx.Response(200, json={'session_id': SESSION})
        if failure == 'estimate':
            return httpx.Response(502, text='private provider data')
        return httpx.Response(200, json={'result': {'summary': 'private provider data'}})
    monkeypatch.setattr(httpx, 'AsyncHTTPTransport', lambda **kwargs: httpx.MockTransport(handle))
    code, out, err = cli(['--transport', 'http', '--base-url', 'http://example.test', '--dataset',
                          dataset(tmp_path, [CASE])], capsys)
    assert (code, err) == (1, '')
    assert json.loads(out) == {'cases': [{'id': 'portal-case', 'passed': False, 'error': 'Evaluation failed.'}]}
    assert 'private' not in out and CASE['transcript'] not in out
    assert paths == (['/sessions'] if failure == 'create' else ['/sessions', f'/sessions/{SESSION}/estimate'])


def test_in_process_uses_public_session_api_with_mock_provider(tmp_path, monkeypatch, capsys):
    calls = []
    async def create(**kwargs):
        calls.append(kwargs)
        system = kwargs['messages'][0]['content']
        content = json.dumps(METADATA if 'project metadata extractor' in system else RESULT)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    context = AsyncMock()
    context.__aenter__.return_value = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=AsyncMock(side_effect=create))))
    monkeypatch.setattr(llm_service, 'AsyncOpenAI', MagicMock(return_value=context))
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, openai_api_key='test')
    try:
        code, out, err = cli(['--transport', 'in-process', '--dataset', dataset(tmp_path, [CASE])], capsys)
        assert (code, err) == (0, '')
        assert json.loads(out) == report('portal-case')
        assert len(calls) == 2
        assert calls[0]['messages'][-1]['content'] == CASE['transcript']

        async def check_prior_contract():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                session = (await client.post('/sessions')).json()['session_id']
                response = await client.post(f'/sessions/{session}/estimate', data={'transcript': CASE['transcript']})
                assert response.status_code == 200
                assert response.json() == {'result': RESULT, 'prompt_version': 'v1',
                                           'cached': False, 'project_metadata': METADATA}
                inspected = await client.get(f'/sessions/{session}')
                assert inspected.status_code == 200
                assert inspected.json() == {'session_id': session, 'history_window_message_count': 2, 'max_turns': 6}
        asyncio.run(check_prior_contract())
    finally:
        app.dependency_overrides.pop(get_settings, None)


def test_invalid_dataset_is_safe_and_does_not_send_requests(tmp_path, monkeypatch, capsys):
    def forbidden(**kwargs):
        pytest.fail('must not construct a transport for invalid dataset')
    monkeypatch.setattr(httpx, 'AsyncHTTPTransport', forbidden)
    code, out, err = cli(['--transport', 'http', '--base-url', 'http://example.test', '--dataset',
                          dataset(tmp_path, [CASE | {'transcript': 'secret'}])], capsys)
    assert (code, out, err) == (2, '', 'Unable to load evaluation dataset.\n')


def test_cli_help_describes_transports(capsys):
    from evals.run import main
    with pytest.raises(SystemExit) as exc:
        main(['--help'])
    assert exc.value.code == 0
    out, err = capsys.readouterr()
    assert '--transport' in out and 'in-process' in out and 'http' in out
    assert '--dataset' in out and '--base-url' in out
    assert err == ''
