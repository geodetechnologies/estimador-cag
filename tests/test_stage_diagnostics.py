"""Public, offline checks for stage diagnostics and unchanged API failure semantics."""
import asyncio
import csv
import io
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from pydantic import SecretStr

from app.config import get_settings
from app.main import app
from app.services import llm_service
from evals.stress import run


class Output(io.StringIO):
    def close(self):
        pass


def csv_rows(output):
    return list(csv.DictReader(io.StringIO(output.getvalue())))


@pytest.fixture
def fake(monkeypatch):
    state = SimpleNamespace(kind=None, calls=[], estimate_messages=[], output=Output(),
                            secret='CREDENTIAL_SENTINEL', estimate_payload=None)

    class Provider(run.FakeProvider):
        def __init__(self, **kwargs):
            state.calls.append(('client', kwargs['max_retries']))
            super().__init__(**kwargs)

        async def create(self, **kwargs):
            metadata = 'project metadata extractor' in kwargs['messages'][0]['content']
            state.calls.append(('metadata' if metadata else 'estimate',))
            if not metadata:
                state.estimate_messages.append(kwargs['messages'])
            if state.kind == 'payload' and not metadata:
                content = json.dumps(state.estimate_payload)
            elif (state.kind == 'estimate' and not metadata) or (state.kind == 'metadata' and metadata):
                content = '{PROVIDER_SENTINEL CREDENTIAL_SENTINEL'
            elif state.kind == 'missing' and metadata:
                content = json.dumps({'project_name': 'FIELD_SENTINEL CREDENTIAL_SENTINEL'})
            else:
                return await super().create(**kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    settings = SimpleNamespace(openai_api_key=SecretStr(state.secret), openai_model='fake', max_turns=6)
    monkeypatch.setattr(llm_service, 'AsyncOpenAI', Provider)
    monkeypatch.setattr(run, 'Settings', lambda **kwargs: settings)
    monkeypatch.setattr(Path, 'open', lambda *args, **kwargs: state.output)
    app.dependency_overrides[get_settings] = lambda: settings
    yield state
    app.dependency_overrides.pop(get_settings, None)


@pytest.mark.parametrize(('kind', 'stage', 'calls'), [
    ('estimate', 'estimate', 1), ('metadata', 'metadata', 2), ('missing', 'metadata', 2),
])
def test_cli_fixed_stage_is_private_and_never_retries(fake, capsys, caplog, kind, stage, calls):
    fake.kind = kind
    with caplog.at_level(logging.DEBUG):
        code = run.main(['--real', '--lengths', '1', '--output', 'report.csv'])
    out, err = capsys.readouterr()
    suffix = '; category: invalid JSON' if kind == 'estimate' else ''
    stdout = 'Wrote 1 real turns (1 rejected).\n' if kind == 'estimate' else ''
    assert (code, out, err) == (1, stdout, f'Error: real provider returned an invalid structured result (stage: {stage}{suffix}).\n')
    assert [row['status'] for row in csv_rows(fake.output)] == (['invalid_estimate'] if kind == 'estimate' else [])
    assert len([c for c in fake.calls if c[0] in ('estimate', 'metadata')]) == calls
    assert all(c[1] == 0 for c in fake.calls if c[0] == 'client')
    assert all(value not in out + err + caplog.text + fake.output.getvalue() for value in
               ('CREDENTIAL_SENTINEL', 'PROVIDER_SENTINEL', 'FIELD_SENTINEL', 'LOG_SENTINEL', 'Traceback'))


def _valid_estimate():
    return {'summary': 'Synthetic project estimate.', 'confidence_pct': 80,
            'phases': [{'name': 'Build', 'duration_weeks': 2, 'cost_eur': 500,
                        'summary': 'Build the synthetic project.'}],
            'total_duration_weeks': 2, 'total_cost_eur': 500}


@pytest.mark.parametrize(('change', 'reason'), [
    ('missing', 'field: summary; category: missing'),
    ('minimum', 'field: confidence_pct; category: below minimum'),
    ('maximum', 'field: total_duration_weeks; category: above maximum'),
    ('phase', 'field: phases.cost_eur; category: below minimum'),
    ('type', 'field: phases.name; category: invalid type'),
    ('nested_missing', 'field: phases.summary; category: missing'),
    ('number_type', 'field: total_cost_eur; category: invalid type'),
    ('list_type', 'field: phases; category: invalid type'),
    ('text_short', 'field: summary; category: too short'),
    ('text_long', 'field: phases.name; category: too long'),
    ('list_short', 'field: phases; category: too short'),
    ('list_long', 'field: phases; category: too long'),
])
def test_cli_estimate_validation_reason_is_fixed_and_private(fake, capsys, caplog, change, reason):
    payload = _valid_estimate()
    if change == 'missing':
        del payload['summary']
    elif change == 'minimum':
        payload['confidence_pct'] = -1
    elif change == 'maximum':
        payload['total_duration_weeks'] = 105
    elif change == 'phase':
        payload['phases'][0]['cost_eur'] = -1
    elif change == 'type':
        payload['phases'][0]['name'] = {'FIELD_SENTINEL': 'PROVIDER_SENTINEL CREDENTIAL_SENTINEL'}
    elif change == 'nested_missing':
        del payload['phases'][0]['summary']
    elif change == 'number_type':
        payload['total_cost_eur'] = 'PROVIDER_SENTINEL CREDENTIAL_SENTINEL'
    elif change == 'list_type':
        payload['phases'] = 'PROVIDER_SENTINEL CREDENTIAL_SENTINEL'
    elif change == 'text_short':
        payload['summary'] = 'short'
    elif change == 'text_long':
        payload['phases'][0]['name'] = 'x' * 65
    elif change == 'list_short':
        payload['phases'] = []
    elif change == 'list_long':
        payload['phases'] = payload['phases'] * 9
    payload['FIELD_SENTINEL'] = 'PROVIDER_SENTINEL CREDENTIAL_SENTINEL'
    fake.kind, fake.estimate_payload = 'payload', payload
    with caplog.at_level(logging.DEBUG):
        code = run.main(['--real', '--lengths', '1', '--output', 'report.csv'])
    out, err = capsys.readouterr()
    assert (code, out, err) == (1, 'Wrote 1 real turns (1 rejected).\n', f'Error: real provider returned an invalid structured result (stage: estimate; {reason}).\n')
    assert [row['status'] for row in csv_rows(fake.output)] == ['invalid_estimate']
    assert [c[0] for c in fake.calls] == ['client', 'estimate']
    assert all(c[1] == 0 for c in fake.calls if c[0] == 'client')
    assert all(secret not in out + err + caplog.text + fake.output.getvalue() for secret in
               ('FIELD_SENTINEL', 'PROVIDER_SENTINEL', 'CREDENTIAL_SENTINEL', 'Traceback'))


@pytest.mark.parametrize(('change', 'suffix'), [
    ('sum', '; rule: phase costs vs total'),
    ('confidence', '; rule: low confidence prefix'),
    ('both', ''),
])
def test_cli_business_rules_are_fixed_and_private(fake, capsys, caplog, change, suffix):
    payload = _valid_estimate()
    if change in ('sum', 'both'):
        payload['total_cost_eur'] = 501
    if change in ('confidence', 'both'):
        payload['confidence_pct'] = 29
    payload['FIELD_SENTINEL'] = 'PROVIDER_SENTINEL CREDENTIAL_SENTINEL'
    fake.kind, fake.estimate_payload = 'payload', payload
    with caplog.at_level(logging.DEBUG):
        code = run.main(['--real', '--lengths', '1', '--output', 'report.csv'])
    out, err = capsys.readouterr()
    assert (code, out, err) == (1, 'Wrote 1 real turns (1 rejected).\n', 'Error: real provider returned an invalid structured result '
                               f'(stage: estimate{suffix}).\n')
    assert [row['status'] for row in csv_rows(fake.output)] == ['invalid_estimate']
    assert [c[0] for c in fake.calls] == ['client', 'estimate']
    assert all(secret not in out + err + caplog.text for secret in
               ('FIELD_SENTINEL', 'PROVIDER_SENTINEL', 'CREDENTIAL_SENTINEL', '500', '501', 'Traceback'))


@pytest.mark.parametrize('kind', ['fabricated', 'actual'])
def test_fake_sdk_spoofed_validation_error_cannot_claim_business_rule(fake, monkeypatch, capsys, caplog, kind):
    from pydantic import ValidationError
    from app.schemas import EstimationResult

    async def fail(self, **kwargs):
        fake.calls.append(('estimate',))
        try:
            if kind == 'actual':
                payload = _valid_estimate()
                payload['total_cost_eur'] = 501
                payload['FIELD_SENTINEL'] = 'PROVIDER_SENTINEL CREDENTIAL_SENTINEL'
                EstimationResult.model_validate_json(json.dumps(payload))
            else:
                raise ValidationError.from_exception_data('EstimationResult', [{
                    'type': 'value_error', 'loc': (), 'input': 'PROVIDER_SENTINEL',
                    'ctx': {'error': ValueError('phases sum (500 EUR) does not match total_cost_eur (501 EUR)')},
                }])
        except ValidationError:
            error = llm_service.EstimationError(502, 'Estimation provider returned an invalid result.')
            error._validation_business_rule = llm_service._PHASE_COST_RULE
            error._validation_origin = object()
            raise error from None

    monkeypatch.setattr(llm_service.AsyncOpenAI, 'create', fail)
    with caplog.at_level(logging.DEBUG):
        code = run.main(['--real', '--lengths', '1', '--output', 'report.csv'])
    out, err = capsys.readouterr()
    assert (code, out, err) == (1, '', 'Error: real provider returned an invalid structured result.\n')
    assert csv_rows(fake.output) == [] and [c[0] for c in fake.calls] == ['client', 'estimate']
    assert all(secret not in out + err + caplog.text for secret in
               ('PROVIDER_SENTINEL', 'CREDENTIAL_SENTINEL', '500', '501', 'Traceback'))


@pytest.mark.parametrize(('confidence', 'summary'), [
    (30, 'Synthetic project estimate.'),
    (29, 'Out of scope: synthetic project.'),
])
def test_business_rule_boundaries_keep_successful_cli_behavior(fake, capsys, confidence, summary):
    payload = _valid_estimate()
    payload.update(confidence_pct=confidence, summary=summary)
    fake.kind, fake.estimate_payload = 'payload', payload
    code = run.main(['--real', '--lengths', '1', '--output', 'report.csv'])
    assert (code, *capsys.readouterr()) == (0, 'Wrote 1 real turns.\n', '')
    assert fake.output.getvalue().count('\n') == 2
    assert [c[0] for c in fake.calls] == ['client', 'estimate', 'client', 'metadata']


def test_spoofed_business_message_from_provider_boundary_stays_generic(fake, monkeypatch, capsys, caplog):
    from pydantic import ValidationError

    async def fail(*args, **kwargs):
        try:
            raise ValidationError.from_exception_data('EstimationResult', [{
                'type': 'value_error', 'loc': (), 'input': 'PROVIDER_SENTINEL',
                'ctx': {'error': ValueError('phases sum (500 EUR) does not match total_cost_eur '
                                            '(501 EUR); adjust either the phases or the total')},
            }])
        except ValidationError:
            raise llm_service.EstimationError(502, 'Estimation provider returned an invalid result.') from None

    # This exception comes from estimate(), before structured_estimate validates JSON.
    monkeypatch.setattr(llm_service, 'estimate', fail)
    with caplog.at_level(logging.DEBUG):
        code = run.main(['--real', '--lengths', '1', '--output', 'report.csv'])
    out, err = capsys.readouterr()
    assert (code, out, err) == (1, '', 'Error: real provider returned an invalid structured result.\n')
    assert csv_rows(fake.output) == [] and fake.calls == []
    assert all(secret not in out + err + caplog.text for secret in
               ('PROVIDER_SENTINEL', 'CREDENTIAL_SENTINEL', '500', '501', 'Traceback'))


@pytest.mark.parametrize(('change', 'reason'), [
    ('multiple', 'field: summary; category: missing; field: confidence_pct; category: missing'),
    ('truncated', 'field: summary; category: missing; field: confidence_pct; category: missing; '
                  'field: total_duration_weeks; category: missing; additional errors omitted'),
    ('duplicate', 'field: phases.name; category: missing; field: phases.duration_weeks; category: missing; '
                  'field: phases.cost_eur; category: missing; additional errors omitted'),
])
def test_cli_multiple_errors_are_bounded_and_private(fake, capsys, caplog, change, reason):
    payload = _valid_estimate()
    if change in ('multiple', 'truncated'):
        del payload['summary']
        del payload['confidence_pct']
        if change == 'truncated':
            del payload['total_duration_weeks']
            del payload['total_cost_eur']
    else:
        payload['phases'] = [{'FIELD_SENTINEL': 'PROVIDER_SENTINEL CREDENTIAL_SENTINEL'}] * 2
    payload['FIELD_SENTINEL'] = 'PROVIDER_SENTINEL CREDENTIAL_SENTINEL'
    fake.kind, fake.estimate_payload = 'payload', payload
    with caplog.at_level(logging.DEBUG):
        code = run.main(['--real', '--lengths', '1', '--output', 'report.csv'])
    out, err = capsys.readouterr()
    assert (code, out, err) == (1, 'Wrote 1 real turns (1 rejected).\n', f'Error: real provider returned an invalid structured result (stage: estimate; {reason}).\n')
    assert [row['status'] for row in csv_rows(fake.output)] == ['invalid_estimate']
    assert [c[0] for c in fake.calls] == ['client', 'estimate']
    assert all(c[1] == 0 for c in fake.calls if c[0] == 'client')
    assert all(secret not in out + err + caplog.text for secret in
               ('FIELD_SENTINEL', 'PROVIDER_SENTINEL', 'CREDENTIAL_SENTINEL', 'Traceback'))


def test_cli_mixed_unknown_and_recognized_errors_do_not_echo_unknown(fake, monkeypatch, capsys, caplog):
    from pydantic import ValidationError
    from app.services import estimation

    async def fail(*args, **kwargs):
        try:
            raise ValidationError.from_exception_data('PROVIDER_SENTINEL', [
                {'type': 'missing', 'loc': ('summary',), 'input': 'CREDENTIAL_SENTINEL'},
                {'type': 'missing', 'loc': ('FIELD_SENTINEL',), 'input': 'PROVIDER_SENTINEL'},
            ])
        except ValidationError:
            error = llm_service.EstimationError(502, 'Estimation provider returned an invalid result.')
            error._validation_stage = estimation._INVALID_ESTIMATE_STAGE
            raise error from None
    monkeypatch.setattr(run.estimation, 'estimate_conversational', fail)
    with caplog.at_level(logging.DEBUG):
        code = run.main(['--real', '--lengths', '1', '--output', 'report.csv'])
    out, err = capsys.readouterr()
    assert (code, out, err) == (1, '', 'Error: real provider returned an invalid structured result.\n')
    assert csv_rows(fake.output) == [] and fake.calls == []
    assert all(secret not in out + err + caplog.text for secret in
               ('FIELD_SENTINEL', 'PROVIDER_SENTINEL', 'CREDENTIAL_SENTINEL', 'Traceback'))


def test_cli_unknown_location_does_not_echo_field_or_error(fake, monkeypatch, capsys, caplog):
    from pydantic import ValidationError
    from app.services import estimation

    async def fail(*args, **kwargs):
        try:
            raise ValidationError.from_exception_data('PROVIDER_SENTINEL', [{
                'type': 'missing', 'loc': ('FIELD_SENTINEL',),
                'input': 'PROVIDER_SENTINEL CREDENTIAL_SENTINEL',
            }])
        except ValidationError:
            error = llm_service.EstimationError(502, 'Estimation provider returned an invalid result.')
            error._validation_stage = estimation._INVALID_ESTIMATE_STAGE
            raise error from None
    monkeypatch.setattr(run.estimation, 'estimate_conversational', fail)
    with caplog.at_level(logging.DEBUG):
        code = run.main(['--real', '--lengths', '1', '--output', 'report.csv'])
    out, err = capsys.readouterr()
    assert (code, out, err) == (1, '', 'Error: real provider returned an invalid structured result.\n')
    assert csv_rows(fake.output) == [] and fake.calls == []
    assert all(secret not in out + err + caplog.text for secret in
               ('FIELD_SENTINEL', 'PROVIDER_SENTINEL', 'CREDENTIAL_SENTINEL', 'Traceback'))


def test_cli_unknown_invalid_stage_keeps_existing_fixed_fallback(fake, monkeypatch, capsys):
    async def fail(*args, **kwargs):
        error = llm_service.EstimationError(502, 'Estimation provider returned an invalid result.')
        error._validation_stage = 'FIELD_SENTINEL CREDENTIAL_SENTINEL'
        raise error
    monkeypatch.setattr(run.estimation, 'estimate_conversational', fail)
    code = run.main(['--real', '--lengths', '1', '--output', 'report.csv'])
    assert (code, *capsys.readouterr()) == (1, '', 'Error: real provider returned an invalid structured result.\n')
    assert csv_rows(fake.output) == [] and fake.calls == []


@pytest.mark.parametrize('kind', ['estimate', 'metadata', 'missing', 'sum', 'confidence', 'both'])
def test_normal_api_keeps_exact_error_and_atomic_state(fake, capsys, caplog, kind):
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            session = (await client.post('/sessions')).json()['session_id']
            url = f'/sessions/{session}'
            initial = await client.post(url + '/estimate', data={'transcript': 'Harbor initial turn'})
            assert initial.status_code == 200 and initial.json()['project_metadata']['project_name'] == 'Harbor'
            before = (await client.get(url)).json()
            if kind in ('sum', 'confidence', 'both'):
                payload = _valid_estimate()
                if kind in ('sum', 'both'):
                    payload['total_cost_eur'] = 501
                if kind in ('confidence', 'both'):
                    payload['confidence_pct'] = 29
                payload['FIELD_SENTINEL'] = 'PROVIDER_SENTINEL CREDENTIAL_SENTINEL'
                fake.kind, fake.estimate_payload = 'payload', payload
            else:
                fake.kind = kind
            response = await client.post(url + '/estimate', data={'transcript': 'discard this turn'})
            assert response.status_code == 502
            assert response.json() == {'detail': 'Estimation provider returned an invalid result.'}
            assert (await client.get(url)).json() == before
            fake.kind = None
            success = await client.post(url + '/estimate', data={'transcript': 'fresh turn'})
            assert success.status_code == 200
            assert (await client.get(url)).json()['history_window_message_count'] == 4
            assert success.json()['project_metadata']['project_name'] == 'Harbor'
            assert 'discard this turn' not in json.dumps(fake.estimate_messages[-1])
            assert 'Harbor initial turn' in json.dumps(fake.estimate_messages[-1])
            return response.text
    with caplog.at_level(logging.INFO):
        response_text = asyncio.run(check())
    out, err = capsys.readouterr()
    assert out == err == ''
    assert all(value not in response_text + out + err + caplog.text for value in
               ('PROVIDER_SENTINEL', 'FIELD_SENTINEL', 'CREDENTIAL_SENTINEL'))
    assert 'stage' not in response_text
    observations = [record.getMessage() for record in caplog.records if record.getMessage().startswith('turn_observed ')]
    assert len(observations) == 3
    assert [json.loads(item.split(' ', 1)[1])['turn_index'] for item in observations] == [1, None, 2]
