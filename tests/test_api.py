from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app
from app.services import llm_service


@pytest.fixture
def client():
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, openai_api_key=None)
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


payload = {'description': 'A customer portal for scheduling appointments', 'project_type': 'web_saas',
           'detail_level': 'summary', 'output_format': 'narrative'}


def test_health_and_docs(client):
    assert client.get('/health').json() == {'status': 'ok'}
    assert client.get('/docs').status_code == 200
    operation = client.get('/openapi.json').json()['paths']['/api/v1/estimate']['post']
    assert operation['tags'] == ['estimations']
    assert client.post('/estimate', json=payload).status_code == 404


@pytest.mark.parametrize('body', [{}, payload | {'description': 'short'},
    payload | {'description': 'a' * 2001}, payload | {'project_type': 'bad'},
    payload | {'detail_level': 'bad'}, payload | {'output_format': 'bad'},
    payload | {'description': ' ' * 20}])
def test_invalid_input(client, body):
    assert client.post('/api/v1/estimate', json=body).status_code == 422


def test_missing_key(client):
    response = client.post('/api/v1/estimate', json=payload)
    assert response.status_code == 503
    assert 'OPENAI_API_KEY' in response.json()['detail']


def test_success(client, monkeypatch):
    result = {'summary': 'Estimación de proyecto', 'confidence_pct': 80,
              'phases': [{'name': 'Desarrollo', 'duration_weeks': 2, 'cost_eur': 500,
                          'summary': 'Desarrollo del proyecto'}],
              'total_duration_weeks': 2, 'total_cost_eur': 500}
    estimate = AsyncMock(return_value=result)
    monkeypatch.setattr(llm_service, 'structured_estimate', estimate)
    response = client.post('/api/v1/estimate', json=payload)
    assert response.status_code == 200
    assert response.json() == {'result': result, 'prompt_version': 'v1', 'cached': False}
    assert payload['description'] in estimate.call_args.args[0]
    assert estimate.call_args.kwargs['system']


def test_safe_service_error(client, monkeypatch):
    monkeypatch.setattr(llm_service, 'structured_estimate', AsyncMock(
        side_effect=llm_service.EstimationError(502, 'Provider unavailable.')))
    response = client.post('/api/v1/estimate', json=payload)
    assert response.status_code == 502
    assert response.json() == {'detail': 'Provider unavailable.'}
