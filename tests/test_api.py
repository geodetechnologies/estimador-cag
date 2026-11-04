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
    estimate = AsyncMock(return_value='Estimación: 20 horas')
    monkeypatch.setattr(llm_service, 'estimate', estimate)
    response = client.post('/api/v1/estimate', json=payload)
    assert response.status_code == 200
    assert response.json() == {'text': 'Estimación: 20 horas', 'prompt_version': 'v1'}
    assert payload['description'] in estimate.call_args.args[0]
    assert estimate.call_args.kwargs['system']


def test_safe_service_error(client, monkeypatch):
    monkeypatch.setattr(llm_service, 'estimate', AsyncMock(
        side_effect=llm_service.EstimationError(502, 'Provider unavailable.')))
    response = client.post('/api/v1/estimate', json=payload)
    assert response.status_code == 502
    assert response.json() == {'detail': 'Provider unavailable.'}
