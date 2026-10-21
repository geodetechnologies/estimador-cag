from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app
from app.services import llm_service


@pytest.fixture
def client():
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None)
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_health_and_docs(client):
    assert client.get('/health').json() == {'status': 'ok'}
    assert client.get('/docs').status_code == 200


@pytest.mark.parametrize('payload', [{}, {'transcription': ''},
    {'transcription': '  \n '}, {'transcription': 'a' * 20001},
    {'transcription': None}, {'transcription': 42}])
def test_invalid_input(client, payload):
    assert client.post('/api/v1/estimate', json=payload).status_code == 422


def test_missing_key(client):
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, openai_api_key=None)
    response = client.post('/api/v1/estimate', json={'transcription': 'A portal'})
    assert response.status_code == 503
    assert 'OPENAI_API_KEY' in response.json()['detail']


def test_success(client, monkeypatch):
    estimate = AsyncMock(return_value='Estimación: 20 horas')
    monkeypatch.setattr(llm_service, 'estimate', estimate)
    response = client.post('/api/v1/estimate', json={'transcription': ' A portal '})
    assert response.status_code == 200
    assert response.json() == {'estimation': 'Estimación: 20 horas',
                               'model': 'gpt-4o-mini', 'provider': 'openai'}
    assert estimate.call_args.args[0] == 'A portal'


def test_safe_service_error(client, monkeypatch):
    monkeypatch.setattr(llm_service, 'estimate', AsyncMock(
        side_effect=llm_service.EstimationError(502, 'Provider unavailable.')))
    response = client.post('/api/v1/estimate', json={'transcription': 'A portal'})
    assert response.status_code == 502
    assert response.json() == {'detail': 'Provider unavailable.'}
