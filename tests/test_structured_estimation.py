import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.main import app
from app.schemas import EstimationResult
from app.services import llm_service


RESULT = {
    'summary': 'A detailed project estimate', 'confidence_pct': 80,
    'phases': [{'name': 'Build', 'duration_weeks': 2, 'cost_eur': 500,
                'summary': 'Build the project'}],
    'total_duration_weeks': 2, 'total_cost_eur': 500,
}
PAYLOAD = {'description': 'A customer portal for scheduling appointments',
           'project_type': 'web_saas', 'detail_level': 'summary', 'output_format': 'narrative'}


def test_business_rules():
    assert EstimationResult.model_validate(RESULT).total_cost_eur == 500
    with pytest.raises(ValidationError, match='phases sum'):
        EstimationResult.model_validate(RESULT | {'total_cost_eur': 501})
    with pytest.raises(ValidationError, match='confidence_pct'):
        EstimationResult.model_validate(RESULT | {'confidence_pct': 29})
    assert EstimationResult.model_validate(RESULT | {'confidence_pct': 29,
        'summary': 'Out of scope: too little information'}).confidence_pct == 29
    assert EstimationResult.model_validate(RESULT | {'confidence_pct': 30}).confidence_pct == 30


def test_endpoint_returns_structured_result(monkeypatch):
    client_context = AsyncMock()
    create = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=json.dumps(RESULT)))]))
    client_context.__aenter__.return_value = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(llm_service, 'AsyncOpenAI', MagicMock(return_value=client_context))
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, openai_api_key='test')
    try:
        with TestClient(app) as client:
            response = client.post('/api/v1/estimate', json=PAYLOAD)
        assert response.status_code == 200
        assert response.json() == {'result': RESULT, 'prompt_version': 'v1', 'cached': False}
        assert create.call_args.kwargs['response_format'] == {'type': 'json_object'}
    finally:
        app.dependency_overrides.clear()


def test_invalid_model_output_is_safe(monkeypatch):
    context = AsyncMock()
    create = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=json.dumps(RESULT | {'total_cost_eur': 999})))]))
    context.__aenter__.return_value = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(llm_service, 'AsyncOpenAI', MagicMock(return_value=context))
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, openai_api_key='test')
    try:
        with TestClient(app) as client:
            response = client.post('/api/v1/estimate', json=PAYLOAD)
        assert response.status_code == 502
        assert response.json() == {'detail': 'Estimation provider returned an invalid result.'}
    finally:
        app.dependency_overrides.clear()
