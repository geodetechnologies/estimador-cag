import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.main import app
import app.embedding_pipeline.router as ingest_router


@pytest.fixture
def sample():
    return json.loads(Path('data/budgets_sample.json').read_text(encoding='utf-8'))


@pytest.fixture
def provider(monkeypatch):
    calls = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls.append(('init', kwargs))
            self.embeddings = self

        def create(self, **kwargs):
            calls.append(('create', kwargs))
            return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[float(i)] * 1536)
                                         for i, _ in enumerate(kwargs['input'])])

    monkeypatch.setattr(ingest_router, 'OpenAI', FakeOpenAI, raising=False)
    monkeypatch.setattr(ingest_router, 'get_settings', lambda: SimpleNamespace(openai_api_key=SecretStr('synthetic-key')), raising=False)
    return calls


def test_sample_ingest(provider, sample):
    response = TestClient(app).post('/embeddings/ingest', json={'budgets': sample})
    assert response.status_code == 200
    body = response.json()
    ids = [f"{b['budget_id']}::{c['component_id']}" for b in sample for c in b['components']]
    assert [c['chunk_id'] for c in body['chunks']] == ids
    assert len(ids) == 52
    assert all(len(c['embedding']) == 1536 for c in body['chunks'])
    tokens = sum(c['token_count'] for c in body['chunks'])
    assert body['stats'] == {'total_budgets': 15, 'total_chunks': 52, 'total_tokens': tokens, 'estimated_cost_usd': tokens * 0.02 / 1_000_000}
    assert provider[0] == ('init', {'api_key': 'synthetic-key', 'max_retries': 0})
    assert len(provider) == 2
    assert provider[1][1]['input'] == [c['text'] for c in body['chunks']]


def test_invalid_input_never_calls_provider(provider):
    response = TestClient(app).post('/embeddings/ingest', json={'budgets': [42]})
    assert response.status_code == 422
    assert provider == []


@pytest.mark.parametrize('empty_components', [False, True])
def test_empty_needs_no_configuration_or_client(monkeypatch, sample, empty_components):
    def forbidden():
        raise AssertionError('configuration accessed')
    monkeypatch.setattr(ingest_router, 'get_settings', forbidden, raising=False)
    budgets = [{**sample[0], 'components': []}] if empty_components else []
    response = TestClient(app).post('/embeddings/ingest', json={'budgets': budgets})
    assert response.status_code == 200
    assert response.json() == {'chunks': [], 'stats': {'total_budgets': len(budgets), 'total_chunks': 0, 'total_tokens': 0, 'estimated_cost_usd': 0.0}}


@pytest.mark.parametrize('stage', ['settings', 'constructor', 'provider'])
def test_errors_are_generic_and_logs_do_not_leak(monkeypatch, sample, capsys, stage):
    secret = 'PRIVATE-provider-message-and-key'
    def fail(*args, **kwargs):
        raise RuntimeError(secret)
    monkeypatch.setattr(ingest_router, 'get_settings', fail if stage == 'settings' else lambda: SimpleNamespace(openai_api_key=SecretStr('synthetic')), raising=False)
    class Client:
        def __init__(self, **kwargs):
            if stage == 'constructor':
                fail()
            self.embeddings = self
        create = fail
    monkeypatch.setattr(ingest_router, 'OpenAI', Client, raising=False)
    response = TestClient(app).post('/embeddings/ingest', json={'budgets': sample[:1]})
    assert response.status_code == 500
    assert response.json() == {'detail': 'Embedding generation failed'}
    captured = capsys.readouterr()
    assert secret not in response.text + captured.out + captured.err


def test_openapi_and_health():
    client = TestClient(app)
    assert 'post' in client.get('/openapi.json').json()['paths']['/embeddings/ingest']
    assert client.get('/docs').status_code == 200
    assert client.get('/health').json() == {'status': 'ok'}
