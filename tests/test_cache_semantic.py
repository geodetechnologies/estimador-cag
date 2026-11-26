import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from app.cache.semantic import SemanticCache, semantic_cache_key
from app.schemas import EstimationRequest, EstimationResult


@pytest.fixture
def estimation_request():
    return EstimationRequest(description='Build a secure planning portal for a team.',
                             project_type='web_saas', detail_level='medium', output_format='narrative')


@pytest.fixture
def result():
    return EstimationResult.model_validate({
        'summary': 'A reasonable project estimate.', 'confidence_pct': 80,
        'phases': [{'name': 'Build', 'duration_weeks': 2, 'cost_eur': 1000,
                    'summary': 'Implement the planning portal.'}],
        'total_duration_weeks': 2, 'total_cost_eur': 1000,
    })


ARGS = dict(bucket='tenant-a', prompt_version='v1', model='model', embedding_model='embed-v1')


def run(coro):
    return asyncio.run(coro)


class Backend:
    def __init__(self):
        self.data = {}
        self.calls = []

    async def get(self, key):
        self.calls.append(('get', key))
        return self.data.get(key)

    async def setex(self, key, ttl, value):
        self.calls.append(('setex', key, ttl))
        self.data[key] = value


class Embedder:
    def __init__(self, vector=(1.0, 0.0)):
        self.vector = vector
        self.embed = AsyncMock(side_effect=lambda text: self.vector)


def test_disabled_and_missing_dependencies_never_call_backend_or_embedder(estimation_request, result):
    request = estimation_request
    backend = Backend()
    embedder = Embedder()
    for cache in (SemanticCache(backend, embedder), SemanticCache(backend, None, enabled=True),
                  SemanticCache(None, embedder, enabled=True)):
        assert run(cache.get(request, **ARGS)) is None
        assert run(cache.set(request, result, **ARGS)) is None
    assert backend.calls == []
    embedder.embed.assert_not_awaited()


def test_bucket_key_is_private_and_isolates_identity_fields(estimation_request):
    request = estimation_request
    key = semantic_cache_key(request, **ARGS)
    assert key.startswith('estimation:semantic:v1:')
    assert len(key.rsplit(':', 1)[-1]) == 64
    int(key.rsplit(':', 1)[-1], 16)
    assert request.description not in key and ARGS['bucket'] not in key
    assert key == semantic_cache_key(request.model_copy(update={'description': 'A different description'}), **ARGS)
    for field, value in [('bucket', 'tenant-b'), ('prompt_version', 'v2'),
                         ('model', 'other'), ('embedding_model', 'embed-v2')]:
        assert key != semantic_cache_key(request, **{**ARGS, field: value})
    for field, value in [('project_type', 'mobile_app'), ('detail_level', 'summary'),
                         ('output_format', 'line_items')]:
        changed = EstimationRequest.model_validate({**request.model_dump(), field: value})
        assert key != semantic_cache_key(changed, **ARGS)


def test_similarity_threshold_ttl_and_single_entry_replacement(estimation_request, result):
    request = estimation_request
    backend, embedder = Backend(), Embedder()
    cache = SemanticCache(backend, embedder, enabled=True, threshold=0.8, ttl_seconds=42)
    assert run(cache.get(request, **ARGS)) is None
    assert run(cache.set(request, result, **ARGS)) is None
    assert backend.calls[-1][0::2] == ('setex', 42)
    assert embedder.embed.await_args.args == (request.description,)
    similar = request.model_copy(update={'description': 'A similar but different project.'})
    embedder.vector = (0.8, 0.6)
    assert run(cache.get(similar, **ARGS)) == result
    embedder.vector = (0.79, 0.613106)
    assert run(cache.get(similar, **ARGS)) is None
    embedder.vector = (-1.0, 0.0)
    assert run(cache.get(similar, **ARGS)) is None
    assert run(cache.set(similar, result, **ARGS)) is None
    assert len(backend.data) == 1
    assert run(cache.get(request, **ARGS)) == result


def test_bucket_isolation_on_read(estimation_request, result):
    request = estimation_request
    backend, embedder = Backend(), Embedder()
    cache = SemanticCache(backend, embedder, enabled=True)
    run(cache.set(request, result, **ARGS))
    for field, value in [('bucket', 'tenant-b'), ('prompt_version', 'v2'),
                         ('model', 'other'), ('embedding_model', 'embed-v2')]:
        assert run(cache.get(request, **{**ARGS, field: value})) is None
    changed = EstimationRequest.model_validate({**request.model_dump(), 'output_format': 'line_items'})
    assert run(cache.get(changed, **ARGS)) is None
    assert run(cache.get(request, **ARGS)) == result


@pytest.mark.parametrize('bad', [[], [0, 0], [True, 1], ['1', 0], [float('nan'), 1],
                                 [float('inf'), 1], [1.7e308, 1.7e308], [1] * 4097])
def test_invalid_embedding_is_never_written_or_matched(estimation_request, result, bad):
    request = estimation_request
    backend, embedder = Backend(), Embedder(bad)
    cache = SemanticCache(backend, embedder, enabled=True)
    assert run(cache.set(request, result, **ARGS)) is None
    assert backend.data == {} and backend.calls == []
    embedder.vector = (1.0, 0.0)
    run(cache.set(request, result, **ARGS))
    before = dict(backend.data)
    embedder.vector = bad
    assert run(cache.get(request, **ARGS)) is None
    assert backend.data == before


def test_untrusted_payload_and_mismatched_dimensions_are_misses(estimation_request, result):
    request = estimation_request
    backend, embedder = Backend(), Embedder()
    cache = SemanticCache(backend, embedder, enabled=True)
    run(cache.set(request, result, **ARGS))
    key = semantic_cache_key(request, **ARGS)
    valid = json.loads(backend.data[key])
    for payload in ('', '{', 'null', '[]', '{}', b'\xff',
                    json.dumps({**valid, 'key': 'wrong'}),
                    json.dumps({**valid, 'vector': [0, 0]}),
                    json.dumps({**valid, 'vector': [float('nan'), 0]}),
                    json.dumps({**valid, 'vector': [True, 1]}),
                    json.dumps({**valid, 'vector': [1, 0, 0]}),
                    json.dumps({**valid, 'result': {}}),
                    json.dumps({**valid, 'result': {**valid['result'], 'total_cost_eur': 0}}),
                    json.dumps({**valid, 'extra': 'untrusted'})):
        backend.data[key] = payload
        assert run(cache.get(request, **ARGS)) is None
        assert backend.data[key] == payload
    backend.data[key] = json.dumps(valid).encode()
    assert run(cache.get(request, **ARGS)) == result


def test_invalid_results_and_failures_are_silent_noops(estimation_request, result, caplog):
    request = estimation_request
    backend, embedder = Backend(), Embedder()
    cache = SemanticCache(backend, embedder, enabled=True)
    invalid = result.model_copy(update={'total_cost_eur': 0})
    assert run(cache.set(request, invalid, **ARGS)) is None
    assert backend.calls == [] and backend.data == {}
    embedder.embed.assert_not_awaited()
    failing = AsyncMock(side_effect=RuntimeError('private error'))
    embedder.embed = failing
    assert run(cache.set(request, result, **ARGS)) is None
    assert backend.data == {} and backend.calls == []
    embedder.embed = AsyncMock(return_value=(1, 0))
    run(cache.set(request, result, **ARGS))
    before = dict(backend.data)
    embedder.embed = failing
    assert run(cache.get(request, **ARGS)) is None
    assert backend.data == before
    embedder.embed = AsyncMock(return_value=(1, 0))
    broken = type('Broken', (), {'get': failing, 'setex': failing})()
    cache = SemanticCache(broken, embedder, enabled=True)
    assert run(cache.get(request, **ARGS)) is None
    assert run(cache.set(request, result, **ARGS)) is None
    assert 'private error' not in caplog.text
    assert request.description not in caplog.text and result.summary not in caplog.text


@pytest.mark.parametrize('kwargs', [dict(ttl_seconds=0), dict(ttl_seconds=True),
                                   dict(ttl_seconds=1.5), dict(threshold=-0.1),
                                   dict(threshold=1.1), dict(threshold=True),
                                   dict(threshold=float('nan'))])
def test_invalid_configuration(kwargs):
    with pytest.raises((TypeError, ValueError)):
        SemanticCache(**kwargs)


def test_invalid_identity_does_not_access_dependencies(estimation_request, result):
    backend, embedder = Backend(), Embedder()
    cache = SemanticCache(backend, embedder, enabled=True)
    for field in ARGS:
        args = {**ARGS, field: '  '}
        assert run(cache.get(estimation_request, **args)) is None
        assert run(cache.set(estimation_request, result, **args)) is None
    assert backend.calls == [] and backend.data == {}
    embedder.embed.assert_not_awaited()


def test_cancellation_propagates(estimation_request, result):
    request = estimation_request
    backend, embedder = Backend(), Embedder()
    embedder.embed.side_effect = asyncio.CancelledError()
    cache = SemanticCache(backend, embedder, enabled=True)
    with pytest.raises(asyncio.CancelledError):
        run(cache.set(request, result, **ARGS))
    assert backend.data == {}
