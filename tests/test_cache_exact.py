import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from app.schemas import EstimationRequest, EstimationResult


@pytest.fixture
def cache_api():
    from app.cache.exact import ExactCache, exact_cache_key

    return ExactCache, exact_cache_key


@pytest.fixture
def estimation_request():
    return EstimationRequest(
        description='Build a secure planning portal for a team.',
        project_type='web_saas', detail_level='medium', output_format='narrative',
    )


@pytest.fixture
def result():
    return EstimationResult.model_validate({
        'summary': 'A reasonable project estimate.',
        'confidence_pct': 80,
        'phases': [{'name': 'Build', 'duration_weeks': 2, 'cost_eur': 1000,
                    'summary': 'Implement the planning portal.'}],
        'total_duration_weeks': 2,
        'total_cost_eur': 1000,
    })


def run(coro):
    return asyncio.run(coro)


class Backend:
    def __init__(self):
        self.data = {}
        self.calls = []

    async def get(self, key):
        self.calls.append(('get', key))
        return self.data.get(key)

    async def setex(self, key, ttl_seconds, value):
        self.calls.append(('setex', key, ttl_seconds, value))
        self.data[key] = value
        return True


def test_default_disabled_is_noop_even_with_explosive_backend(cache_api, estimation_request, result):
    ExactCache, _ = cache_api
    backend = type('Explosive', (), {
        'get': AsyncMock(side_effect=AssertionError('backend accessed')),
        'setex': AsyncMock(side_effect=AssertionError('backend accessed')),
    })()
    cache = ExactCache(backend)
    assert run(cache.get(estimation_request, prompt_version='v1', model='model')) is None
    assert run(cache.set(estimation_request, result, prompt_version='v1', model='model')) is None
    backend.get.assert_not_awaited()
    backend.setex.assert_not_awaited()


def test_key_is_stable_private_and_separates_every_typed_input(cache_api, estimation_request):
    _, key_for = cache_api
    key = key_for(estimation_request, prompt_version='v1', model='model')
    assert key == key_for(estimation_request.model_copy(), prompt_version='v1', model='model')
    assert estimation_request.description not in key
    assert key.startswith('estimation:exact:v1:')
    assert len(key.removeprefix('estimation:exact:v1:')) == 64
    int(key.rsplit(':', 1)[-1], 16)
    for field, value in {
        'description': 'Build a different secure planning portal.',
        'project_type': 'mobile_app', 'detail_level': 'summary', 'output_format': 'line_items',
    }.items():
        changed = EstimationRequest.model_validate({**estimation_request.model_dump(), field: value})
        assert key != key_for(changed, prompt_version='v1', model='model')
    assert key != key_for(estimation_request, prompt_version='v2', model='model')
    assert key != key_for(estimation_request, prompt_version='v1', model='other')
    assert key_for(estimation_request, prompt_version='ab', model='c') != key_for(
        estimation_request, prompt_version='a', model='bc',
    )


def test_write_and_read_validated_result_with_default_ttl(cache_api, estimation_request, result):
    ExactCache, key_for = cache_api
    backend = Backend()
    cache = ExactCache(backend, enabled=True)
    assert run(cache.get(estimation_request, prompt_version='v1', model='model')) is None
    assert run(cache.set(estimation_request, result, prompt_version='v1', model='model')) is None
    key = key_for(estimation_request, prompt_version='v1', model='model')
    _, written_key, ttl, payload = backend.calls[-1]
    assert written_key == key
    assert ttl == 86400
    assert isinstance(payload, str)
    assert run(cache.get(estimation_request, prompt_version='v1', model='model')) == result
    backend.data[key] = payload.encode()
    assert run(cache.get(estimation_request, prompt_version='v1', model='model')) == result
    assert run(cache.get(estimation_request, prompt_version='v2', model='model')) is None


def test_bad_payloads_and_stale_identity_are_misses(cache_api, estimation_request, result):
    ExactCache, key_for = cache_api
    backend = Backend()
    cache = ExactCache(backend, enabled=True)
    run(cache.set(estimation_request, result, prompt_version='v1', model='model'))
    key = key_for(estimation_request, prompt_version='v1', model='model')
    valid = json.loads(backend.data[key])
    bad_result = dict(valid['result'], total_cost_eur=0)
    bad_values = [
        '', '{', 'null', '[]', '{}', b'\xff',
        json.dumps({**valid, 'key': 'stale'}),
        json.dumps({**valid, 'result': bad_result}),
        json.dumps({**valid, 'result': {}}),
    ]
    for value in bad_values:
        backend.data[key] = value
        assert run(cache.get(estimation_request, prompt_version='v1', model='model')) is None


def test_only_valid_results_are_written(cache_api, estimation_request, result):
    ExactCache, _ = cache_api
    backend = Backend()
    cache = ExactCache(backend, enabled=True)
    invalid = result.model_copy(update={'total_cost_eur': 0})
    assert run(cache.set(estimation_request, invalid, prompt_version='v1', model='model')) is None
    assert backend.calls == []
    assert backend.data == {}
    assert run(cache.get(estimation_request, prompt_version='v1', model='model')) is None


def test_backend_failures_and_missing_backend_are_best_effort(cache_api, estimation_request, result, caplog):
    ExactCache, _ = cache_api
    backend = type('Broken', (), {
        'get': AsyncMock(side_effect=RuntimeError('private backend error')),
        'setex': AsyncMock(side_effect=RuntimeError('private backend error')),
    })()
    for cache in (ExactCache(enabled=True), ExactCache(backend, enabled=True)):
        assert run(cache.get(estimation_request, prompt_version='v1', model='model')) is None
        assert run(cache.set(estimation_request, result, prompt_version='v1', model='model')) is None
    backend.get.assert_awaited_once()
    backend.setex.assert_awaited_once()
    assert 'private backend error' not in caplog.text
    assert estimation_request.description not in caplog.text
    assert result.summary not in caplog.text


def test_configurable_ttl_and_validation(cache_api, estimation_request, result):
    ExactCache, _ = cache_api
    backend = Backend()
    cache = ExactCache(backend, enabled=True, ttl_seconds=42)
    run(cache.set(estimation_request, result, prompt_version='v1', model='model'))
    assert backend.calls[-1][2] == 42
    for invalid in (0, -1, True, 1.5, '42'):
        with pytest.raises((ValueError, TypeError)):
            ExactCache(ttl_seconds=invalid)
