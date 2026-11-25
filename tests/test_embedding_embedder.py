"""Behavioral contract for the OpenAI embedding adapter; no network calls."""

from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from openai import RateLimitError

from app.embedding_pipeline.embedder import OpenAIEmbedder, estimated_cost_usd
from app.embedding_pipeline.schemas import Chunk


def chunk(i):
    return Chunk(chunk_id=str(i), text=f"private text {i}", metadata={"id": i}, token_count=i + 1)


def vector(i):
    return [float(i)] * 1536


def response(count):
    return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=vector(i)) for i in reversed(range(count))])


def client_with(effect):
    return SimpleNamespace(embeddings=SimpleNamespace(create=Mock(side_effect=effect)))


def test_batches_preserve_input_order_and_data():
    client = client_with(lambda **kw: response(len(kw['input'])))
    chunks = [chunk(i) for i in range(101)]
    before = [c.model_dump() for c in chunks]
    result = OpenAIEmbedder(client=client).embed_many(chunks)
    assert [call.kwargs for call in client.embeddings.create.call_args_list] == [
        {"model": "text-embedding-3-small", "input": [c.text for c in chunks[:100]]},
        {"model": "text-embedding-3-small", "input": [chunks[100].text]},
    ]
    assert [r.embedding for r in result] == [vector(i) for i in range(100)] + [vector(0)]
    assert [r.model_dump(exclude={'embedding'}) for r in result] == before
    assert [c.model_dump() for c in chunks] == before


def test_one_and_empty():
    client = client_with(lambda **kw: response(1))
    embedder = OpenAIEmbedder(client=client)
    assert embedder.embed_many([]) == []
    client.embeddings.create.assert_not_called()
    assert embedder.embed_one('hello') == vector(0)
    client.embeddings.create.assert_called_once_with(model='text-embedding-3-small', input=['hello'])


def rate_error():
    return RateLimitError('private error', response=httpx.Response(429, request=httpx.Request('POST', 'https://example.test')), body=None)


def test_rate_limit_retries_and_exhaustion():
    sleep = Mock()
    client = client_with([rate_error(), rate_error(), rate_error(), response(1)])
    assert OpenAIEmbedder(client=client, sleep=sleep).embed_one('secret') == vector(0)
    assert sleep.call_args_list == [((1,),), ((2,),), ((4,),)]
    assert client.embeddings.create.call_count == 4
    client = client_with([rate_error()] * 4)
    sleep = Mock()
    with pytest.raises(RateLimitError):
        OpenAIEmbedder(client=client, sleep=sleep).embed_one('secret')
    assert client.embeddings.create.call_count == 4
    assert sleep.call_count == 3


def test_other_errors_propagate_without_retry(monkeypatch):
    logger = Mock()
    monkeypatch.setattr('app.embedding_pipeline.embedder.logger', logger)
    error = ValueError('private error')
    client = client_with(error)
    sleep = Mock()
    with pytest.raises(ValueError) as exc:
        OpenAIEmbedder(client=client, sleep=sleep).embed_one('secret')
    assert exc.value is error
    assert client.embeddings.create.call_count == 1
    sleep.assert_not_called()
    logger.info.assert_not_called()


def test_default_client_is_lazy_and_disables_sdk_retries(monkeypatch):
    client = client_with([response(1)])
    constructor = Mock(return_value=client)
    monkeypatch.setattr('app.embedding_pipeline.embedder.OpenAI', constructor)
    embedder = OpenAIEmbedder()
    assert embedder.embed_many([]) == []
    constructor.assert_not_called()
    assert embedder.embed_one('hello') == vector(0)
    constructor.assert_called_once_with(max_retries=0)


@pytest.mark.parametrize('data', [
    [],
    [SimpleNamespace(index=1, embedding=vector(0))],
    [SimpleNamespace(index=0, embedding=[1.0])],
    [SimpleNamespace(index=0, embedding=[float('nan')] * 1536)],
    [SimpleNamespace(index=0, embedding=vector(0)), SimpleNamespace(index=0, embedding=vector(0))],
    [SimpleNamespace(index=0, embedding=['bad'] * 1536)],
])
def test_malformed_provider_data_rejected(data):
    client = client_with([SimpleNamespace(data=data)])
    with pytest.raises(ValueError):
        OpenAIEmbedder(client=client).embed_one('secret')


def test_content_free_batch_log_and_cost(monkeypatch):
    logger = Mock()
    monkeypatch.setattr('app.embedding_pipeline.embedder.logger', logger, raising=False)
    client = client_with([response(1)])
    OpenAIEmbedder(client=client).embed_many([chunk(2)])
    assert logger.info.call_count == 1
    args, fields = logger.info.call_args
    assert set(fields) == {'batch_count', 'estimated_input_tokens', 'elapsed_seconds', 'model'}
    assert fields['batch_count'] == 1 and fields['estimated_input_tokens'] == 3
    assert fields['model'] == 'text-embedding-3-small' and fields['elapsed_seconds'] >= 0
    assert 'private' not in repr((args, fields))
    assert estimated_cost_usd(1_000_000) == 0.02
