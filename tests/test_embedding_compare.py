import importlib.util
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_compare():
    spec = importlib.util.spec_from_file_location('compare', ROOT / 'scripts/compare.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cosine_similarity():
    compare = load_compare()
    assert compare.cosine_similarity([1, 0], [0, 1]) == 0
    assert compare.cosine_similarity([1, 1], [1, 1]) == pytest.approx(1)
    assert compare.cosine_similarity([1], [-1]) == -1


@pytest.mark.parametrize('a,b', [([], []), ([1], [1, 2]), ([0], [1]), ([1], [0]), ([math.nan], [1]), ([math.inf], [1])])
def test_cosine_rejects_invalid_vectors(a, b):
    with pytest.raises(ValueError):
        load_compare().cosine_similarity(a, b)


def test_main_embeds_twice_and_prints(monkeypatch, capsys):
    compare = load_compare()
    calls = []
    monkeypatch.setattr(compare, 'get_settings', lambda: SimpleNamespace(openai_api_key=SimpleNamespace(get_secret_value=lambda: 'fake')))
    monkeypatch.setattr(compare, 'OpenAI', lambda **kw: calls.append(kw) or object())
    class Embedder:
        def __init__(self, client): calls.append(client)
        def embed_one(self, text):
            calls.append(text)
            return [1, 0] if text == 'first' else [0, 1]
    monkeypatch.setattr(compare, 'OpenAIEmbedder', Embedder)
    assert compare.main(['--text-a', 'first', '--text-b', 'second']) == 0
    assert calls[0] == {'api_key': 'fake', 'max_retries': 0}
    assert calls[2:] == ['first', 'second']
    assert capsys.readouterr() == ('Text A: first\nText B: second\nCosine similarity: 0.0000\n', '')


def test_failure_is_generic(monkeypatch, capsys):
    compare = load_compare()
    def fail(): raise RuntimeError('secret raw provider error')
    monkeypatch.setattr(compare, 'get_settings', fail)
    assert compare.main(['--text-a', 'a', '--text-b', 'b']) == 1
    assert capsys.readouterr() == ('', 'Embedding comparison failed\n')


def test_provider_failure_is_generic_and_stops_after_first_request(monkeypatch, capsys):
    compare = load_compare()
    calls = []
    client = object()
    key = 'fake-private-key'
    text = 'private-text-marker'
    monkeypatch.setattr(compare, 'get_settings', lambda: SimpleNamespace(
        openai_api_key=SimpleNamespace(get_secret_value=lambda: key)))
    monkeypatch.setattr(compare, 'OpenAI', lambda **kwargs: calls.append(kwargs) or client)

    def fail_embedding(self, value):
        calls.append((self._client, value))
        raise RuntimeError('private-provider-error ' + key + ' ' + value)

    monkeypatch.setattr(compare.OpenAIEmbedder, 'embed_one', fail_embedding)
    assert compare.main(['--text-a', text, '--text-b', 'second']) == 1
    assert calls == [{'api_key': key, 'max_retries': 0}, (client, text)]
    output = capsys.readouterr()
    assert output.out == ''
    assert output.err == 'Embedding comparison failed\n'
    assert all(marker not in output.out + output.err for marker in
               ('private-provider-error', key, text))


def test_argument_validation_before_settings(monkeypatch, capsys):
    compare = load_compare()
    monkeypatch.setattr(compare, 'get_settings', lambda: pytest.fail('settings accessed'))
    with pytest.raises(SystemExit) as exc:
        compare.main(['--text-a', 'a'])
    assert exc.value.code == 2
    output = capsys.readouterr()
    assert output.out == '' and '--text-b' in output.err


def test_help_without_configuration():
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/compare.py'), '--help'], cwd='/tmp', capture_output=True, text=True)
    assert result.returncode == 0
    assert '--text-a' in result.stdout and '--text-b' in result.stdout
    assert 'paid' in result.stdout and 'retries' in result.stdout
    assert result.stderr == ''
