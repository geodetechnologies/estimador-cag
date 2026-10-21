import pytest
from pydantic import ValidationError

from app.config import Settings


def test_app_env_defaults_and_blank_values(tmp_path, monkeypatch):
    monkeypatch.delenv('APP_ENV', raising=False)
    assert Settings(_env_file=None).app_env == 'development'
    dotenv = tmp_path / 'settings.txt'
    dotenv.write_text('APP_ENV=\n')
    monkeypatch.setenv('APP_ENV', '')
    assert Settings(_env_file=dotenv).app_env == 'development'


def test_app_env_dotenv_and_environment(tmp_path, monkeypatch):
    monkeypatch.delenv('APP_ENV', raising=False)
    dotenv = tmp_path / 'settings.txt'
    dotenv.write_text('APP_ENV=staging\n')
    assert Settings(_env_file=dotenv).app_env == 'staging'
    monkeypatch.setenv('APP_ENV', 'production')
    assert Settings(_env_file=dotenv).app_env == 'production'
    monkeypatch.setenv('APP_ENV', '')
    assert Settings(_env_file=dotenv).app_env == 'staging'


def test_defaults(monkeypatch):
    for name in ('OPENAI_API_KEY', 'OPENAI_MODEL', 'LLM_PROVIDER'):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None)
    assert settings.llm_provider == 'openai'
    assert settings.openai_model == 'gpt-4o-mini'
    assert settings.openai_api_key is None


def test_dotenv_and_environment(tmp_path, monkeypatch):
    for name in ('OPENAI_API_KEY', 'OPENAI_MODEL', 'LLM_PROVIDER'):
        monkeypatch.delenv(name, raising=False)
    dotenv = tmp_path / 'settings.txt'
    dotenv.write_text('OPENAI_API_KEY=test-only\nLLM_PROVIDER=openai\nOPENAI_MODEL=gpt-4o-mini\n')
    settings = Settings(_env_file=dotenv)
    assert settings.openai_api_key.get_secret_value() == 'test-only'
    assert 'test-only' not in repr(settings)
    monkeypatch.setenv('OPENAI_API_KEY', 'environment-only')
    assert Settings(_env_file=dotenv).openai_api_key.get_secret_value() == 'environment-only'


def test_blank_values_use_defaults(tmp_path):
    dotenv = tmp_path / 'settings.txt'
    dotenv.write_text('OPENAI_API_KEY=\nLLM_PROVIDER=\nOPENAI_MODEL=\n')
    settings = Settings(_env_file=dotenv)
    assert settings.llm_provider == 'openai'
    assert settings.openai_model == 'gpt-4o-mini'


@pytest.mark.parametrize('kwargs', [{'llm_provider': 'other'}, {'openai_model': 'gpt-4o'}])
def test_only_supported_provider_and_model(kwargs):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **kwargs)
