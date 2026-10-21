"""Load server configuration and cache it for reuse across requests."""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Resolve constructor values, environment, .env, then defaults in priority order.

    Nonempty environment values override .env values; empty values in either
    source are ignored. A missing API key permits startup, but not estimation.
    """

    model_config = SettingsConfigDict(
        env_file='.env', env_file_encoding='utf-8', env_ignore_empty=True,
        extra='ignore',
    )

    llm_provider: Literal['openai'] = 'openai'
    openai_model: Literal['gpt-4o-mini'] = 'gpt-4o-mini'
    openai_api_key: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    """Reuse settings after first load; restart to pick up configuration changes.

    Explicitly clearing this function's cache also forces a reload on next use.
    """
    return Settings()
