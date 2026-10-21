"""Combine static context with user text and translate provider failures safely."""

from openai import AsyncOpenAI, OpenAIError, APITimeoutError, RateLimitError

from app.config import Settings
from app.prompts import SYSTEM_PROMPT


class EstimationError(Exception):
    """Carry an HTTP status and a client-facing detail without provider diagnostics."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


async def estimate(transcription: str, settings: Settings) -> str:
    """Request an estimate with a per-call async client that closes on exit.

    Static system context and untrusted user text remain separate messages;
    this does not guarantee protection against prompt injection. The SDK uses
    a 30-second timeout and no automatic retries.

    Missing credentials yield 503; SDK timeouts yield 504, rate limits 503,
    and other SDK failures 502, with fixed details hiding provider diagnostics.
    Return stripped text, or raise a 502 error for missing or blank output.
    """
    key = settings.openai_api_key
    if key is None or not key.get_secret_value().strip():
        raise EstimationError(503, 'Set OPENAI_API_KEY on the server to enable estimation.')

    try:
        async with AsyncOpenAI(
            api_key=key.get_secret_value(), timeout=30.0, max_retries=0,
        ) as client:
            response = await client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': transcription},
                ],
            )
    except APITimeoutError:
        raise EstimationError(504, 'Estimation provider timed out. Try again later.') from None
    except RateLimitError:
        raise EstimationError(503, 'Estimation provider is busy. Try again later.') from None
    except OpenAIError:
        raise EstimationError(502, 'Estimation provider unavailable. Try again later.') from None

    content = response.choices[0].message.content if response.choices else None
    if not content or not content.strip():
        raise EstimationError(502, 'Estimation provider returned an empty result.')
    return content.strip()
