"""Combine static context with user text and translate provider failures safely."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from time import perf_counter

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


@dataclass
class CallMetrics:
    """Provider usage and full elapsed time, published only after success."""

    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    elapsed_seconds: float | None = None


async def stream_estimate(
    messages: list[dict[str, str]], settings: Settings,
    *, metrics: CallMetrics | None = None,
) -> AsyncIterator[str]:
    """Yield provider text deltas using the API's CAG prompt and configuration.

    The chat supplies completed user/assistant turns and the new transcript.
    Client and stream close even on provider failure. Partial output must not
    be saved as a completed answer by the caller. Optional metrics request
    provider usage and measure the entire stream, not time to first token.
    """
    key = settings.openai_api_key
    if key is None or not key.get_secret_value().strip():
        raise EstimationError(503, 'Set OPENAI_API_KEY on the server to enable estimation.')

    has_text = False
    started = perf_counter() if metrics is not None else None
    model = settings.openai_model
    input_tokens = output_tokens = None
    options = {'stream_options': {'include_usage': True}} if metrics is not None else {}
    try:
        async with AsyncOpenAI(
            api_key=key.get_secret_value(), timeout=30.0, max_retries=0,
        ) as client:
            stream = await client.chat.completions.create(
                model=settings.openai_model,
                messages=[{'role': 'system', 'content': SYSTEM_PROMPT}, *messages],
                stream=True,
                **options,
            )
            async with stream:
                async for chunk in stream:
                    if metrics is not None:
                        model = getattr(chunk, 'model', None) or model
                        usage = getattr(chunk, 'usage', None)
                        if usage is not None:
                            input_tokens = usage.prompt_tokens
                            output_tokens = usage.completion_tokens
                    content = chunk.choices[0].delta.content if chunk.choices else None
                    if content:
                        has_text = has_text or bool(content.strip())
                        yield content
    except APITimeoutError:
        raise EstimationError(504, 'Estimation provider timed out. Try again later.') from None
    except RateLimitError:
        raise EstimationError(503, 'Estimation provider is busy. Try again later.') from None
    except OpenAIError:
        raise EstimationError(502, 'Estimation provider unavailable. Try again later.') from None

    if not has_text:
        raise EstimationError(502, 'Estimation provider returned an empty result.')
    if metrics is not None:
        metrics.model = model
        metrics.input_tokens = input_tokens
        metrics.output_tokens = output_tokens
        metrics.elapsed_seconds = perf_counter() - started
