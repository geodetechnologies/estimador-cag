"""Combine static context with user text and translate provider failures safely."""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from time import perf_counter

import anyio
from openai import AsyncOpenAI, OpenAIError, APITimeoutError, RateLimitError
from pydantic import ValidationError

from app.config import Settings
from app.prompts import SYSTEM_PROMPT
from app.schemas import EstimationResult


# Identity-only markers set exclusively at the actual JSON validation boundary.
_PHASE_COST_RULE = object()
_LOW_CONFIDENCE_RULE = object()


def _business_rule(error: ValidationError) -> object | None:
    """Identify one actual model validator without inspecting its message or values.

    Pydantic's root error retains the original validator's traceback. Its frame
    supplies the already field-validated model, so the existing validators can
    establish whether exactly one rule failed without duplicating either rule.
    Any unfamiliar error shape or validator failure remains unclassified.
    """
    try:
        if type(error) is not ValidationError or error.error_count() != 1:
            return None
        items = error.errors(include_input=False, include_url=False)
        if len(items) != 1 or items[0].get('loc') != () or items[0].get('type') != 'value_error':
            return None
        original = items[0].get('ctx', {}).get('error')
        if type(original) is not ValueError:
            return None
        trace = original.__traceback__
        if trace is None or trace.tb_next is not None:
            return None
        validators = ((EstimationResult.phases_sum_matches_total, _PHASE_COST_RULE),
                      (EstimationResult.low_confidence_requires_out_of_scope_prefix, _LOW_CONFIDENCE_RULE))
        origin = next((marker for validator, marker in validators
                       if trace.tb_frame.f_code is validator.__code__), None)
        model = trace.tb_frame.f_locals.get('self')
        if origin is None or type(model) is not EstimationResult:
            return None
        failures = []
        for validator, marker in validators:
            try:
                validator(model)
            except ValueError:
                failures.append(marker)
        return origin if failures == [origin] else None
    except Exception:
        return None


class EstimationError(Exception):
    """Carry an HTTP status and a client-facing detail without provider diagnostics."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


async def estimate(transcription: str, settings: Settings, *, system: str = SYSTEM_PROMPT,
                   json_output: bool = False,
                   history: list[dict[str, str]] | None = None,
                   metrics: CallMetrics | None = None) -> str:
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

    started = perf_counter() if metrics is not None else None
    try:
        async with AsyncOpenAI(
            api_key=key.get_secret_value(), timeout=30.0, max_retries=0,
        ) as client:
            response = await client.chat.completions.create(
                model=settings.openai_model,
                **({'response_format': {'type': 'json_object'}} if json_output else {}),
                messages=[
                    {'role': 'system', 'content': system},
                    *(history or []),
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
    if metrics is not None:
        model = getattr(response, 'model', None)
        usage = getattr(response, 'usage', None)
        metrics.model = model if isinstance(model, str) and model else None
        metrics.input_tokens = _token_count(getattr(usage, 'prompt_tokens', None))
        metrics.output_tokens = _token_count(getattr(usage, 'completion_tokens', None))
        metrics.elapsed_seconds = perf_counter() - started
    return content.strip()


async def structured_estimate(transcription: str, settings: Settings, *, system: str,
                              history: list[dict[str, str]] | None = None,
                              metrics: CallMetrics | None = None,
                              _validation_origin: object | None = None) -> EstimationResult:
    """Validate provider JSON and enforce business rules before returning it."""
    schema = EstimationResult.model_json_schema()
    instructions = (f'\nReturn a JSON object matching this schema: {json.dumps(schema)}. '
                    'The total_cost_eur must equal the sum of phase costs. '
                    'When confidence_pct is below 30, summary must start with Out of scope:. '
                    'Provide phases before totals. All monetary amounts are in euros.')
    options = {'history': history} if history is not None else {}
    pending = CallMetrics() if metrics is not None else None
    if pending is not None:
        options['metrics'] = pending
    result = await estimate(transcription, settings, system=system + instructions,
                            json_output=True, **options)
    try:
        validated = EstimationResult.model_validate_json(result)
    except ValidationError as validation_error:
        error = EstimationError(502, 'Estimation provider returned an invalid result.')
        if _validation_origin is not None:
            error._validation_origin = _validation_origin
            error._validation_business_rule = _business_rule(validation_error)
        raise error from None
    if metrics is not None:
        metrics.model = pending.model
        metrics.input_tokens = pending.input_tokens
        metrics.output_tokens = pending.output_tokens
        metrics.elapsed_seconds = pending.elapsed_seconds
    return validated


@asynccontextmanager
async def _shielded_exit(manager):
    """Allow async provider cleanup to finish under level cancellation."""
    value = await manager.__aenter__()
    try:
        yield value
    except BaseException:
        with anyio.CancelScope(shield=True):
            if not await manager.__aexit__(*sys.exc_info()):
                raise
    else:
        with anyio.CancelScope(shield=True):
            await manager.__aexit__(None, None, None)


@dataclass
class CallMetrics:
    """Provider usage and full elapsed time, published only after success."""

    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    elapsed_seconds: float | None = None

    @property
    def cost_usd(self) -> float | None:
        """Estimate uncached text-token cost only for explicitly priced models."""
        prices = {'gpt-4o-mini-2024-07-18': (0.15, 0.60)}  # USD per million tokens
        price = prices.get(self.model)
        if price is None or self.input_tokens is None or self.output_tokens is None:
            return None
        return (self.input_tokens * price[0] + self.output_tokens * price[1]) / 1_000_000


def _token_count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


async def stream_estimate(
    messages: list[dict[str, str]], settings: Settings,
    *, metrics: CallMetrics | None = None, system: str = SYSTEM_PROMPT,
) -> AsyncIterator[str]:
    """Yield provider text deltas using the default or supplied system prompt.

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
        async with _shielded_exit(AsyncOpenAI(
            api_key=key.get_secret_value(), timeout=30.0, max_retries=0,
        )) as client:
            stream = await client.chat.completions.create(
                model=settings.openai_model,
                messages=[{'role': 'system', 'content': system}, *messages],
                stream=True,
                **options,
            )
            async with _shielded_exit(stream):
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
