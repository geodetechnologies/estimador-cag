"""Orchestrate an atomic conversational estimation turn."""

import json
import logging

from pydantic import ValidationError

from app.config import Settings
from app.memory import anchors, summarizer
from app.memory.tier import resolve_tier
from app.prompts.loader import render_metadata_prompt, render_session_estimation_system
from app.services import llm_service
from app.sessions import ProjectMetadata, Session


logger = logging.getLogger(__name__)

# Opaque markers carry only the validation stage, never provider data.
_INVALID_ESTIMATE_STAGE = object()
_INVALID_METADATA_STAGE = object()
_INVALID_RESULT_DETAIL = 'Estimation provider returned an invalid result.'


def _invalid_metadata_error() -> llm_service.EstimationError:
    error = llm_service.EstimationError(502, _INVALID_RESULT_DETAIL)
    error._validation_stage = _INVALID_METADATA_STAGE
    return error


def _log_metrics(operation: str, metrics: llm_service.CallMetrics) -> None:
    # Even a syntactically valid provider model can contain confidential text.
    model = metrics.model if metrics.model in {'gpt-4o-mini', 'gpt-4o-mini-2024-07-18'} else None
    try:
        logger.info('provider_usage operation=%s model=%s input_tokens=%s output_tokens=%s '
                    'cost_usd=%s elapsed_seconds=%s', operation, model, metrics.input_tokens,
                    metrics.output_tokens, metrics.cost_usd, metrics.elapsed_seconds)
    except Exception:
        # Telemetry is best effort and must not invalidate an already committed turn.
        pass


async def estimate_conversational(session: Session, user: str, settings: Settings, *,
                                  observation: dict | None = None,
                                  _validation_origin: object | None = None) -> dict:
    """Prepare memory and publish the whole turn only after all calls succeed."""
    async with session.lock:
        history = session.history.to_messages_list('')[1:]
        previous_turn_limit = session.history.max_turns - 1
        evicted = history[:2] if len(history) // 2 > previous_turn_limit else []
        history = history[-2 * previous_turn_limit:] if previous_turn_limit else []
        summary_metrics = llm_service.CallMetrics()
        summary = await summarizer.summarize(session.summary, evicted, settings, metrics=summary_metrics)
        promoted = session.anchors
        if evicted and anchors.is_anchor(evicted[0]['content']):
            promoted = [*session.anchors, (evicted[0]['content'], evicted[1]['content'])]
        tier, _ = resolve_tier(transcript=user, metadata=session.metadata)
        system = render_session_estimation_system(session.metadata, summary=summary,
                                                   anchors=promoted, tier=tier.value)
        estimation_metrics = llm_service.CallMetrics()
        metadata_metrics = llm_service.CallMetrics()
        try:
            validation_options = ({'_validation_origin': _validation_origin}
                                  if _validation_origin is not None else {})
            result = await llm_service.structured_estimate(user, settings, system=system, history=history,
                                                           metrics=estimation_metrics, **validation_options)
        except llm_service.EstimationError as error:
            # The per-call capability is attached only by the JSON-validation catch.
            if (type(error) is llm_service.EstimationError and error.status_code == 502
                    and error.detail == _INVALID_RESULT_DETAIL and error.__suppress_context__
                    and error.__cause__ is None and isinstance(error.__context__, ValidationError)
                    and _validation_origin is not None
                    and getattr(error, '_validation_origin', None) is _validation_origin):
                error._validation_stage = _INVALID_ESTIMATE_STAGE
            raise
        result_json = result.model_dump_json()
        extractor_input = json.dumps({'previous_metadata': session.metadata.model_dump(),
                                      'user': user, 'assistant': result.model_dump()}, ensure_ascii=False)
        extracted = await llm_service.estimate(extractor_input, settings,
            system=render_metadata_prompt(), json_output=True, metrics=metadata_metrics)
        try:
            metadata = ProjectMetadata.model_validate_json(extracted)
            if metadata.model_fields_set != set(ProjectMetadata.model_fields):
                raise _invalid_metadata_error()
        except ValidationError:
            raise _invalid_metadata_error() from None
        session.history.add_turn(user, result_json)
        session.metadata = metadata
        session.summary = summary
        session.anchors = promoted
        session.tier = tier.value
        session.committed_turns += 1
        if observation is not None:
            metrics = ([summary_metrics] if evicted else []) + [estimation_metrics, metadata_metrics]
            def total(attribute: str):
                values = [getattr(item, attribute) for item in metrics]
                return sum(values) if all(value is not None for value in values) else None
            observation.update(turn_index=session.committed_turns,
                messages_in_window=len(session.history.to_messages_list('')) - 1,
                anchors_count=len(session.anchors), summary_chars=len(session.summary),
                tokens_in=total('input_tokens'), tokens_out=total('output_tokens'),
                cost_usd=total('cost_usd'), cache_hit_kind='none', last_resolved_tier=session.tier)
        if evicted:
            _log_metrics('summary', summary_metrics)
        _log_metrics('estimation', estimation_metrics)
        _log_metrics('metadata', metadata_metrics)
        return {'result': result.model_dump(), 'prompt_version': 'v1', 'cached': False,
                'project_metadata': metadata.model_dump()}
