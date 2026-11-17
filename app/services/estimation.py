"""Orchestrate an atomic conversational estimation turn."""

import json
import logging
import re

from pydantic import ValidationError

from app.config import Settings
from app.prompts.loader import render_metadata_prompt, render_session_estimation_system
from app.services import llm_service
from app.sessions import ProjectMetadata, Session


logger = logging.getLogger(__name__)


def _log_metrics(operation: str, metrics: llm_service.CallMetrics) -> None:
    # Provider values are untrusted; never log free-form text or control characters.
    model = metrics.model
    if model is not None and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,127}', model):
        model = None
    try:
        logger.info('provider_usage operation=%s model=%s input_tokens=%s output_tokens=%s '
                    'cost_usd=%s elapsed_seconds=%s', operation, model, metrics.input_tokens,
                    metrics.output_tokens, metrics.cost_usd, metrics.elapsed_seconds)
    except Exception:
        # Telemetry is best effort and must not invalidate an already committed turn.
        pass


async def estimate_conversational(session: Session, user: str, settings: Settings) -> dict:
    """Estimate and update history and metadata together after both calls succeed."""
    async with session.lock:
        system = render_session_estimation_system(session.metadata)
        history = session.history.to_messages_list(system)[1:]
        previous_turn_limit = session.history.max_turns - 1
        history = history[-2 * previous_turn_limit:] if previous_turn_limit else []
        estimation_metrics = llm_service.CallMetrics()
        metadata_metrics = llm_service.CallMetrics()
        result = await llm_service.structured_estimate(user, settings, system=system, history=history,
                                                       metrics=estimation_metrics)
        result_json = result.model_dump_json()
        extractor_input = json.dumps({'previous_metadata': session.metadata.model_dump(),
                                      'user': user, 'assistant': result.model_dump()}, ensure_ascii=False)
        extracted = await llm_service.estimate(extractor_input, settings,
            system=render_metadata_prompt(), json_output=True, metrics=metadata_metrics)
        try:
            metadata = ProjectMetadata.model_validate_json(extracted)
            if metadata.model_fields_set != set(ProjectMetadata.model_fields):
                raise llm_service.EstimationError(502, 'Estimation provider returned an invalid result.')
        except ValidationError:
            raise llm_service.EstimationError(502, 'Estimation provider returned an invalid result.') from None
        session.history.add_turn(user, result_json)
        session.metadata = metadata
        _log_metrics('estimation', estimation_metrics)
        _log_metrics('metadata', metadata_metrics)
        return {'result': result.model_dump(), 'prompt_version': 'v1', 'cached': False,
                'project_metadata': metadata.model_dump()}
