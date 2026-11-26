"""Session creation and atomic, structured conversational estimates."""

import json
import logging
from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.routing import APIRoute

from app.config import Settings, get_settings
from app.services import estimation, llm_service
from app.services.attachments import AttachmentError, enrich_transcript
from app.sessions import create_session, get_session

logger = logging.getLogger(__name__)


class ObservedRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()
        if self.path != '/sessions/{session_id}/estimate' or 'POST' not in self.methods:
            return handler

        async def observed(request: Request):
            started = perf_counter()
            session_id = request.path_params['session_id']
            # Only registered server-generated IDs are safe to log; the path is untrusted.
            observation = dict.fromkeys(('turn_index', 'session_id', 'enriched_transcript_chars',
                'attachments_total_chars', 'messages_in_window', 'anchors_count', 'summary_chars',
                'tokens_in', 'tokens_out', 'cost_usd', 'latency_ms', 'cache_hit_kind', 'last_resolved_tier'))
            observation['session_id'] = session_id if get_session(session_id) is not None else None
            request.state.turn_observation = observation
            try:
                return await handler(request)
            finally:
                observation['latency_ms'] = (perf_counter() - started) * 1000
                try:
                    logger.info('turn_observed %s', json.dumps(observation, separators=(',', ':')))
                except Exception:
                    # Instrumentation must not change the response or a committed turn.
                    pass
        return observed


router = APIRouter(prefix='/sessions', tags=['sessions'], route_class=ObservedRoute)


@router.post('')
async def create(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, str]:
    return {'session_id': create_session(settings.max_turns)}


@router.get('/{session_id}')
async def inspect_session(session_id: str) -> dict[str, str | int]:
    """Return the retained message-window size and turn limit, not conversation content."""
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail='Session not found.')
    async with session.lock:
        return {
            'session_id': session_id,
            # The helper includes one system message, which is not retained history.
            'history_window_message_count': len(session.history.to_messages_list('')) - 1,
            'max_turns': session.history.max_turns,
        }


@router.post('/{session_id}/estimate')
async def estimate_session(
    session_id: str,
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    transcript: Annotated[str, Form(min_length=1)],
    attachments: Annotated[list[UploadFile] | None, File()] = None,
) -> dict:
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail='Session not found.')
    if not transcript.strip():
        raise HTTPException(status_code=422, detail='Transcript must not be blank.')
    try:
        user = await enrich_transcript(transcript, attachments or [], observation=request.state.turn_observation)
    except AttachmentError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None

    try:
        return await estimation.estimate_conversational(session, user, settings,
                                                        observation=request.state.turn_observation)
    except llm_service.EstimationError as error:
        raise HTTPException(status_code=error.status_code, detail=error.detail) from None
