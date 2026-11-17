"""Session creation and atomic, structured conversational estimates."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.config import Settings, get_settings
from app.services import estimation, llm_service
from app.services.attachments import AttachmentError, enrich_transcript
from app.sessions import create_session, get_session

router = APIRouter(prefix='/sessions', tags=['sessions'])


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
        user = await enrich_transcript(transcript, attachments or [])
    except AttachmentError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None

    try:
        return await estimation.estimate_conversational(session, user, settings)
    except llm_service.EstimationError as error:
        raise HTTPException(status_code=error.status_code, detail=error.detail) from None
